"""
Automated browser-ready surf forecast.

Data sources
------------
- NOAA GFS Wave at a fixed 0.25-degree model point
- A local direction-period wave-transfer matrix
- MSQ/BOM predicted interval tide data using the original notebook method

Public entry point
------------------
generate_final_plot(
    date_string="YYYYMMDD",
    cycle="00",
    output_path="docs/final_surf_forecast.svg",
)

The module creates only the requested final plot. It does not create
intermediate CSV files or figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import tempfile
import time
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import cfgrib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xarray as xr


#### PATHS #####################################################################

MODULE_DIRECTORY = Path(__file__).resolve().parent

TRANSFER_MATRIX_PATH = (
    MODULE_DIRECTORY
    / "wave_transfer_matrix.txt"
)


#### LOCATION CONFIGURATION ####################################################

SPOT_NAME = "Heron Island Surf Forecast"

SPOT_LAT = -23.348043530796062
SPOT_LON = 152.618531665867

# Known nearest GFS Wave grid point.
GFS_GRID_LAT = -23.25
GFS_GRID_LON = 152.50

LOCAL_TZ = "Australia/Brisbane"
LOCAL_TZ_INFO = ZoneInfo(LOCAL_TZ)

TIDE_STATION = "Heron Island"


#### FORECAST CONFIGURATION ####################################################

FORECAST_LENGTH_HOURS = 168
FORECAST_STEP_HOURS = 3

BBOX_PADDING_DEGREES = 0.02

REQUEST_TIMEOUT_SECONDS = 90
REQUEST_PAUSE_SECONDS = 0.10

TIDE_MATCH_TOLERANCE = pd.Timedelta("20min")


#### REMOTE DATA SOURCES #######################################################

NOMADS_WAVE_FILTER = (
    "https://nomads.ncep.noaa.gov/"
    "cgi-bin/filter_gfswave.pl"
)

# Original notebook tide-source settings.
CKAN_BASE = (
    "https://www.data.qld.gov.au/"
    "api/3/action"
)

DATASTORE_DUMP_BASE = (
    "https://www.data.qld.gov.au/"
    "datastore/dump"
)


#### PLOT CONFIGURATION ########################################################

MAX_DIRECTION_ANNOTATIONS = 20
QUALITY_COLOUR_MAP = "RdYlGn"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.25,
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    }
)


#### TRANSFER-MATRIX MAPPINGS ##################################################

ROW_MAPPING = {
    15: 0,
    30: 1,
    45: 2,
    60: 3,
    75: 4,
    90: 5,
    105: 6,
    120: 7,
    135: 8,
    150: 9,
    165: 10,
    180: 11,
    195: 12,
    210: 13,
    225: 14,
    240: 15,
    255: 16,
    270: 17,
    285: 18,
    300: 19,
    315: 20,
    330: 21,
    345: 22,
    360: 23,
}

COL_MAPPING = {
    24: 0,
    23: 1,
    22: 1,
    21: 2,
    20: 2,
    19: 3,
    18: 3,
    17: 4,
    16: 4,
    15: 5,
    14: 6,
    13: 7,
    12: 7,
    11: 8,
    10: 9,
    9: 10,
    8: 11,
    7: 13,
    6: 15,
    5: 16,
    4: 17,
}


#### QUALITY CONFIGURATION #####################################################

QUALITY_WEIGHTS = {
    "wave_height": 0.25,
    "wave_period": 0.20,
    "wave_direction": 0.15,
    "tide": 0.15,
    "wind_speed": 0.10,
    "wind_direction": 0.15,
}

QUALITY_COMPONENTS = (
    "wave_height",
    "wave_period",
    "wave_direction",
    "tide",
    "wind_speed",
    "wind_direction",
)

COMPASS_POINTS = np.array(
    [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ],
    dtype=object,
)


#### TIDE RESOURCE INFORMATION #################################################

@dataclass
class TideResource:
    station_name: str
    package_name: str
    resource_id: str
    resource_name: str
    year: int
    url: str | None


#### HTTP SESSION FOR GFS ######################################################

def create_http_session() -> requests.Session:
    """
    Create a reusable session with retries for GFS downloads.
    """
    retry_policy = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset(
            ["GET"]
        ),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_policy,
        pool_connections=4,
        pool_maxsize=4,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent": (
                "Heron Island automated surf forecast"
            ),
        }
    )

    return session


#### GENERAL HELPERS ###########################################################

def scalar(
    data_array: xr.DataArray,
) -> float:
    """
    Convert a scalar xarray value into a Python float.
    """
    return float(
        np.asarray(
            data_array.values
        ).squeeze()
    )


def longitude_to_180(
    longitude: float,
) -> float:
    """
    Convert longitude to the -180 to 180 convention.
    """
    return (
        (longitude + 180.0) % 360.0
    ) - 180.0


def forecast_hours() -> range:
    """
    Return all requested GFS forecast lead times.
    """
    return range(
        0,
        FORECAST_LENGTH_HOURS + 1,
        FORECAST_STEP_HOURS,
    )


def direction_to_compass(
    directions_deg,
) -> np.ndarray:
    """
    Convert bearings to 16-point compass labels.
    """
    directions = np.asarray(
        directions_deg,
        dtype=float,
    )

    labels = np.full(
        directions.shape,
        "",
        dtype=object,
    )

    valid = np.isfinite(
        directions
    )

    indices = (
        (
            directions[valid]
            + 11.25
        )
        // 22.5
    ).astype(int) % 16

    labels[valid] = COMPASS_POINTS[
        indices
    ]

    return labels


#### TRANSFER MATRIX ###########################################################

def load_transfer_matrix(
    path: Path = TRANSFER_MATRIX_PATH,
) -> np.ndarray:
    """
    Load and validate the 24-row by 18-column transfer matrix.
    """
    if not path.exists():
        raise FileNotFoundError(
            "Wave-transfer matrix was not found:\n"
            f"{path.resolve()}"
        )

    matrix = np.loadtxt(
        path,
        dtype=float,
    )

    expected_shape = (
        len(ROW_MAPPING),
        max(COL_MAPPING.values()) + 1,
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            f"Transfer matrix has shape {matrix.shape}, "
            f"but expected {expected_shape}."
        )

    if not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            "Transfer matrix contains NaN or infinite values."
        )

    if np.any(
        matrix < 0
    ):
        raise ValueError(
            "Transfer matrix contains negative values."
        )

    return matrix


def apply_transfer_matrix(
    forecast: pd.DataFrame,
    transfer_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Convert offshore GFS Hs to estimated nearshore Hs.
    """
    result = forecast.copy()

    wave_directions = result[
        "wave_direction_deg"
    ].to_numpy(
        dtype=float
    )

    wave_periods = result[
        "wave_period_s"
    ].to_numpy(
        dtype=float
    )

    mapped_directions = (
        np.floor(
            (
                wave_directions % 360.0
            )
            / 15.0
            + 0.5
        )
        * 15.0
    ).astype(int)

    mapped_directions[
        (
            mapped_directions == 0
        )
        | (
            mapped_directions >= 360
        )
    ] = 360

    mapped_periods = np.floor(
        wave_periods + 0.5
    ).astype(int)

    mapped_periods = np.clip(
        mapped_periods,
        4,
        24,
    )

    row_indices = np.fromiter(
        (
            ROW_MAPPING[
                direction
            ]
            for direction in mapped_directions
        ),
        dtype=int,
        count=len(mapped_directions),
    )

    column_indices = np.fromiter(
        (
            COL_MAPPING[
                period
            ]
            for period in mapped_periods
        ),
        dtype=int,
        count=len(mapped_periods),
    )

    coefficients = transfer_matrix[
        row_indices,
        column_indices,
    ]

    result[
        "matrix_direction_deg"
    ] = mapped_directions

    result[
        "matrix_period_s"
    ] = mapped_periods

    result[
        "transfer_coefficient"
    ] = coefficients

    result[
        "nearshore_wave_height_m"
    ] = (
        result[
            "wave_height_m"
        ].to_numpy(
            dtype=float
        )
        * coefficients
    )

    return result


#### GFS WAVE DOWNLOAD #########################################################

BOUNDING_BOX = {
    "leftlon": round(
        GFS_GRID_LON
        - BBOX_PADDING_DEGREES,
        4,
    ),
    "rightlon": round(
        GFS_GRID_LON
        + BBOX_PADDING_DEGREES,
        4,
    ),
    "bottomlat": round(
        GFS_GRID_LAT
        - BBOX_PADDING_DEGREES,
        4,
    ),
    "toplat": round(
        GFS_GRID_LAT
        + BBOX_PADDING_DEGREES,
        4,
    ),
}


def build_wave_url(
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> str:
    """
    Build a NOMADS URL for one small wave-and-wind subset.
    """
    filename = (
        f"gfswave.t{cycle}z."
        f"global.0p25."
        f"f{forecast_hour:03d}.grib2"
    )

    directory = (
        f"/gfs.{date_string}/"
        f"{cycle}/wave/gridded"
    )

    parameters = {
        "dir": directory,
        "file": filename,
        "var_HTSGW": "on",
        "var_PERPW": "on",
        "var_DIRPW": "on",
        "var_WIND": "on",
        "var_WDIR": "on",
        "lev_surface": "on",
        "subregion": "",
        **BOUNDING_BOX,
    }

    return (
        f"{NOMADS_WAVE_FILTER}?"
        f"{urlencode(parameters)}"
    )


def download_grib(
    session: requests.Session,
    url: str,
    destination: Path,
) -> None:
    """
    Download and validate one GFS Wave GRIB subset.
    """
    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    content = response.content

    if b"GRIB" not in content[:1000]:
        preview = content[
            :500
        ].decode(
            errors="ignore"
        )

        raise RuntimeError(
            "NOMADS did not return GRIB data.\n\n"
            f"URL:\n{url}\n\n"
            f"Response preview:\n{preview}"
        )

    destination.write_bytes(
        content
    )


def open_grib(
    path: Path,
) -> xr.Dataset:
    """
    Open the downloaded surface GRIB, merging cfgrib groups if needed.
    """
    backend_kwargs = {
        "indexpath": "",
        "filter_by_keys": {
            "typeOfLevel": "surface",
        },
    }

    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            decode_timedelta=True,
            backend_kwargs=backend_kwargs,
        )

    except Exception:
        datasets = cfgrib.open_datasets(
            path,
            backend_kwargs=backend_kwargs,
        )

        if not datasets:
            raise RuntimeError(
                f"No GRIB datasets could be opened from {path}."
            )

        try:
            merged = xr.merge(
                datasets,
                compat="override",
                join="outer",
                combine_attrs="override",
            )

            merged.load()

        finally:
            for dataset in datasets:
                dataset.close()

        return merged


def find_variable(
    dataset: xr.Dataset,
    exact_names: tuple[str, ...],
    metadata_terms: tuple[str, ...],
) -> xr.DataArray:
    """
    Find a GRIB variable by name and then by metadata.
    """
    for variable_name in exact_names:
        if variable_name in dataset.data_vars:
            return dataset[
                variable_name
            ]

    for _, data_array in dataset.data_vars.items():
        metadata = " ".join(
            str(
                data_array.attrs.get(
                    attribute,
                    "",
                )
            )
            for attribute in (
                "GRIB_shortName",
                "GRIB_name",
                "GRIB_cfName",
                "long_name",
                "standard_name",
            )
        ).lower()

        if any(
            term.lower() in metadata
            for term in metadata_terms
        ):
            return data_array

    raise KeyError(
        f"Could not find variable {exact_names}.\n"
        f"Available variables: {list(dataset.data_vars)}"
    )


def select_downloaded_grid_point(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Select the single point returned by the NOMADS subset.
    """
    latitude_name = (
        "latitude"
        if "latitude" in dataset.coords
        else "lat"
    )

    longitude_name = (
        "longitude"
        if "longitude" in dataset.coords
        else "lon"
    )

    latitude_values = np.atleast_1d(
        dataset[
            latitude_name
        ].values
    )

    longitude_values = np.atleast_1d(
        dataset[
            longitude_name
        ].values
    )

    if (
        latitude_values.size != 1
        or longitude_values.size != 1
    ):
        raise RuntimeError(
            "The GRIB subset contains more than one grid point.\n"
            f"Latitudes: {latitude_values}\n"
            f"Longitudes: {longitude_values}"
        )

    indexers: dict[str, int] = {}

    if latitude_name in dataset.dims:
        indexers[
            latitude_name
        ] = 0

    if longitude_name in dataset.dims:
        indexers[
            longitude_name
        ] = 0

    if indexers:
        return dataset.isel(
            indexers
        )

    return dataset


def get_valid_time(
    dataset: xr.Dataset,
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> pd.Timestamp:
    """
    Return the valid forecast time in UTC.
    """
    if "valid_time" in dataset.coords:
        valid_time = np.asarray(
            dataset[
                "valid_time"
            ].values
        ).squeeze()

        return pd.Timestamp(
            pd.to_datetime(
                valid_time,
                utc=True,
            )
        )

    run_time = pd.to_datetime(
        f"{date_string}{cycle}",
        format="%Y%m%d%H",
        utc=True,
    )

    return pd.Timestamp(
        run_time
        + pd.Timedelta(
            hours=forecast_hour
        )
    )


def read_forecast_record(
    grib_path: Path,
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> dict[str, object]:
    """
    Read one GFS wave-and-wind record.
    """
    dataset = open_grib(
        grib_path
    )

    try:
        point = select_downloaded_grid_point(
            dataset
        )

        wave_height = find_variable(
            point,
            exact_names=(
                "swh",
                "htsgw",
            ),
            metadata_terms=(
                "significant height of combined wind waves and swell",
                "significant wave height",
            ),
        )

        wave_period = find_variable(
            point,
            exact_names=(
                "perpw",
            ),
            metadata_terms=(
                "primary wave mean period",
                "primary wave period",
            ),
        )

        wave_direction = find_variable(
            point,
            exact_names=(
                "dirpw",
            ),
            metadata_terms=(
                "primary wave direction",
            ),
        )

        wind_speed = find_variable(
            point,
            exact_names=(
                "wind",
            ),
            metadata_terms=(
                "wind speed",
            ),
        )

        wind_direction = find_variable(
            point,
            exact_names=(
                "wdir",
            ),
            metadata_terms=(
                "wind direction",
            ),
        )

        latitude_name = (
            "latitude"
            if "latitude" in point.coords
            else "lat"
        )

        longitude_name = (
            "longitude"
            if "longitude" in point.coords
            else "lon"
        )

        downloaded_latitude = scalar(
            point[
                latitude_name
            ]
        )

        downloaded_longitude = scalar(
            point[
                longitude_name
            ]
        )

        if downloaded_longitude > 180:
            downloaded_longitude = (
                longitude_to_180(
                    downloaded_longitude
                )
            )

        if not np.isclose(
            downloaded_latitude,
            GFS_GRID_LAT,
            atol=0.01,
        ):
            raise RuntimeError(
                "Unexpected GFS latitude returned.\n"
                f"Expected: {GFS_GRID_LAT}\n"
                f"Received: {downloaded_latitude}"
            )

        if not np.isclose(
            downloaded_longitude,
            GFS_GRID_LON,
            atol=0.01,
        ):
            raise RuntimeError(
                "Unexpected GFS longitude returned.\n"
                f"Expected: {GFS_GRID_LON}\n"
                f"Received: {downloaded_longitude}"
            )

        wind_speed_ms = scalar(
            wind_speed
        )

        return {
            "forecast_hour": forecast_hour,
            "time_utc": get_valid_time(
                dataset=dataset,
                date_string=date_string,
                cycle=cycle,
                forecast_hour=forecast_hour,
            ),
            "wave_height_m": scalar(
                wave_height
            ),
            "wave_period_s": scalar(
                wave_period
            ),
            "wave_direction_deg": (
                scalar(
                    wave_direction
                )
                % 360.0
            ),
            "wind_speed_ms": wind_speed_ms,
            "wind_speed_knots": (
                wind_speed_ms
                * 1.943844
            ),
            "wind_direction_deg": (
                scalar(
                    wind_direction
                )
                % 360.0
            ),
            "grid_latitude": downloaded_latitude,
            "grid_longitude": downloaded_longitude,
        }

    finally:
        dataset.close()


def collect_forecast(
    transfer_matrix: np.ndarray,
    date_string: str,
    cycle: str,
) -> pd.DataFrame:
    """
    Download and process the full selected GFS Wave cycle.
    """
    if (
        len(date_string) != 8
        or not date_string.isdigit()
    ):
        raise ValueError(
            "date_string must use YYYYMMDD format."
        )

    if cycle not in {
        "00",
        "06",
        "12",
        "18",
    }:
        raise ValueError(
            "cycle must be 00, 06, 12 or 18."
        )

    records: list[
        dict[str, object]
    ] = []

    with create_http_session() as session:
        with tempfile.TemporaryDirectory() as temporary_directory:
            grib_path = (
                Path(
                    temporary_directory
                )
                / "gfswave_point.grib2"
            )

            for forecast_hour in forecast_hours():
                download_grib(
                    session=session,
                    url=build_wave_url(
                        date_string=date_string,
                        cycle=cycle,
                        forecast_hour=forecast_hour,
                    ),
                    destination=grib_path,
                )

                records.append(
                    read_forecast_record(
                        grib_path=grib_path,
                        date_string=date_string,
                        cycle=cycle,
                        forecast_hour=forecast_hour,
                    )
                )

                grib_path.unlink(
                    missing_ok=True
                )

                if REQUEST_PAUSE_SECONDS > 0:
                    time.sleep(
                        REQUEST_PAUSE_SECONDS
                    )

    if not records:
        raise RuntimeError(
            "No GFS forecast records were collected."
        )

    forecast = (
        pd.DataFrame(
            records
        )
        .sort_values(
            "time_utc"
        )
        .drop_duplicates(
            subset=[
                "time_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    expected_records = (
        FORECAST_LENGTH_HOURS
        // FORECAST_STEP_HOURS
        + 1
    )

    if len(forecast) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} GFS records, "
            f"but received {len(forecast)}."
        )

    forecast = apply_transfer_matrix(
        forecast=forecast,
        transfer_matrix=transfer_matrix,
    )

    forecast[
        "spot_name"
    ] = SPOT_NAME

    forecast[
        "spot_latitude"
    ] = SPOT_LAT

    forecast[
        "spot_longitude"
    ] = SPOT_LON

    forecast[
        "run_date"
    ] = date_string

    forecast[
        "run_cycle"
    ] = cycle

    return forecast


#### ORIGINAL NOTEBOOK TIDE METHOD #############################################

def ckan_get(
    action: str,
    params: dict,
    timeout: int = 45,
) -> dict:
    """
    Call the Queensland Government CKAN API.

    This preserves the original notebook method.
    """
    url = f"{CKAN_BASE}/{action}"

    response = requests.get(
        url,
        params=params,
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):
        raise RuntimeError(
            f"CKAN API call failed: {payload}"
        )

    return payload[
        "result"
    ]


def search_msq_tide_package(
    station_name: str,
) -> dict:
    """
    Search for the station's predicted interval tide dataset.

    Important:
        The broad fallback below is retained exactly from the working
        notebook. It is intentionally less strict than the failed GitHub
        version.
    """
    queries = [
        f'"{station_name}" "predicted interval data" tide',
        f'"{station_name}" "tide gauge" "predicted interval"',
        f"{station_name} tide gauge predicted interval data",
    ]

    station_lower = station_name.lower()

    for query in queries:
        result = ckan_get(
            "package_search",
            params={
                "q": query,
                "rows": 10,
            },
        )

        packages = result.get(
            "results",
            [],
        )

        if not packages:
            continue

        # Preferred exact match.
        for package in packages:
            title = package.get(
                "title",
                "",
            ).lower()

            name = package.get(
                "name",
                "",
            ).lower()

            if (
                station_lower in title
                and "predicted" in title
                and "interval" in title
            ):
                return package

            if (
                station_lower.replace(
                    " ",
                    "-",
                )
                in name
                and "predicted" in title
            ):
                return package

        # Broad fallback from the working notebook.
        for package in packages:
            package_text = (
                f"{package.get('title', '')} "
                f"{package.get('notes', '')}"
            ).lower()

            if (
                "tide" in package_text
                and "predicted" in package_text
            ):
                return package

    raise RuntimeError(
        "Could not find an MSQ predicted interval tide package "
        f"for {station_name}."
    )


def find_msq_predicted_interval_resource(
    station_name: str,
    year: int,
) -> TideResource:
    """
    Find the annual predicted interval CSV/API resource.

    This follows the original notebook selection logic.
    """
    package = search_msq_tide_package(
        station_name
    )

    resources = package.get(
        "resources",
        [],
    )

    year_text = str(
        year
    )

    candidates = []

    for resource in resources:
        name = str(
            resource.get(
                "name",
                "",
            )
        )

        description = str(
            resource.get(
                "description",
                "",
            )
        )

        resource_format = str(
            resource.get(
                "format",
                "",
            )
        )

        resource_id = resource.get(
            "id"
        )

        resource_url = str(
            resource.get(
                "url",
                "",
            )
        )

        resource_text = (
            f"{name} "
            f"{description} "
            f"{resource_format} "
            f"{resource_url}"
        ).lower()

        if not resource_id:
            continue

        if year_text not in resource_text:
            continue

        if "predicted" not in resource_text:
            continue

        if "interval" not in resource_text:
            continue

        if (
            "csv" not in resource_text
            and "api" not in resource_text
        ):
            continue

        candidates.append(
            resource
        )

    if not candidates:
        available_resources = "\n".join(
            f"  - {resource.get('name', '')}"
            for resource in resources
        )

        raise RuntimeError(
            f"Could not find a {year} predicted interval CSV/API "
            f"resource for {station_name}.\n\n"
            f"Available resources:\n"
            f"{available_resources}"
        )

    resource = candidates[
        0
    ]

    return TideResource(
        station_name=station_name,
        package_name=package.get(
            "name",
            "",
        ),
        resource_id=resource[
            "id"
        ],
        resource_name=resource.get(
            "name",
            "",
        ),
        year=year,
        url=resource.get(
            "url"
        ),
    )


def download_msq_tide_csv(
    resource: TideResource,
) -> pd.DataFrame:
    """
    Download an MSQ predicted interval tide CSV.

    The datastore dump is attempted first, followed by the catalogue URL.
    """
    urls_to_try = [
        (
            f"{DATASTORE_DUMP_BASE}/"
            f"{resource.resource_id}?format=csv"
        ),
    ]

    if resource.url:
        urls_to_try.append(
            resource.url
        )

    errors = []

    for url in urls_to_try:
        try:
            response = requests.get(
                url,
                timeout=90,
            )

            response.raise_for_status()

            text = response.text

            if (
                "Date" not in text
                or "Time" not in text
                or "Reading" not in text
            ):
                raise RuntimeError(
                    "Response did not look like the expected MSQ "
                    "tide CSV.\n\n"
                    f"First 500 characters:\n{text[:500]}"
                )

            return pd.read_csv(
                StringIO(
                    text
                )
            )

        except Exception as error:
            errors.append(
                f"{url}\n{error}"
            )

    raise RuntimeError(
        "Failed to download MSQ tide CSV from all endpoints:\n\n"
        + "\n\n".join(
            errors
        )
    )


def parse_msq_tide_dataframe(
    raw_df: pd.DataFrame,
    station_name: str,
) -> pd.DataFrame:
    """
    Parse the predicted interval tide CSV.

    Expected columns:
        Date
        Time
        Reading
    """
    tide = raw_df.copy()

    tide.columns = [
        str(
            column
        ).strip()
        for column in tide.columns
    ]

    required_columns = {
        "Date",
        "Time",
        "Reading",
    }

    missing_columns = (
        required_columns
        - set(
            tide.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"Missing expected MSQ columns: {missing_columns}.\n"
            f"Columns found: {list(tide.columns)}"
        )

    datetime_text = (
        tide[
            "Date"
        ]
        .astype(str)
        .str.strip()
        + " "
        + tide[
            "Time"
        ]
        .astype(str)
        .str.strip()
    )

    tide[
        "valid_time_local"
    ] = pd.to_datetime(
        datetime_text,
        dayfirst=True,
        errors="coerce",
    )

    tide[
        "tide_height_m"
    ] = pd.to_numeric(
        tide[
            "Reading"
        ],
        errors="coerce",
    )

    tide = tide.dropna(
        subset=[
            "valid_time_local",
            "tide_height_m",
        ]
    ).copy()

    if tide.empty:
        raise RuntimeError(
            "The MSQ tide CSV contained no valid tide records."
        )

    if tide[
        "valid_time_local"
    ].dt.tz is None:
        tide[
            "valid_time_local"
        ] = (
            tide[
                "valid_time_local"
            ]
            .dt.tz_localize(
                LOCAL_TZ
            )
        )

    tide[
        "station_name"
    ] = station_name

    tide[
        "tide_source"
    ] = (
        "MSQ Open Data predicted interval, BOM-produced"
    )

    tide[
        "tide_method"
    ] = (
        "msq_10min_predicted_interval"
    )

    return (
        tide[
            [
                "station_name",
                "valid_time_local",
                "tide_height_m",
                "tide_source",
                "tide_method",
            ]
        ]
        .sort_values(
            "valid_time_local"
        )
        .reset_index(
            drop=True
        )
    )


def get_msq_tide_for_wave_forecast(
    forecast: pd.DataFrame,
    station_name: str = TIDE_STATION,
) -> pd.DataFrame:
    """
    Download predicted tide data covering forecast["time_utc"].

    This is the original notebook workflow, adapted only to return data
    to the web pipeline instead of writing a CSV or tide-only plot.
    """
    if "time_utc" not in forecast.columns:
        raise KeyError(
            "forecast must contain a 'time_utc' column."
        )

    forecast_times_utc = pd.to_datetime(
        forecast[
            "time_utc"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    if forecast_times_utc.empty:
        raise ValueError(
            "forecast['time_utc'] contains no valid timestamps."
        )

    start_time = (
        forecast_times_utc.min()
        .tz_convert(
            LOCAL_TZ
        )
        .floor(
            "10min"
        )
    )

    end_time = (
        forecast_times_utc.max()
        .tz_convert(
            LOCAL_TZ
        )
        .ceil(
            "10min"
        )
    )

    years_needed = sorted(
        {
            start_time.year,
            end_time.year,
        }
    )

    tide_frames = []

    for year in years_needed:
        resource = find_msq_predicted_interval_resource(
            station_name=station_name,
            year=year,
        )

        raw_tide = download_msq_tide_csv(
            resource
        )

        parsed_tide = parse_msq_tide_dataframe(
            raw_df=raw_tide,
            station_name=station_name,
        )

        parsed_tide[
            "resource_id"
        ] = resource.resource_id

        parsed_tide[
            "resource_name"
        ] = resource.resource_name

        parsed_tide[
            "resource_year"
        ] = year

        tide_frames.append(
            parsed_tide
        )

    if not tide_frames:
        raise RuntimeError(
            "No tide data were downloaded."
        )

    tide_df = pd.concat(
        tide_frames,
        ignore_index=True,
    )

    tide_df = (
        tide_df
        .drop_duplicates(
            subset=[
                "valid_time_local"
            ]
        )
        .sort_values(
            "valid_time_local"
        )
    )

    tide_df = tide_df.loc[
        (
            tide_df[
                "valid_time_local"
            ]
            >= start_time
        )
        & (
            tide_df[
                "valid_time_local"
            ]
            <= end_time
        )
    ].copy()

    if tide_df.empty:
        raise RuntimeError(
            "Downloaded MSQ tide data successfully, but no rows "
            "matched the forecast window.\n\n"
            f"Start: {start_time}\n"
            f"End: {end_time}"
        )

    tide_df[
        "forecast_hour"
    ] = (
        tide_df[
            "valid_time_local"
        ]
        - start_time
    ).dt.total_seconds() / 3600.0

    tide_df[
        "time_utc"
    ] = (
        tide_df[
            "valid_time_local"
        ]
        .dt.tz_convert(
            "UTC"
        )
    )

    return (
        tide_df[
            [
                "station_name",
                "valid_time_local",
                "time_utc",
                "forecast_hour",
                "tide_height_m",
                "tide_source",
                "tide_method",
                "resource_id",
                "resource_name",
                "resource_year",
            ]
        ]
        .sort_values(
            "valid_time_local"
        )
        .reset_index(
            drop=True
        )
    )


#### QUALITY SCORING ###########################################################

def circular_distance_deg(
    angle_a,
    angle_b,
) -> np.ndarray:
    """
    Return the smallest angular distance between bearings.
    """
    angle_a = np.asarray(
        angle_a,
        dtype=float,
    )

    angle_b = np.asarray(
        angle_b,
        dtype=float,
    )

    return np.abs(
        (
            angle_a
            - angle_b
            + 180.0
        )
        % 360.0
        - 180.0
    )


def direction_in_sector(
    direction_deg,
    sector_start: float,
    sector_end: float,
) -> np.ndarray:
    """
    Test whether bearings lie within a circular sector.
    """
    direction = np.mod(
        np.asarray(
            direction_deg,
            dtype=float,
        ),
        360.0,
    )

    sector_start %= 360.0
    sector_end %= 360.0

    if sector_start <= sector_end:
        return (
            (direction >= sector_start)
            & (direction <= sector_end)
        )

    return (
        (direction >= sector_start)
        | (direction <= sector_end)
    )


def score_wave_height(
    height_m,
) -> np.ndarray:
    """
    Score transformed wave height from 0 to 1.
    """
    height = np.asarray(
        height_m,
        dtype=float,
    )

    scores = np.where(
        height < 1.0,
        0.0,
        np.where(
            height < 2.0,
            (
                height
                - 1.0
            )
            * 0.7,
            0.7
            + (
                height
                - 2.0
            )
            / 1.5
            * 0.3,
        ),
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    scores[
        ~np.isfinite(
            height
        )
    ] = np.nan

    return scores


def score_wave_period(
    period_s,
) -> np.ndarray:
    """
    Score primary wave period from 0 to 1.
    """
    period = np.asarray(
        period_s,
        dtype=float,
    )

    scores = np.where(
        period < 6.0,
        0.0,
        np.where(
            period < 10.0,
            (
                period
                - 6.0
            )
            / 4.0
            * 0.7,
            0.7
            + (
                period
                - 10.0
            )
            / 6.0
            * 0.3,
        ),
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    scores[
        ~np.isfinite(
            period
        )
    ] = np.nan

    return scores


def score_wave_direction(
    direction_deg,
    good_start: float = 145.0,
    good_end: float = 45.0,
) -> np.ndarray:
    """
    Score wave direction from 0 to 1.
    """
    direction = np.asarray(
        direction_deg,
        dtype=float,
    )

    inside_sector = direction_in_sector(
        direction,
        good_start,
        good_end,
    )

    nearest_edge_distance = np.minimum(
        circular_distance_deg(
            direction,
            good_start,
        ),
        circular_distance_deg(
            direction,
            good_end,
        ),
    )

    scores = np.where(
        inside_sector,
        1.0,
        1.0
        - nearest_edge_distance
        / 90.0,
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    scores[
        ~np.isfinite(
            direction
        )
    ] = np.nan

    return scores


def score_tide(
    tide_m,
) -> np.ndarray:
    """
    Score tide height from 0 to 1.
    """
    tide = np.asarray(
        tide_m,
        dtype=float,
    )

    scores = np.where(
        tide <= 1.0,
        0.0,
        np.where(
            tide < 2.0,
            (
                tide
                - 1.0
            )
            * 0.4,
            0.7
            + (
                tide
                - 2.0
            )
            / 1.5
            * 0.3,
        ),
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    scores[
        ~np.isfinite(
            tide
        )
    ] = np.nan

    return scores


def score_wind_speed(
    wind_speed_knots,
) -> np.ndarray:
    """
    Score wind speed from 0 to 1.
    """
    speed = np.asarray(
        wind_speed_knots,
        dtype=float,
    )

    scores = np.select(
        [
            speed <= 5.0,
            speed <= 10.0,
            speed <= 15.0,
            speed <= 20.0,
        ],
        [
            1.0,
            0.8,
            0.5,
            0.2,
        ],
        default=0.0,
    ).astype(float)

    scores[
        ~np.isfinite(
            speed
        )
    ] = np.nan

    return scores


def score_wind_direction(
    direction_deg,
) -> np.ndarray:
    """
    Score wind direction from 0 to 1.
    """
    direction = np.asarray(
        direction_deg,
        dtype=float,
    )

    good_sector = direction_in_sector(
        direction,
        135.0,
        225.0,
    )

    distance_from_north = (
        circular_distance_deg(
            direction,
            0.0,
        )
    )

    distance_from_south = (
        circular_distance_deg(
            direction,
            180.0,
        )
    )

    scores = np.where(
        good_sector,
        1.0,
        np.where(
            distance_from_north <= 45.0,
            0.0,
            1.0
            - distance_from_south
            / 135.0,
        ),
    )

    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    scores[
        ~np.isfinite(
            direction
        )
    ] = np.nan

    return scores


def rating_from_score(
    scores,
) -> np.ndarray:
    """
    Convert 0-100 scores to descriptive ratings.
    """
    values = np.asarray(
        scores,
        dtype=float,
    )

    ratings = np.full(
        values.shape,
        "No rating",
        dtype=object,
    )

    valid = np.isfinite(
        values
    )

    ratings[
        valid
    ] = np.select(
        [
            values[valid] < 15.0,
            values[valid] < 30.0,
            values[valid] < 45.0,
            values[valid] < 60.0,
            values[valid] < 75.0,
            values[valid] < 90.0,
        ],
        [
            "Not working",
            "Poor",
            "Okay",
            "Not bad",
            "Pretty good",
            "Very good",
        ],
        default="Pumping",
    )

    return ratings


def prepare_quality_inputs(
    forecast: pd.DataFrame,
    tide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match every GFS forecast time to the nearest 10-minute tide value.
    """
    forecast_data = forecast.copy()

    forecast_data[
        "time_utc"
    ] = pd.to_datetime(
        forecast_data[
            "time_utc"
        ],
        utc=True,
        errors="coerce",
    )

    numeric_forecast_columns = [
        "nearshore_wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wind_speed_knots",
        "wind_direction_deg",
    ]

    for column in numeric_forecast_columns:
        forecast_data[
            column
        ] = pd.to_numeric(
            forecast_data[
                column
            ],
            errors="coerce",
        )

    forecast_data = (
        forecast_data
        .dropna(
            subset=[
                "time_utc",
                *numeric_forecast_columns,
            ]
        )
        .sort_values(
            "time_utc"
        )
        .drop_duplicates(
            subset=[
                "time_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    tide_data = pd.DataFrame(
        {
            "time_utc": pd.to_datetime(
                tide[
                    "time_utc"
                ],
                utc=True,
                errors="coerce",
            ),
            "tide_height_m": pd.to_numeric(
                tide[
                    "tide_height_m"
                ],
                errors="coerce",
            ),
        }
    )

    tide_data = (
        tide_data
        .dropna(
            subset=[
                "time_utc",
                "tide_height_m",
            ]
        )
        .sort_values(
            "time_utc"
        )
        .drop_duplicates(
            subset=[
                "time_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )
    # pandas 3 may preserve different datetime resolutions depending on
    # the original data source. merge_asof requires identical key dtypes.
    utc_nanosecond_dtype = pd.DatetimeTZDtype(
        unit="ns",
        tz="UTC",
    )

    forecast_data["time_utc"] = (
        forecast_data["time_utc"]
        .astype(utc_nanosecond_dtype)
    )

    tide_data["time_utc"] = (
        tide_data["time_utc"]
        .astype(utc_nanosecond_dtype)
    )

    combined = pd.merge_asof(
        forecast_data,
        tide_data,
        on="time_utc",
        direction="nearest",
        tolerance=TIDE_MATCH_TOLERANCE,
    )

    missing_tide_count = int(
        combined[
            "tide_height_m"
        ]
        .isna()
        .sum()
    )

    if missing_tide_count:
        raise RuntimeError(
            f"No nearby tide value was found for "
            f"{missing_tide_count} GFS forecast times."
        )

    combined[
        "time_local"
    ] = (
        combined[
            "time_utc"
        ]
        .dt.tz_convert(
            LOCAL_TZ
        )
    )

    return combined


def score_forecast_times(
    forecast: pd.DataFrame,
    tide: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Calculate vectorised surf-quality scores.
    """
    if weights is None:
        weights = QUALITY_WEIGHTS

    missing_weights = (
        set(
            QUALITY_COMPONENTS
        )
        - set(
            weights
        )
    )

    if missing_weights:
        raise KeyError(
            f"Missing quality weights: "
            f"{sorted(missing_weights)}"
        )

    weight_values = np.array(
        [
            weights[
                name
            ]
            for name in QUALITY_COMPONENTS
        ],
        dtype=float,
    )

    if (
        not np.all(
            np.isfinite(
                weight_values
            )
        )
        or np.any(
            weight_values < 0
        )
        or weight_values.sum() <= 0
    ):
        raise ValueError(
            "Quality weights must be finite, non-negative "
            "and have a positive total."
        )

    result = prepare_quality_inputs(
        forecast=forecast,
        tide=tide,
    )

    component_scores = np.column_stack(
        [
            score_wave_height(
                result[
                    "nearshore_wave_height_m"
                ]
            ),
            score_wave_period(
                result[
                    "wave_period_s"
                ]
            ),
            score_wave_direction(
                result[
                    "wave_direction_deg"
                ]
            ),
            score_tide(
                result[
                    "tide_height_m"
                ]
            ),
            score_wind_speed(
                result[
                    "wind_speed_knots"
                ]
            ),
            score_wind_direction(
                result[
                    "wind_direction_deg"
                ]
            ),
        ]
    )

    valid_rows = np.all(
        np.isfinite(
            component_scores
        ),
        axis=1,
    )

    quality_scores = np.full(
        len(
            result
        ),
        np.nan,
        dtype=float,
    )

    quality_scores[
        valid_rows
    ] = (
        component_scores[
            valid_rows
        ]
        @ weight_values
        / weight_values.sum()
        * 100.0
    )

    result[
        "quality_score"
    ] = quality_scores

    result[
        "rating"
    ] = rating_from_score(
        quality_scores
    )

    result[
        "wave_direction_compass"
    ] = direction_to_compass(
        result[
            "wave_direction_deg"
        ]
    )

    result[
        "wind_direction_compass"
    ] = direction_to_compass(
        result[
            "wind_direction_deg"
        ]
    )

    return result


#### FINAL PLOT DATA ###########################################################

def prepare_forecast_plot_data(
    quality_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare aligned wave, wind, tide and rating data for plotting.
    """
    required_columns = {
        "time_local",
        "nearshore_wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wave_direction_compass",
        "wind_speed_knots",
        "wind_direction_deg",
        "wind_direction_compass",
        "tide_height_m",
        "quality_score",
        "rating",
    }

    missing_columns = (
        required_columns
        - set(
            quality_forecast.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "Quality forecast is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = quality_forecast.copy()

    plot_times = pd.to_datetime(
        data[
            "time_local"
        ],
        errors="coerce",
    )

    if plot_times.dt.tz is None:
        plot_times = (
            plot_times
            .dt.tz_localize(
                LOCAL_TZ
            )
        )

    else:
        plot_times = (
            plot_times
            .dt.tz_convert(
                LOCAL_TZ
            )
        )

    data[
        "plot_time"
    ] = plot_times

    numeric_columns = [
        "nearshore_wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wind_speed_knots",
        "wind_direction_deg",
        "tide_height_m",
        "quality_score",
    ]

    for column in numeric_columns:
        data[
            column
        ] = pd.to_numeric(
            data[
                column
            ],
            errors="coerce",
        )

    data = (
        data
        .dropna(
            subset=[
                "plot_time",
                *numeric_columns,
            ]
        )
        .sort_values(
            "plot_time"
        )
        .drop_duplicates(
            subset=[
                "plot_time"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if data.empty:
        raise RuntimeError(
            "No valid forecast values are available for plotting."
        )

    return data


def prepare_tide_plot_data(
    tide: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Prepare the original 10-minute tide series for plotting.
    """
    data = tide.copy()

    data[
        "plot_time"
    ] = (
        pd.to_datetime(
            data[
                "time_utc"
            ],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(
            LOCAL_TZ
        )
    )

    data[
        "tide_height_m"
    ] = pd.to_numeric(
        data[
            "tide_height_m"
        ],
        errors="coerce",
    )

    buffer = pd.Timedelta(
        hours=1
    )

    data = (
        data
        .dropna(
            subset=[
                "plot_time",
                "tide_height_m",
            ]
        )
        .loc[
            lambda frame:
            (
                frame[
                    "plot_time"
                ]
                >= start_time - buffer
            )
            & (
                frame[
                    "plot_time"
                ]
                <= end_time + buffer
            )
        ]
        .sort_values(
            "plot_time"
        )
        .drop_duplicates(
            subset=[
                "plot_time"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if data.empty:
        raise RuntimeError(
            "No tide values overlap the forecast plot period."
        )

    return data


#### PLOT HELPERS ##############################################################

def calculate_time_edges(
    times: pd.Series,
) -> pd.DatetimeIndex:
    """
    Calculate interval edges around each GFS forecast timestamp.
    """
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(
            times
        )
    )

    if len(
        timestamps
    ) == 1:
        half_step = pd.Timedelta(
            hours=FORECAST_STEP_HOURS / 2
        )

        return pd.DatetimeIndex(
            [
                timestamps[
                    0
                ]
                - half_step,
                timestamps[
                    0
                ]
                + half_step,
            ]
        )

    midpoints = (
        timestamps[
            :-1
        ]
        + (
            timestamps[
                1:
            ]
            - timestamps[
                :-1
            ]
        )
        / 2
    )

    first_edge = (
        timestamps[
            0
        ]
        - (
            timestamps[
                1
            ]
            - timestamps[
                0
            ]
        )
        / 2
    )

    last_edge = (
        timestamps[
            -1
        ]
        + (
            timestamps[
                -1
            ]
            - timestamps[
                -2
            ]
        )
        / 2
    )

    return pd.DatetimeIndex(
        [
            first_edge,
            *midpoints,
            last_edge,
        ]
    )


def calculate_bar_width_days(
    times: pd.Series,
) -> float:
    """
    Calculate a suitable bar width from the forecast interval.
    """
    time_numbers = mdates.date2num(
        pd.to_datetime(
            times
        )
    )

    if len(
        time_numbers
    ) < 2:
        return (
            FORECAST_STEP_HOURS
            / 24.0
            * 0.74
        )

    return float(
        np.median(
            np.diff(
                time_numbers
            )
        )
        * 0.74
    )


def choose_annotation_step(
    number_of_points: int,
) -> int:
    """
    Thin direction annotations to avoid overlaps.
    """
    return max(
        1,
        int(
            np.ceil(
                number_of_points
                / MAX_DIRECTION_ANNOTATIONS
            )
        ),
    )


def plot_quality_strip(
    axis: plt.Axes,
    times: pd.Series,
    quality_scores: pd.Series,
    date_string: str,
    cycle: str,
) -> None:
    """
    Plot a red-yellow-green quality strip.
    """
    scores = np.asarray(
        quality_scores,
        dtype=float,
    )

    time_edges = calculate_time_edges(
        times
    )

    numeric_edges = mdates.date2num(
        time_edges.to_pydatetime()
    )

    axis.pcolormesh(
        numeric_edges,
        np.array(
            [
                0.0,
                1.0,
            ]
        ),
        scores.reshape(
            1,
            -1,
        ),
        cmap=plt.get_cmap(
            QUALITY_COLOUR_MAP
        ),
        norm=Normalize(
            vmin=0.0,
            vmax=100.0,
            clip=True,
        ),
        shading="flat",
        rasterized=False,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_yticks([])

    axis.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    axis.set_ylabel(
        "Quality",
        rotation=0,
        ha="right",
        va="center",
        labelpad=26,
    )

    axis.set_title(
        "Surf quality · red = low · yellow = moderate · green = high",
        loc="left",
        fontsize=9,
        fontweight="normal",
        pad=4,
    )

    axis.set_title(
        f"GFS Wave run {date_string} {cycle}Z",
        loc="right",
        fontsize=9,
        fontweight="normal",
        pad=4,
    )

    for spine in axis.spines.values():
        spine.set_visible(
            False
        )


def add_direction_annotations(
    axis: plt.Axes,
    times: pd.Series,
    directions_from_deg: pd.Series,
    compass_labels: pd.Series,
    annotation_step: int,
) -> None:
    """
    Add non-overlapping travel arrows and source-direction labels.
    """
    transform = blended_transform_factory(
        axis.transData,
        axis.transAxes,
    )

    for index in range(
        0,
        len(
            times
        ),
        annotation_step,
    ):
        direction_from = float(
            directions_from_deg.iloc[
                index
            ]
        )

        if not np.isfinite(
            direction_from
        ):
            continue

        travel_direction = (
            direction_from
            + 180.0
        ) % 360.0

        axis.text(
            times.iloc[
                index
            ],
            0.91,
            "↑",
            transform=transform,
            rotation=(
                -travel_direction
            ),
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            clip_on=True,
            zorder=8,
        )

        axis.text(
            times.iloc[
                index
            ],
            0.79,
            str(
                compass_labels.iloc[
                    index
                ]
            ),
            transform=transform,
            ha="center",
            va="center",
            fontsize=7,
            clip_on=True,
            zorder=8,
        )


def style_primary_axis(
    axis: plt.Axes,
) -> None:
    """
    Apply consistent publication-style formatting.
    """
    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
        zorder=0,
    )

    axis.spines[
        "top"
    ].set_visible(
        False
    )

    axis.spines[
        "right"
    ].set_visible(
        False
    )

    axis.tick_params(
        direction="out",
        length=4,
    )

    axis.margins(
        x=0
    )


def add_day_divisions(
    axes: tuple[
        plt.Axes,
        ...,
    ],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> None:
    """
    Add subtle local-midnight separators.
    """
    for day in pd.date_range(
        start=start_time.floor(
            "D"
        ),
        end=end_time.ceil(
            "D"
        ),
        freq="1D",
    ):
        for axis in axes:
            axis.axvline(
                day,
                linewidth=0.8,
                alpha=0.15,
                zorder=1,
            )


#### FINAL PUBLICATION PLOT ####################################################

def plot_publication_surf_forecast(
    forecast_data: pd.DataFrame,
    tide: pd.DataFrame,
    date_string: str,
    cycle: str,
    output_path: Path,
) -> plt.Figure:
    """
    Create the final browser-ready surf forecast.

    Layout:
        Quality strip
        Waves: transformed height, primary period and direction
        Wind: speed and direction
        Full-resolution tide
    """
    tide_data = prepare_tide_plot_data(
        tide=tide,
        start_time=forecast_data[
            "plot_time"
        ].min(),
        end_time=forecast_data[
            "plot_time"
        ].max(),
    )

    times = forecast_data[
        "plot_time"
    ]

    time_edges = calculate_time_edges(
        times
    )

    bar_width = calculate_bar_width_days(
        times
    )

    annotation_step = choose_annotation_step(
        len(
            forecast_data
        )
    )

    figure = plt.figure(
        figsize=(
            18,
            10,
        ),
        layout="constrained",
        facecolor="white",
    )

    grid = figure.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[
            0.34,
            3.5,
            2.15,
            1.85,
        ],
        hspace=0.04,
    )

    quality_axis = figure.add_subplot(
        grid[
            0
        ]
    )

    wave_axis = figure.add_subplot(
        grid[
            1
        ],
        sharex=quality_axis,
    )

    wind_axis = figure.add_subplot(
        grid[
            2
        ],
        sharex=quality_axis,
    )

    tide_axis = figure.add_subplot(
        grid[
            3
        ],
        sharex=quality_axis,
    )

    figure.suptitle(
        f"{SPOT_NAME} — seven-day surf forecast",
        x=0.065,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )

    plot_quality_strip(
        axis=quality_axis,
        times=times,
        quality_scores=forecast_data[
            "quality_score"
        ],
        date_string=date_string,
        cycle=cycle,
    )

    #### WAVES #################################################################

    wave_height = forecast_data[
        "nearshore_wave_height_m"
    ]

    wave_period = forecast_data[
        "wave_period_s"
    ]

    maximum_wave_height = max(
        float(
            wave_height.max()
        ),
        0.5,
    )

    wave_axis.bar(
        times,
        wave_height,
        width=bar_width,
        alpha=0.72,
        label="Nearshore wave height",
        zorder=3,
    )

    wave_axis.plot(
        times,
        wave_height,
        linewidth=1.25,
        zorder=4,
    )

    wave_axis.set_ylim(
        0,
        maximum_wave_height
        * 1.65,
    )

    wave_axis.set_ylabel(
        "Wave height (m)"
    )

    wave_axis.set_title(
        "Waves",
        loc="left",
        pad=7,
    )

    style_primary_axis(
        wave_axis
    )

    period_axis = (
        wave_axis.twinx()
    )

    maximum_period = max(
        float(
            wave_period.max()
        ),
        5.0,
    )

    period_axis.plot(
        times,
        wave_period,
        marker="o",
        markersize=3.2,
        linewidth=1.8,
        label="Primary wave period",
        zorder=5,
    )

    period_axis.set_ylim(
        0,
        maximum_period
        * 1.35,
    )

    period_axis.set_ylabel(
        "Wave period (s)"
    )

    period_axis.spines[
        "top"
    ].set_visible(
        False
    )

    add_direction_annotations(
        axis=wave_axis,
        times=times,
        directions_from_deg=forecast_data[
            "wave_direction_deg"
        ],
        compass_labels=forecast_data[
            "wave_direction_compass"
        ],
        annotation_step=annotation_step,
    )

    wave_handles, wave_labels = (
        wave_axis.get_legend_handles_labels()
    )

    period_handles, period_labels = (
        period_axis.get_legend_handles_labels()
    )

    wave_axis.legend(
        wave_handles
        + period_handles,
        wave_labels
        + period_labels,
        loc="upper left",
        bbox_to_anchor=(
            0.0,
            0.70,
        ),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )

    #### WIND ##################################################################

    wind_speed = forecast_data[
        "wind_speed_knots"
    ]

    maximum_wind = max(
        float(
            wind_speed.max()
        ),
        5.0,
    )

    wind_axis.bar(
        times,
        wind_speed,
        width=bar_width,
        alpha=0.65,
        label="Wind speed",
        zorder=3,
    )

    wind_axis.plot(
        times,
        wind_speed,
        linewidth=1.2,
        zorder=4,
    )

    wind_axis.set_ylim(
        0,
        maximum_wind
        * 1.65,
    )

    wind_axis.set_ylabel(
        "Wind speed (kt)"
    )

    wind_axis.set_title(
        "Wind",
        loc="left",
        pad=7,
    )

    style_primary_axis(
        wind_axis
    )

    add_direction_annotations(
        axis=wind_axis,
        times=times,
        directions_from_deg=forecast_data[
            "wind_direction_deg"
        ],
        compass_labels=forecast_data[
            "wind_direction_compass"
        ],
        annotation_step=annotation_step,
    )

    wind_axis.legend(
        loc="upper left",
        bbox_to_anchor=(
            0.0,
            0.70,
        ),
        frameon=False,
    )

    #### TIDE ##################################################################

    tide_height = tide_data[
        "tide_height_m"
    ]

    tide_baseline = float(
        tide_height.min()
    )

    tide_axis.plot(
        tide_data[
            "plot_time"
        ],
        tide_height,
        linewidth=1.9,
        label="Predicted tide",
        zorder=4,
    )

    tide_axis.fill_between(
        tide_data[
            "plot_time"
        ],
        tide_height,
        tide_baseline,
        alpha=0.20,
        zorder=2,
    )

    tide_axis.set_ylabel(
        "Tide height (m)"
    )

    tide_axis.set_title(
        "Tide",
        loc="left",
        pad=7,
    )

    style_primary_axis(
        tide_axis
    )

    tide_axis.legend(
        loc="upper left",
        frameon=False,
    )

    #### SHARED TIME AXIS ######################################################

    tide_axis.xaxis.set_major_locator(
        mdates.DayLocator(
            tz=LOCAL_TZ_INFO
        )
    )

    tide_axis.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%a\n%d %b",
            tz=LOCAL_TZ_INFO,
        )
    )

    tide_axis.xaxis.set_minor_locator(
        mdates.HourLocator(
            byhour=[
                6,
                12,
                18,
            ],
            tz=LOCAL_TZ_INFO,
        )
    )

    tide_axis.tick_params(
        axis="x",
        which="minor",
        length=2.5,
    )

    tide_axis.set_xlabel(
        "Local time — Australia/Brisbane"
    )

    wave_axis.tick_params(
        axis="x",
        labelbottom=False,
    )

    wind_axis.tick_params(
        axis="x",
        labelbottom=False,
    )

    for axis in (
        quality_axis,
        wave_axis,
        wind_axis,
        tide_axis,
    ):
        axis.set_xlim(
            time_edges[
                0
            ],
            time_edges[
                -1
            ],
        )

    add_day_divisions(
        axes=(
            wave_axis,
            wind_axis,
            tide_axis,
        ),
        start_time=times.min(),
        end_time=times.max(),
    )

    figure.align_ylabels(
        [
            wave_axis,
            wind_axis,
            tide_axis,
        ]
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_format = (
        output_path.suffix
        .lower()
        .lstrip(".")
    )

    if output_format not in {
        "svg",
        "png",
    }:
        raise ValueError(
            "The final plot path must end in .svg or .png."
        )

    save_arguments: dict[
        str,
        object,
    ] = {
        "format": output_format,
        "facecolor": "white",
    }

    if output_format == "png":
        save_arguments[
            "dpi"
        ] = 180

    figure.savefig(
        output_path,
        **save_arguments,
    )

    return figure


#### PUBLIC ENTRY POINT ########################################################

def generate_final_plot(
    date_string: str,
    cycle: str,
    output_path: str | Path,
) -> Path:
    """
    Run the full pipeline and save only the final SVG or PNG.
    """
    output_path = Path(
        output_path
    )

    transfer_matrix = load_transfer_matrix(
        TRANSFER_MATRIX_PATH
    )

    forecast_df = collect_forecast(
        transfer_matrix=transfer_matrix,
        date_string=date_string,
        cycle=cycle,
    )

    tide_df = get_msq_tide_for_wave_forecast(
        forecast=forecast_df,
        station_name=TIDE_STATION,
    )

    quality_forecast_df = score_forecast_times(
        forecast=forecast_df,
        tide=tide_df,
        weights=QUALITY_WEIGHTS,
    )

    final_plot_data = prepare_forecast_plot_data(
        quality_forecast_df
    )

    figure = plot_publication_surf_forecast(
        forecast_data=final_plot_data,
        tide=tide_df,
        date_string=date_string,
        cycle=cycle,
        output_path=output_path,
    )

    plt.close(
        figure
    )

    if (
        not output_path.exists()
        or output_path.stat().st_size < 1000
    ):
        raise RuntimeError(
            "The final surf forecast plot was not created correctly."
        )

    return output_path


__all__ = [
    "generate_final_plot",
]
