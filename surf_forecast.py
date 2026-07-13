"""
Generate a browser-ready surf forecast from:

- NOAA GFS Wave data at a fixed model grid point
- A local wave-transfer matrix
- MSQ/BOM predicted tide data for Heron Island
- A weighted surf-quality rating

This module creates only the final forecast plot. It does not write
intermediate CSV files or plots.

The public entry point is:

    generate_final_plot(
        date_string="20260713",
        cycle="00",
        output_path="docs/final_surf_forecast.svg",
    )
"""

from __future__ import annotations


#### HEADLESS MATPLOTLIB #######################################################

import matplotlib

matplotlib.use("Agg")


#### IMPORTS ###################################################################

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

# Name displayed in the final plot.
SPOT_NAME = "Sykemeter"

# Actual surf location.
SPOT_LAT = -23.348043530796062
SPOT_LON = 152.618531665867

# Known nearest GFS Wave model point.
GFS_GRID_LAT = -23.25
GFS_GRID_LON = 152.50

# Tide station used for the forecast.
TIDE_STATION = "Heron Island"

# Queensland does not use daylight-saving time.
LOCAL_TIMEZONE = "Australia/Brisbane"
LOCAL_TZ_INFO = ZoneInfo(LOCAL_TIMEZONE)


#### FORECAST CONFIGURATION ####################################################

FORECAST_LENGTH_HOURS = 168
FORECAST_STEP_HOURS = 3

# This small box contains only the known GFS grid point.
BBOX_PADDING_DEGREES = 0.02

# Brief pause between NOAA requests.
REQUEST_PAUSE_SECONDS = 0.10

REQUEST_TIMEOUT_SECONDS = 90

# Tide predictions are normally spaced at 10-minute intervals.
TIDE_MATCH_TOLERANCE = pd.Timedelta("20min")


#### REMOTE DATA SOURCES #######################################################

NOMADS_WAVE_FILTER = (
    "https://nomads.ncep.noaa.gov/"
    "cgi-bin/filter_gfswave.pl"
)

CKAN_API_BASE = (
    "https://www.data.qld.gov.au/"
    "api/3/action"
)

DATASTORE_DUMP_BASE = (
    "https://www.data.qld.gov.au/"
    "datastore/dump"
)


#### PLOT CONFIGURATION ########################################################

# Limits the number of direction arrows to avoid crowded browser output.
MAX_DIRECTION_ANNOTATIONS = 20

# Red at 0, yellow at 50 and green at 100.
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

# Rows represent primary wave direction.
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

# Columns represent primary wave period.
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


#### QUALITY WEIGHTS ###########################################################

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


#### COMPASS LABELS ############################################################

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


#### TIDE RESOURCE #############################################################

@dataclass(frozen=True)
class TideResource:
    station_name: str
    package_name: str
    resource_id: str
    resource_name: str
    year: int
    url: str | None


#### HTTP SESSION ##############################################################

def create_http_session() -> requests.Session:
    """
    Create one reusable HTTP session with automatic retries.

    Reusing the session avoids creating a new HTTPS connection for every
    GFS forecast lead time.
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
            "Accept": (
                "application/json,"
                "text/csv,"
                "text/plain,"
                "*/*"
            ),
        }
    )

    return session


#### GENERAL HELPERS ###########################################################

def scalar(
    data_array: xr.DataArray,
) -> float:
    """
    Convert a scalar xarray DataArray into a Python float.
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
    Convert bearings into 16-point compass labels.
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


#### TRANSFER-MATRIX LOADING ###################################################

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


#### TRANSFER-MATRIX APPLICATION ###############################################

def apply_transfer_matrix(
    forecast: pd.DataFrame,
    transfer_matrix: np.ndarray,
) -> pd.DataFrame:
    """
    Apply the direction-period transfer matrix.

    Nearshore Hs = offshore GFS Hs × transfer coefficient.
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

    # Round wave direction to the nearest 15 degrees.
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

    # North is represented by 360 in the matrix.
    mapped_directions[
        (
            mapped_directions == 0
        )
        | (
            mapped_directions >= 360
        )
    ] = 360

    # Round period to the nearest whole second.
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

    result["matrix_direction_deg"] = (
        mapped_directions
    )

    result["matrix_period_s"] = (
        mapped_periods
    )

    result["transfer_coefficient"] = (
        coefficients
    )

    result["nearshore_wave_height_m"] = (
        result["wave_height_m"].to_numpy(
            dtype=float
        )
        * coefficients
    )

    return result


#### FIXED GFS BOUNDING BOX ####################################################

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


#### GFS URL CONSTRUCTION ######################################################

def build_wave_url(
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> str:
    """
    Build a NOMADS request for one forecast lead time.

    The response contains:
        Wave height
        Primary wave period
        Primary wave direction
        Wind speed
        Wind direction
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


#### GFS DOWNLOAD ##############################################################

def download_grib(
    session: requests.Session,
    url: str,
    destination: Path,
) -> None:
    """
    Download and validate one small GFS Wave GRIB subset.
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


#### GRIB OPENING ##############################################################

def open_grib(
    path: Path,
) -> xr.Dataset:
    """
    Open a small GFS Wave surface GRIB file.

    The fast xarray path is tried first. If cfgrib separates the fields
    into multiple datasets, they are loaded and merged.
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

            # The files are tiny, so loading immediately releases the
            # underlying GRIB file handles.
            merged.load()

        finally:
            for dataset in datasets:
                dataset.close()

        return merged


#### GRIB VARIABLE LOOKUP ######################################################

def find_variable(
    dataset: xr.Dataset,
    exact_names: tuple[str, ...],
    metadata_terms: tuple[str, ...],
) -> xr.DataArray:
    """
    Find a GRIB variable by name, then by descriptive metadata.
    """
    for variable_name in exact_names:
        if variable_name in dataset.data_vars:
            return dataset[
                variable_name
            ]

    for variable_name, data_array in dataset.data_vars.items():
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


#### FIXED GRID-POINT SELECTION ################################################

def select_downloaded_grid_point(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Select the single point in the downloaded GRIB subset.

    This avoids nearest-neighbour searching and avoids floating-point
    coordinate differences such as 152.500006 versus 152.5.
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


#### FORECAST VALID TIME #######################################################

def get_valid_time(
    dataset: xr.Dataset,
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> pd.Timestamp:
    """
    Return the valid forecast time as a UTC timestamp.
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


#### FORECAST RECORD READING ###################################################

def read_forecast_record(
    grib_path: Path,
    date_string: str,
    cycle: str,
    forecast_hour: int,
) -> dict[str, object]:
    """
    Read wave and wind values from one GFS Wave GRIB file.
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

        wave_direction_deg = (
            scalar(
                wave_direction
            )
            % 360.0
        )

        wind_direction_deg = (
            scalar(
                wind_direction
            )
            % 360.0
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
                wave_direction_deg
            ),
            "wind_speed_ms": (
                wind_speed_ms
            ),
            "wind_speed_knots": (
                wind_speed_ms
                * 1.943844
            ),
            "wind_direction_deg": (
                wind_direction_deg
            ),
            "grid_latitude": (
                downloaded_latitude
            ),
            "grid_longitude": (
                downloaded_longitude
            ),
        }

    finally:
        dataset.close()


#### GFS FORECAST COLLECTION ###################################################

def collect_forecast(
    transfer_matrix: np.ndarray,
    date_string: str,
    cycle: str,
) -> pd.DataFrame:
    """
    Download and process the complete GFS Wave forecast.

    The cycle is supplied externally so the GitHub workflow does not
    rediscover the model cycle during the expensive generation step.
    """
    if len(date_string) != 8 or not date_string.isdigit():
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
                wave_url = build_wave_url(
                    date_string=date_string,
                    cycle=cycle,
                    forecast_hour=forecast_hour,
                )

                download_grib(
                    session=session,
                    url=wave_url,
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

    forecast = pd.DataFrame(
        records
    )

    forecast = (
        forecast
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

    forecast["spot_name"] = (
        SPOT_NAME
    )

    forecast["spot_latitude"] = (
        SPOT_LAT
    )

    forecast["spot_longitude"] = (
        SPOT_LON
    )

    forecast["run_date"] = (
        date_string
    )

    forecast["run_cycle"] = (
        cycle
    )

    return forecast


#### CKAN API ##################################################################

def ckan_get(
    session: requests.Session,
    action: str,
    **parameters,
) -> dict:
    """
    Call the Queensland Government Open Data CKAN API.
    """
    response = session.get(
        f"{CKAN_API_BASE}/{action}",
        params=parameters,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get(
        "success",
        False,
    ):
        raise RuntimeError(
            f"CKAN API request failed for '{action}'."
        )

    return payload[
        "result"
    ]


#### TIDE PACKAGE SEARCH #######################################################

def search_msq_tide_package(
    session: requests.Session,
    station_name: str,
) -> dict:
    """
    Find the MSQ predicted interval tide package for the station.

    The search structure follows the working MSQ catalogue workflow.
    """
    queries = [
        (
            f'"{station_name}" '
            f'"predicted interval data" tide'
        ),
        (
            f'"{station_name}" '
            f'"tide gauge" '
            f'"predicted interval"'
        ),
        (
            f"{station_name} tide gauge "
            f"predicted interval data"
        ),
    ]

    station_lower = (
        station_name.lower()
    )

    for query in queries:
        result = ckan_get(
            session,
            "package_search",
            q=query,
            rows=20,
        )

        packages = result.get(
            "results",
            [],
        )

        # Prefer an exact title or package-name match.
        for package in packages:
            title = str(
                package.get(
                    "title",
                    "",
                )
            ).lower()

            name = str(
                package.get(
                    "name",
                    "",
                )
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

        # Broader match including the package resources.
        for package in packages:
            resource_text = " ".join(
                " ".join(
                    [
                        str(
                            resource.get(
                                "name",
                                "",
                            )
                        ),
                        str(
                            resource.get(
                                "description",
                                "",
                            )
                        ),
                        str(
                            resource.get(
                                "url",
                                "",
                            )
                        ),
                    ]
                )
                for resource in package.get(
                    "resources",
                    [],
                )
            )

            combined_text = " ".join(
                [
                    str(
                        package.get(
                            "title",
                            "",
                        )
                    ),
                    str(
                        package.get(
                            "name",
                            "",
                        )
                    ),
                    str(
                        package.get(
                            "notes",
                            "",
                        )
                    ),
                    resource_text,
                ]
            ).lower()

            if (
                station_lower in combined_text
                and "tide" in combined_text
                and "predicted" in combined_text
            ):
                return package

    raise RuntimeError(
        "Could not find an MSQ predicted interval tide "
        f"package for {station_name}."
    )


#### TIDE RESOURCE SELECTION ###################################################

def select_msq_tide_resource(
    package: dict,
    station_name: str,
    year: int,
) -> TideResource:
    """
    Select the annual predicted interval CSV/API resource.
    """
    year_text = str(
        year
    )

    candidates = []

    for resource in package.get(
        "resources",
        [],
    ):
        resource_id = resource.get(
            "id"
        )

        if not resource_id:
            continue

        resource_text = " ".join(
            [
                str(
                    resource.get(
                        "name",
                        "",
                    )
                ),
                str(
                    resource.get(
                        "description",
                        "",
                    )
                ),
                str(
                    resource.get(
                        "format",
                        "",
                    )
                ),
                str(
                    resource.get(
                        "url",
                        "",
                    )
                ),
            ]
        ).lower()

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
            (
                "  - "
                + str(
                    resource.get(
                        "name",
                        "Unnamed resource",
                    )
                )
            )
            for resource in package.get(
                "resources",
                [],
            )
        )

        raise RuntimeError(
            f"Could not find a {year} predicted interval "
            f"resource for {station_name}.\n\n"
            f"Available resources:\n"
            f"{available_resources}"
        )

    # Prefer active CKAN datastore resources, then explicit CSV resources.
    candidates.sort(
        key=lambda resource: (
            resource.get(
                "datastore_active"
            )
            is not True,
            "csv"
            not in str(
                resource.get(
                    "format",
                    "",
                )
            ).lower(),
        )
    )

    resource = candidates[
        0
    ]

    return TideResource(
        station_name=station_name,
        package_name=str(
            package.get(
                "name",
                "",
            )
        ),
        resource_id=str(
            resource[
                "id"
            ]
        ),
        resource_name=str(
            resource.get(
                "name",
                "",
            )
        ),
        year=year,
        url=resource.get(
            "url"
        ),
    )


#### TIDE DOWNLOAD #############################################################

def download_msq_tide_csv(
    session: requests.Session,
    resource: TideResource,
) -> pd.DataFrame:
    """
    Download one annual MSQ predicted interval tide resource.
    """
    candidate_urls = [
        (
            f"{DATASTORE_DUMP_BASE}/"
            f"{resource.resource_id}"
            f"?format=csv"
        )
    ]

    if resource.url:
        candidate_urls.append(
            resource.url
        )

    candidate_urls = list(
        dict.fromkeys(
            candidate_urls
        )
    )

    errors = []

    for url in candidate_urls:
        try:
            response = session.get(
                url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            text = response.text

            lower_preview = text[
                :5000
            ].lower()

            if not all(
                column_name in lower_preview
                for column_name in (
                    "date",
                    "time",
                    "reading",
                )
            ):
                raise RuntimeError(
                    "Response did not contain the expected "
                    "Date, Time and Reading fields."
                )

            return pd.read_csv(
                StringIO(
                    text
                ),
                low_memory=False,
            )

        except Exception as error:
            errors.append(
                f"{url}\n"
                f"{type(error).__name__}: {error}"
            )

    raise RuntimeError(
        "Failed to download the selected tide resource.\n\n"
        + "\n\n".join(
            errors
        )
    )


#### TIDE COLUMN LOOKUP ########################################################

def find_dataframe_column(
    dataframe: pd.DataFrame,
    possible_names: tuple[str, ...],
) -> str:
    """
    Find a column name without requiring exact capitalisation.
    """
    normalised_columns = {
        str(
            column
        ).strip().lower(): str(
            column
        )
        for column in dataframe.columns
    }

    for possible_name in possible_names:
        key = possible_name.strip().lower()

        if key in normalised_columns:
            return normalised_columns[
                key
            ]

    raise KeyError(
        f"Could not find any of {possible_names}.\n"
        f"Available columns: {list(dataframe.columns)}"
    )


#### TIDE PARSING ##############################################################

def parse_msq_tide_dataframe(
    raw_dataframe: pd.DataFrame,
    resource: TideResource,
) -> pd.DataFrame:
    """
    Parse an MSQ predicted interval CSV into a timezone-aware series.
    """
    raw = raw_dataframe.copy()

    raw.columns = [
        str(
            column
        ).strip()
        for column in raw.columns
    ]

    date_column = find_dataframe_column(
        raw,
        (
            "Date",
            "Prediction date",
        ),
    )

    time_column = find_dataframe_column(
        raw,
        (
            "Time",
            "Prediction time",
        ),
    )

    reading_column = find_dataframe_column(
        raw,
        (
            "Reading",
            "Height",
            "Water level",
            "Predicted height",
        ),
    )

    datetime_text = (
        raw[
            date_column
        ]
        .astype(str)
        .str.strip()
        + " "
        + raw[
            time_column
        ]
        .astype(str)
        .str.strip()
    )

    valid_time_local = pd.to_datetime(
        datetime_text,
        dayfirst=True,
        errors="coerce",
    )

    tide_height = pd.to_numeric(
        raw[
            reading_column
        ],
        errors="coerce",
    )

    tide = pd.DataFrame(
        {
            "valid_time_local": (
                valid_time_local
            ),
            "tide_height_m": (
                tide_height
            ),
        }
    )

    tide = tide.dropna(
        subset=[
            "valid_time_local",
            "tide_height_m",
        ]
    )

    if tide.empty:
        raise RuntimeError(
            "The tide resource was downloaded, but no valid "
            "tide records could be parsed."
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
                LOCAL_TIMEZONE
            )
        )

    else:
        tide[
            "valid_time_local"
        ] = (
            tide[
                "valid_time_local"
            ]
            .dt.tz_convert(
                LOCAL_TIMEZONE
            )
        )

    tide["time_utc"] = (
        tide[
            "valid_time_local"
        ]
        .dt.tz_convert(
            "UTC"
        )
    )

    tide["station_name"] = (
        resource.station_name
    )

    tide["resource_year"] = (
        resource.year
    )

    tide["resource_id"] = (
        resource.resource_id
    )

    tide["resource_name"] = (
        resource.resource_name
    )

    return (
        tide
        .sort_values(
            "valid_time_local"
        )
        .drop_duplicates(
            subset=[
                "valid_time_local"
            ]
        )
        .reset_index(
            drop=True
        )
    )


#### TIDE FORECAST COLLECTION ##################################################

def get_msq_tide_for_wave_forecast(
    forecast: pd.DataFrame,
    station_name: str = TIDE_STATION,
) -> pd.DataFrame:
    """
    Download tide data covering the GFS forecast period.
    """
    forecast_times = pd.to_datetime(
        forecast[
            "time_utc"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    if forecast_times.empty:
        raise ValueError(
            "The GFS forecast contains no valid timestamps."
        )

    start_time = (
        forecast_times.min()
        .tz_convert(
            LOCAL_TIMEZONE
        )
        .floor(
            "10min"
        )
    )

    end_time = (
        forecast_times.max()
        .tz_convert(
            LOCAL_TIMEZONE
        )
        .ceil(
            "10min"
        )
    )

    required_years = range(
        start_time.year,
        end_time.year + 1,
    )

    tide_frames = []

    with create_http_session() as session:
        # Search the catalogue once, even if the forecast crosses a year.
        package = search_msq_tide_package(
            session=session,
            station_name=station_name,
        )

        for year in required_years:
            resource = select_msq_tide_resource(
                package=package,
                station_name=station_name,
                year=year,
            )

            raw_tide = download_msq_tide_csv(
                session=session,
                resource=resource,
            )

            tide_frames.append(
                parse_msq_tide_dataframe(
                    raw_dataframe=raw_tide,
                    resource=resource,
                )
            )

    if not tide_frames:
        raise RuntimeError(
            "No tide data were downloaded."
        )

    tide = pd.concat(
        tide_frames,
        ignore_index=True,
    )

    tide = (
        tide
        .sort_values(
            "valid_time_local"
        )
        .drop_duplicates(
            subset=[
                "valid_time_local"
            ]
        )
    )

    tide = tide.loc[
        tide[
            "valid_time_local"
        ].between(
            start_time,
            end_time,
            inclusive="both",
        )
    ].copy()

    if tide.empty:
        raise RuntimeError(
            "Tide data were downloaded successfully, but no "
            "records overlap the GFS forecast period."
        )

    return tide.reset_index(
        drop=True
    )


#### DIRECTION SCORING HELPERS #################################################

def circular_distance_deg(
    angle_a,
    angle_b,
) -> np.ndarray:
    """
    Return the smallest distance between circular bearings.
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
    Test whether directions lie within a circular sector.

    If sector_start is larger than sector_end, the sector wraps through
    true north.
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


#### WAVE-HEIGHT SCORE #########################################################

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


#### WAVE-PERIOD SCORE #########################################################

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


#### WAVE-DIRECTION SCORE ######################################################

def score_wave_direction(
    direction_deg,
    good_start: float = 145.0,
    good_end: float = 45.0,
) -> np.ndarray:
    """
    Score wave direction from 0 to 1.

    The preferred sector wraps through north from 145° to 45°.
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


#### TIDE SCORE ################################################################

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


#### WIND-SPEED SCORE ##########################################################

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


#### WIND-DIRECTION SCORE ######################################################

def score_wind_direction(
    direction_deg,
) -> np.ndarray:
    """
    Score wind direction from 0 to 1.

    Preferred winds are SE through S to SW: 135° to 225°.
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


#### RATING LABELS #############################################################

def rating_from_score(
    scores,
) -> np.ndarray:
    """
    Convert numerical quality scores to descriptive ratings.
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


#### QUALITY INPUT ALIGNMENT ###################################################

def prepare_quality_inputs(
    forecast: pd.DataFrame,
    tide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match each GFS forecast time to the nearest tide prediction.
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

    combined["time_local"] = (
        combined[
            "time_utc"
        ]
        .dt.tz_convert(
            LOCAL_TIMEZONE
        )
    )

    return combined


#### QUALITY SCORING ###########################################################

def score_forecast_times(
    forecast: pd.DataFrame,
    tide: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Calculate the quality score for every GFS forecast time.

    All score calculations are vectorised.
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
            "Quality weights must be finite, non-negative, "
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

    result["quality_score"] = (
        quality_scores
    )

    result["rating"] = (
        rating_from_score(
            quality_scores
        )
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


#### FORECAST PLOT DATA ########################################################

def prepare_forecast_plot_data(
    quality_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare combined wave, wind, tide and rating values for plotting.
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
                LOCAL_TIMEZONE
            )
        )

    else:
        plot_times = (
            plot_times
            .dt.tz_convert(
                LOCAL_TIMEZONE
            )
        )

    data["plot_time"] = (
        plot_times
    )

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


#### FULL-RESOLUTION TIDE PLOT DATA ############################################

def prepare_tide_plot_data(
    tide: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Prepare the original 10-minute tide series for the final plot.
    """
    data = tide.copy()

    data["plot_time"] = (
        pd.to_datetime(
            data[
                "time_utc"
            ],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(
            LOCAL_TIMEZONE
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


#### PLOT TIME HELPERS #########################################################

def calculate_time_edges(
    times: pd.Series,
) -> pd.DatetimeIndex:
    """
    Calculate the time interval represented by each forecast record.
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
    Calculate a suitable bar width from the GFS interval.
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
    Thin direction annotations automatically to avoid overlapping text.
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


#### QUALITY COLOUR STRIP ######################################################

def plot_quality_strip(
    axis: plt.Axes,
    times: pd.Series,
    quality_scores: pd.Series,
    date_string: str,
    cycle: str,
) -> None:
    """
    Plot the red-yellow-green surf-quality strip.
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

    colour_map = plt.get_cmap(
        QUALITY_COLOUR_MAP
    )

    normaliser = Normalize(
        vmin=0.0,
        vmax=100.0,
        clip=True,
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
        cmap=colour_map,
        norm=normaliser,
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


#### DIRECTION ANNOTATIONS #####################################################

def add_direction_annotations(
    axis: plt.Axes,
    times: pd.Series,
    directions_from_deg: pd.Series,
    compass_labels: pd.Series,
    annotation_step: int,
) -> None:
    """
    Add a reserved row of direction arrows and compass labels.

    Arrows point in the direction of travel. Compass labels show the
    direction from which the wave or wind originates.
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


#### PLOT STYLING ##############################################################

def style_primary_axis(
    axis: plt.Axes,
) -> None:
    """
    Apply consistent publication-style axis formatting.
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
    Create and save the final browser-ready forecast plot.

    Layout:
        Surf-quality strip
        Wave height, period and direction
        Wind speed and direction
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

    #### QUALITY ###############################################################

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

    #### SAVE ##################################################################

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
    Run the complete forecast pipeline and save only the final plot.

    Parameters
    ----------
    date_string:
        GFS run date in YYYYMMDD format.

    cycle:
        GFS cycle hour: 00, 06, 12 or 18.

    output_path:
        Final SVG or PNG destination.

    Returns
    -------
    pathlib.Path
        The completed final plot path.
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


#### EXPORTED FUNCTIONS ########################################################

__all__ = [
    "generate_final_plot",
]
