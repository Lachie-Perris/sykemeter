"""
Check whether a new, complete GFS Wave cycle is available.

This is deliberately lightweight. It uses only Python's standard
library, so GitHub Actions does not need to install the scientific
Python dependencies unless a new model cycle is found.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


#### CONFIGURATION #############################################################

NOMADS_DIRECTORY_BASE = (
    "https://nomads.ncep.noaa.gov/"
    "pub/data/nccf/com/gfs/prod"
)

# The full forecast should only run after the final required file exists.
FORECAST_LENGTH_HOURS = 168

# Stored in the repository after a successful website deployment.
STATE_PATH = Path(
    ".forecast_state/latest_gfs_cycle.txt"
)

# The first run should generate a plot even if a state file exists.
PLOT_PATH = Path(
    "docs/final_surf_forecast.svg"
)

MAX_LOOKBACK_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 30


#### GITHUB ACTIONS OUTPUT #####################################################

def write_github_output(
    name: str,
    value: str,
) -> None:
    """
    Expose a value to later GitHub Actions steps.
    """
    github_output = os.environ.get(
        "GITHUB_OUTPUT"
    )

    if github_output:
        with open(
            github_output,
            "a",
            encoding="utf-8",
        ) as output_file:
            output_file.write(
                f"{name}={value}\n"
            )

    print(
        f"{name}={value}"
    )


#### COMPLETE GFS CYCLE CHECK ##################################################

def latest_complete_cycle() -> tuple[str, str]:
    """
    Find the newest GFS Wave cycle containing the final forecast hour.

    Looking for f168 means the workflow waits until the full seven-day
    forecast has been uploaded rather than running when only f000 exists.
    """
    now_utc = datetime.now(
        timezone.utc
    )

    cycle_hours = (
        18,
        12,
        6,
        0,
    )

    for day_offset in range(
        MAX_LOOKBACK_DAYS + 1
    ):
        candidate_date = (
            now_utc
            - timedelta(
                days=day_offset
            )
        )

        date_string = candidate_date.strftime(
            "%Y%m%d"
        )

        for cycle_hour in cycle_hours:
            cycle_time = datetime(
                year=candidate_date.year,
                month=candidate_date.month,
                day=candidate_date.day,
                hour=cycle_hour,
                tzinfo=timezone.utc,
            )

            if cycle_time > now_utc:
                continue

            cycle = f"{cycle_hour:02d}"

            directory_url = (
                f"{NOMADS_DIRECTORY_BASE}/"
                f"gfs.{date_string}/"
                f"{cycle}/wave/gridded/"
            )

            final_filename = (
                f"gfswave.t{cycle}z."
                f"global.0p25."
                f"f{FORECAST_LENGTH_HOURS:03d}.grib2"
            )

            request = Request(
                directory_url,
                headers={
                    "User-Agent": (
                        "Heron Island surf forecast "
                        "cycle checker"
                    )
                },
            )

            try:
                with urlopen(
                    request,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                ) as response:
                    directory_text = (
                        response.read()
                        .decode(
                            "utf-8",
                            errors="ignore",
                        )
                    )

                if final_filename in directory_text:
                    return (
                        date_string,
                        cycle,
                    )

            except (
                HTTPError,
                URLError,
                TimeoutError,
            ):
                continue

    raise RuntimeError(
        "No complete recent GFS Wave cycle was found."
    )


#### UPDATE DECISION ###########################################################

def main() -> None:
    """
    Determine whether the full surf forecast needs to run.
    """
    date_string, cycle = (
        latest_complete_cycle()
    )

    cycle_key = (
        f"{date_string}-{cycle}Z"
    )

    previous_cycle = ""

    if STATE_PATH.exists():
        previous_cycle = (
            STATE_PATH.read_text(
                encoding="utf-8"
            )
            .strip()
        )

    force_update = (
        os.environ.get(
            "FORCE_UPDATE",
            "false",
        )
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )

    needs_update = (
        force_update
        or cycle_key != previous_cycle
        or not PLOT_PATH.exists()
    )

    write_github_output(
        "date_string",
        date_string,
    )

    write_github_output(
        "cycle",
        cycle,
    )

    write_github_output(
        "cycle_key",
        cycle_key,
    )

    write_github_output(
        "needs_update",
        str(
            needs_update
        ).lower(),
    )

    if needs_update:
        print(
            f"Forecast update required for {cycle_key}."
        )
    else:
        print(
            f"The website already uses {cycle_key}."
        )


#### RUN #######################################################################

if __name__ == "__main__":
    main()
