"""
Generate the final website SVG for a specified GFS Wave cycle.

The SVG is first written to a temporary file. The public file and model
cycle marker are replaced only after successful forecast generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from surf_forecast import (
    TIDE_TRANSFER_MATRIX_LONG_PATH,
    TIDE_TRANSFER_MATRIX_PATHS,
    generate_final_plot,
)


#### PATHS #####################################################################

OUTPUT_PATH = Path(
    "docs/final_surf_forecast.svg"
)

STATE_PATH = Path(
    ".forecast_state/latest_gfs_cycle.txt"
)


#### PREFLIGHT #################################################################

def validate_required_runtime_files() -> None:
    """
    Fail early if no tide-aware SWAN matrix source is in the repository.
    """
    config_path = Path(__file__).resolve().parent / "model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for filename, expected in config["matrix_sha256"].items():
        matrix_path = config_path.parent / filename
        if not matrix_path.exists() or hashlib.sha256(matrix_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise RuntimeError(f"Transfer matrix does not match model_config.json: {filename}")

    missing_paths = [
        path
        for path in TIDE_TRANSFER_MATRIX_PATHS.values()
        if not path.exists()
    ]

    if not missing_paths:
        return

    if TIDE_TRANSFER_MATRIX_LONG_PATH.exists():
        return

    missing_list = "\n".join(
        f"- {path.resolve()}"
        for path in missing_paths
    )

    raise FileNotFoundError(
        "The tide-aware SWAN transfer matrix files are required "
        "to generate the forecast, but one or more are missing.\n\n"
        f"Missing files:\n{missing_list}\n\n"
        "Either upload these files to the repository root, beside "
        "surf_forecast.py and generate_site.py, or upload the long "
        "table fallback:\n"
        f"- {TIDE_TRANSFER_MATRIX_LONG_PATH.resolve()}"
    )


#### COMMAND-LINE ARGUMENTS ####################################################

def parse_arguments() -> argparse.Namespace:
    """
    Read the GFS date and cycle supplied by GitHub Actions.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the public surf forecast "
            "for one GFS Wave cycle."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help=(
            "GFS run date in YYYYMMDD format."
        ),
    )

    parser.add_argument(
        "--cycle",
        required=True,
        choices=[
            "00",
            "06",
            "12",
            "18",
        ],
        help=(
            "GFS model cycle hour."
        ),
    )

    return parser.parse_args()


#### WEBSITE GENERATION ########################################################

def main() -> None:
    """
    Generate the SVG and update the cycle marker after success.
    """
    arguments = parse_arguments()

    cycle_key = (
        f"{arguments.date}-"
        f"{arguments.cycle}Z"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_name(
        "final_surf_forecast.tmp.svg"
    )

    temporary_path.unlink(
        missing_ok=True
    )

    try:
        validate_required_runtime_files()

        generate_final_plot(
            date_string=arguments.date,
            cycle=arguments.cycle,
            output_path=temporary_path,
        )

        if (
            not temporary_path.exists()
            or temporary_path.stat().st_size < 1000
        ):
            raise RuntimeError(
                "The generated SVG is missing or unexpectedly small."
            )

        # Replace the public plot atomically.
        temporary_path.replace(
            OUTPUT_PATH
        )

        STATE_PATH.write_text(
            f"{cycle_key}\n",
            encoding="utf-8",
        )

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )

        raise

    print(
        f"Generated public forecast for {cycle_key}."
    )

    print(
        f"Output: {OUTPUT_PATH.resolve()}"
    )


#### RUN #######################################################################

if __name__ == "__main__":
    main()
