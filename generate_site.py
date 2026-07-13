"""
Generate the final website SVG for a specified GFS Wave cycle.

The SVG is first written to a temporary file. The public file and model
cycle marker are replaced only after successful forecast generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from surf_forecast import generate_final_plot


#### PATHS #####################################################################

OUTPUT_PATH = Path(
    "docs/final_surf_forecast.svg"
)

STATE_PATH = Path(
    ".forecast_state/latest_gfs_cycle.txt"
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
