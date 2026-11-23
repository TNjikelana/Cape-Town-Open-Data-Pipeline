"""Interactive runner for the System Energy data pipeline.

Pipeline:  raw CSV -> 1. clean -> 2. transform -> 3. visualise

Usage:
    python main.py             # interactive menu
    python main.py --rebuild   # ignore saved files and rebuild from the raw data

Each stage saves its output to data/processed/. When you ask for a stage, the
program looks for the saved output of the stage before it and only runs that
earlier stage if the file is missing. It asks before moving on to the next stage.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.clean import clean_data, cleaning_report
from src.transform import transform_data, transformation_report

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "System_Energy.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "system_energy_clean.csv"
TRANSFORMED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "system_energy_transformed.csv"
)
FIGURES_DIR = PROJECT_ROOT / "figures"

CLEAN_CODE = PROJECT_ROOT / "src" / "clean.py"
TRANSFORM_CODE = PROJECT_ROOT / "src" / "transform.py"

MENU = """
System Energy pipeline
  1. Clean & transform data
  2. Visualise data
  3. Exit
"""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def confirm(question: str, default: bool = True) -> bool:
    """Ask a yes/no question in the terminal. Enter accepts the default."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def warn_if_stale(data_path: Path, *source_files: Path) -> None:
    """Warn (never act) if a saved file is older than what produced it."""
    newest_source = max(f.stat().st_mtime for f in source_files if f.exists())
    if data_path.stat().st_mtime < newest_source:
        print(
            f"  Note: {data_path.name} is older than the data or code that "
            "produces it. Run 'python main.py --rebuild' to refresh it."
        )


def load_processed(path: Path) -> pd.DataFrame:
    """Reload a saved stage output, restoring the Date column's type."""
    return pd.read_csv(path, parse_dates=["Date"])


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def stage_clean() -> pd.DataFrame:
    """Stage 1: clean the raw data and save it."""
    print("\n--- Stage 1: cleaning ---")
    raw = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    cleaned = clean_data(raw)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Loaded {len(raw)} rows from {RAW_DATA_PATH}")
    print(f"Saved {len(cleaned)} cleaned rows to {CLEANED_DATA_PATH}\n")
    print(cleaning_report(raw, cleaned).to_string())
    return cleaned


def stage_transform(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: transform the cleaned data and save it."""
    print("\n--- Stage 2: transformation ---")
    transformed = transform_data(cleaned)

    TRANSFORMED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(TRANSFORMED_DATA_PATH, index=False)

    print(f"Saved {len(transformed)} rows to {TRANSFORMED_DATA_PATH}\n")
    print(transformation_report(cleaned, transformed).to_string())
    return transformed


def stage_visualise(transformed: pd.DataFrame) -> None:
    """Stage 3: exploration and visualisation (src/explore.py)."""
    print("\n--- Stage 3: exploration & visualisation ---")
    try:
        from src.explore import run_exploration
    except ModuleNotFoundError as error:
        if error.name != "src.explore":
            raise  # a real missing library, not a missing stage
        print("The visualisation stage (src/explore.py) is not implemented yet.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    run_exploration(transformed, FIGURES_DIR)
    print(f"Figures saved to {FIGURES_DIR}")


# --------------------------------------------------------------------------
# Menu flows
# --------------------------------------------------------------------------
def run_clean_and_transform(rebuild: bool) -> None:
    """Option 1: use saved cleaned data if present, otherwise clean first."""
    if CLEANED_DATA_PATH.exists() and not rebuild:
        print(f"\nFound cleaned data: {CLEANED_DATA_PATH.name} (skipping cleaning)")
        warn_if_stale(CLEANED_DATA_PATH, RAW_DATA_PATH, CLEAN_CODE)
        cleaned = load_processed(CLEANED_DATA_PATH)
    else:
        cleaned = stage_clean()
        if not confirm("\nContinue to transformation?"):
            return

    transformed = stage_transform(cleaned)
    if confirm("\nContinue to visualisation?"):
        stage_visualise(transformed)


def run_visualise(rebuild: bool) -> None:
    """Option 2: use saved transformed data; run missing earlier stages if asked."""
    if TRANSFORMED_DATA_PATH.exists() and not rebuild:
        print(f"\nFound transformed data: {TRANSFORMED_DATA_PATH.name}")
        warn_if_stale(
            TRANSFORMED_DATA_PATH,
            RAW_DATA_PATH,
            CLEAN_CODE,
            CLEANED_DATA_PATH,
            TRANSFORM_CODE,
        )
        stage_visualise(load_processed(TRANSFORMED_DATA_PATH))
        return

    print("\nTransformed data not found." if not rebuild else "\nRebuilding.")
    if CLEANED_DATA_PATH.exists() and not rebuild:
        print(f"Found cleaned data: {CLEANED_DATA_PATH.name}")
        if not confirm("Transform it now?"):
            return
        cleaned = load_processed(CLEANED_DATA_PATH)
    else:
        if not confirm("Cleaned data not found either. Clean the raw data now?"):
            return
        cleaned = stage_clean()
        if not confirm("\nContinue to transformation?"):
            return

    transformed = stage_transform(cleaned)
    if confirm("\nContinue to visualisation?"):
        stage_visualise(transformed)


def main() -> None:
    parser = argparse.ArgumentParser(description="System Energy data pipeline")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore saved files in data/processed and rebuild from the raw data",
    )
    rebuild = parser.parse_args().rebuild

    try:
        while True:
            print(MENU)
            choice = input("Choose an option (1-3): ").strip()
            if choice == "1":
                run_clean_and_transform(rebuild)
            elif choice == "2":
                run_visualise(rebuild)
            elif choice == "3":
                print("Goodbye.")
                return
            else:
                print("Please enter 1, 2 or 3.")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")


if __name__ == "__main__":
    main()