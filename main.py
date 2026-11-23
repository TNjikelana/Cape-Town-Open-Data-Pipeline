from pathlib import Path

import pandas as pd

from src.clean import clean_data


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "System_Energy.csv"
PROCESSED_DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "system_energy_clean.csv")


def run_pipeline() -> Path:
    """Run the implemented cleaning stage and save its processed dataset."""
    raw_data = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    cleaned_data = clean_data(raw_data)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Loaded {len(raw_data)} rows from {RAW_DATA_PATH}")
    print(f"Saved {len(cleaned_data)} cleaned rows to {PROCESSED_DATA_PATH}")
    flag_columns = [
        column for column in cleaned_data.columns if column.startswith("flag_")
    ]
    print("Review flags:\n", cleaned_data[flag_columns].sum().to_string())
    return PROCESSED_DATA_PATH


if __name__ == "__main__":
    run_pipeline()