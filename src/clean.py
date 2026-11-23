"""Stage 1: data cleaning for the City of Cape Town System Energy dataset.

Design principle: the raw measurements are never altered or deleted. Problems
are made visible as ``flag_`` columns (and one ``total_diff_KWH`` column) so
every cleaning decision can be explained, checked, and reversed.
"""

import pandas as pd

NUMERIC_COLUMNS = (
    "Eskom_SPS_KWH",
    "SPS_Import_KWH",
    "SPS_OUT_KWH",
    "AGT_OUT_KWH",
    "RGT_OUT_KWH",
    "Total_System_OUT_KWH",
)

# Stored as text like "-3.93%". Converted to numbers in percentage points
# (-3.93, not -0.0393), so these columns are in "%" units.
PERCENT_COLUMNS = ("System_Energy_Growth", "F12_Month_Ave")

# The source column name contains a typo ("Sytem"); it is renamed on load.
COLUMN_RENAMES = {"Sytem_Energy_Growth": "System_Energy_Growth"}

# Growth is a constant -1.00% in the most recent months, which looks like a
# placeholder rather than a measurement.
PLACEHOLDER_GROWTH = -1.0

# The reported total should equal supply - imports + outputs. Allow 1 kWh
# for rounding; anything larger is flagged.
TOTAL_TOLERANCE_KWH = 1


def _iqr_outliers(values: pd.Series) -> pd.Series:
    """Flag values outside 1.5 * IQR of the quartiles.

    Caveat: IQR assumes one roughly bell-shaped cluster. AGT/RGT output has two
    regimes (small negative values most months, occasional large positive
    ones), so "unusual" here means "unlike the typical month", not "wrong".
    """
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the source data without altering measurements."""
    df = raw.copy().rename(columns=COLUMN_RENAMES)

    required = {"Date", *NUMERIC_COLUMNS, *PERCENT_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Dates are stored as UTC text (e.g. "2017/07/30 22:00:00+00"). 22:00 UTC is
    # midnight in South Africa, so converting the timezone gives the true
    # month-end date (31 July).
    df["Date"] = (
        pd.to_datetime(df["Date"], utc=True, errors="raise")
        .dt.tz_convert("Africa/Johannesburg")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    if df["Date"].isna().any():
        raise ValueError("Date contains missing or invalid values.")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # Strip the "%" sign and convert to plain float64 (not pandas' nullable
    # Float64) so plotting and scikit-learn behave predictably.
    for column in PERCENT_COLUMNS:
        text = df[column].astype("string").str.strip().str.rstrip("%")
        df[column] = pd.to_numeric(text, errors="raise").astype("float64")

    df = df.drop(columns=["ObjectId"], errors="ignore")  # row counter only

    # --- Flags: mark issues, never delete rows or change measurements ---
    df["flag_duplicate_record"] = df.duplicated(keep=False)
    df["flag_SPS_OUT_zero"] = df["SPS_OUT_KWH"].eq(0)
    df["flag_import_nonpositive"] = df["SPS_Import_KWH"].le(0)
    df["flag_AGT_unusual"] = _iqr_outliers(df["AGT_OUT_KWH"])
    df["flag_RGT_unusual"] = _iqr_outliers(df["RGT_OUT_KWH"])

    # F12_Month_Ave is missing for the first 66 months. The gap is structural
    # (the figure doesn't exist yet), not random, so it is flagged and left
    # as NaN instead of being imputed.
    df["flag_F12_missing"] = df["F12_Month_Ave"].isna()

    calculated_total = (
        df["Eskom_SPS_KWH"]
        - df["SPS_Import_KWH"]
        + df["SPS_OUT_KWH"]
        + df["AGT_OUT_KWH"]
        + df["RGT_OUT_KWH"]
    )
    df["total_diff_KWH"] = (calculated_total - df["Total_System_OUT_KWH"]).abs()
    df["flag_total_mismatch"] = df["total_diff_KWH"] > TOTAL_TOLERANCE_KWH

    df["flag_growth_placeholder"] = df["System_Energy_Growth"].eq(PLACEHOLDER_GROWTH)

    return df


def cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Summarise what cleaning found: row counts, missing values, flag totals."""
    rows = {"rows in": len(raw), "rows out": len(cleaned)}
    missing = {
        f"missing: {col}": int(n)
        for col, n in cleaned.isna().sum().items()
        if n > 0
    }
    flags = {
        col: int(cleaned[col].sum())
        for col in cleaned.columns
        if col.startswith("flag_")
    }
    summary = {**rows, **missing, **flags}
    return pd.Series(summary, name="count").rename_axis("check").to_frame()