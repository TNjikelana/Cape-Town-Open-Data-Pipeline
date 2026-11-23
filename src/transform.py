"""Notes for teamates for better understandingStage

2: data transformation for the City of Cape Town System Energy dataset.

Takes the output of ``clean.clean_data`` and reshapes it for analysis. Every
transformation adds new columns and leaves the cleaned measurements untouched,
so each step can be traced back to its source column.

What this stage does, and why:
    1. Unit conversion       kWh -> GWh, so numbers are readable.
    2. Time features         Year / Month / Quarter / Season, so we can group
                             and compare periods.
    3. Energy features       Month-on-month and year-on-year change, a 12-month
                             rolling average, and the non-Eskom share.
    4. Reconstruction        Recompute the 12-month growth average that the
                             source leaves blank, and validate it.
    5. Skew correction       Yeo-Johnson power transform, applied only to
                             columns whose skewness justifies it.
    6. Scaling               Min-max and z-score versions, so columns that are
                             hundreds of GWh apart can share one axis.
    7. Encoding              One-hot for Season, sine/cosine for Month.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler

KWH_PER_GWH = 1_000_000

ENERGY_COLUMNS = (
    "Eskom_SPS",
    "SPS_Import",
    "SPS_OUT",
    "AGT_OUT",
    "RGT_OUT",
    "Total_System_OUT",
)

# Southern-hemisphere seasons, keyed by month number.
SEASONS = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn", 4: "Autumn", 5: "Autumn",
    6: "Winter", 7: "Winter", 8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}

# Columns that sit on very different scales (hundreds of GWh vs. tens of GWh).
# AGT/RGT are excluded: they are near zero and handled by the power transform.
SCALE_COLUMNS = (
    "Eskom_SPS_GWh",
    "SPS_Import_GWh",
    "SPS_OUT_GWh",
    "Total_System_OUT_GWh",
)

# |skewness| above 1 is a common rule of thumb for "highly skewed".
SKEW_THRESHOLD = 1.0

# Our recomputed year-on-year change should match the supplied growth column
# to within this many percentage points; larger gaps are flagged.
GROWTH_TOLERANCE_PCT = 0.1


def _require_consecutive_months(df: pd.DataFrame) -> None:
    """Rolling and percent-change features need one row per month, in order."""
    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")
    month_number = df["Date"].dt.year * 12 + df["Date"].dt.month
    if not month_number.diff().dropna().eq(1).all():
        raise ValueError(
            "Expected exactly one row per consecutive month, sorted by Date."
        )


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Add *_GWh columns (1 GWh = 1,000,000 kWh). The kWh originals are kept."""
    df = df.copy()
    for name in ENERGY_COLUMNS:
        df[f"{name}_GWh"] = df[f"{name}_KWH"] / KWH_PER_GWH
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Year, Month, Quarter and Season, so periods can be grouped and compared."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["Season"] = df["Month"].map(SEASONS)
    return df


def add_energy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived measures of change, trend and mix."""
    df = df.copy()
    total = df["Total_System_OUT_GWh"]

    # Month-on-month change. Not the same as the supplied growth column,
    # which is a year-on-year figure.
    df["MoM_Change_Pct"] = total.pct_change() * 100

    # Year-on-year change, recomputed so it can be checked against the source.
    df["YoY_Change_Pct"] = total.pct_change(12) * 100

    # 12-month rolling average smooths out the seasonal cycle. The first 11
    # rows are empty because a full year of history is needed.
    df["Rolling_12M_GWh"] = total.rolling(12).mean()

    # Eskom is ~100% of total output, so its share barely varies. The share
    # supplied by everything else (SPS, AGT, RGT) is the more informative mix.
    local = df["SPS_OUT_GWh"] + df["AGT_OUT_GWh"] + df["RGT_OUT_GWh"]
    df["Non_Eskom_Share_Pct"] = local / total * 100
    return df


def add_growth_reconstruction(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute F12_Month_Ave (12-month average of growth) where it is blank.

    The source leaves it empty for the first 66 months, but it matches the
    rolling 12-month mean of System_Energy_Growth almost exactly, so it can be
    rebuilt wherever 12 months of growth exist. Reconstructed values are
    flagged and kept in a separate column; the original is never overwritten.
    """
    df = df.copy()
    df["Growth_12M_Avg"] = df["System_Energy_Growth"].rolling(12).mean()
    df["F12_Month_Ave_filled"] = df["F12_Month_Ave"].fillna(df["Growth_12M_Avg"])
    df["flag_F12_reconstructed"] = (
        df["F12_Month_Ave"].isna() & df["Growth_12M_Avg"].notna()
    )

    # Validation: our year-on-year change vs. the supplied growth column.
    df["growth_vs_yoy_diff"] = (df["System_Energy_Growth"] - df["YoY_Change_Pct"]).abs()
    df["flag_growth_yoy_mismatch"] = df["growth_vs_yoy_diff"] > GROWTH_TOLERANCE_PCT
    return df


def skewness_table(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Skewness of each column, most skewed first (by absolute value)."""
    skew = df[columns].skew()
    return skew.reindex(skew.abs().sort_values(ascending=False).index)


def add_power_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Yeo-Johnson transform for highly skewed energy columns.

    The decision is data-driven: only columns with |skewness| above
    SKEW_THRESHOLD are transformed. Yeo-Johnson is used instead of a plain log
    because AGT/RGT contain negative values, which a log cannot handle. The
    output is standardised (mean 0, std 1).
    """
    df = df.copy()
    gwh_columns = [f"{name}_GWh" for name in ENERGY_COLUMNS]
    skew = skewness_table(df, gwh_columns)
    for column in skew.index[skew.abs() > SKEW_THRESHOLD]:
        transformer = PowerTransformer(method="yeo-johnson")
        df[f"{column}_YJ"] = transformer.fit_transform(df[[column]]).ravel()
    return df


def add_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max (0 to 1) and z-score (mean 0, std 1) versions of key columns.

    Scalers are fitted on the full dataset, which is fine for exploration. If
    this data were later used to train a model, fit on the training split only
    to avoid data leakage.
    """
    df = df.copy()
    columns = list(SCALE_COLUMNS)
    df[[f"{c}_MinMax" for c in columns]] = MinMaxScaler().fit_transform(df[columns])
    df[[f"{c}_Z" for c in columns]] = StandardScaler().fit_transform(df[columns])
    return df


def add_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode Season; encode Month cyclically.

    Season has no natural order, so it is one-hot encoded. One category
    (Autumn) is dropped as the baseline: the dummies would otherwise be
    redundant, which matters for linear models.

    Month is circular: December (12) sits next to January (1). Plain numbers
    put them 11 apart, so sine and cosine place months around a circle.
    """
    df = df.copy()
    seasons = pd.get_dummies(df["Season"], prefix="Season", drop_first=True, dtype=int)
    df = pd.concat([df, seasons], axis=1)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    return df


def transform_data(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Run the full transformation stage on the output of ``clean_data``."""
    _require_consecutive_months(cleaned)
    df = convert_units(cleaned)
    df = add_time_features(df)
    df = add_energy_features(df)
    df = add_growth_reconstruction(df)
    df = add_power_transforms(df)
    df = add_scaled_features(df)
    df = add_encodings(df)
    return df


def transformation_report(cleaned: pd.DataFrame, transformed: pd.DataFrame) -> pd.DataFrame:
    """List every new column with its type and how many values are missing."""
    new_columns = [c for c in transformed.columns if c not in cleaned.columns]
    return pd.DataFrame(
        {
            "dtype": transformed[new_columns].dtypes.astype(str),
            "missing": transformed[new_columns].isna().sum(),
        }
    ).rename_axis("new_column")