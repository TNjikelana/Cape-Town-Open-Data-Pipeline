

# 1. LOAD THE RAW DATA
import pandas as pd

raw = pd.read_csv("System_Energy.csv", encoding="utf-8-sig")  # utf-8-sig removes the hidden BOM
df = raw.copy()                                               # always keep the raw copy for before/after

print(df.shape)
print(df.head())
df.info()

# 2. INSPECTING THE PROBLEMS (before cleaning)
print("Missing values:\n", df.isna().sum(), "\n")
print("Duplicate rows:", df.duplicated().sum())
print("SPS_OUT_KWH equal to 0:", (df["SPS_OUT_KWH"] == 0).sum())
print("SPS_Import_KWH <= 0:", (df["SPS_Import_KWH"] <= 0).sum())
print(df.dtypes)

# 3. FIXING DATA TYPES AND COLUMN NAMES
# Dates are text in UTC (22:00 UTC = midnight South African time, i.e. the month end)
df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("Africa/Johannesburg")
df["Date"] = df["Date"].dt.tz_localize(None).dt.normalize()   # keep just the date

# Percentages are stored as text like "-3.93%"
for col in ["Sytem_Energy_Growth", "F12_Month_Ave"]:
    df[col] = df[col].str.rstrip("%").astype(float)

# Fixing the typo in the column name
df = df.rename(columns={"Sytem_Energy_Growth": "System_Energy_Growth"})

# ObjectId is just a row counter, so it adds no information
df = df.drop(columns=["ObjectId"])

# 4. FLAG SUSPICIOUS VALUES 
# 4.1 Zero readings in SPS_OUT_KWH
df["flag_SPS_OUT_zero"] = df["SPS_OUT_KWH"] == 0

# 4.2 Zero or negative imports (an import should not be negative)
df["flag_import_nonpositive"] = df["SPS_Import_KWH"] <= 0

# 4.3 Outliers in AGT_OUT_KWH and RGT_OUT_KWH using the IQR rule
def iqr_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

df["flag_AGT_outlier"] = iqr_outliers(df["AGT_OUT_KWH"])
df["flag_RGT_outlier"] = iqr_outliers(df["RGT_OUT_KWH"])

# 4.4 Does the total add up? (Total = Eskom - Import + SPS_OUT + AGT_OUT + RGT_OUT)
calc_total = (df["Eskom_SPS_KWH"] - df["SPS_Import_KWH"] + df["SPS_OUT_KWH"]
              + df["AGT_OUT_KWH"] + df["RGT_OUT_KWH"])
df["flag_total_mismatch"] = calc_total != df["Total_System_OUT_KWH"]

# 4.5 Growth values of exactly -1.00% (look like placeholders)
df["flag_growth_placeholder"] = df["System_Energy_Growth"] == -1.00

flag_cols = [c for c in df.columns if c.startswith("flag_")]
print(df[flag_cols].sum())          # how many rows each flag catches

# 5. DATA CLEANING DECISIONS 
# Decision 1: placeholder growth values are not real so set to missing (NaN)
#             (the Transformation stage can recalculate year-on-year growth from the totals)
df.loc[df["flag_growth_placeholder"], "System_Energy_Growth"] = float("nan")

# Decision 2: remove duplicate rows (there are none, but this shows the step)
before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))      # prints 0
 
# Decision 3: a negative import is not logical so treat them as missing (NaN)
#             (check the portal metadata first; remove this line if negatives are valid)
df.loc[df["SPS_Import_KWH"] < 0, "SPS_Import_KWH"] = float("nan")
 
 
# Decision 4: outliers in AGT_OUT_KWH and RGT_OUT_KWH are cap extreme values at the
#             5th and 95th percentiles.  
for col in ["AGT_OUT_KWH", "RGT_OUT_KWH"]:
    lo, hi = df[col].quantile([0.05, 0.95])
    df[col] = df[col].clip(lo, hi)


# Decision 5: totals that do not add up so KEPT and flagged (published figure, not ours to change).
 
# Decision 6: F12_Month_Ave is blank for the first 66 months. We leave it as NaN
#             because the figure does not exist for that period (check the portal metadata).
 


# 6. BEFORE / AFTER CHECK
print("BEFORE:"); raw.info()
print("AFTER:");  df.info()
print(df.isna().sum())

# 7. SAVE THE CLEANED FILE
df.to_csv("energy_clean.csv", index=False)
print("Saved energy_clean.csv with", df.shape[0], "rows and", df.shape[1], "columns")