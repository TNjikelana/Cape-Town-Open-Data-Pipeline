import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ==========================================
# 1. LOAD THE DATASET
# ==========================================

df = pd.read_csv("System_Energy.csv")

print("ORIGINAL DATASET:")
print(df.head())

print("\nORIGINAL DATA TYPES:")
print(df.dtypes)


# ==========================================
# 2. STANDARDISE COLUMN NAMES
# ==========================================

# Correct the spelling mistake in the original dataset
df.rename(
    columns={'Sytem_Energy_Growth': 'System_Energy_Growth'},
    inplace=True
)

print("\nUPDATED COLUMN NAMES:")
print(df.columns.tolist())


# ==========================================
# 3. CONVERT THE DATE COLUMN
# ==========================================

# Convert the text into a proper datetime (the file stores UTC time)
df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')

# 22:00 UTC is midnight in South Africa (UTC+2), so each record is
# really a month-end. Convert to local time so the date is correct.
df['Date'] = (
    df['Date']
    .dt.tz_convert('Africa/Johannesburg')
    .dt.tz_localize(None)
    .dt.normalize()
)

print("\nDATE COLUMN AFTER CONVERSION:")
print(df['Date'].head())


# ==========================================
# 4. CONVERT PERCENTAGE COLUMNS
# ==========================================

# Remove the % symbol and convert text into numbers
for col in ['System_Energy_Growth', 'F12_Month_Ave']:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace('%', '', regex=False),
        errors='coerce'
    )

print("\nTRANSFORMED PERCENTAGE COLUMNS:")
print(df[['System_Energy_Growth', 'F12_Month_Ave']].head())


# ==========================================
# 5. REMOVE UNNEEDED COLUMN
# ==========================================

# ObjectId is only a row number and has no analytical value
df.drop(columns=['ObjectId'], inplace=True)


# ==========================================
# 6. UNIT CONVERSION (kWh -> GWh)
# ==========================================

# 1 GWh = 1,000,000 kWh. Smaller numbers are easier to read and chart.
kwh_cols = [c for c in df.columns if c.endswith('_KWH')]

for col in kwh_cols:
    df[col.replace('_KWH', '_GWh')] = df[col] / 1_000_000

print("\nUNIT CONVERSION:")
print(df[['Total_System_OUT_KWH', 'Total_System_OUT_GWh']].head())


# ==========================================
# 7. FEATURE ENGINEERING - TIME FEATURES
# ==========================================

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter

# South African seasons (southern hemisphere)
season_map = {
    12: 'Summer', 1: 'Summer', 2: 'Summer',
    3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
    6: 'Winter', 7: 'Winter', 8: 'Winter',
    9: 'Spring', 10: 'Spring', 11: 'Spring'
}
df['Season'] = df['Month'].map(season_map)

print("\nTIME FEATURES:")
print(df[['Date', 'Year', 'Month', 'Quarter', 'Season']].head())


# ==========================================
# 8. FEATURE ENGINEERING - ENERGY FEATURES
# ==========================================

# Eskom's share of the total system output (%)
df['Eskom_Share_Pct'] = (
    df['Eskom_SPS_GWh'] / df['Total_System_OUT_GWh'] * 100
)

# Month-on-month change in total output (%)
df['MoM_Change_Pct'] = df['Total_System_OUT_GWh'].pct_change() * 100

# 12-month rolling average (smooths out seasonality).
# The first 11 rows are empty because 12 months are needed.
df['Rolling_12M_GWh'] = df['Total_System_OUT_GWh'].rolling(12).mean()

print("\nENERGY FEATURES:")
print(df[['Eskom_Share_Pct', 'MoM_Change_Pct', 'Rolling_12M_GWh']].head(14))


# ==========================================
# 9. SCALING
# ==========================================

scale_cols = [
    'Eskom_SPS_GWh', 'SPS_Import_GWh',
    'SPS_OUT_GWh', 'Total_System_OUT_GWh'
]

# Min-Max scaling: values rescaled to the range 0 to 1
df[[c + '_MinMax' for c in scale_cols]] = (
    MinMaxScaler().fit_transform(df[scale_cols])
)

# Standardisation (z-score): mean 0 and standard deviation 1
df[[c + '_Z' for c in scale_cols]] = (
    StandardScaler().fit_transform(df[scale_cols])
)

print("\nSCALING CHECK:")
print(df[[c + '_MinMax' for c in scale_cols]].agg(['min', 'max']))
print(df[[c + '_Z' for c in scale_cols]].agg(['mean', 'std']).round(3))


# ==========================================
# 10. ENCODING
# ==========================================

# One-hot encoding for Season (categories with no natural order)
season_dummies = pd.get_dummies(df['Season'], prefix='Season', dtype=int)
df = pd.concat([df, season_dummies], axis=1)

# Cyclical encoding for Month: December (12) is next to January (1),
# so sine and cosine are used instead of plain numbers
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

print("\nENCODING:")
print(df[['Season', 'Season_Summer', 'Season_Winter',
          'Month', 'Month_Sin', 'Month_Cos']].head())


# ==========================================
# 11. DISPLAY TRANSFORMED DATASET
# ==========================================

print("\nTRANSFORMED DATASET:")
print(df.head())

print("\nDATASET INFORMATION:")
df.info()


# ==========================================
# 12. SAVE TRANSFORMED DATASET
# ==========================================

df.to_csv("System_Energy_transformed.csv", index=False)

print("\nData transformation completed successfully!")
