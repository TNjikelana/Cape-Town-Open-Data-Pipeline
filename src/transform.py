import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Load the dataset
df = pd.read_csv("System_Energy.csv")
print(df.head())

