import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_sea_corals_6186_86ea_f111.csv")
df = pd.read_csv(data_path, low_memory=False)
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
df = df.dropna(subset=['longitude', 'latitude'])

x = df['longitude'].values
y = df['latitude'].values

# Check how many coral points are even in Tampa Bay bounds
mask = (x >= -83.5) & (x <= -82.0) & (y >= 27.0) & (y <= 28.5)
print("Coral points in Tampa Bay bounds:", mask.sum())
print("Total coral points:", len(x))
print("Lon range:", x.min(), x.max())
print("Lat range:", y.min(), y.max())