import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# Load data once
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep_sea_corals_6186_86ea_f111.csv")
df = pd.read_csv(data_path, low_memory=False)
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['latitude']  = pd.to_numeric(df['latitude'],  errors='coerce')
df['DepthInMeters'] = pd.to_numeric(df['DepthInMeters'], errors='coerce')
df = df.dropna(subset=['longitude', 'latitude'])

x = df['longitude'].values
y = df['latitude'].values
z = df['DepthInMeters'].values

# Grid
# lat_vec = np.linspace(27.0, 28.5, 50)
# lon_vec = np.linspace(-83.5, -82.0, 50)
lat_vec = np.linspace(22.875, 31.0867, 50)
lon_vec = np.linspace(-88.3, -78.9333, 50)
LonGrid, LatGrid = np.meshgrid(lon_vec, lat_vec)

# Run griddata once across all points
coral_depth = griddata(points=(x, y), values=z, xi=(LonGrid, LatGrid), method='linear')

# Use KDTree for fast nearest neighbor distance across all grid points
tree = cKDTree(np.column_stack([x, y]))
grid_points = np.column_stack([LonGrid.ravel(), LatGrid.ravel()])
distances, _ = tree.query(grid_points)
distances = distances.reshape(50, 50)

# coral_presence = (distances <= 0.0000269).astype(int)
coral_presence = (distances <= 0.1).astype(int)

# Plot
plt.figure(figsize=(8,5))
plt.contourf(LonGrid, LatGrid, coral_presence, cmap='YlOrRd')
plt.colorbar(label='Coral Presence')
plt.title('Coral Presence - Tampa Bay')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()