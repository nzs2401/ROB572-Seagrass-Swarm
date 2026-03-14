def coral(lon, lat):

    import pandas as pd
    import numpy as np
    from scipy.spatial import cKDTree
    from pathlib import Path

    # Load Coral Data
    repo_root = Path(__file__).parent
    coral = pd.read_csv(repo_root / "deep_sea_corals_6186_86ea_f111.csv", low_memory=False)

    # Convert data from strings to numbers
    coral['longitude'] = pd.to_numeric(coral['longitude'], errors='coerce')
    coral['latitude']  = pd.to_numeric(coral['latitude'],  errors='coerce')

    # Drop rows where conversion failed
    coral = coral.dropna(subset=['longitude', 'latitude'])

    # Extract arrays
    x = coral['longitude'].values
    y = coral['latitude'].values

    # Flatten the meshgrid for vectorized computation
    lon_flat = lat_flat = None
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    # Compute coral presence with nearest neighbor
    tree = cKDTree(np.c_[x, y])
    distances, _ = tree.query(np.c_[lon_flat, lat_flat])
    coralpresence_flat = (distances <= 0.0000269).astype(int)
    coralpresence = coralpresence_flat.reshape(lon.shape)

    return coralpresence