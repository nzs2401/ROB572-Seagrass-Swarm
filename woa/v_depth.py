import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import rowcol
from pathlib import Path
import glob
import subprocess
import os

def depth(lon, lat):
    repo_root = Path(__file__).parent
    tif_files = sorted(glob.glob(str(repo_root / "tif_files" / "more_data*.tif"))) + \
            sorted(glob.glob(str(repo_root / "tif_files" / "reprojected_*.tif")))

    if len(tif_files) == 0:
        raise FileNotFoundError("No .tif files found in tif_files folder.")

    print(f"  Found {len(tif_files)} bathymetric tiles, building VRT...")

    # Build a VRT (virtual mosaic) instead of loading all tiles into memory
    vrt_path = str(repo_root / "tif_files" / "merged.vrt")
    subprocess.run(["gdalbuildvrt", vrt_path] + tif_files, 
                   check=True, capture_output=True)

    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    with rasterio.open(vrt_path) as src:
        rows, cols = rowcol(src.transform, lon_flat, lat_flat)
        n_rows, n_cols = src.height, src.width
        rows = np.clip(rows, 0, n_rows - 1)
        cols = np.clip(cols, 0, n_cols - 1)
        
        # Sample in chunks instead of loading entire raster
        depth_flat = np.full(len(lon_flat), np.nan)
        chunk_size = 10000
        for i in range(0, len(lon_flat), chunk_size):
            chunk_rows = rows[i:i+chunk_size]
            chunk_cols = cols[i:i+chunk_size]
            # Use rasterio sample method - reads only needed pixels
            coords = [(lon_flat[j], lat_flat[j]) 
                    for j in range(i, min(i+chunk_size, len(lon_flat)))]
            samples = list(src.sample(coords))
            for j, sample in enumerate(samples):
                depth_flat[i+j] = sample[0]
        
        nodata = src.nodata
        if nodata is not None:
            depth_flat[depth_flat == nodata] = np.nan

    depth_flat = -depth_flat
    depth_flat[depth_flat < 0] = np.nan

    depth_grid = depth_flat.reshape(lon.shape)

    print(f"  Depth range: {np.nanmin(depth_grid):.1f}m to {np.nanmax(depth_grid):.1f}m")
    print(f"  Cells in 1-3m seagrass zone: {np.sum((depth_grid > 1) & (depth_grid < 3))}")

    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(depth_grid)
    if nan_mask.any():
        distance, indices = distance_transform_edt(nan_mask, return_indices=True)
        fill_mask = nan_mask & (distance <= 20)
        depth_grid[fill_mask] = depth_grid[tuple(indices[:, fill_mask])]

    return depth_grid