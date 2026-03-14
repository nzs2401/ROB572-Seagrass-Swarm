def depth(lon, lat):
    """
    Loads and merges CUDEM bathymetric GeoTIFF tiles and interpolates
    depth values onto the provided lon/lat grid.

    CUDEM convention: negative values = below sea level (water depth)
    We flip the sign so that depth is a positive number in metres.

    Parameters:
        lon : 2D numpy array of longitudes
        lat : 2D numpy array of latitudes

    Returns:
        depth_grid : 2D numpy array of water depth in metres (positive = deeper)
                     NaN where data is missing or on land
    """
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import rowcol
    from pathlib import Path
    import glob

    repo_root = Path(__file__).parent

    # Find all CUDEM tif tiles in the repo folder
    tif_files = sorted(glob.glob(str(repo_root / "thirdarcsec_DEM_J1342746*.tif")))

    if len(tif_files) == 0:
        raise FileNotFoundError(
            "No CUDEM .tif files found in repo folder. "
            "Make sure all thirdarcsec_DEM_J1342746*.tif files are present."
        )

    print(f"  Found {len(tif_files)} bathymetric tiles, merging...")

    # Open and merge all tiles into one seamless raster
    src_files = [rasterio.open(f) for f in tif_files]
    merged, merged_transform = merge(src_files)

    # merged shape is (1, rows, cols) — squeeze to 2D
    merged_data = merged[0].astype(float)

    # Get nodata value and mask it
    nodata = src_files[0].nodata
    if nodata is not None:
        merged_data[merged_data == nodata] = np.nan

    # Close all open files
    for src in src_files:
        src.close()

    # Get the CRS info from the merged transform
    # Map each lon/lat query point to a pixel in the merged raster
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    # Convert lon/lat to pixel row/col indices
    rows, cols = rowcol(merged_transform, lon_flat, lat_flat)

    # Clamp indices to valid range
    n_rows, n_cols = merged_data.shape
    rows = np.clip(rows, 0, n_rows - 1)
    cols = np.clip(cols, 0, n_cols - 1)

    # Sample depth at each grid point
    depth_flat = merged_data[rows, cols]

    # CUDEM: negative = below sea level, positive = above (land)
    # Flip sign so water depth is positive, set land (positive elevation) to NaN
    depth_flat = -depth_flat
    depth_flat[depth_flat < 0] = np.nan  # was above sea level (land)

    depth_grid = depth_flat.reshape(lon.shape)

    print(f"  Depth range: {np.nanmin(depth_grid):.1f}m to {np.nanmax(depth_grid):.1f}m")
    print(f"  Cells in 1-3m seagrass zone: {np.sum((depth_grid > 1) & (depth_grid < 3))}")

    return depth_grid