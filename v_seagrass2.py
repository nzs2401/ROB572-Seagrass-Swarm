def seagrass(lon, lat):  # lon and lat are 2D arrays
    print("Loading packages")
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box, Point
    from pathlib import Path

    print("Loading seagrass data")
    repo_root = Path(__file__).parent
    gdf = gpd.read_file(repo_root / "Seagrass.gpkg")

    print("Taking subset")
    minx, miny, maxx, maxy = -85, 24, -78, 32
    bbox = box(minx, miny, maxx, maxy)
    subset = gdf[gdf.intersects(bbox)].copy()

    # Build spatial index for faster lookups
    print("Building spatial index")
    subset_sindex = subset.sindex

    print("Creating points and array")
    # Flatten the lon/lat grids and create GeoDataFrame of points
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    points_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(lon_flat, lat_flat)],
        crs=subset.crs
    )

    # Spatial join: find which polygon contains each point
    print("Performing spatial join")
    joined = gpd.sjoin(points_gdf, subset[['geometry', 'cover']], how='left', predicate='within')

    # Initialize cover_grid with default "Unknown"
    cover_grid = np.full(lon.shape, "Unknown", dtype=object)
    # Fill cover_grid with joined results
    cover_vals = joined['cover'].values
    cover_grid.flat[:] = cover_vals

    print("Done")
    return cover_grid