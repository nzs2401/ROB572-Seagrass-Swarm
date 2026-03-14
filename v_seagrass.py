#This function takes in the meshed location and returns the values at each point..
def seagrass(lon, lat): #lon and lat are 2D arrays 

    #Libraries Needed
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box
    from shapely.geometry import Point

    #Load the Seagrass data
    # Depends on what computer your are running on...
    ### Comment this out if not running on specific computer ###
    # gdf = gpd.read_file(r"C:\Users\gamcs\Downloads\Seagrass (1)\Seagrass.gpkg")
    ### Comment this out if not running on specific computer ###

    ### Comment this out if this doesn't work for you ###
    from pathlib import Path
    repo_root = Path(__file__).parent
    gdf = gpd.read_file(repo_root / "Seagrass.gpkg")
    ### Comment this out if this doesn't work for you ###

    gdf.head()

    #Define a subset of the Seagrass data
    gdf.total_bounds
    minx, miny, maxx, maxy = -85, 24, -78, 32
    bbox = box(minx, miny, maxx, maxy)
    #Filter dataset to only that subset
    subset = gdf[gdf.intersects(bbox)].copy()

    #DCreate empty array of same size as lon/lat grid
    cover_grid = np.empty(lon.shape, dtype=object)
    cover_grid[:] = "Unknown"  # default if no point found
    
    # Flatten grids for iteration (Grid to vector)
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()

    #Loop through coordinates
    for i, (lon, lat) in enumerate(zip(lon_flat, lat_flat)): #For each coordinate pair
        point = Point(lon, lat) #Convert coordinate to a geometric point
        res = subset[subset.contains(point)] #Check what polygon contains it
        if not res.empty:
            cover_grid.flat[i] = res.iloc[0]["cover"]  # If polygon exists take cover attribute and store it
    
    return cover_grid





