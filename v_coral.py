#This function takes in the meshed location and returns the values at each point..
def coral(lon, lat):#lon and lat are 2D arrays 

    #Libraries Needed
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree


    #Load Coral Data
    coral = pd.read_csv(r"C:\Users\gamcs\Downloads\deep_sea_corals_6186_86ea_f111.csv", low_memory=False)
    
    #Convert Data to from strings to numbers
    coral['longitude'] = pd.to_numeric(coral['longitude'], errors='coerce')
    coral['latitude'] = pd.to_numeric(coral['latitude'], errors='coerce')
    coral['DepthInMeters'] = pd.to_numeric(coral['DepthInMeters'], errors='coerce')
    
    #Drop rows where conversion failed (NaN)
    coral = coral.dropna(subset=['longitude', 'latitude', 'DepthInMeters'])

    #Extract Arrays
    x = coral['longitude'].values
    y = coral['latitude'].values
    z = coral['DepthInMeters'].values
    
    #Flatten the meshgrid for vectorized computation
    lon_flat = lon.flatten()
    lat_flat = lat.flatten()
    
    # Interpolate depth over the grid (griddata estimates depth at every grid point)
    #### HERE IS THE SOURCE OF OUR PROBLEMS!!!!!!!!!
    z_grid_flat = griddata(points=(x, y), values=z, xi=(lon_flat, lat_flat), method='linear')
    #Convert back to 2D grid
    z_grid = z_grid_flat.reshape(lon.shape)
    
    # Compute coral presence with nearest neighbor
    tree = cKDTree(np.c_[x, y]) #Create spatial index of coral locations
    distances, indices = tree.query(np.c_[lon_flat, lat_flat]) #For each grid point find nearest coral & give us its info and the distance to it
    coralpresence_flat = (distances <= 0.0000269).astype(int) #If coral is less than 3m gives 1, if greater than 3m gives 0.
    coralpresence = coralpresence_flat.reshape(lon.shape)
    
    return coralpresence, z_grid
