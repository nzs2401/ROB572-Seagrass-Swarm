# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:10:21 2026

@author: gamcs
"""

def coral(lon, lat):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from shapely.geometry import Point
    from scipy.interpolate import griddata


# Coral Data
    coral = pd.read_csv(r"C:\Users\gamcs\Downloads\deep_sea_corals_6186_86ea_f111.csv")
#Convert Data to form we can use
    coral['longitude'] = pd.to_numeric(coral['longitude'], errors='coerce')
    coral['latitude'] = pd.to_numeric(coral['latitude'], errors='coerce')
    coral['DepthInMeters'] = pd.to_numeric(coral['DepthInMeters'], errors='coerce')
# Drop rows where conversion failed (NaN)
    coral = coral.dropna(subset=['longitude', 'latitude'])
#Now define  values....
    x = coral['longitude'].values
    y = coral['latitude'].values
    z = coral['DepthInMeters'].values

#Define Mesh Space
    latq = np.linspace(y.min(),y.max(),400)
    lonq = np.linspace(x.min(),y.max(),400)
    LonGrid, LatGrid = np.meshgrid(lonq,latq)
#Mesh for entire area... plotted with values   
    #Zgrid = griddata(points=(x, y), values=z, xi=(LonGrid, LatGrid), method='linear')
    #fig = plt.figure(figsize=(10,7))
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x, y, z, c='m', marker='o', label='Data Points')
    #ax.plot_surface(LonGrid, LatGrid, Zgrid, cmap='viridis', alpha=0.7)
    #ax.set_xlabel('Longitude')
    #ax.set_ylabel('Latitude')
    #ax.set_zlabel('Depth (m)')
    #ax.set_title('Natural Neighbor Interpolation (linear approx)')
    #plt.show()
#Defining values
    lon= -83.134
    lat = 29.29063
#Use neighbors/linear to find depth at areas
    z_new = griddata(points=(x, y), values=z, xi=(lon, lat),method='linear') #nearest, cubic
    #print("Interpolated depth:", z_new)

#By hand nearest neighbor
    coral['distance'] = np.sqrt((coral['longitude']-lon)**2+(coral['latitude']-lat)**2)
    nearest = coral.loc[coral['distance'].idxmin()]
    #print("Nearest Depth (m):", nearest['DepthInMeters'])

    if nearest['distance'] > 0.0000269:
        coralpresence = 0
    else: coralpresence = 1

    #print(coralpresence)
    
###################################
#Original plotting of data    
      #plt.figure(figsize=(8,6))
      #sc = plt.scatter(x, y, c=z, s=20, cmap='viridis', edgecolor='k')  # s=marker size
      #cb = plt.colorbar(sc)
      #cb.set_label('depth (m)')
      #plt.clim(0, 1000)

      #plt.xlabel('Longitude')
      #plt.ylabel('Latitude')
      #plt.title('Coral Locations')
      #plt.grid(True)
      #plt.show()  
    
    
    return(coralpresence,z_new)