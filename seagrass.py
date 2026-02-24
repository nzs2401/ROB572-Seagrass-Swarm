# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:35:45 2026

@author: gamcs
"""
 
def seagrass(lon, lat):
    
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from shapely.geometry import box
    from shapely.geometry import Point

# Seagrass Data
    gdf = gpd.read_file(r"C:\Users\gamcs\Downloads\Seagrass (1)\Seagrass.gpkg")
    gdf.head()

    gdf.total_bounds
    minx, miny, maxx, maxy = -85, 24, -78, 32
    bbox = box(minx, miny, maxx, maxy)
    subset = gdf[gdf.intersects(bbox)]
    #subset.plot()
    #plt.show()


    lon = -83.1346
    lat = 29.29063

    point = Point(lon, lat)
    result = subset[subset.contains(point)]
    print(result["cover"].values)

return{result["cover"].values}


