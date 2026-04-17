#Mapping of Viable Sites
import numpy as np
import v_seagrass2
import v_coral
import v_depth
import random
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# trying now
latq = np.linspace(24, 32, 600)
lonq = np.linspace(-86, -80, 600)

LonGrid, LatGrid = np.meshgrid(lonq,latq)
print('Grid Made!')
seagrass_coverage = v_seagrass2.seagrass(LonGrid, LatGrid)
print('Seagrass Collected!  Now for Coral...')
coral_presence = v_coral.coral(LonGrid, LatGrid)
print('Depth & Coral Presence Collected')
# add depth...
depth = v_depth.depth(LonGrid, LatGrid)

percover = np.zeros_like(depth)
good_depth = (depth > 1) & (depth < 3)


# DEBUG - add these lines
print('Cells with good depth:', np.sum(good_depth))
print('Unique seagrass values at good depth cells:')
#print(np.unique(seagrass_coverage[good_depth])) #THIS WILL CAUSE PROBLEMS!

# Assign viability scores based on seagrass coverage and depth

percover[good_depth & (seagrass_coverage == "Nan")] = 1
percover[good_depth & (seagrass_coverage == "51 - 100%")] = 0.25
percover[good_depth & (seagrass_coverage == "90 - 100%")] = 0.5
percover[good_depth & (seagrass_coverage == "1 - 89%")] = 0.55
percover[good_depth & (seagrass_coverage == "10 - 50%")] = 0.7
percover[good_depth & (seagrass_coverage == "Continuous")] = 0
percover[good_depth & (seagrass_coverage == "Discontinuous")] = 0.5
percover[good_depth & (seagrass_coverage == "<50%")] = 0.75
percover[good_depth & (seagrass_coverage == "Unknown")] = 0.5
percover[good_depth & (seagrass_coverage == "")] = 0.5
percover[good_depth & (seagrass_coverage == ">50%")] = 0.25
percover[good_depth & (seagrass_coverage == "Continuous Seagrass")] = 0
percover[good_depth & (seagrass_coverage == "Patchy (Discontinuous) Seagrass")] = 0.5
percover[depth > 3] = 0 #Too Deep
percover[depth <= 1] = 0 #Too Shallow  

# Plot coastline
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-86, -80, 24, 32], crs=ccrs.PlateCarree())
mesh = ax.pcolormesh(LonGrid, LatGrid, percover, shading='auto', 
                     transform=ccrs.PlateCarree(), cmap='viridis')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
plt.colorbar(mesh, ax=ax, label='Viability Score')
plt.title('Viability of Sites for Seagrass Restoration')
# # Use when run in background
plt.savefig('viability_map.png', dpi=150, bbox_inches='tight')
plt.close()
# # Use when run in background

#For error checking...
print(np.nanmin(percover), np.nanmax(percover)) #Shows max and min viability score
print(np.nanmin(depth), np.nanmax(depth)) #Shows max and min depth
#This second line shows NAN when coral function uses linear for z_grid_flat
#This second line shows 0 and 3000 when coral function uses nearest for z_grid_flat

#Looking at depth scores to see where we have NaN values 
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.pcolormesh(LonGrid, LatGrid, np.isnan(depth), shading='auto')
plt.colorbar(label='True = NaN depth')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations without depth data')
plt.show()

# Use when run in the background
plt.savefig('nan_depth_map.png')
plt.close()
# Use when run in the background

# Save grids for AFSA, WOA, and MPA integrations
print("Saving Grids...")
np.save('percover.npy', percover)
np.save('depth_grid.npy', depth)
np.save('lon_grid.npy', LonGrid)
np.save('lat_grid.npy', LatGrid)
print("Grids saved!")
