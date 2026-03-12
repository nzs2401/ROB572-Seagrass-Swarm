#Mapping of Viable Sites
import numpy as np
import v_seagras
import v_coral
import random
import matplotlib.pyplot as plt

latq = np.linspace(24,32,400)
lonq = np.linspace(-86,-80,400)
LonGrid, LatGrid = np.meshgrid(lonq,latq)
print('Grid Made!')
seagrass_coverage = v_seagras.seagrass(LonGrid, LatGrid)
print('Seagrass Collected!  Now for Coral...')
coral_presence, depth = coral(LonGrid, LatGrid)
print('Depth & Coral Presence Collected')

percover = np.zeros_like(depth)
good_depth = (depth > 1) & (depth < 3)
percover[good_depth & (seagrass_coverage == "Nan")] = 1 #0
percover[good_depth & (seagrass_coverage == "51 - 100%  ")] = 0.25 #0.75
percover[good_depth & (seagrass_coverage == "90 - 100% ")] = 0.5 #0.95
percover[good_depth & (seagrass_coverage == "1 - 89% ")] = 0.55 #0.45
percover[good_depth & (seagrass_coverage == "10 - 50% ")] = 0.7 #0.30
percover[good_depth & (seagrass_coverage == "Continuous ")] = 0 #1
percover[good_depth & (seagrass_coverage == "Discontinuous ")] = 0.5 #0.5
percover[good_depth & (seagrass_coverage == "<50% ")] = 0.75 #0.25
percover[good_depth & (seagrass_coverage == "Unknown ")] = np.random.rand(np.sum(good_depth & (seagrass_coverage == "Unknown " )))
percover[good_depth & (seagrass_coverage == ">50% ")] = 0.25 #.75
percover[good_depth & (seagrass_coverage == "Continuous Seagrass ")] = 0 #1
percover[good_depth & (seagrass_coverage == "Patchy (Discontinuous) Seagrass ")] = 0.5 #0.5
percover[depth > 3] = 0 #Too Deep
percover[depth <= 1] = 0 #Too Shallow   

#Plotting!
plt.figure(figsize=(10,7))
plt.pcolormesh(LonGrid, LatGrid, percover, shading='auto')
plt.colorbar(label='Viability Score')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Viability of Sites for Seagrass Restoration')
plt.show()
#This shows 0 viability when coral function uses linear for z_grid_flat
#This shows irregular viability when coral function uses nearest for z_grid_flat


#For error checking...
print(np.min(percover), np.max(percover)) #Shows max and min viability score
print(np.min(depth), np.max(depth)) #Shows max and min depth
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
