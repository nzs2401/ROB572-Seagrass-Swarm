# ROB572-Seagrass-Swarm

### First Time Setup
#### Environment Setup
Before beginning make sure your environment has the following packages installed: python=3.10, geopandas, rasterio, matplotlib, scikit-learn, cartopy. If these are not previously installed or you have other packages that conflict with these we recommend creating a new clean environment using the following code:
```bash
conda create -n rob572_env python=3.10
conda activate rob572_env
conda install -c conda-forge geopandas rasterio matplotlib scikit-learn cartopy gdal

```
#### Collect Code
You can then either download and unzip the code from this github or branch it using the following steps: 
```bash

# Clone the repository
git clone git@github.com:nzs2401/ROB572-Seagrass-Swarm.git

# Enter the folder
cd ROB572-Seagrass-Swarm
```

##### Before you start working:
```bash
# Get latest changes from teammates
git pull
```

##### After making changes:
```bash
# See what you changed
git status

# Add your changed files
git add .

# Commit with a descriptive message
git commit -m "Describe what you changed"

# Share with team
git push
```
#### Collect Data that's too large for github
There are four files/folders that were too large to be put in github that are needed for this simulation. Below are links to the relevant data (they should be accessible to anyone at University of Michigan)

Seagrass (1).zip

https://drive.google.com/file/d/1GRa9SPnZ6kV8lzeaap5Qp1-DkcCPXhui/view?usp=sharing

thirdarcsec_DEM_J1342746.zip

https://drive.google.com/file/d/1aTzdPDOaz1STeH7RIT_daWG83DVHfdcS/view?usp=sharing

more_data_on_seagrass_growth_usace2022_gulf_coast_dem_J1342825.zip

https://drive.google.com/file/d/1w7oY9kzRPG9kuJ9MhR1kfSXRYlVt2tou/view?usp=sharing

ninearcres_ncei_nintharcsec_dem_J1343192.zip

https://drive.google.com/file/d/1QqajA6yp7ny6u4e2nJs9Q8qC0MouCs1O/view?usp=sharing

These files should all be downloaded and unzipped then...
- Move seagrass.gpkg to the main folder (ROB572-Seagrass-Swarm-main)
- Add the remaining three zip files in one of the following ways depending on your computer system:
1. For MAC: Use your terminal and the following commands to add the three zip files
```bash
cd tif_files

for f in thirdarcsec_DEM_J1342746*.tif; do
    gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 "$f" "reprojected_${f}"
done

for f in ninearcres_ncei_nintharcsec_dem_J1343192*.tif; do
    gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 "$f" "reprojected_${f}"
done

for f in more_data_on_seagrass_growth_usace2022_gulf_coast_dem_J1342825*.tif; do
    gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 "$f" "reprojected_${f}"
done

cd ..
```
2. For Windows: Unzip all the files and move only the tiff files in each file into tif_files. (it may say that these files are already there but they have a bad habit of depreciating so move them in any way). Then in your terminal run the following code:
```bash
cd tif_files

Get-ChildItem "more_data_on_seagrass_growth_usace2022_gulf_coast_dem_J1342825*.tif" | ForEach-Object {gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 $_.FullName ("reprojected_" + $_.Name)}

Get-ChildItem "thirdarcsec_DEM_J1342746*.tif" | ForEach-Object {gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 $_.FullName ("reprojected_" + $_.Name)}

Get-ChildItem "ninearcres_ncei_nintharcsec_dem_J1343192*.tif" | ForEach-Object {gdalwarp -t_srs EPSG:6318 -tr 9.259e-05 9.259e-05 $_.FullName ("reprojected_" + $_.Name)}

```
#### Code
Run Mapping_of_Viable_Sites.py to build environment files needed to run the rest of the code:
``` bash
conda activate rob572_env
python Mapping_of_Viable_Sites.py
```
*** When run this produces the numpy files needed to run the rest of the files and a viability_map.png (which is included in this github). If your viability_map.png doesn't show all the planting sites visible in github file some of the tif_files were not added correctly (most often this occurs with the tif_files in "more_data_on_seagrass_growth_..."). In order to fix this you might have to manually select and drag the tif files from more_data_on_seagrass_growth_usace2022_gulf_coast_dem_J1342825 into the tif_files!

Following this you should be able to navigate into the afsa (Artificial Fish Swarm Algorithm), mpa (Marine Predator Algorithm), and woa (Whale Optimization Algorithm) folders and run the python files contained within. Each file simulates the mission with agents exploring the environment using the titular algorithm with the goal of identifying and mapping seagrass coverage. At the end of each run these files create a png file with the results of the simulated mission.
