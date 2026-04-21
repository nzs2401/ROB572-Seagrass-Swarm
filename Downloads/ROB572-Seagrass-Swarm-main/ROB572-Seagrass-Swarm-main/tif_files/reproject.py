import glob
import subprocess

# First batch
for f in glob.glob("thirdarcsec_DEM_J1342746*.tif"):
    out = f"reprojected_{f}"
    subprocess.run(["gdalwarp", "-t_srs", "EPSG:6318", "-tr", "9.259e-05", "9.259e-05", f, out])

# Second batch
for f in glob.glob("ninearcres_ncei_nintharcsec_dem_J1343192*.tif"):
    out = f"reprojected_{f}"
    subprocess.run(["gdalwarp", "-t_srs", "EPSG:6318", "-tr", "9.259e-05", "9.259e-05", f, out])