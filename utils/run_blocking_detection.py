import numpy as np
import xarray as xr

import github.utils.blocking_detection as blocking_detection


input_file  = "path/to/LWA_input.nc"
output_file = "path/to/blocking_output.nc"

ds    = xr.open_dataset(input_file)
lwa   = ds['LWA_pv'].values
lwa_a = ds['LWA_pv_a'].values
lwa_c = ds['LWA_pv_c'].values
time  = ds['time'].values
lon   = ds['x'].values
lat   = ds['y'].values
ds.close()

nlon = len(lon)
nlat = len(lat)

LR   = 720e3
dlon = np.diff(lon)[0] * LR
dlat = np.diff(lat)[0] * LR
area = np.ones((nlat, nlon)) * dlon * dlat

var_type           = 0
thresh_option      = 0
duration_threshold = 5
size_factor        = 1

blocking_detection.blocking_detection(
    lwa, lwa_a, lwa_c, time, area, dlon, dlat, LR,
    duration_threshold, size_factor, output_file, var_type, thresh_option
)

print('Blocking detection finished successfully.')