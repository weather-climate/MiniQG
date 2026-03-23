import os
import time
import numpy as np
import xarray as xr
import psutil
from multiprocessing import Pool

import github.utils.lwa as lwa


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def process_chunk(args):
    n, pv, z, time_name, chunk_size, nlon, nlat, dlon, dlat, area = args

    mem_before = get_memory_usage()

    lwa_pv_chunk   = np.zeros((chunk_size, nlat, nlon))
    lwa_pv_a_chunk = np.zeros((chunk_size, nlat, nlon))
    lwa_pv_c_chunk = np.zeros((chunk_size, nlat, nlon))

    pv_chunk = pv.isel(**{time_name: slice(n, n + chunk_size)})
    z_chunk  = z.isel(**{time_name: slice(n, n + chunk_size)})

    for t in range(chunk_size):
        pv_2d = pv_chunk.isel(**{time_name: t}).values
        z_2d  = z_chunk.isel(**{time_name: t}).values
        lwa_pv_t, lwa_pv_a_t, lwa_pv_c_t, _, _, _, _, _ = lwa.LWA(
            pv_2d, z_2d, nlon, nlat, dlon, dlat, area
        )
        lwa_pv_chunk[t]   = lwa_pv_t
        lwa_pv_a_chunk[t] = lwa_pv_a_t
        lwa_pv_c_chunk[t] = lwa_pv_c_t

    print(f"Task {n} used {get_memory_usage() - mem_before:.2f} MB")
    return n, lwa_pv_chunk, lwa_pv_a_chunk, lwa_pv_c_chunk


input_file  = "path/to/input.npz"
output_file = "path/to/output.nc"

chunk_days = 6
n_core     = 1

data = np.load(input_file)
q    = data['predictions']
q    = q[:, :, :, 0]

x = np.array([-23, -22.64062, -22.28125, -21.92188, -21.5625, -21.20312, -20.84375,
              -20.48438, -20.125, -19.76562, -19.40625, -19.04688, -18.6875, -18.32812,
              -17.96875, -17.60938, -17.25, -16.89062, -16.53125, -16.17188, -15.8125,
              -15.45312, -15.09375, -14.73438, -14.375, -14.01562, -13.65625,
              -13.29688, -12.9375, -12.57812, -12.21875, -11.85938, -11.5, -11.14062,
              -10.78125, -10.42188, -10.0625, -9.703125, -9.34375, -8.984375, -8.625,
              -8.265625, -7.90625, -7.546875, -7.1875, -6.828125, -6.46875, -6.109375,
              -5.75, -5.390625, -5.03125, -4.671875, -4.3125, -3.953125, -3.59375,
              -3.234375, -2.875, -2.515625, -2.15625, -1.796875, -1.4375, -1.078125,
              -0.71875, -0.359375, 0, 0.359375, 0.71875, 1.078125, 1.4375, 1.796875,
              2.15625, 2.515625, 2.875, 3.234375, 3.59375, 3.953125, 4.3125, 4.671875,
              5.03125, 5.390625, 5.75, 6.109375, 6.46875, 6.828125, 7.1875, 7.546875,
              7.90625, 8.265625, 8.625, 8.984375, 9.34375, 9.703125, 10.0625, 10.42188,
              10.78125, 11.14062, 11.5, 11.85938, 12.21875, 12.57812, 12.9375,
              13.29688, 13.65625, 14.01562, 14.375, 14.73438, 15.09375, 15.45312,
              15.8125, 16.17188, 16.53125, 16.89062, 17.25, 17.60938, 17.96875,
              18.32812, 18.6875, 19.04688, 19.40625, 19.76562, 20.125, 20.48438,
              20.84375, 21.20312, 21.5625, 21.92188, 22.28125, 22.64062])
lon  = x[1::2]
lat0 = lon

tim = np.arange(q.shape[0])

pv = xr.DataArray(
    q,
    dims=["time", "y", "x"],
    coords={"time": tim, "y": lat0, "x": lon},
)
pv = pv.sel(y=slice(-17.5, 17.5))
lat = pv.y.values

z = pv

time_name = 'time'
nlon = len(lon)
nlat = len(lat)
ny   = len(lat0)

LR   = 720e3
dlon = np.diff(lon)[0] * LR
dlat = np.diff(lat)[0] * LR
area = np.ones((nlat, nlon)) * dlon * dlat

ndays       = len(tim)
numlist     = list(range(0, ndays - ndays % chunk_days, chunk_days))
extra_chunk = ndays % chunk_days

LWA_pv  = np.zeros((ndays, nlat, nlon), dtype=np.float32)
LWA_pv_a = np.zeros((ndays, nlat, nlon), dtype=np.float32)
LWA_pv_c = np.zeros((ndays, nlat, nlon), dtype=np.float32)

if __name__ == "__main__":
    start = time.time()

    args = [(n, pv, z, time_name, chunk_days, nlon, nlat, dlon, dlat, area) for n in numlist]
    with Pool(processes=n_core) as pool:
        results = pool.map(process_chunk, args)

    for n, lwa_pv, lwa_pv_a, lwa_pv_c in results:
        LWA_pv[n:n+chunk_days]   = lwa_pv
        LWA_pv_a[n:n+chunk_days] = lwa_pv_a
        LWA_pv_c[n:n+chunk_days] = lwa_pv_c

    if extra_chunk != 0:
        n        = ndays - extra_chunk
        pv_chunk = pv.isel(**{time_name: slice(n, n + extra_chunk)})
        z_chunk  = z.isel(**{time_name: slice(n, n + extra_chunk)})
        buf_pv   = np.zeros((extra_chunk, nlat, nlon))
        buf_pv_a = np.zeros((extra_chunk, nlat, nlon))
        buf_pv_c = np.zeros((extra_chunk, nlat, nlon))

        for t in range(extra_chunk):
            pv_2d = pv_chunk.isel(**{time_name: t}).values
            z_2d  = z_chunk.isel(**{time_name: t}).values
            lwa_pv_t, lwa_pv_a_t, lwa_pv_c_t, _, _, _, _, _ = lwa.LWA(
                pv_2d, z_2d, nlon, nlat, dlon, dlat, area
            )
            buf_pv[t]   = lwa_pv_t
            buf_pv_a[t] = lwa_pv_a_t
            buf_pv_c[t] = lwa_pv_c_t

        LWA_pv[n:n+extra_chunk]   = buf_pv
        LWA_pv_a[n:n+extra_chunk] = buf_pv_a
        LWA_pv_c[n:n+extra_chunk] = buf_pv_c

    LWA_PV  = np.zeros((ndays, ny, nlon), dtype=np.float32)
    LWA_PV_A = np.zeros((ndays, ny, nlon), dtype=np.float32)
    LWA_PV_C = np.zeros((ndays, ny, nlon), dtype=np.float32)

    dy = (ny - nlat) // 2
    LWA_PV[:,  dy:-dy, :] = LWA_pv
    LWA_PV_A[:, dy:-dy, :] = LWA_pv_a
    LWA_PV_C[:, dy:-dy, :] = LWA_pv_c

    ds_LWA = xr.Dataset(
        {
            "LWA_pv":   (["time", "y", "x"], LWA_PV),
            "LWA_pv_a": (["time", "y", "x"], LWA_PV_A),
            "LWA_pv_c": (["time", "y", "x"], LWA_PV_C),
        },
        coords={"time": tim, "y": lat0, "x": lon},
        attrs={"description": "LWA calculated from QGPV and Psi", "source": "Two-layer QGPV model"},
    )

    encoding = {
        "LWA_pv":   {"dtype": "float32"},
        "LWA_pv_a": {"dtype": "float32"},
        "LWA_pv_c": {"dtype": "float32"},
    }

    ds_LWA.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
    print(f"Saved to {output_file} ({time.time() - start:.2f}s)")