import numpy as np
import os
import xarray as xr


OUTPUT_DIR = 'output/data'


def combine_and_save(ds, output_dir=OUTPUT_DIR, label=0.0):
    os.makedirs(output_dir, exist_ok=True)

    q1 = ds['q1']
    q2 = ds['q2']

    new_time = np.arange(1, q1.shape[0] + 1)

    combined = xr.DataArray(
        np.stack([q1.values, q2.values], axis=1),
        dims=('time', 'channel', 'y', 'x'),
        coords={
            'time': new_time,
            'channel': ['q1', 'q2'],
            'y': ds['y'],
            'x': ds['x'],
        },
        name='q1q2'
    )

    combined_ds = combined.to_dataset(name='q1q2')
    out_path = os.path.join(output_dir, f'q1q2_combined_{label}.nc')
    combined_ds.to_netcdf(out_path)