import os
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


# ── Pipeline switches ────────────────────────────────────────────────────────
ENABLE_SPATIAL_CROP       = False
ENABLE_SPATIAL_POOL       = False
ENABLE_TEMPORAL_SUBSAMPLE = False
ENABLE_SPECTRAL_FILTER    = False
ENABLE_DETREND            = False
ENABLE_NORMALIZATION      = False
ENABLE_RANDOM_SUBSAMPLE   = False
ENABLE_BLOCKING_REMOVAL   = False
# ─────────────────────────────────────────────────────────────────────────────


def _spatial_crop(ds, config):
    return ds.isel(
        y=slice(config.get('H_start', 0), config.get('H_end') or ds['q1q2'].shape[2]),
        x=slice(config.get('W_start', 0), config.get('W_end') or ds['q1q2'].shape[3]),
    )


def _spatial_pool(ds, config):
    factor = config.get('pool_factor', 2)
    return ds.coarsen(y=factor, x=factor, boundary="trim").mean()


def _temporal_subsample(ds, config):
    stride = config.get('time_stride', 2)
    times = ds['time'].values
    keep = (times % stride) == (times[0] % stride)
    return ds.isel(time=keep)


def _spectral_filter(ds, config):
    sigma = config.get('spectral_sigma', 1)
    data = ds['q1q2'].values.copy()
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            data[t, c] = uniform_filter(data[t, c], size=sigma)
    return ds.assign({'q1q2': (ds['q1q2'].dims, data)})


def _detrend(ds, config):
    data = ds['q1q2'].values.copy()
    T = data.shape[0]
    t = np.arange(T)
    for c in range(data.shape[1]):
        flat = data[:, c].reshape(T, -1)
        coeffs = np.polyfit(t, flat, deg=1)
        trend = np.outer(coeffs[0], t).T.reshape(data[:, c].shape) + coeffs[1].reshape(1, *data.shape[2:])
        data[:, c] -= trend
    return ds.assign({'q1q2': (ds['q1q2'].dims, data)})


def _normalize(time_series, train_indices, config):
    method = config.get('normalization_method', 'global')
    epsilon = 1e-8
    C, H, W = time_series.shape[1], time_series.shape[2], time_series.shape[3]
    train_data = time_series[train_indices]

    if method == "global":
        norm_mean = np.zeros(C, dtype=float)
        norm_std  = np.zeros(C, dtype=float)
        for ch in range(C):
            m = train_data[:, ch].mean()
            s = max(train_data[:, ch].std(), epsilon)
            time_series[:, ch] = (time_series[:, ch] - m) / s
            norm_mean[ch], norm_std[ch] = m, s

    elif method == "per_location":
        norm_mean = np.zeros((C, H, W), dtype=float)
        norm_std  = np.zeros((C, H, W), dtype=float)
        for ch in range(C):
            for y in range(H):
                for x in range(W):
                    m = train_data[:, ch, y, x].mean()
                    s = max(train_data[:, ch, y, x].std(), epsilon)
                    time_series[:, ch, y, x] = (time_series[:, ch, y, x] - m) / s
                    norm_mean[ch, y, x], norm_std[ch, y, x] = m, s

    elif method == "none":
        norm_mean = np.zeros((C, H, W), dtype=float)
        norm_std  = np.ones((C, H, W),  dtype=float)

    else:
        raise ValueError(f"Unknown normalization_method: {method}")

    return time_series, norm_mean, norm_std


def _remove_blocking_days(ds, blocking_days):
    keep = ~np.isin(ds['time'].values, blocking_days)
    return ds.isel(time=keep), list(blocking_days)


def _random_subsample(ds, target_size, config, original_train_end, random_seed=None):
    n = ds['q1q2'].shape[0]
    samples_to_remove = n - target_size
    if samples_to_remove <= 0:
        return ds.copy(), []

    num_blocks = config.get('num_random_blocks', 10)
    available = np.arange(0, min(original_train_end, n))

    if random_seed is not None:
        np.random.seed(random_seed)

    base = samples_to_remove // num_blocks
    rem  = samples_to_remove % num_blocks
    block_sizes = [base + (1 if i < rem else 0) for i in range(num_blocks)]

    indices_to_remove, used_ranges = [], []
    for block_size in block_sizes:
        for _ in range(1000):
            max_start = len(available) - block_size
            start = np.random.randint(0, max_start + 1)
            end   = start + block_size
            if all(end <= s or start >= e for s, e in used_ranges):
                indices_to_remove.extend(available[start:end])
                used_ranges.append((start, end))
                break
        else:
            raise RuntimeError("Could not place all random blocks without overlap.")

    indices_to_remove = np.array(indices_to_remove)
    keep = np.ones(n, dtype=bool)
    keep[indices_to_remove] = False
    removed_times = ds['time'].values[indices_to_remove].tolist()
    return ds.isel(time=keep), removed_times


def _compute_splits(time_coords, original_time_coords, original_train_end, original_valid_end):
    t_train = original_time_coords[original_train_end - 1]
    t_valid = original_time_coords[original_valid_end - 1]
    train_idx = np.where((time_coords >= original_time_coords[0]) & (time_coords <= t_train))[0]
    valid_idx = np.where((time_coords > t_train) & (time_coords <= t_valid))[0]
    test_idx  = np.where(time_coords > t_valid)[0]
    return train_idx, valid_idx, test_idx


def prepare_dataset(
    input_path,
    output_dir,
    config,
    blocking_days=None,
    train_ratio=0.8,
    valid_ratio=0.1,
    test_ratio=0.1,
    random_seed=None,
):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    ds = xr.open_dataset(input_path)
    T_orig = ds['q1q2'].shape[0]
    original_train_end = int(T_orig * train_ratio)
    original_valid_end = int(T_orig * (train_ratio + valid_ratio))
    original_time_coords = ds['time'].values

    datasets = {}
    removed_info = {}

    if ENABLE_BLOCKING_REMOVAL and blocking_days:
        ds_b, removed = _remove_blocking_days(ds, blocking_days)
        datasets['blocking'] = ds_b
        removed_info['blocking'] = removed

    if ENABLE_RANDOM_SUBSAMPLE:
        target = datasets['blocking']['q1q2'].shape[0] if 'blocking' in datasets else ds['q1q2'].shape[0]
        ds_s, removed = _random_subsample(ds, target, config, original_train_end, random_seed)
        datasets['subsampled'] = ds_s
        removed_info['subsampled'] = removed

    if not datasets:
        raise ValueError("No datasets to process.")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for dtype, ds_cur in datasets.items():

        if ENABLE_SPATIAL_CROP:
            ds_cur = _spatial_crop(ds_cur, config)

        original_mean = ds_cur['q1q2'].mean(dim='time')
        original_std  = ds_cur['q1q2'].std(dim='time')

        if ENABLE_SPATIAL_POOL:
            ds_cur = _spatial_pool(ds_cur, config)

        if ENABLE_DETREND:
            ds_cur = _detrend(ds_cur, config)

        if ENABLE_TEMPORAL_SUBSAMPLE:
            ds_cur = _temporal_subsample(ds_cur, config)

        if ENABLE_SPECTRAL_FILTER:
            ds_cur = _spectral_filter(ds_cur, config)

        time_series = ds_cur['q1q2'].values
        time_coords = ds_cur['time'].values

        train_idx, valid_idx, test_idx = _compute_splits(
            time_coords, original_time_coords, original_train_end, original_valid_end
        )

        if ENABLE_NORMALIZATION:
            time_series, norm_mean, norm_std = _normalize(time_series, train_idx, config)
        else:
            C, H, W = time_series.shape[1], time_series.shape[2], time_series.shape[3]
            norm_mean = np.zeros((C, H, W), dtype=float)
            norm_std  = np.ones((C, H, W),  dtype=float)

        out_path = os.path.join(output_dir, f"{dtype}_{len(removed_info[dtype])}removed.npz")

        np.savez_compressed(
            out_path,
            time_series=time_series,
            time_coords=time_coords,
            train_indices=train_idx,
            valid_indices=valid_idx,
            test_indices=test_idx,
            normalization_mean=norm_mean,
            normalization_std=norm_std,
            original_mean=original_mean.values,
            original_std=original_std.values,
            removed_days=removed_info[dtype],
        )

        results[dtype] = {
            'path':   out_path,
            'shape':  time_series.shape,
            'splits': (len(train_idx), len(valid_idx), len(test_idx)),
        }

    return results