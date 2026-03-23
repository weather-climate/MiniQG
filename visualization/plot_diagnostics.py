import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d


SPATIAL_STEP = None
MAX_PLOTS = 256


def plot_histograms(bins, hist_q1, hist_q2, label):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(bins[:-1], hist_q1, width=np.diff(bins), alpha=0.7, color='blue')
    plt.xlabel('q1 Value')
    plt.ylabel('Density')
    plt.title(f'{label} q1 Distribution')

    plt.subplot(1, 2, 2)
    plt.bar(bins[:-1], hist_q2, width=np.diff(bins), alpha=0.7, color='green')
    plt.xlabel('q2 Value')
    plt.ylabel('Density')
    plt.title(f'{label} q2 Distribution')

    plt.tight_layout()
    plt.show()


def plot_pixelwise_curves(filename):
    ds = xr.open_dataset(filename)
    q1q2 = ds['q1q2']

    q1_data = q1q2.sel(channel='q1')[1:].values
    q2_data = q1q2.sel(channel='q2')[1:].values

    time_steps, height, width = q1_data.shape
    step = SPATIAL_STEP if SPATIAL_STEP is not None else max(1, height // 16)

    spatial_points = [(y, x) for y in range(0, height, step) for x in range(0, width, step)]
    if len(spatial_points) > MAX_PLOTS:
        spatial_points = spatial_points[:MAX_PLOTS]

    time = np.arange(time_steps)
    time_smooth = np.linspace(0, time_steps - 1, 500)
    n_cols = 3
    n_rows = (len(spatial_points) + n_cols - 1) // n_cols

    all_q1_ranges = [np.max(q1_data[:, y, x]) - np.min(q1_data[:, y, x]) for y, x in spatial_points]
    all_q2_ranges = [np.max(q2_data[:, y, x]) - np.min(q2_data[:, y, x]) for y, x in spatial_points]
    q1_span = np.max(all_q1_ranges) * 1.05
    q2_span = np.max(all_q2_ranges) * 1.05

    for (q_data, q_span, color, label) in [
        (q1_data, q1_span, 'blue', 'Q1'),
        (q2_data, q2_span, 'red', 'Q2'),
    ]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        fig.suptitle(f'{label} Time Series at Spatial Locations', fontsize=16, y=0.98)

        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (y, x) in enumerate(spatial_points):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            vals = q_data[:, y, x]
            f_interp = interp1d(time, vals, kind='cubic')
            ax.plot(time_smooth, f_interp(time_smooth), color=color, alpha=0.8, linewidth=1.5)
            center = (np.min(vals) + np.max(vals)) / 2
            ax.set_ylim(center - q_span / 2, center + q_span / 2)
            ax.set_title(f'{label}: y={y}, x={x}', fontsize=10)
            ax.set_xlabel('Time Step', fontsize=8)
            ax.set_ylabel(f'{label} Value', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        for idx in range(len(spatial_points), n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        plt.tight_layout()
        plt.show()

    ds.close()


def plot_field_histograms(filename, label, bin_range=None, n_bins=100):
    ds = xr.open_dataset(filename)
    arr = ds['q1q2'].values

    q1 = arr[:, 0, :, :].ravel()
    q2 = arr[:, 1, :, :].ravel()
    q1 = q1[np.isfinite(q1)]
    q2 = q2[np.isfinite(q2)]

    if bin_range is None:
        lo = min(q1.min(), q2.min())
        hi = max(q1.max(), q2.max())
        bin_range = (lo, hi)

    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_width = bins[1] - bins[0]

    hist_q1, _ = np.histogram(q1, bins=bins, density=False)
    hist_q2, _ = np.histogram(q2, bins=bins, density=False)

    hist_q1 = hist_q1 / (len(q1) * bin_width) if len(q1) > 0 else hist_q1
    hist_q2 = hist_q2 / (len(q2) * bin_width) if len(q2) > 0 else hist_q2

    plot_histograms(bins, hist_q1, hist_q2, label)

    ds.close()