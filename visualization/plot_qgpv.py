import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter


def plot_qgpv_snapshot(ds, time_step, label=None, cmap='seismic', quantile_min=0.0, quantile_max=1.0):
    q1 = ds['q1q2'].sel(time=time_step).values[0]
    q2 = ds['q1q2'].sel(time=time_step).values[1]
    x = ds['x'].values
    y = ds['y'].values

    vmin = min(np.quantile(q1, quantile_min), np.quantile(q2, quantile_min))
    vmax = max(np.quantile(q1, quantile_max), np.quantile(q2, quantile_max))

    if vmin == vmax:
        vmax = vmin + 1e-5

    levels = np.linspace(vmin, vmax, 100)
    title_suffix = f', L={label}' if label is not None else ''

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(x, y, q1, levels=levels, cmap=cmap, extend='both')
    plt.colorbar(label='QGPV (q1)')
    plt.title(f'QGPV q1 (upper layer) at t={time_step}{title_suffix}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.contourf(x, y, q2, levels=levels, cmap=cmap, extend='both')
    plt.colorbar(label='QGPV (q2)')
    plt.title(f'QGPV q2 (lower layer) at t={time_step}{title_suffix}')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()


def plot_qgpv_snapshot_with_pooling(ds, time_step, pool_factor=2,
                                    cmap='seismic', quantile_min=0.0, quantile_max=1.0):
    q1 = ds['q1q2'].sel(time=time_step).values[0]
    q2 = ds['q1q2'].sel(time=time_step).values[1]

    def downsample_field(field, factor, mode='average'):
        h, w = field.shape
        assert h % factor == 0 and w % factor == 0
        if mode == 'average':
            return uniform_filter(field, size=factor, mode='nearest')[::factor, ::factor]
        elif mode == 'max':
            return maximum_filter(field, size=factor, mode='nearest')[::factor, ::factor]
        raise ValueError("mode must be 'average' or 'max'")

    q1_avg = downsample_field(q1, pool_factor, 'average')
    q2_avg = downsample_field(q2, pool_factor, 'average')
    q1_max = downsample_field(q1, pool_factor, 'max')
    q2_max = downsample_field(q2, pool_factor, 'max')

    all_vals = np.concatenate([q1.ravel(), q2.ravel(), q1_avg.ravel(), q2_avg.ravel(),
                                q1_max.ravel(), q2_max.ravel()])
    vmin = np.quantile(all_vals, quantile_min)
    vmax = np.quantile(all_vals, quantile_max)
    levels = np.linspace(vmin, vmax, 100)

    x_orig = np.arange(q1.shape[1])
    y_orig = np.arange(q1.shape[0])
    x_small = np.arange(q1_avg.shape[1])
    y_small = np.arange(q1_avg.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    panels = [
        (axes[0, 0], x_orig, y_orig, q1, 'q1 original'),
        (axes[0, 1], x_small, y_small, q1_avg, f'q1 avg pooled (×{pool_factor})'),
        (axes[0, 2], x_small, y_small, q1_max, f'q1 max pooled (×{pool_factor})'),
        (axes[1, 0], x_orig, y_orig, q2, 'q2 original'),
        (axes[1, 1], x_small, y_small, q2_avg, f'q2 avg pooled (×{pool_factor})'),
        (axes[1, 2], x_small, y_small, q2_max, f'q2 max pooled (×{pool_factor})'),
    ]

    for ax, x, y, data, title in panels:
        cf = ax.contourf(x, y, data, levels=levels, cmap=cmap, extend='both')
        ax.set_title(title)
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)

    fig.suptitle(f'QGPV Snapshot (t={time_step})', fontsize=16)
    plt.show()