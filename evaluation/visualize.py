import numpy as np
import matplotlib.pyplot as plt
import torch

from evaluation.metrics import compute_metrics


def _contour_quad(axes, fields, titles, cmap='seismic', n_levels=100):
    vmax = max(np.abs(f).max() for f in fields)
    vmin = -vmax
    for ax, field, title in zip(axes, fields, titles):
        ny, nx = field.shape
        lvls = np.linspace(vmin, vmax, n_levels)
        ax.contourf(np.arange(nx), np.arange(ny), field,
                    levels=lvls, cmap=cmap, extend='both')
        ax.set_title(title)



def visualize_prediction(model, x_test, y_test, sample_idx, denormalizer,
                         device, channel_names=None, error_threshold=0.05):

    channel_names = channel_names or ['q1', 'q2']
    x_s = x_test[sample_idx:sample_idx + 1].to(device)
    y_s = y_test[sample_idx:sample_idx + 1].to(device)

    with torch.no_grad():
        pred = model(x_s)

    denormalizer.cpu()
    pred_dn = denormalizer.decode(pred.cpu()).numpy()
    true_dn = denormalizer.decode(y_s.cpu()).numpy()
    denormalizer.cuda()

    for c, name in enumerate(channel_names):
        p = pred_dn[0, :, :, c]
        t = true_dn[0, :, :, c]
        d = p - t

        pct_err     = np.abs(d) / (np.abs(t) + 1e-10) * 100
        filtered    = np.where(pct_err > error_threshold * 100, np.abs(d), 0.0)

        max_abs     = max(np.abs(p).max(), np.abs(t).max())
        diff_max    = np.abs(d).max()
        filt_max    = filtered.max() if filtered.max() > 0 else 1.0

        ny, nx = p.shape
        x_ax, y_ax = np.arange(nx), np.arange(ny)
        lvl_main = np.linspace(-max_abs, max_abs, 100)
        lvl_diff = np.linspace(-diff_max, diff_max, 100)
        lvl_filt = np.linspace(0, filt_max, 100)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].contourf(x_ax, y_ax, p, levels=lvl_main, cmap='seismic', extend='both')
        axes[0].set_title(f'Predicted {name}')
        axes[1].contourf(x_ax, y_ax, t, levels=lvl_main, cmap='seismic', extend='both')
        axes[1].set_title(f'True {name}')
        axes[2].contourf(x_ax, y_ax, d, levels=lvl_diff, cmap='seismic', extend='both')
        axes[2].set_title(f'Signed Error {name}')
        axes[3].contourf(x_ax, y_ax, filtered, levels=lvl_filt, cmap='gray_r', extend='max')
        axes[3].set_title(f'|Error| > {int(error_threshold*100)}% — {name}')

        for ax in axes:
            plt.colorbar(ax.collections[0] if ax.collections else ax.images[0], ax=ax)

        fig.suptitle(f'Sample {sample_idx} — {name}', fontsize=13)
        plt.tight_layout()
        plt.show()


def autoregressive_rollout(model, x_init, steps, device):
    model.eval()
    x = x_init.to(device).clone()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = model(x)
            preds.append(x.cpu())
    return preds


def visualize_autoregressive(model, x_test, y_test, sample_idx, steps,
                              denormalizer, device, channel_names=None,
                              time_gap=1):
    
    channel_names = channel_names or ['q1', 'q2']
    if sample_idx + steps >= len(y_test):
        print(f"Not enough future steps for sample {sample_idx}.")
        return

    x_init  = x_test[sample_idx:sample_idx + 1]
    y_true  = y_test[sample_idx:sample_idx + steps + 1].to(device)
    preds   = autoregressive_rollout(model, x_init, steps, device)

    denormalizer.cpu()

    for t, pred_t in enumerate(preds, start=1):
        pred_dn = denormalizer.decode(pred_t).numpy()
        true_dn = denormalizer.decode(y_true[t - 1:t].cpu()).numpy()

        for c, name in enumerate(channel_names):
            p, t_f = pred_dn[0, :, :, c], true_dn[0, :, :, c]
            d       = p - t_f

            metrics = compute_metrics(p, t_f)

            max_abs  = max(np.abs(p).max(), np.abs(t_f).max())
            diff_max = np.abs(d).max() or 1e-8
            ny, nx   = p.shape
            xa, ya   = np.arange(nx), np.arange(ny)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].contourf(xa, ya, p,           levels=np.linspace(-max_abs, max_abs, 100),  cmap='seismic', extend='both')
            axes[0].set_title(f'Pred {name} t+{t}')
            axes[1].contourf(xa, ya, t_f,         levels=np.linspace(-max_abs, max_abs, 100),  cmap='seismic', extend='both')
            axes[1].set_title(f'True {name} t+{t}')
            axes[2].contourf(xa, ya, d,           levels=np.linspace(-diff_max, diff_max, 100), cmap='seismic', extend='both')
            axes[2].set_title('Signed Error')
            axes[3].contourf(xa, ya, np.abs(d),   levels=np.linspace(0, diff_max, 100),         cmap='gray_r',  extend='max')
            axes[3].set_title('Abs Error')

            for ax in axes:
                plt.colorbar(ax.collections[0], ax=ax)

            fig.suptitle(
                f'Sample {sample_idx} | Step t+{t} | {name} '
                f'(R²={metrics["R2"]:.3f}, RelL2={metrics["RelL2"]:.3f})',
                fontsize=11,
            )
            plt.tight_layout()
            plt.show()

    denormalizer.cuda()


def _extract_hovmoller_slice(data, spatial_dim, spatial_idx):
    if spatial_dim == 'x':
        return data[:, spatial_idx, :, :]
    return data[:, :, spatial_idx, :]


def hovmoller_rollout(model, x_init, denormalizer, steps, device):
    preds = autoregressive_rollout(model, x_init, steps, device)
    denormalizer.cpu()
    out = np.stack([denormalizer.decode(p).numpy() for p in preds], axis=0)
    denormalizer.cuda()
    return out


def plot_hovmoller_comparison(pred_seq, true_seq, spatial_dim='x',
                              spatial_idx=None, channel_names=None):

    channel_names = channel_names or ['q1', 'q2']
    T, _, H, W, C = pred_seq.shape

    if spatial_idx is None:
        spatial_idx = H // 2 if spatial_dim == 'x' else W // 2

    pred_hov = _extract_hovmoller_slice(pred_seq[:, 0], spatial_dim, spatial_idx)
    true_hov = _extract_hovmoller_slice(true_seq[:, 0], spatial_dim, spatial_idx)

    space_size  = W if spatial_dim == 'x' else H
    axis_label  = 'X index' if spatial_dim == 'x' else 'Y index'
    time_coords = np.arange(T)
    space_coords= np.arange(space_size)
    S, TT       = np.meshgrid(space_coords, time_coords)

    for c, name in enumerate(channel_names):
        p = pred_hov[:, :, c]
        t = true_hov[:, :, c]
        d = p - t

        vmax     = max(np.abs(p).max(), np.abs(t).max())
        diff_max = np.abs(d).max() or 1e-8

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, field, title, vlo, vhi in [
            (axes[0], p, f'Predicted {name}',    -vmax,     vmax),
            (axes[1], t, f'True {name}',          -vmax,     vmax),
            (axes[2], d, f'Difference {name}',    -diff_max, diff_max),
        ]:
            ax.contourf(S, TT, field, levels=50, cmap='seismic',
                        vmin=vlo, vmax=vhi, extend='both')
            ax.invert_yaxis()
            ax.set_xlabel(axis_label)
            ax.set_ylabel('Time step')
            ax.set_title(title)
            plt.colorbar(ax.collections[0], ax=ax)

        fig.suptitle(f'Hovmöller — {name}', fontsize=13)
        plt.tight_layout()
        plt.show()