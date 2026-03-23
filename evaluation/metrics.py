import numpy as np
import torch


def to_numpy(x):
    return x.numpy() if torch.is_tensor(x) else x


def compute_metrics(pred, true):

    pred = to_numpy(pred).ravel().astype(float)
    true = to_numpy(true).ravel().astype(float)

    residuals = pred - true
    rmse      = float(np.sqrt(np.mean(residuals ** 2)))
    mae       = float(np.mean(np.abs(residuals)))

    value_range = true.max() - true.min()
    nrmse       = rmse / (value_range + 1e-8)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    r2     = float(1.0 - ss_res / (ss_tot + 1e-8))

    rel_l2 = float(
        np.sqrt(np.sum(residuals ** 2)) / (np.sqrt(np.sum(true ** 2)) + 1e-8)
    )

    within_5pct = np.mean(np.abs(residuals) <= 0.05 * (np.abs(true) + 1e-8))

    return {
        'RMSE':      rmse,
        'NRMSE':     nrmse,
        'MAE':       mae,
        'R2':        r2,
        'RelL2':     rel_l2,
        'Acc5pct':   float(within_5pct),
    }


def evaluate_all_channels(predictions, targets, channel_names=None):

    predictions = to_numpy(predictions)
    targets     = to_numpy(targets)
    n_channels  = predictions.shape[-1]
    names       = channel_names or [f'ch{i}' for i in range(n_channels)]

    results = {}
    for i, name in enumerate(names):
        results[name] = compute_metrics(predictions[..., i], targets[..., i])
    results['overall'] = compute_metrics(predictions, targets)
    return results