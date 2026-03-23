import os
import sys
import gc
import numpy as np
import torch
import torch.utils.data as td

from models.afno    import AFNONet
from models.losses  import RelativeLpLoss, FieldDenormalizer
from training.trainer import (
    LRFinder, train, fine_tune, load_checkpoint, save_checkpoint
)
from evaluation.metrics    import evaluate_all_channels
from evaluation.visualize  import (
    visualize_prediction, visualize_autoregressive,
    hovmoller_rollout, plot_hovmoller_comparison
)


torch.cuda.empty_cache()
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(0)
np.random.seed(0)

cfg = {
    'data_path':  'path/to/your_dataset.npz',
    'save_dir':   'path/to/model_output',

    'in_chans':         2,
    'out_chans':        2,
    'patch_size':       None,
    'embed_dim':        None,
    'depth':            None,
    'n_groups':         None,
    'mlp_ratio':        4.0,
    'dropout':          0.0,
    'drop_path_rate':   0.1,
    'shrink_threshold': 0.01,
    'mode_fraction':    1.0,

    'batch_size':       None,
    'epochs':           None,
    'learning_rate':    None,
    'weight_decay':     1e-4,
    'patience':         50,
    'scheduler':        'cosine',
    't_max':            None,
    'eta_min':          1e-7,

    'use_lr_finder':    True,
    'lr_find_start':    1e-7,
    'lr_find_end':      1.0,
    'lr_find_iter':     100,

    'fine_tune':        False,
    'epochs_ft':        100,
    't_max_ft':         100,
    'learning_rate_ft': None,
    'alpha':            None,
    'beta':             None,
    'warmup_epochs':    5,
    'patience_ft':      30,

    'time_gap':         None,
    'channel_names':    ['q1', 'q2'],
}


def load_data(cfg):
    data = np.load(cfg['data_path'], allow_pickle=True)

    ts            = data['time_series']
    time_coords   = data['time_coords']
    train_idx     = data['train_indices']
    valid_idx     = data['valid_indices']
    test_idx      = data['test_indices']
    norm_mean     = data['normalization_mean']
    norm_std      = data['normalization_std']

    cfg['grid_size'] = (ts.shape[2], ts.shape[3])

    ts = ts.transpose(0, 2, 3, 1)

    def consecutive_pairs(indices, time_gap):
        times = time_coords[indices]
        mask  = np.where(np.diff(times) == time_gap)[0]
        return ts[indices[mask]], ts[indices[mask + 1]]

    gap = cfg['time_gap']
    x_tr, y_tr = consecutive_pairs(train_idx, gap)
    x_va, y_va = consecutive_pairs(valid_idx, gap)
    x_te, y_te = consecutive_pairs(test_idx,  gap)

    def to_tensor(arr):
        return torch.from_numpy(arr).float()

    tensors = {
        'x_train': to_tensor(x_tr), 'y_train': to_tensor(y_tr),
        'x_val':   to_tensor(x_va), 'y_val':   to_tensor(y_va),
        'x_test':  to_tensor(x_te), 'y_test':  to_tensor(y_te),
        'time_coords': time_coords,
        'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx,
    }
    denorm = FieldDenormalizer(norm_mean, norm_std).cuda()
    return tensors, denorm


def make_loaders(tensors, cfg):
    bs  = cfg['batch_size']
    def loader(x, y, shuffle):
        return td.DataLoader(td.TensorDataset(x.to(device), y.to(device)),
                             batch_size=bs, shuffle=shuffle)
    return (
        loader(tensors['x_train'], tensors['y_train'], True),
        loader(tensors['x_val'],   tensors['y_val'],   False),
        loader(tensors['x_test'],  tensors['y_test'],  False),
    )


def make_triplet_loaders(tensors, time_coords, cfg):
    """Build loaders for the fine-tuning stage (consecutive triplets)."""
    gap = cfg['time_gap']

    def consecutive_triplets(indices):
        times = time_coords[indices]
        mask  = np.where((np.diff(times[:-1]) == gap) & (np.diff(times[1:]) == gap))[0]
        if len(mask) == 0:
            empty = torch.empty(0, *tensors['x_train'].shape[1:])
            return empty, empty, empty
        ts_full = tensors['x_train'].numpy()
        xs  = torch.from_numpy(ts_full[indices[mask    ]]).float()
        y1s = torch.from_numpy(ts_full[indices[mask + 1]]).float()
        y2s = torch.from_numpy(ts_full[indices[mask + 2]]).float()
        return xs, y1s, y2s

    def loader(x, y1, y2, shuffle):
        return td.DataLoader(
            td.TensorDataset(x.to(device), y1.to(device), y2.to(device)),
            batch_size=cfg['batch_size'], shuffle=shuffle,
        )

    x_tr, y1_tr, y2_tr = consecutive_triplets(tensors['train_idx'])
    x_va, y1_va, y2_va = consecutive_triplets(tensors['valid_idx'])
    x_te, y1_te, y2_te = consecutive_triplets(tensors['test_idx'])

    return (
        loader(x_tr, y1_tr, y2_tr, True),
        loader(x_va, y1_va, y2_va, False),
        loader(x_te, y1_te, y2_te, False),
    )


def build_model(cfg):
    model = AFNONet(
        grid_size        = cfg['grid_size'],
        patch_size       = cfg['patch_size'],
        in_chans         = cfg['in_chans'],
        out_chans        = cfg['out_chans'],
        embed_dim        = cfg['embed_dim'],
        depth            = cfg['depth'],
        mlp_ratio        = cfg['mlp_ratio'],
        dropout          = cfg['dropout'],
        drop_path_rate   = cfg['drop_path_rate'],
        n_groups         = cfg['n_groups'],
        shrink_threshold = cfg['shrink_threshold'],
        mode_fraction    = cfg['mode_fraction'],
    ).to(device)
    return model


def build_scheduler(optimizer, cfg):
    s = cfg['scheduler']
    if s == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['t_max'], eta_min=cfg['eta_min'])
    if s == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.get('step_size', 100), gamma=cfg.get('gamma', 0.5))
    if s == 'exp':
        dr, ds = cfg.get('decay_rate', 0.95), cfg.get('decay_steps', 10)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda ep: dr ** (ep / ds))
    return None


if __name__ == '__main__':
    tensors, denorm = load_data(cfg)
    tr_loader, va_loader, te_loader = make_loaders(tensors, cfg)

    model     = build_model(cfg)
    criterion = RelativeLpLoss(size_average=False)

    lr = cfg['learning_rate']
    if cfg['use_lr_finder']:
        tmp_opt    = torch.optim.Adam(model.parameters(), lr=1e-7)
        finder     = LRFinder(model, tmp_opt, criterion, device)
        finder.range_test(tr_loader,
                          start_lr=cfg['lr_find_start'],
                          end_lr=cfg['lr_find_end'],
                          n_iter=cfg['lr_find_iter'])
        finder.plot()
        lr = finder.suggest() or lr

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=max(lr, 1e-5),
                                 weight_decay=cfg['weight_decay'])
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch, skip = load_checkpoint(model, optimizer, cfg['save_dir'])
    if not skip:
        train_losses, val_losses, test_errors = train(
            model, optimizer, scheduler, criterion,
            tr_loader, va_loader, te_loader,
            cfg, cfg['save_dir'], device,
            start_epoch=start_epoch,
        )

    if cfg['fine_tune']:
        ft_tr, ft_va, ft_te = make_triplet_loaders(
            tensors, tensors['time_coords'], cfg)
        ft_optimizer = torch.optim.Adam(model.parameters(),
                                        lr=cfg['learning_rate_ft'],
                                        weight_decay=cfg['weight_decay'])
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            ft_optimizer, T_max=cfg['t_max_ft'], eta_min=0)
        fine_tune(model, ft_optimizer, ft_scheduler, criterion,
                  ft_tr, ft_va, ft_te,
                  cfg, cfg['save_dir'], device)

    model.eval()
    x_te = tensors['x_test'].to(device)
    y_te = tensors['y_test'].to(device)

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x, y = x.to(device), y.to(device)
            all_preds.append(model(x).cpu())
            all_true.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_true  = torch.cat(all_true)

    denorm.cpu()
    preds_dn = denorm.decode(all_preds)
    true_dn  = denorm.decode(all_true)
    denorm.cuda()

    results = evaluate_all_channels(preds_dn, true_dn, cfg['channel_names'])
    for ch, mets in results.items():
        print(f'\n{ch}:')
        for k, v in mets.items():
            print(f'  {k}: {v:.4f}')

    for idx in range(3):
        visualize_prediction(model, x_te, y_te, idx, denorm, device,
                             cfg['channel_names'])

    for idx in range(2):
        visualize_autoregressive(model, x_te, y_te, idx, steps=8,
                                 denormalizer=denorm, device=device,
                                 channel_names=cfg['channel_names'],
                                 time_gap=cfg['time_gap'])

    for idx in range(2):
        x_init = x_te[idx:idx + 1]
        preds  = hovmoller_rollout(model, x_init, denorm, steps=8, device=device)

        denorm.cpu()
        true_hov = np.stack([
            denorm.decode(y_te[idx + t:idx + t + 1].cpu()).numpy()
            for t in range(8)
        ])
        denorm.cuda()

        plot_hovmoller_comparison(preds, true_hov,
                                  channel_names=cfg['channel_names'])