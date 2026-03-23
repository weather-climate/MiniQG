import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer


class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.should_stop = False
        self.best_state  = None
        self.best_opt    = None
        self.best_epoch  = None

    def __call__(self, val_loss, model, optimizer, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = model.state_dict()
            self.best_opt   = optimizer.state_dict()
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def save_checkpoint(model, optimizer, epoch, save_dir, filename='checkpoint.pth'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, filename))


def load_checkpoint(model, optimizer, save_dir):
    best_path  = os.path.join(save_dir, 'best_model.pth')
    ckpt_path  = os.path.join(save_dir, 'checkpoint.pth')

    if os.path.exists(best_path):
        ckpt = torch.load(best_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch'], True

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt['epoch'] + 1, False

    return 0, False


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model     = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device
        self._lrs, self._losses = [], []

    def range_test(self, loader, start_lr=1e-7, end_lr=1.0,
                   n_iter=100, smooth=0.05, diverge_th=5):
        model_state = self.model.state_dict()
        opt_state   = self.optimizer.state_dict()

        for g in self.optimizer.param_groups:
            g['lr'] = start_lr

        lr_mult  = (end_lr / start_lr) ** (1.0 / n_iter)
        lr       = start_lr
        avg_loss = 0.0
        best     = float('inf')
        beta     = 1.0 - smooth

        it = iter(loader)
        for step in range(n_iter):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out  = self.model(x)
            loss = self.criterion(out.reshape(x.size(0), -1), y.reshape(x.size(0), -1))

            avg_loss = beta * avg_loss + (1 - beta) * loss.item() if step else loss.item()
            smoothed = avg_loss / (1 - beta ** (step + 1))

            self._lrs.append(lr)
            self._losses.append(smoothed)

            if smoothed > diverge_th * best:
                break
            if smoothed < best:
                best = smoothed

            loss.backward()
            self.optimizer.step()

            lr *= lr_mult
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(opt_state)

    def suggest(self, skip_start=10, skip_end=5):
        lrs    = self._lrs[skip_start: -skip_end or None]
        losses = self._losses[skip_start: -skip_end or None]
        if not lrs:
            return None
        idx = losses.index(min(losses))
        return lrs[idx] / 8

    def plot(self, skip_start=10, skip_end=5, log_scale=True):
        lrs    = self._lrs[skip_start: -skip_end or None]
        losses = self._losses[skip_start: -skip_end or None]
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Smoothed Loss')
        plt.title('LR Range Test')
        if log_scale:
            plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def train(model, optimizer, scheduler, criterion, train_loader, val_loader,
          test_loader, cfg, save_dir, device, start_epoch=0):

    stopper = EarlyStopping(patience=cfg.get('patience', 50))
    train_losses, val_losses, test_errors = [], [], []

    for ep in range(start_epoch, cfg['epochs']):
        model.train()
        t0 = default_timer()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out.reshape(x.size(0), -1), y.reshape(x.size(0), -1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out.reshape(x.size(0), -1),
                                      y.reshape(x.size(0), -1)).item()

        n_val     = sum(len(b[0]) for b in val_loader)
        val_loss /= n_val

        stopper(val_loss, model, optimizer, ep)

        test_err = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_err += criterion(out.reshape(x.size(0), -1),
                                      y.reshape(x.size(0), -1)).item()

        n_train     = sum(len(b[0]) for b in train_loader)
        n_test      = sum(len(b[0]) for b in test_loader)
        train_loss /= n_train
        test_err   /= n_test

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_errors.append(test_err)

        print(f"Epoch {ep:4d} | {default_timer()-t0:.1f}s | "
              f"Train {train_loss:.6f} | Val {val_loss:.6f} | Test {test_err:.6f}")

        if ep % 10 == 0:
            save_checkpoint(model, optimizer, ep, save_dir)

        if stopper.should_stop:
            print(f"Early stopping at epoch {ep} (best: {stopper.best_epoch})")
            break

    model.load_state_dict(stopper.best_state)
    optimizer.load_state_dict(stopper.best_opt)
    save_checkpoint(model, optimizer, stopper.best_epoch, save_dir, 'best_model.pth')

    return train_losses, val_losses, test_errors


def fine_tune(model, optimizer, scheduler, criterion, train_loader, val_loader,
              test_loader, cfg, save_dir, device):

    alpha   = cfg['alpha']
    beta    = cfg['beta']
    warmup  = cfg.get('warmup_epochs', 5)
    epochs  = cfg.get('epochs_ft', 100)
    patience= cfg.get('patience_ft', 30)

    stopper = EarlyStopping(patience=patience)
    train_losses, val_losses, test_losses = [], [], []

    for ep in range(epochs):
        model.train()
        t0 = default_timer()
        tr_total = tr_n = 0

        for x_t, y_t1, y_t2 in train_loader:
            x_t, y_t1, y_t2 = x_t.to(device), y_t1.to(device), y_t2.to(device)
            bs = x_t.size(0)
            optimizer.zero_grad()

            pred1 = model(x_t)
            l1    = criterion(pred1.reshape(bs, -1), y_t1.reshape(bs, -1))

            if ep < warmup:
                pred2  = model(y_t1)
                w2     = 0.1
            else:
                pred2  = model(pred1)
                w2     = beta

            l2   = criterion(pred2.reshape(bs, -1), y_t2.reshape(bs, -1))
            loss = alpha * l1 + w2 * l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_total += loss.item()
            tr_n     += bs

        scheduler.step()

        model.eval()
        vl_total = vl_n = 0
        with torch.no_grad():
            for x_t, y_t1, y_t2 in val_loader:
                x_t, y_t1, y_t2 = x_t.to(device), y_t1.to(device), y_t2.to(device)
                bs   = x_t.size(0)
                p1   = model(x_t)
                p2   = model(p1)
                l1   = criterion(p1.reshape(bs, -1), y_t1.reshape(bs, -1))
                l2   = criterion(p2.reshape(bs, -1), y_t2.reshape(bs, -1))
                vl_total += (alpha * l1 + beta * l2).item()
                vl_n     += bs

        te_total = te_n = 0
        with torch.no_grad():
            for x_t, y_t1, y_t2 in test_loader:
                x_t, y_t1, y_t2 = x_t.to(device), y_t1.to(device), y_t2.to(device)
                bs   = x_t.size(0)
                p1   = model(x_t)
                p2   = model(p1)
                l1   = criterion(p1.reshape(bs, -1), y_t1.reshape(bs, -1))
                l2   = criterion(p2.reshape(bs, -1), y_t2.reshape(bs, -1))
                te_total += (alpha * l1 + beta * l2).item()
                te_n     += bs

        tr = tr_total / tr_n
        vl = vl_total / vl_n
        te = te_total / te_n

        train_losses.append(tr)
        val_losses.append(vl)
        test_losses.append(te)

        phase = 'warm' if ep < warmup else 'auto'
        print(f"[FT {ep:3d}] {phase} | {default_timer()-t0:.1f}s | "
              f"Train {tr:.6f} | Val {vl:.6f} | Test {te:.6f}")

        stopper(vl, model, optimizer, ep)
        if ep % 10 == 0:
            save_checkpoint(model, optimizer, ep, save_dir, 'checkpoint_ft.pth')

        if stopper.should_stop:
            print(f"Fine-tuning early stop at epoch {ep}")
            break

    model.load_state_dict(stopper.best_state)
    optimizer.load_state_dict(stopper.best_opt)

    torch.save({
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch':                stopper.best_epoch,
        'val_loss':             stopper.best_loss,
        'train_losses':         train_losses,
        'val_losses':           val_losses,
        'test_losses':          test_losses,
    }, os.path.join(save_dir, 'best_model_ft.pth'))

    return train_losses, val_losses, test_losses