import torch
import torch.nn as nn


class RelativeLpLoss:

    def __init__(self, p=2, size_average=True, reduction=True):
        assert p > 0
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def _relative(self, x, y):
        n = x.size(0)
        diff = torch.norm(x.reshape(n, -1) - y.reshape(n, -1), self.p, dim=1)
        base = torch.norm(y.reshape(n, -1), self.p, dim=1)
        errors = diff / (base + 1e-8)
        if self.reduction:
            return torch.mean(errors) if self.size_average else torch.sum(errors)
        return errors

    def _absolute(self, x, y):
        n = x.size(0)
        h = 1.0 / (x.size(1) - 1.0)
        norms = (h ** (2 / self.p)) * torch.norm(
            x.view(n, -1) - y.view(n, -1), self.p, dim=1
        )
        if self.reduction:
            return torch.mean(norms) if self.size_average else torch.sum(norms)
        return norms

    def __call__(self, x, y):
        return self._relative(x, y)


class FieldDenormalizer:

    def __init__(self, mean, std, eps=1e-8):
        self.mean = torch.from_numpy(mean).float()
        self.std = torch.from_numpy(std).float()
        self.eps = eps

    def decode(self, x):
        x = x.to(self.mean.device)
        if self.mean.ndim == 1:
            mu  = self.mean.view(1, 1, 1, -1)
            sig = self.std.view(1, 1, 1, -1)
        elif self.mean.ndim == 3:
            mu  = self.mean.permute(1, 2, 0).unsqueeze(0)
            sig = self.std.permute(1, 2, 0).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported normalization shape: {self.mean.shape}")
        return x * (sig + self.eps) + mu

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std  = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std  = self.std.cpu()
        return self