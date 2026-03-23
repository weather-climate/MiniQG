"""
Adaptive Fourier Neural Operator (AFNO) backbone.

Architecture reference:
    Pathak, J., Subramanian, S., Harrington, P., Raja, S., Chattopadhyay, A.,
    Mardani, M., Kurth, T., Hall, D., Li, Z., Azizzadenesheli, K.,
    Hassanzadeh, P., Kashinath, K., & Anandkumar, A. (2022).
    FourCastNet: A global data-driven high-resolution weather model using
    adaptive Fourier neural operators.
    arXiv preprint arXiv:2202.11214.
"""

import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _init_truncated_normal(tensor, mean, std, lo, hi):
    with torch.no_grad():
        tensor.normal_(mean, std).clamp_(min=lo, max=hi)
    return tensor


def _stochastic_depth(x, drop_p, training):
    if drop_p == 0.0 or not training:
        return x
    survival = 1.0 - drop_p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
    return x / survival * mask


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _stochastic_depth(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, act, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FrequencyMixer(nn.Module):

    def __init__(self, channels, n_groups, shrink_threshold, mode_fraction):
        super().__init__()
        assert channels % n_groups == 0, \
            f"channels ({channels}) must be divisible by n_groups ({n_groups})"

        self.channels         = channels
        self.n_groups         = n_groups
        self.group_dim        = channels // n_groups
        self.shrink_threshold = shrink_threshold
        self.mode_fraction    = mode_fraction

        self.W_in  = nn.Parameter(torch.empty(2, n_groups, self.group_dim, self.group_dim))
        self.b_in  = nn.Parameter(torch.zeros(2, n_groups, self.group_dim))
        self.W_out = nn.Parameter(torch.empty(2, n_groups, self.group_dim, self.group_dim))
        self.b_out = nn.Parameter(torch.zeros(2, n_groups, self.group_dim))

        for part in range(2):
            nn.init.kaiming_normal_(self.W_in[part],  mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.W_out[part], mode='fan_out', nonlinearity='relu')

    def _cmul(self, x_re, x_im, W_re, W_im, b_re, b_im):
        out_re = F.relu(
            torch.einsum('...gi,gio->...go', x_re, W_re)
            - torch.einsum('...gi,gio->...go', x_im, W_im)
            + b_re
        )
        out_im = F.relu(
            torch.einsum('...gi,gio->...go', x_im, W_re)
            + torch.einsum('...gi,gio->...go', x_re, W_im)
            + b_im
        )
        return out_re, out_im

    def forward(self, x):
        shortcut = x
        dtype    = x.dtype
        x        = x.float()
        B, H, W, C = x.shape

        X = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        X = X.reshape(B, H, W // 2 + 1, self.n_groups, self.group_dim)

        total_modes = H // 2 + 1
        kept_modes  = int(total_modes * self.mode_fraction)
        lo, hi      = total_modes - kept_modes, total_modes + kept_modes

        mid_re = torch.zeros_like(X.real)
        mid_im = torch.zeros_like(X.imag)
        out_re = torch.zeros_like(X.real)
        out_im = torch.zeros_like(X.imag)

        mid_re[:, lo:hi, :kept_modes], mid_im[:, lo:hi, :kept_modes] = self._cmul(
            X.real[:, lo:hi, :kept_modes], X.imag[:, lo:hi, :kept_modes],
            self.W_in[0], self.W_in[1], self.b_in[0], self.b_in[1]
        )

        out_re[:, lo:hi, :kept_modes] = (
            torch.einsum('...gi,gio->...go', mid_re[:, lo:hi, :kept_modes], self.W_out[0])
            - torch.einsum('...gi,gio->...go', mid_im[:, lo:hi, :kept_modes], self.W_out[1])
            + self.b_out[0]
        )
        out_im[:, lo:hi, :kept_modes] = (
            torch.einsum('...gi,gio->...go', mid_im[:, lo:hi, :kept_modes], self.W_out[0])
            + torch.einsum('...gi,gio->...go', out_re[:, lo:hi, :kept_modes], self.W_out[1])
            + self.b_out[1]
        )

        out = F.softshrink(torch.stack([out_re, out_im], dim=-1), lambd=self.shrink_threshold)
        out = torch.view_as_complex(out.contiguous())
        out = out.reshape(B, H, W // 2 + 1, C)

        out = torch.fft.irfft2(out, s=(H, W), dim=(1, 2), norm='ortho').type(dtype)
        return out + shortcut


class AFNOBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, drop_path, n_groups, shrink_threshold, mode_fraction):
        super().__init__()
        self.norm1  = nn.LayerNorm(dim, eps=1e-6)
        self.mixer  = FrequencyMixer(dim, n_groups, shrink_threshold, mode_fraction)
        self.norm2  = nn.LayerNorm(dim, eps=1e-6)
        self.ffn    = FeedForward(dim, int(dim * mlp_ratio), dim, nn.GELU, dropout)
        self.drop_p = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.drop_p(self.ffn(self.norm2(x)))
        return x


class GridPatchEmbed(nn.Module):

    def __init__(self, grid_size, patch_size, in_chans, embed_dim):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(grid_size, int):
            grid_size  = (grid_size, grid_size)

        self.grid_size   = grid_size
        self.patch_size  = patch_size
        self.num_patches = (grid_size[0] // patch_size[0]) * (grid_size[1] // patch_size[1])
        self.proj        = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.grid_size, \
            f"Input ({H}×{W}) does not match expected grid {self.grid_size}"
        return self.proj(x).flatten(2).transpose(1, 2)


class AFNONet(nn.Module):

    def __init__(
        self,
        grid_size,
        patch_size,
        in_chans,
        out_chans,
        embed_dim,
        depth,
        mlp_ratio,
        dropout,
        drop_path_rate,
        n_groups,
        shrink_threshold,
        mode_fraction,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.grid_size  = grid_size
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        self.patch_embed = GridPatchEmbed(grid_size, patch_size, in_chans, embed_dim)
        n_patches        = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.pos_drop  = nn.Dropout(p=dropout)

        dpr = [v.item() for v in torch.linspace(0, drop_path_rate, depth)]

        self.n_h = grid_size[0] // patch_size[0]
        self.n_w = grid_size[1] // patch_size[1]

        self.blocks = nn.ModuleList([
            AFNOBlock(
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                n_groups=n_groups,
                shrink_threshold=shrink_threshold,
                mode_fraction=mode_fraction,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, out_chans * patch_size[0] * patch_size[1], bias=False)

        _init_truncated_normal(self.pos_embed, mean=0.0, std=0.02, lo=-2.0, hi=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            _init_truncated_normal(m.weight, mean=0.0, std=0.02, lo=-2.0, hi=2.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def _encode(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed
        x = self.pos_drop(x).reshape(B, self.n_h, self.n_w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self._encode(x)
        x = self.head(x)
        x = rearrange(
            x,
            'b h w (p1 p2 c) -> b c (h p1) (w p2)',
            p1=self.patch_size[0], p2=self.patch_size[1],
            h=self.grid_size[0] // self.patch_size[0],
            w=self.grid_size[1] // self.patch_size[1],
        )
        return x.permute(0, 2, 3, 1)

    def param_count(self):
        return sum(reduce(operator.mul, p.size()) for p in self.parameters())