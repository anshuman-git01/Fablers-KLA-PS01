"""Restoration metrics.

PSNR and SSIM are implemented directly here so training/validation needs no extra dependency
and everything runs on-GPU (KLA scores end-to-end throughput, and CPU metric round-trips are a
classic bottleneck). LPIPS requires a pretrained network and is added later for reporting only.

GT is normalized to [0,1], so ``data_range=1.0`` throughout.
"""

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB, averaged over the batch.

    Computed per-image then averaged, not over the pooled MSE — pooling would let one easy
    image flatter a batch containing a hard one.
    """
    pred, target = pred.float(), target.float()
    dims = tuple(range(1, pred.ndim))
    mse = ((pred - target) ** 2).mean(dim=dims).clamp_min(1e-12)
    return (10.0 * torch.log10(data_range ** 2 / mse)).mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).view(1, 1, window_size, window_size)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Mean SSIM over the batch, single-channel, Gaussian-windowed (Wang et al. 2004)."""
    pred, target = pred.float(), target.float()
    w = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, w, padding=pad)
    mu_t = F.conv2d(target, w, padding=pad)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

    sigma_p = F.conv2d(pred * pred, w, padding=pad) - mu_p2
    sigma_t = F.conv2d(target * target, w, padding=pad) - mu_t2
    sigma_pt = F.conv2d(pred * target, w, padding=pad) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_pt + c1) * (2 * sigma_pt + c2)
    den = (mu_p2 + mu_t2 + c1) * (sigma_p + sigma_t + c2)
    return (num / den).mean()
