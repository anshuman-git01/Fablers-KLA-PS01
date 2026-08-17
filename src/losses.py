"""Loss functions for restoration training.

Motivation (CLAUDE.md §15c failure analysis): pure L1 regresses to the conditional mean and
over-smooths stochastic texture — the model's dominant failure mode. SSIM and LPIPS are two
thirds of KLA's scored metric and both penalise exactly that texture loss, so optimising them
directly is the indicated fix.

Weighting: the three terms have very different natural magnitudes at our operating point
(L1 ~0.029, 1-SSIM ~0.21, LPIPS ~0.30). Default weights are chosen so each contributes roughly
equally rather than letting the larger-magnitude perceptual terms swamp pixel fidelity:

    1.00 * 0.029  = 0.029
    0.15 * 0.21   = 0.032
    0.10 * 0.30   = 0.030

Re-tune if the operating point moves substantially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ssim


class CombinedLoss(nn.Module):
    """w_l1 * L1  +  w_ssim * (1 - SSIM)  +  w_lpips * LPIPS.

    Any weight set to 0 disables that term (and skips its computation entirely, so setting
    ``w_lpips=0`` avoids constructing the LPIPS network at all).

    The LPIPS network is frozen and kept in eval mode; its parameters are never handed to the
    optimizer. LPIPS expects 3-channel input in [-1, 1], so predictions are clamped to [0, 1],
    replicated to 3 channels and rescaled. The clamp is deliberate: it keeps LPIPS inside the
    domain it was trained on. Gradient still flows for every in-range pixel.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_ssim: float = 0.15,
        w_lpips: float = 0.10,
        lpips_net: str = "alex",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.w_l1, self.w_ssim, self.w_lpips = w_l1, w_ssim, w_lpips
        self.lpips = None
        if w_lpips > 0:
            import warnings

            import lpips as lpips_lib

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net = lpips_lib.LPIPS(net=lpips_net, verbose=False)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            if device is not None:
                net = net.to(device)
            # assigned via object.__setattr__ so it is not registered as a submodule and can
            # never leak into model.parameters()
            object.__setattr__(self, "lpips", net)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (total_loss, per-term scalars for logging)."""
        parts: dict[str, float] = {}
        total = pred.new_zeros(())

        if self.w_l1 > 0:
            l1 = F.l1_loss(pred, target)
            total = total + self.w_l1 * l1
            parts["l1"] = l1.item()

        if self.w_ssim > 0:
            s = ssim(pred.clamp(0, 1), target)
            total = total + self.w_ssim * (1.0 - s)
            parts["ssim"] = s.item()

        if self.w_lpips > 0:
            p3 = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            t3 = target.repeat(1, 3, 1, 1) * 2 - 1
            lp = self.lpips(p3, t3).mean()
            total = total + self.w_lpips * lp
            parts["lpips"] = lp.item()

        parts["total"] = total.item()
        return total, parts


def build_loss(name: str, device: torch.device, **kw) -> CombinedLoss:
    """``l1`` reproduces the original baseline exactly; ``l1_ssim_lpips`` is the combined loss."""
    if name == "l1":
        return CombinedLoss(w_l1=1.0, w_ssim=0.0, w_lpips=0.0, device=device)
    if name == "l1_ssim":
        return CombinedLoss(
            w_l1=kw.get("w_l1", 1.0), w_ssim=kw.get("w_ssim", 0.15), w_lpips=0.0, device=device
        )
    if name == "l1_ssim_lpips":
        return CombinedLoss(
            w_l1=kw.get("w_l1", 1.0),
            w_ssim=kw.get("w_ssim", 0.15),
            w_lpips=kw.get("w_lpips", 0.10),
            device=device,
        )
    raise ValueError(f"unknown loss {name!r}")


LOSSES = ("l1", "l1_ssim", "l1_ssim_lpips")
