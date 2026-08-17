"""A deliberately small baseline restoration network.

This is the *sanity-check / baseline* model, not the final architecture. Its only jobs are to
prove the pipeline wires up correctly and to give us a recorded baseline to beat (required by
the submission: "compare at least one baseline with the final method").

Design:
  * Fully convolutional — no fixed input size. Although the shipped data is uniformly
    128x128 -> 256x256 (CLAUDE.md §3a), the docs claim 512x512 may appear, so nothing here
    hardcodes a resolution.
  * Residual formulation: the network predicts a *correction* on top of a bilinear x2 upsample.
    That means it starts near a sensible solution and only has to learn the detail and the
    denoising, which converges much faster than predicting pixels from scratch.
  * Output is left **unclipped** by default. KLA does not clip or renormalize before scoring
    (CLAUDE.md §6), so range handling is an explicit, deliberate choice made at inference time
    via ``clamp_output``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyRestorer(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        num_blocks: int = 4,
        scale: int = 2,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.clamp_output = clamp_output

        body: list[nn.Module] = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            body += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*body)

        # sub-pixel upsample: cheaper and artefact-free vs transposed conv
        self.to_residual = nn.Conv2d(channels, scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

        # start as a pure bilinear upsampler: zero-init the residual head
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        residual = self.shuffle(self.to_residual(self.body(x)))
        out = base + residual
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out


class ResBlock(nn.Module):
    """Pre-activation-free residual block: conv-relu-conv + identity."""

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.relu(self.c1(x), inplace=True))


class UNetRestorer(nn.Module):
    """Multi-scale encoder-decoder restorer.

    Motivation (CLAUDE.md §15c): the flat-conv ``TinyRestorer`` baseline plateaued at 28.36 dB
    while still *underfitting* — its ~17 px receptive field gives it no multi-scale context, so
    it cannot separate speckle from genuine fine texture. Downsampling to 1/2 and 1/4 resolution
    widens the effective receptive field by roughly an order of magnitude at modest cost, and
    the skip connections carry the high-frequency detail that super-resolution needs back to the
    output.

    Shape flow for the shipped data (128x128 -> 256x256):
        input 128 -> enc1 128 -> enc2 64 -> enc3 32 -> bottleneck 32
                  -> dec3 64 -> dec2 128 -> PixelShuffle x2 -> 256

    Retains the two tricks that worked on the baseline:
      * global residual on top of a bilinear x2 upsample,
      * zero-init output head, so the network *starts* as an exact bilinear upsampler.

    Fully convolutional: any input whose spatial dims are divisible by 4 works, so a 256x256
    input (-> 512x512) is handled without changes.
    """

    def __init__(
        self,
        base: int = 32,
        blocks_per_level: int = 2,
        scale: int = 2,
        clamp_output: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.clamp_output = clamp_output
        c1, c2, c3 = base, base * 2, base * 4

        def stack(ch: int) -> nn.Sequential:
            return nn.Sequential(*[ResBlock(ch) for _ in range(blocks_per_level)])

        self.stem = nn.Conv2d(1, c1, 3, padding=1)
        self.enc1 = stack(c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = stack(c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.bottleneck = stack(c3)

        # nearest-upsample + conv rather than transposed conv: avoids checkerboard artefacts,
        # which matter here because KLA penalises "artificial patterns or ringing".
        self.up2 = nn.Conv2d(c3, c2, 3, padding=1)
        self.fuse2 = nn.Conv2d(c2 * 2, c2, 1)
        self.dec2 = stack(c2)
        self.up1 = nn.Conv2d(c2, c1, 3, padding=1)
        self.fuse1 = nn.Conv2d(c1 * 2, c1, 1)
        self.dec1 = stack(c1)

        self.to_residual = nn.Conv2d(c1, scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        nn.init.zeros_(self.to_residual.weight)
        nn.init.zeros_(self.to_residual.bias)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(self.stem(x))            # full res
        s2 = self.enc2(self.down1(s1))          # 1/2
        b = self.bottleneck(self.down2(s2))     # 1/4

        d2 = self.up2(self._up(b))              # 1/2
        d2 = self.dec2(self.fuse2(torch.cat([d2, s2], dim=1)))
        d1 = self.up1(self._up(d2))             # full res
        d1 = self.dec1(self.fuse1(torch.cat([d1, s1], dim=1)))

        base = F.interpolate(
            x, scale_factor=self.scale, mode="bilinear", align_corners=False
        )
        out = base + self.shuffle(self.to_residual(d1))
        if self.clamp_output:
            out = out.clamp(0.0, 1.0)
        return out


ARCHITECTURES = {"tiny": TinyRestorer, "unet": UNetRestorer}


def build_model(arch: str, **kwargs) -> nn.Module:
    """Construct a model by name, passing only the kwargs that architecture accepts."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    cls = ARCHITECTURES[arch]
    import inspect

    accepted = set(inspect.signature(cls).parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
