"""Dataset and loading helpers for paired NoisyLR -> GT restoration.

Design notes:
  * Arrays on disk are float32 2-D ``(H, W)``. We add a channel axis to get ``(1, H, W)``.
  * **NoisyLR values are NOT clipped.** Out-of-range values carry information about local
    speckle strength (see CLAUDE.md §3b), and KLA explicitly warned against clipping them away.
    Any range handling belongs at the model output, not the input.
  * Stem lists come from configs/split_*.txt so the split is defined in exactly one place and
    a directory glob can never silently pull in the wrong files.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .paths import SUFFIX


def read_stems(split_file: Path) -> list[str]:
    """Read a configs/split_*.txt file into a sorted list of stems."""
    return sorted(Path(split_file).read_text().split())


def load_array(path: Path) -> torch.Tensor:
    """Load one .npy into a float32 tensor of shape (1, H, W)."""
    a = np.load(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected 2-D array, got shape {a.shape}")
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)


def dihedral(
    lr: torch.Tensor, gt: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one of the 8 dihedral transforms (4 rotations x optional flip) to BOTH tensors.

    The identical geometric transform must hit LR and GT or the pair stops corresponding.
    These are the safe augmentations for this task: they are exactly label-preserving, they do
    not resample pixels (so they introduce no interpolation blur and do not disturb the noise
    statistics we are trying to learn), and they multiply the effective dataset by 8.

    ``k`` in [0, 8).
    """
    if k & 4:
        lr, gt = torch.flip(lr, dims=[-1]), torch.flip(gt, dims=[-1])
    r = k & 3
    if r:
        lr, gt = torch.rot90(lr, r, dims=[-2, -1]), torch.rot90(gt, r, dims=[-2, -1])
    return lr.contiguous(), gt.contiguous()


class PairedRestorationDataset(Dataset):
    """Yields ``(lr, gt)`` tensors, each shaped ``(1, H, W)``.

    Parameters
    ----------
    gt_dir, lr_dir:
        Directories holding matched ``<stem>.npy`` files.
    stems:
        Explicit stem list. If ``None``, every stem present in ``gt_dir`` is used — prefer
        passing an explicit list from a split file.
    augment:
        Apply random dihedral transforms. Train only — never on validation, or the metric
        stops being comparable between epochs.
    cache:
        Load every array into RAM once up front. The full 3200-pair dataset is only ~950 MB,
        and KLA scores data-pipeline efficiency: caching removes disk I/O from the training
        loop entirely so time is spent in forward/backward instead.
    """

    def __init__(
        self,
        gt_dir: Path,
        lr_dir: Path,
        stems: list[str] | None = None,
        augment: bool = False,
        cache: bool = False,
    ):
        self.gt_dir, self.lr_dir = Path(gt_dir), Path(lr_dir)
        self.augment = augment
        if stems is None:
            stems = sorted(p.stem for p in self.gt_dir.glob(f"*{SUFFIX}"))
        self.stems = list(stems)
        if not self.stems:
            raise ValueError(f"no samples found for {self.gt_dir}")

        missing = [
            s for s in self.stems
            if not (self.gt_dir / f"{s}{SUFFIX}").exists()
            or not (self.lr_dir / f"{s}{SUFFIX}").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} stems missing a GT or NoisyLR file, e.g. {missing[:5]}"
            )

        self._cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        if cache:
            self._cache = [
                (
                    load_array(self.lr_dir / f"{s}{SUFFIX}"),
                    load_array(self.gt_dir / f"{s}{SUFFIX}"),
                )
                for s in self.stems
            ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            lr, gt = self._cache[i]
        else:
            stem = self.stems[i]
            lr = load_array(self.lr_dir / f"{stem}{SUFFIX}")
            gt = load_array(self.gt_dir / f"{stem}{SUFFIX}")
        if self.augment:
            lr, gt = dihedral(lr, gt, int(torch.randint(0, 8, (1,)).item()))
        return lr, gt
