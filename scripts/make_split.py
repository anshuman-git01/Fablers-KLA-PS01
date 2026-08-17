"""Build a deterministic, leak-free 90/10 train/val split from data/train/.

Operates ONLY on data/train/. The held-out test set (data/NoisyLR/) is guarded and verified
intact at both ends of the run.

Steps:
  1. Verify GT and NoisyLR stem sets in data/train/ are identical.
  2. Build configs/manifest.csv by reading .npy headers only (no full array reads).
  3. Stratify by GT resolution, shuffle with a fixed seed, split 90/10.
  4. Write split_train.txt / split_val.txt BEFORE moving anything (fully reversible).
  5. Move each val pair's GT and NoisyLR together.

Idempotent: exits early if data/val/ is already populated.

Usage:  python scripts/make_split.py [--seed 42] [--val-frac 0.1]
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import (  # noqa: E402
    MANIFEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SUFFIX,
    TRAIN_GT,
    TRAIN_LR,
    VAL_GT,
    VAL_LR,
    assert_heldout_intact,
    assert_not_heldout,
    stems,
)


def npy_shape(path: Path) -> tuple[int, ...]:
    """Read an .npy array shape without loading the data into memory.

    ``mmap_mode='r'`` reads only the header and maps the rest lazily, so this stays fast
    across all 3200 pairs.
    """
    return np.load(path, mmap_mode="r").shape


def build_manifest(all_stems: list[str]) -> list[dict]:
    rows, violations = [], []
    for i, stem in enumerate(all_stems, 1):
        gt_shape = npy_shape(TRAIN_GT / f"{stem}{SUFFIX}")
        lr_shape = npy_shape(TRAIN_LR / f"{stem}{SUFFIX}")

        if len(gt_shape) != 2 or len(lr_shape) != 2:
            violations.append(f"{stem}: expected 2-D arrays, got GT{gt_shape} LR{lr_shape}")
            continue

        gt_h, gt_w = gt_shape
        lr_h, lr_w = lr_shape
        if (gt_h, gt_w) != (2 * lr_h, 2 * lr_w):
            violations.append(f"{stem}: GT{gt_shape} is not exactly 2x LR{lr_shape}")

        rows.append(
            {"stem": stem, "gt_h": gt_h, "gt_w": gt_w, "lr_h": lr_h, "lr_w": lr_w}
        )
        if i % 800 == 0:
            print(f"    ...{i}/{len(all_stems)} headers read")

    if violations:
        print(f"\n  !! {len(violations)} shape violations (NOT dropped, reported only):")
        for v in violations[:20]:
            print(f"       {v}")
        if len(violations) > 20:
            print(f"       ... and {len(violations) - 20} more")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    # Guard: this script must never see the held-out test set.
    assert_not_heldout(TRAIN_GT, TRAIN_LR, VAL_GT, VAL_LR)
    assert_heldout_intact()
    print("[guard] held-out test set data/NoisyLR/ verified intact (400 files), untouched\n")

    if any(VAL_GT.glob(f"*{SUFFIX}")):
        n_val = len(list(VAL_GT.glob(f"*{SUFFIX}")))
        print(f"data/val/GT/ already contains {n_val} files — split already done. Exiting.")
        return 0

    # 1. stem parity ---------------------------------------------------------------------------
    gt_stems, lr_stems = stems(TRAIN_GT), stems(TRAIN_LR)
    if gt_stems != lr_stems:
        only_gt = sorted(set(gt_stems) - set(lr_stems))
        only_lr = sorted(set(lr_stems) - set(gt_stems))
        raise AssertionError(
            f"GT/NoisyLR stem mismatch in data/train/.\n"
            f"  only in GT ({len(only_gt)}): {only_gt[:10]}\n"
            f"  only in NoisyLR ({len(only_lr)}): {only_lr[:10]}"
        )
    print(f"[1/5] stem parity OK — {len(gt_stems)} matched pairs in data/train/")

    # 2. manifest ------------------------------------------------------------------------------
    print("[2/5] reading .npy headers to build configs/manifest.csv ...")
    rows = build_manifest(gt_stems)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stem", "gt_h", "gt_w", "lr_h", "lr_w"])
        w.writeheader()
        w.writerows(rows)

    strata: dict[tuple[int, int], list[str]] = {}
    for r in rows:
        strata.setdefault((r["gt_h"], r["gt_w"]), []).append(r["stem"])
    print(f"      wrote {MANIFEST.relative_to(MANIFEST.parent.parent)} ({len(rows)} rows)")
    print("      GT resolution mix:")
    for (h, w), members in sorted(strata.items()):
        pct = 100 * len(members) / len(rows)
        print(f"        {h}x{w}: {len(members)} pairs ({pct:.1f}%)")

    # 3. stratified split ----------------------------------------------------------------------
    train_stems, val_stems = [], []
    for key in sorted(strata):
        members = sorted(strata[key])
        random.Random(args.seed).shuffle(members)
        n_val = round(len(members) * args.val_frac)
        val_stems += members[:n_val]
        train_stems += members[n_val:]
    train_stems, val_stems = sorted(train_stems), sorted(val_stems)

    assert not (set(train_stems) & set(val_stems)), "train/val overlap!"
    assert set(train_stems) | set(val_stems) == set(gt_stems), "split does not cover all stems"
    print(f"[3/5] stratified split (seed={args.seed}): "
          f"{len(train_stems)} train / {len(val_stems)} val")

    # 4. record BEFORE moving ------------------------------------------------------------------
    SPLIT_TRAIN.write_text("\n".join(train_stems) + "\n")
    SPLIT_VAL.write_text("\n".join(val_stems) + "\n")
    print("[4/5] wrote configs/split_train.txt and configs/split_val.txt (move is reversible)")

    # 5. move val pairs together ---------------------------------------------------------------
    for d in (VAL_GT, VAL_LR):
        d.mkdir(parents=True, exist_ok=True)
    for stem in val_stems:
        fn = f"{stem}{SUFFIX}"
        shutil.move(TRAIN_GT / fn, VAL_GT / fn)
        shutil.move(TRAIN_LR / fn, VAL_LR / fn)
        assert (VAL_GT / fn).exists() and (VAL_LR / fn).exists(), f"{stem} did not land in val"
        assert not (TRAIN_GT / fn).exists() and not (TRAIN_LR / fn).exists(), \
            f"{stem} still present in train"
    print(f"[5/5] moved {len(val_stems)} GT+NoisyLR pairs into data/val/")

    # final verification -----------------------------------------------------------------------
    assert stems(TRAIN_GT) == stems(TRAIN_LR) == train_stems, "train dirs inconsistent"
    assert stems(VAL_GT) == stems(VAL_LR) == val_stems, "val dirs inconsistent"
    assert_heldout_intact()
    print("\nDone. train=%d  val=%d  held-out test=400 (untouched)"
          % (len(train_stems), len(val_stems)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
