"""Canonical path constants for the project.

Every script resolves paths through this module rather than hardcoding strings, so that the
held-out test set has exactly one definition and one guard.

⚠️  THE TWO-``NoisyLR`` TRAP
    ``data/train/NoisyLR/``  -> TRAINING inputs, paired with ``data/train/GT/``
    ``data/NoisyLR/``        -> HELD-OUT TEST SET, 400 files, no GT exists

``HELDOUT_TEST_LR`` may only be imported by ``inference.py``. It must never enter a training or
validation split, model selection, or any reported metric. Use ``assert_not_heldout`` in any
script that touches directories.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"
CONFIGS = PROJECT_ROOT / "configs"
RESULTS = PROJECT_ROOT / "results"
INSPECTION = RESULTS / "inspection"
WEIGHTS = PROJECT_ROOT / "weights"

# --- training / validation / inspection ------------------------------------------------------
TRAIN_GT = DATA_ROOT / "train" / "GT"
TRAIN_LR = DATA_ROOT / "train" / "NoisyLR"

VAL_GT = DATA_ROOT / "val" / "GT"
VAL_LR = DATA_ROOT / "val" / "NoisyLR"

SAMPLE_GT = DATA_ROOT / "sample" / "GT"
SAMPLE_LR = DATA_ROOT / "sample" / "NoisyLR"

# --- split bookkeeping ------------------------------------------------------------------------
MANIFEST = CONFIGS / "manifest.csv"
SPLIT_TRAIN = CONFIGS / "split_train.txt"
SPLIT_VAL = CONFIGS / "split_val.txt"

# --- ⛔ held out: final inference only ---------------------------------------------------------
HELDOUT_TEST_LR = DATA_ROOT / "NoisyLR"
HELDOUT_TEST_COUNT = 400

SUFFIX = ".npy"


def assert_not_heldout(*paths: Path) -> None:
    """Fail loudly if any path resolves to (or inside) the held-out test set."""
    forbidden = HELDOUT_TEST_LR.resolve()
    for p in paths:
        rp = Path(p).resolve()
        if rp == forbidden or forbidden in rp.parents:
            raise AssertionError(
                f"Refusing to operate on the held-out test set: {rp}\n"
                "data/NoisyLR/ is reserved for final inference only."
            )


def assert_heldout_intact() -> None:
    """Verify the held-out test set still has exactly the expected number of files."""
    n = len(list(HELDOUT_TEST_LR.glob(f"*{SUFFIX}")))
    if n != HELDOUT_TEST_COUNT:
        raise AssertionError(
            f"Held-out test set changed: expected {HELDOUT_TEST_COUNT} files, found {n}."
        )


def stems(directory: Path) -> list[str]:
    """Sorted list of filename stems for the .npy files in ``directory``."""
    return sorted(p.stem for p in Path(directory).glob(f"*{SUFFIX}"))
