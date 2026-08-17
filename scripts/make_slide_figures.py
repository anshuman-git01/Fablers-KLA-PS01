"""Generate the presentation figures as transparent PNGs sized for the slide template.

Outputs to results/figures/:
    pipeline_diagram.png     end-to-end restoration flow (slide 4)
    unet_architecture.png    encoder/decoder with skips (slide 7)
    metric_bars.png          baseline vs final on all three metrics (slide 6)

Transparent background so they composite onto the template's dark circuit artwork.
Series colours are validated for dark-surface CVD separation (see scripts/ notes):
    #6C77C4 baseline   #68A828 ours
Both sit inside the L 0.48-0.67 band, clear the chroma floor, and separate by
dEE 25 under deuteranopia against surface #0D1338.

Usage:  python scripts/make_slide_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import RESULTS  # noqa: E402

OUT = RESULTS / "figures"

# palette
BASE = "#6C77C4"   # baseline series
OURS = "#68A828"   # our model series
LIME = "#9EE84F"   # template accent, used for strokes/labels only (not a data series)
INK = "#FFFFFF"
INK_DIM = "#AEB6D6"
BOX_FILL = "#1B2559"
BOX_EDGE = "#4A6EE0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK_DIM,
    "ytick.color": INK_DIM,
})


def rounded_box(ax, x, y, w, h, label, sub=None, fill=BOX_FILL, edge=BOX_EDGE,
                fs=15, subfs=11, lw=2.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))
    # Offsets are fractions of the box height so the two lines never collide,
    # whatever height the caller uses.
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center",
            va="center", fontsize=fs, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=subfs, color=INK_DIM, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=LIME, lw=2.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=18, linewidth=lw,
        color=color, linestyle=ls, zorder=4,
        connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- pipeline ---
def pipeline():
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4.2); ax.axis("off")

    y, h = 1.95, 1.0
    boxes = [
        (0.25, 2.5, "NoisyLR input", "128 x 128  .npy"),
        (3.35, 3.0, "U-Net encoder-decoder", "3 scales  |  97 px context"),
        (7.05, 2.5, "PixelShuffle x2", "sub-pixel upsample"),
        (12.15, 3.55, "Restored output", "256 x 256  .npy float32"),
    ]
    for x, w, lab, sub in boxes:
        rounded_box(ax, x, y, w, h, lab, sub)

    # main path
    arrow(ax, 2.75, y + h / 2, 3.35, y + h / 2)
    arrow(ax, 6.35, y + h / 2, 7.05, y + h / 2)
    arrow(ax, 9.55, y + h / 2, 10.35, y + h / 2)

    # sum node
    cx, cy = 10.75, y + h / 2
    ax.add_patch(plt.Circle((cx, cy), 0.4, facecolor=BOX_FILL, edgecolor=LIME,
                            linewidth=2.4, zorder=3))
    ax.text(cx, cy, "+", ha="center", va="center", fontsize=26,
            fontweight="bold", color=LIME, zorder=4)
    arrow(ax, cx + 0.4, cy, 12.15, cy)

    # residual branch: bilinear base added back in
    arrow(ax, 1.5, y, 1.5, 0.95, rad=0)
    ax.plot([1.5, 9.4], [0.95, 0.95], color=LIME, lw=2.2, zorder=2)
    arrow(ax, 9.4, 0.95, cx, cy - 0.4, rad=-0.35)
    rounded_box(ax, 3.9, 0.5, 3.3, 0.9, "bilinear x2 base", None,
                fill="#152047", edge=LIME, fs=13, lw=1.8)

    ax.text(8.0, 3.62, "The network predicts only the correction on top of the "
                       "bilinear base",
            ha="center", fontsize=13.5, color=LIME, style="italic")
    ax.text(8.0, 0.12, "single pass  |  no degradation-order detection  |  "
                       "fully convolutional, any input size",
            ha="center", fontsize=12, color=INK_DIM)

    fig.tight_layout()
    fig.savefig(OUT / "pipeline_diagram.png", dpi=200, transparent=True,
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ architecture ---
def architecture():
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0.2, 15.8)
    ax.set_ylim(0.2, 6.1)
    ax.axis("off")

    mid = 2.45

    # Increased box widths (1.5 -> 1.75) and block heights
    levels = [
        (0.6, 1.75, 2.4, "enc 1", "128 x 128", "32 ch", BASE),
        (3.0, 1.75, 1.7, "enc 2", "64 x 64", "64 ch", BASE),
        (5.4, 1.75, 1.1, "bottleneck", "32 x 32", "128 ch", LIME),
        (7.8, 1.75, 1.7, "dec 2", "64 x 64", "64 ch", OURS),
        (10.2, 1.75, 2.4, "dec 1", "128 x 128", "32 ch", OURS),
    ]

    for x, w, h, lab, res, ch, col in levels:
        ax.add_patch(
            FancyBboxPatch(
                (x, mid - h / 2),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.06",
                linewidth=2.5,
                edgecolor=col,
                facecolor=BOX_FILL,
                zorder=2,
            )
        )
        # Scaled up font sizes inside blocks
        ax.text(
            x + w / 2,
            mid + 0.18,
            lab,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=INK,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            mid - 0.18,
            res,
            ha="center",
            va="center",
            fontsize=12.5,
            color=INK_DIM,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            mid - h / 2 - 0.32,
            ch,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=col,
            zorder=3,
        )

    # Thicker connecting arrows between blocks
    for i in range(len(levels) - 1):
        x1 = levels[i][0] + levels[i][1]
        x2 = levels[i + 1][0]
        arrow(ax, x1, mid, x2, mid, color=INK_DIM, lw=2.2)

    # Skip connection arcs
    for a, b, lift in ((0, 4, 1.45), (1, 3, 0.95)):
        xa = levels[a][0] + levels[a][1] / 2
        xb = levels[b][0] + levels[b][1] / 2
        ya = mid + levels[a][2] / 2
        yb = mid + levels[b][2] / 2
        ax.annotate(
            "",
            xy=(xb, yb),
            xytext=(xa, ya),
            arrowprops=dict(
                arrowstyle="-|>",
                color=LIME,
                lw=2.4,
                connectionstyle=f"arc3,rad=-{lift/4.2}",
            ),
            zorder=5,
        )

    # Scaled text labels
    # Sits above the outer arc's apex (~y=5.3) rather than across it.
    ax.text(
        6.7,
        5.72,
        "skip connections carry high-frequency detail forward",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=LIME,
        style="italic",
    )

    # PixelShuffle head box
    ps_x = 12.50
    arrow(ax, levels[4][0] + levels[4][1], mid, ps_x, mid, color=INK_DIM, lw=2.2)
    rounded_box(
        ax,
        ps_x,
        mid - 0.75,
        1.8,
        1.50,
        "PixelShuffle",
        "x2 head",
        fill=BOX_FILL,
        edge=OURS,
        fs=14.5,
        subfs=12,
    )

    # Output text
    arrow(ax, ps_x + 1.8, mid, ps_x + 2.3, mid, color=INK_DIM, lw=2.2)
    ax.text(
        ps_x + 2.8,
        mid + 0.18,
        "256 x 256",
        ha="center",
        fontsize=14.5,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        ps_x + 2.8,
        mid - 0.18,
        "restored",
        ha="center",
        fontsize=12.5,
        color=INK_DIM,
    )

    # Parameter text
    ax.text(
        6.7,
        0.35,
        "1,156,164 parameters  |  4.6 MB  |  receptive field 97 px vs 21 px for a flat CNN",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=INK_DIM,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "unet_architecture.png",
        dpi=150,
        transparent=True,
        bbox_inches="tight",
    )
    plt.close(fig)

# -------------------------------------------------------------- metric bars ---
def metric_bars():
    """Three metrics on three different scales -> small multiples, never one axis."""
    panels = [
        ("PSNR", "dB   higher is better", 24.953, 28.680, "{:.2f}", 34.0, "+3.73 dB"),
        ("SSIM", "higher is better", 0.6215, 0.7833, "{:.3f}", 1.0, "+0.16"),
        ("LPIPS", "lower is better", 0.3842, 0.1556, "{:.3f}", 0.50, "-59%"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, (name, sub, base_v, our_v, fmt, top, delta) in zip(axes, panels):
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)

        xs = [0.32, 0.98]
        vals = [base_v, our_v]
        cols = [BASE, OURS]
        for x, v, c in zip(xs, vals, cols):
            # Square ends. Rounding sized in data units flares horizontally when
            # the x and y scales differ by 30x, which is worse than no rounding.
            ax.add_patch(Rectangle((x - 0.22, 0), 0.44, v,
                                   linewidth=0, facecolor=c, zorder=2))
            ax.text(x, v + top * 0.035, fmt.format(v), ha="center", va="bottom",
                    fontsize=17, fontweight="bold", color=INK, zorder=4)

        # Bars start at zero (never truncated), so state the delta explicitly.
        ax.text(0.65, top * 0.90, delta, ha="center", va="center", fontsize=15,
                fontweight="bold", color=OURS, zorder=5)

        ax.set_xlim(0, 1.3); ax.set_ylim(0, top)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=19, fontweight="bold", color=INK, pad=26)
        ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center",
                fontsize=11.5, color=LIME if "lower" in sub else INK_DIM)
        ax.axhline(0, color=INK_DIM, lw=1.2, alpha=0.55)

    handles = [
        plt.Line2D([], [], marker="s", markersize=13, linestyle="none",
                   color=BASE, label="Bilinear x2 baseline"),
        plt.Line2D([], [], marker="s", markersize=13, linestyle="none",
                   color=OURS, label="Our model"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=14, labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "metric_bars.png", dpi=200, transparent=True,
                bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    architecture()
    metric_bars()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(OUT.parent.parent)}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
