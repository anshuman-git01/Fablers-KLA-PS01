"""Render the animated prototype/simulation video: results/prototype_video.mp4.

This is a true motion-graphics render — every frame is drawn individually at 30 fps with eased
animation, not a slideshow of static cards.

Scenes
    1. INTRO       title builds in over a live denoising backdrop
    2. SIMULATION  the degradation model applied step by step to a clean image
                   (speckle -> Gaussian -> 2x downsample), then restored by the model
    3. EXAMPLES    before/after slider wipe over real validation pairs, with metric counters
                   that animate from the bilinear baseline up to our score
    4. RESULTS     bars grow to final values, throughput counter ticks up

Usage:
    python scripts/make_prototype_video.py                 # full render
    python scripts/make_prototype_video.py --preview 3.0   # single frame at t=3.0s
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from matplotlib import font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import build_model  # noqa: E402
from src.paths import RESULTS, VAL_GT, VAL_LR  # noqa: E402

W, H, FPS = 1920, 1080, 30
BG = (7, 11, 18)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
ACCENT = (88, 166, 255)
GOOD = (63, 185, 80)
WARN = (219, 109, 40)
LINE = (48, 54, 61)

_REG = fm.findfont("DejaVu Sans")
_BOLD = str(Path(_REG).with_name("DejaVuSans-Bold.ttf"))
_MONO = fm.findfont("DejaVu Sans Mono")
_font_cache: dict = {}


def F(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono)
    if key not in _font_cache:
        path = _MONO if mono else (_BOLD if bold else _REG)
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


# ---------------------------------------------------------------- easing / timing
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease_out(t: float) -> float:
    return 1 - (1 - clamp01(t)) ** 3


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def seg(t: float, start: float, dur: float) -> float:
    """Progress of a sub-animation starting at ``start`` lasting ``dur``."""
    return clamp01((t - start) / dur) if dur > 0 else 1.0


# ---------------------------------------------------------------- drawing helpers
def text(dr, xy, s, font, fill, anchor="mm", alpha=1.0):
    if alpha <= 0.01:
        return
    col = tuple(int(c) for c in fill) + (int(255 * clamp01(alpha)),)
    dr.text(xy, s, font=font, fill=col, anchor=anchor)


def gray_to_rgb(arr: np.ndarray, size: int, blocky: bool = False) -> Image.Image:
    a = np.clip(arr, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8), mode="L")
    im = im.resize((size, size), Image.NEAREST if blocky else Image.LANCZOS)
    return im.convert("RGB")


def kb(arr: np.ndarray, size: int, t01: float, amp: float = 0.055,
       blocky: bool = False) -> Image.Image:
    """Ken Burns: slowly push in on the image so no shot is ever frozen.

    All panels in a scene share t01, so the before/after wipe stays pixel-aligned.
    """
    h, w = arr.shape
    z = 1.0 - amp * clamp01(t01)
    ch, cw = max(8, int(h * z)), max(8, int(w * z))
    oy, ox = (h - ch) // 2, (w - cw) // 2
    return gray_to_rgb(arr[oy:oy + ch, ox:ox + cw], size, blocky)


def panel(base: Image.Image, im: Image.Image, box, border=LINE, glow=0.0):
    x, y = box
    if glow > 0:
        g = Image.new("RGB", (im.width + 24, im.height + 24), BG)
        gd = ImageDraw.Draw(g)
        gd.rectangle([0, 0, g.width - 1, g.height - 1],
                     outline=tuple(int(c * glow) for c in ACCENT), width=6)
        g = g.filter(ImageFilter.GaussianBlur(8))
        base.paste(g, (x - 12, y - 12))
    base.paste(im, (x, y))
    ImageDraw.Draw(base).rectangle([x - 1, y - 1, x + im.width, y + im.height],
                                   outline=border, width=2)


def vignette_bg() -> Image.Image:
    """Subtle radial background, built once and reused."""
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    v = np.clip(1.0 - 0.45 * d, 0, 1)[..., None]
    base = np.array(BG, dtype=np.float32)[None, None, :]
    img = (base * v + np.array([2, 4, 8]) * (1 - v)).astype(np.uint8)
    return Image.fromarray(img, "RGB")


_BGCACHE = None


def frame_base() -> Image.Image:
    global _BGCACHE
    if _BGCACHE is None:
        _BGCACHE = vignette_bg()
    return _BGCACHE.copy()


def overlay(base: Image.Image):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return ov, ImageDraw.Draw(ov)


def finish(base: Image.Image, ov: Image.Image) -> Image.Image:
    return Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")


def fade(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    return Image.blend(Image.new("RGB", img.size, BG), img, clamp01(a))


# ---------------------------------------------------------------- degradation simulation
def simulate(gt: np.ndarray, stage: float, rng: np.random.Generator):
    """Progressively apply the three degradations. stage in [0,3]."""
    x = gt.copy()
    if stage > 0:  # speckle: multiplicative, intensity dependent
        s = min(stage, 1.0)
        x = x * (1 + s * 0.35 * rng.standard_normal(x.shape))
    if stage > 1:  # additive Gaussian
        s = min(stage - 1, 1.0)
        x = x + s * 0.09 * rng.standard_normal(x.shape)
    return x


# ---------------------------------------------------------------- scenes
class Show:
    def __init__(self, hero, examples, summ, ref, timing):
        self.hero = hero
        self.ex = examples
        self.summ, self.ref, self.timing = summ, ref, timing
        self.rng = np.random.default_rng(0)

        self.T_INTRO = 4.5
        self.T_SIM = 9.0
        self.T_EX = 5.0
        self.T_SUM = 8.5
        self.total = self.T_INTRO + self.T_SIM + len(self.ex) * self.T_EX + self.T_SUM

    # ---------- 1. intro
    def intro(self, t):
        base = frame_base()
        gt, lr = self.hero["gt"], self.hero["lr"]
        p = ease_in_out(seg(t, 0.3, 3.0))
        blend = (1 - p) * np.clip(np.kron(lr, np.ones((2, 2))), 0, 1) + p * gt
        bg = gray_to_rgb(blend, 1080).resize((1920, 1080), Image.LANCZOS)
        bg = bg.filter(ImageFilter.GaussianBlur(2))
        base = Image.blend(base, bg, 0.16 + 0.06 * p)

        ov, dr = overlay(base)
        a1 = ease_out(seg(t, 0.4, 1.0))
        dy = int(28 * (1 - a1))
        text(dr, (W // 2, 430 + dy), "FABLERS", F(96, True), FG, alpha=a1)
        a2 = ease_out(seg(t, 0.9, 1.0))
        text(dr, (W // 2, 545 + int(20 * (1 - a2))),
             "AI-Based Restoration of Degraded Images", F(40), ACCENT, alpha=a2)
        a3 = ease_out(seg(t, 1.4, 1.0))
        text(dr, (W // 2, 615), "KLA Problem Statement PS01  ·  SEMICON India Hackathon 2026",
             F(22), MUTED, alpha=a3)

        a4 = ease_out(seg(t, 2.1, 1.0))
        if a4 > 0:
            wln = int(620 * a4)
            dr.line([(W // 2 - wln // 2, 680), (W // 2 + wln // 2, 680)],
                    fill=ACCENT + (int(200 * a4),), width=2)
        a5 = ease_out(seg(t, 2.5, 1.0))
        text(dr, (W // 2, 730), "U-Net restorer  ·  1.16M parameters  ·  L1 + SSIM + LPIPS",
             F(24), FG, alpha=a5)
        return finish(base, ov)

    # ---------- 2. degradation simulation
    def sim(self, t):
        base = frame_base()
        ov, dr = overlay(base)
        gt, lr = self.hero["gt"], self.hero["lr"]
        S = 560
        left, top = 380, 265

        # restoration phase progress; the degradation UI fades out as it takes over
        p_rest = seg(t, 5.6, 2.2)
        e_rest = ease_in_out(p_rest)
        deg_a = 1.0 - e_rest   # degradation chrome cross-fades directly into restoration

        text(dr, (W // 2, 104), "THE DEGRADATION MODEL", F(36, True), FG, alpha=deg_a)
        text(dr, (W // 2, 158), "three mechanisms, applied in an undisclosed order",
             F(21), MUTED, alpha=deg_a)
        text(dr, (W // 2, 104), "AI RESTORATION", F(36, True), GOOD, alpha=e_rest)
        text(dr, (W // 2, 158), "one forward pass recovers detail and resolution",
             F(21), MUTED, alpha=e_rest)

        p_spk = seg(t, 0.7, 1.5)
        p_gau = seg(t, 2.3, 1.5)
        p_dwn = seg(t, 3.9, 1.2)
        stage = min(p_spk, 1.0) + min(p_gau, 1.0)
        deg = simulate(gt, stage, np.random.default_rng(7))
        if p_dwn > 0:
            small = np.kron(lr, np.ones((2, 2)))
            deg = (1 - ease_in_out(p_dwn)) * deg + ease_in_out(p_dwn) * small
        zs = t / self.T_SIM
        dimg = kb(deg, S, zs, blocky=p_dwn > 0.5)

        if p_rest <= 0:
            panel(base, dimg, (left, top))
        else:
            rimg = kb(self.hero["pred"], S, zs)
            comp = dimg.copy()
            xw = int(S * e_rest)
            if xw > 0:
                comp.paste(rimg.crop((0, 0, xw, S)), (0, 0))
            panel(base, comp, (left, top), glow=0.7 * e_rest)
            if 0 < xw < S:
                dr.line([(left + xw, top), (left + xw, top + S)],
                        fill=(255, 255, 255, 240), width=4)
                dr.ellipse([left + xw - 16, top + S // 2 - 16,
                            left + xw + 16, top + S // 2 + 16], fill=ACCENT + (255,))

        text(dr, (left + S // 2, top - 30), "SIMULATED DEGRADATION", F(21, True), MUTED,
             alpha=deg_a)
        text(dr, (left + S // 2, top - 30), "RESTORED OUTPUT", F(21, True), GOOD, alpha=e_rest)

        cx = left + S + 100
        chips = [("SPECKLE NOISE", "multiplicative  ·  intensity dependent", p_spk),
                 ("GAUSSIAN NOISE", "additive  ·  zero mean", p_gau),
                 ("2x DOWNSAMPLE", "resolution loss", p_dwn)]
        if deg_a > 0.02:
            for i, (name, sub, prog) in enumerate(chips):
                y = top + 30 + i * 130
                on = prog > 0.02
                a = (0.25 + 0.75 * ease_out(prog)) * deg_a
                dr.rounded_rectangle([cx, y, cx + 470, y + 96], radius=12,
                                     outline=(WARN if on else LINE) + (int(255 * a),), width=3)
                wfill = int(464 * ease_out(prog))
                if wfill > 4:
                    dr.rounded_rectangle([cx + 3, y + 3, cx + 3 + wfill, y + 93], radius=10,
                                         fill=(WARN[0], WARN[1], WARN[2], int(42 * deg_a)))
                text(dr, (cx + 24, y + 36), name, F(25, True), FG if on else MUTED,
                     anchor="lm", alpha=a)
                text(dr, (cx + 24, y + 66), sub, F(18), MUTED, anchor="lm", alpha=a)

        if e_rest > 0.02:
            y = top + 30
            dr.rounded_rectangle([cx, y, cx + 470, y + 96], radius=12,
                                 outline=GOOD + (int(255 * e_rest),), width=3)
            dr.rounded_rectangle([cx + 3, y + 3, cx + 467, y + 93], radius=10,
                                 fill=(GOOD[0], GOOD[1], GOOD[2], int(34 * e_rest)))
            text(dr, (cx + 24, y + 36), "U-NET RESTORER", F(25, True), FG,
                 anchor="lm", alpha=e_rest)
            text(dr, (cx + 24, y + 66), "1.16M params  ·  single forward pass  ·  4.6 ms",
                 F(18), MUTED, anchor="lm", alpha=e_rest)
            a2 = ease_out(seg(t, 6.2, 0.8))
            for j, (lab, val) in enumerate([("receptive field", "97 x 97 px"),
                                            ("output", "2x resolution, float32")]):
                yy = top + 30 + (j + 1) * 130
                dr.rounded_rectangle([cx, yy, cx + 470, yy + 96], radius=12,
                                     outline=LINE + (int(210 * a2),), width=2)
                text(dr, (cx + 24, yy + 36), lab.upper(), F(20, True), MUTED,
                     anchor="lm", alpha=a2)
                text(dr, (cx + 24, yy + 66), val, F(22), FG, anchor="lm", alpha=a2)

        return finish(base, ov)

    # ---------- 3. examples with slider wipe
    def example(self, t, k):
        ex = self.ex[k]
        base = frame_base()
        ov, dr = overlay(base)
        S = 560
        left, top = 170, 250

        text(dr, (W // 2, 96), f"RESTORATION  {k+1} / {len(self.ex)}", F(30, True), FG)
        text(dr, (W // 2, 146), f"validation image {ex['stem']}   ·   "
             f"{ex['pct']}th percentile of the PSNR-gain distribution", F(20), MUTED)

        z = t / self.T_EX  # shared Ken Burns phase keeps the wipe aligned
        deg = kb(np.kron(ex["lr"], np.ones((2, 2))), S, z, blocky=True)
        res = kb(ex["pred"], S, z)
        gtim = kb(ex["gt"], S, z)

        # slider sweeps left->right, holds, then GT panel slides in
        p = ease_in_out(seg(t, 0.7, 2.1))
        comp = deg.copy()
        xw = int(S * p)
        if xw > 0:
            comp.paste(res.crop((0, 0, xw, S)), (0, 0))
        panel(base, comp, (left, top), glow=0.55 * p)
        if 0.001 < p < 0.999:
            x = left + xw
            dr.line([(x, top), (x, top + S)], fill=(255, 255, 255, 235), width=4)
            dr.ellipse([x - 17, top + S // 2 - 17, x + 17, top + S // 2 + 17],
                       fill=ACCENT + (255,))
            text(dr, (x, top + S // 2), "‹ ›", F(20, True), (255, 255, 255), alpha=1)

        text(dr, (left + 14, top + 20), "DEGRADED", F(21, True), (255, 255, 255),
             anchor="lm", alpha=(1 - p) ** 1.5)
        text(dr, (left + S - 14, top + 20), "RESTORED", F(21, True), GOOD,
             anchor="rm", alpha=p)

        ag = ease_out(seg(t, 2.5, 0.9))
        if ag > 0:
            gx = left + S + 110 + int(60 * (1 - ag))
            panel(base, fade(gtim, ag), (gx, top))
            text(dr, (gx + S // 2, top - 30), "GROUND TRUTH", F(20, True), MUTED, alpha=ag)

        # animated metric counters, baseline -> ours
        am = ease_in_out(seg(t, 1.5, 1.8))
        mx = left + 30
        my = top + S + 78
        for i, (lab, b, v, fmt) in enumerate([
                ("PSNR", ex["psnr_bilinear"], ex["psnr"], "{:.2f} dB"),
                ("SSIM", ex["ssim_bilinear"], ex["ssim"], "{:.4f}"),
                ("LPIPS", ex["lpips_bilinear"], ex["lpips"], "{:.4f}")]):
            x = mx + i * 300
            cur = lerp(b, v, am)
            text(dr, (x, my), lab, F(19, True), MUTED, anchor="lm")
            text(dr, (x, my + 44), fmt.format(cur), F(38, True, mono=True), GOOD,
                 anchor="lm")
        d = ex["psnr"] - ex["psnr_bilinear"]
        text(dr, (mx + 3 * 300 + 40, my + 24), f"{d:+.2f} dB", F(46, True), ACCENT,
             anchor="lm", alpha=am)
        text(dr, (mx + 3 * 300 + 40, my + 68), "vs bilinear x2", F(18), MUTED,
             anchor="lm", alpha=am)
        return finish(base, ov)

    # ---------- 4. results
    def results(self, t):
        base = frame_base()
        ov, dr = overlay(base)
        text(dr, (W // 2, 118), "RESULTS", F(52, True), FG)
        text(dr, (W // 2, 178), "320-image held-out validation split", F(22), MUTED)

        rows = [("PSNR", self.ref["psnr"], self.summ["psnr"], "{:.3f} dB", False),
                ("SSIM", self.ref["ssim"], self.summ["ssim"], "{:.4f}", False),
                ("LPIPS", self.ref["lpips"], self.summ["lpips"], "{:.4f}", True)]
        # Bar length = RELATIVE improvement over the bilinear baseline, on a shared scale.
        # (Plotting the raw values would fill every bar and hide which metric actually moved.)
        rels = [((b - v) / b if lower else (v - b) / b) for _, b, v, _, lower in rows]
        rmax = max(rels)
        bx, bw = 470, 820
        for i, ((lab, b, v, fmt, lower), rel) in enumerate(zip(rows, rels)):
            y = 300 + i * 132
            p = ease_in_out(seg(t, 0.35 + i * 0.28, 1.25))
            text(dr, (bx - 40, y + 16), lab, F(30, True), FG, anchor="rm")
            text(dr, (bx - 40, y + 52), "baseline " + fmt.format(b), F(17), MUTED,
                 anchor="rm", alpha=0.9 * p)
            dr.rounded_rectangle([bx, y, bx + bw, y + 44], radius=8, fill=(22, 27, 34, 255))
            wpx = int(bw * (rel / rmax) * p)
            if wpx > 4:
                dr.rounded_rectangle([bx, y, bx + wpx, y + 44], radius=8, fill=GOOD + (235,))
            text(dr, (bx + bw + 34, y + 22), fmt.format(lerp(b, v, p)),
                 F(30, True, mono=True), FG, anchor="lm")
            delta = (v - b) if not lower else -(b - v)
            dtxt = (f"+{delta:.3f} dB" if lab == "PSNR"
                    else (f"+{delta:.4f}" if delta > 0 else f"−{abs(delta):.4f}"))
            text(dr, (bx + bw + 250, y + 8), dtxt, F(27, True), ACCENT, anchor="lm", alpha=p)
            text(dr, (bx + bw + 250, y + 40), f"{100*rel:.0f}% better", F(19), MUTED,
                 anchor="lm", alpha=p)
        text(dr, (bx + bw // 2, 300 + 3 * 132 - 44),
             "bar length = relative improvement over the bilinear baseline",
             F(17), MUTED, alpha=ease_out(seg(t, 1.5, 0.8)))

        pt = ease_in_out(seg(t, 2.6, 1.5))
        text(dr, (W // 2, 790), "END-TO-END THROUGHPUT", F(26, True), ACCENT, alpha=pt)
        ms = lerp(0, self.timing["ms_per_image"], pt)
        text(dr, (W // 2, 862), f"{ms:.2f} ms / image", F(56, True, mono=True), FG, alpha=pt)
        text(dr, (W // 2, 924),
             f"{self.timing['n_images']} images in "
             f"{self.timing['end_to_end_seconds']:.2f} s  ·  batch "
             f"{self.timing['batch_size']}  ·  read → preprocess → GPU → write",
             F(19), MUTED, alpha=pt)
        text(dr, (W // 2, 1010),
             "Examples chosen by percentile of the gain distribution — not hand-picked.",
             F(17), MUTED, alpha=ease_out(seg(t, 4.6, 1.0)))
        return finish(base, ov)

    def _progress(self, img: Image.Image, t: float) -> Image.Image:
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 5, W, H], fill=(18, 22, 29))
        d.rectangle([0, H - 5, int(W * clamp01(t / self.total)), H], fill=ACCENT)
        return img

    # ---------- dispatch
    def render(self, t: float) -> Image.Image:
        return self._progress(self._render(t), t)

    def _render(self, t: float) -> Image.Image:
        XF = 0.45  # crossfade length
        b1 = self.T_INTRO
        b2 = b1 + self.T_SIM
        if t < b1:
            img = self.intro(t)
            if t > b1 - XF:
                img = Image.blend(img, self.sim(0.0), ease_in_out((t - (b1 - XF)) / XF) * 0.9)
            return img
        if t < b2:
            lt = t - b1
            img = self.sim(lt)
            if lt < XF:
                img = Image.blend(self.intro(b1), img, ease_in_out(lt / XF))
            return img
        te = t - b2
        k = int(te // self.T_EX)
        if k < len(self.ex):
            lt = te - k * self.T_EX
            img = self.example(lt, k)
            if lt < XF:
                prev = self.sim(self.T_SIM) if k == 0 else self.example(self.T_EX, k - 1)
                img = Image.blend(prev, img, ease_in_out(lt / XF))
            return img
        lt = te - len(self.ex) * self.T_EX
        img = self.results(lt)
        if lt < XF:
            img = Image.blend(self.example(self.T_EX, len(self.ex) - 1), img,
                              ease_in_out(lt / XF))
        return img


# ---------------------------------------------------------------- data
def load(stems_pct, ckpt: Path):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["arch"], base=cfg.get("base", 32),
                        blocks_per_level=cfg.get("blocks_per_level", 2),
                        channels=cfg.get("channels", 64), num_blocks=cfg.get("blocks", 8))
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    eval_dir = RESULTS / "eval" / ckpt.stem
    per = {r["stem"]: r for r in csv.DictReader(open(eval_dir / "per_image_metrics.csv"))}

    out = []
    for stem, pct in stems_pct:
        lr = np.load(VAL_LR / f"{stem}.npy")
        gt = np.load(VAL_GT / f"{stem}.npy")
        with torch.no_grad():
            pred = model(torch.from_numpy(lr)[None, None].to(device)).clamp(0, 1)[0, 0]
        r = per[stem]
        out.append({"stem": stem, "pct": pct, "lr": lr, "gt": gt,
                    "pred": pred.cpu().numpy(),
                    **{k: float(r[k]) for k in
                       ("psnr", "ssim", "lpips", "psnr_bilinear", "ssim_bilinear",
                        "lpips_bilinear")}})
    return out, eval_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("weights/unet_ssimlpips_b32_best.pt"))
    ap.add_argument("--out", type=Path, default=RESULTS / "prototype_video.mp4")
    ap.add_argument("--preview", type=float, default=None,
                    help="render a single frame at this timestamp and exit")
    ap.add_argument("--preview-times", type=float, nargs="+", default=None)
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    picks = [("000414", 20), ("000769", 65), ("003173", 70), ("000214", 95)]
    hero_stem = ("000539", 85)

    data, eval_dir = load(picks + [hero_stem], args.checkpoint)
    hero = data[-1]
    examples = data[:-1]
    summ = json.load(open(eval_dir / "summary.json"))["all"]
    ref = {"psnr": summ["psnr_bilinear"], "ssim": summ["ssim_bilinear"],
           "lpips": summ["lpips_bilinear"]}
    timing = json.load(open(RESULTS / "inference_timing.json"))

    show = Show(hero, examples, summ, ref, timing)
    print(f"scenes: intro {show.T_INTRO}s + sim {show.T_SIM}s + "
          f"{len(examples)}x{show.T_EX}s + results {show.T_SUM}s = {show.total:.1f}s")

    if args.preview is not None or args.preview_times:
        outdir = RESULTS / "prototype_preview"
        outdir.mkdir(parents=True, exist_ok=True)
        for tt in (args.preview_times or [args.preview]):
            p = outdir / f"t{tt:05.2f}.png"
            show.render(tt).save(p)
            print("  wrote", p)
        return 0

    tmp = RESULTS / "_proto_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    n = int(show.total * args.fps)
    for i in range(n):
        show.render(i / args.fps).save(tmp / f"{i:05d}.png")
        if i % 60 == 0:
            print(f"  frame {i}/{n}  ({100*i/n:.0f}%)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "%05d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr[-2500:])
        return 1
    shutil.rmtree(tmp)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
