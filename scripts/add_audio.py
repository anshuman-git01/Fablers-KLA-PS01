"""Add voice narration and light background music to the prototype video.

Everything is generated locally and originally — nothing is downloaded:
  * narration  : macOS `say` (offline system TTS)
  * music      : an original ambient pad synthesised with numpy in this file
                 (slow chord progression, no samples, no third-party audio)

This matters for the submission: KLA requires every external resource to be disclosed with a
licence. Self-generated audio has nothing to disclose.

Mixing: the music sits well under the voice and is further ducked by a sidechain compressor
keyed off the narration, so speech always stays intelligible.

Usage:
    python scripts/add_audio.py                 # write results/prototype_video_narrated.mp4
    python scripts/add_audio.py --in-place      # overwrite results/prototype_video.mp4
    python scripts/add_audio.py --check         # report narration timing fit only
"""

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RESULTS  # noqa: E402

SR = 44100
VOICE = "Samantha"
RATE = 178  # words per minute

# (start_second, text) — starts are aligned to the video's scene boundaries.
# Scene layout: intro 0-4.5 | simulation 4.5-13.5 | 4 examples 5s each 13.5-33.5 | results 33.5-42
# Run with --check after editing: it verifies no line overruns into the next one.
SCRIPT = [
    (0.70, "Fablers. A.I. based restoration of degraded images."),
    (5.20, "Three degradations. Speckle noise, Gaussian noise, "
           "and two times downsampling."),
    (11.90, "One forward pass reverses all three."),
    (14.10, "Dense stochastic texture. The hardest case in the set."),
    (19.10, "Edges and structure recovered, nearly four decibels above baseline."),
    (24.10, "Fine detail returns across the full frame."),
    (29.10, "At best, close to ten decibels over the baseline."),
    (34.10, "Three point seven decibels better P S N R. "
            "Sixty percent better L P I P S. "
            "Four point three milliseconds per image, end to end."),
]

VIDEO = RESULTS / "prototype_video.mp4"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stderr[-2000:]}")
    return r


def duration(path: Path) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    return float(r.stdout.strip())


def tts(text: str, out: Path) -> np.ndarray:
    """Render one narration line to a mono float array at SR."""
    aiff = out.with_suffix(".aiff")
    run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text])
    wav = out.with_suffix(".wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff),
         "-ar", str(SR), "-ac", "1", str(wav)])
    with wave.open(str(wav), "rb") as w:
        raw = w.readframes(w.getnframes())
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return a


def ambient_bed(total_s: float) -> np.ndarray:
    """Original ambient pad: a slow chord progression of soft detuned sines.

    Deliberately sparse and low — it is a bed, not a track. No samples, no external audio.
    """
    n = int(total_s * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float32)

    # A minor 9 -> F major 9 -> C major -> G sus2, one chord per ~9.25 s
    chords = [[220.00, 261.63, 329.63, 493.88],
              [174.61, 261.63, 349.23, 440.00],
              [130.81, 196.00, 261.63, 329.63],
              [196.00, 220.00, 293.66, 392.00]]
    seg = total_s / len(chords)
    for i, chord in enumerate(chords):
        s0, s1 = i * seg, (i + 1) * seg
        m = (t >= s0 - 1.2) & (t < s1 + 1.2)
        if not m.any():
            continue
        lt = t[m] - s0
        # smooth crossfading envelope so chords bleed into each other
        env = np.clip(np.minimum((lt + 1.2) / 1.8, (seg + 1.2 - lt) / 1.8), 0, 1) ** 1.5
        for j, f in enumerate(chord):
            for det in (-0.12, 0.12):  # gentle detune = width
                amp = 0.16 / (j + 1.6)
                out[m] += (amp * env *
                           np.sin(2 * np.pi * (f + det) * t[m] +
                                  0.6 * np.sin(2 * np.pi * 0.07 * t[m]))).astype(np.float32)

    # slow breathing + soft low shelf via cumulative smoothing
    out *= (0.85 + 0.15 * np.sin(2 * np.pi * 0.045 * t)).astype(np.float32)
    k = 64
    out = np.convolve(out, np.ones(k, dtype=np.float32) / k, mode="same")

    fade = int(2.5 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    peak = float(np.abs(out).max()) or 1.0
    return (out / peak * 0.30).astype(np.float32)


def write_wav(path: Path, a: np.ndarray):
    a = np.clip(a, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((a * 32767).astype(np.int16).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, default=VIDEO)
    ap.add_argument("--out", type=Path, default=RESULTS / "prototype_video_narrated.mp4")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--check", action="store_true", help="report timing fit and exit")
    ap.add_argument("--music-db", type=float, default=-11.0,
                    help="bed level before ducking; -11 gives a ~17 dB "
                         "speech-to-bed separation after loudnorm")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    vdur = duration(args.video)
    print(f"video: {args.video.name}  {vdur:.2f}s")

    tmp = RESULTS / "_audio_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # ---- narration track ---------------------------------------------------------------
    total_n = int(vdur * SR) + SR
    voice = np.zeros(total_n, dtype=np.float32)
    print(f"\nnarration ({VOICE}, {RATE} wpm):")
    problems = []
    for i, (start, txt) in enumerate(SCRIPT):
        a = tts(txt, tmp / f"seg{i:02d}")
        dur = len(a) / SR
        nxt = SCRIPT[i + 1][0] if i + 1 < len(SCRIPT) else vdur
        budget = nxt - start
        fit = "ok" if dur <= budget + 0.02 else f"OVERRUNS by {dur - budget:.2f}s"
        print(f"  {start:5.2f}s  {dur:5.2f}s / {budget:5.2f}s  {fit}   \"{txt[:52]}...\"")
        if dur > budget + 0.02:
            problems.append((i, start, dur, budget))
        s = int(start * SR)
        e = min(s + len(a), total_n)
        voice[s:e] += a[: e - s]

    if problems:
        print(f"\n  ⚠ {len(problems)} segment(s) overrun their slot — "
              f"they will overlap the next line.")
    if args.check:
        return 1 if problems else 0

    voice = voice[: int(vdur * SR)]
    peak = float(np.abs(voice).max()) or 1.0
    voice = voice / peak * 0.92
    write_wav(tmp / "voice.wav", voice)

    # ---- music bed ---------------------------------------------------------------------
    print("\nsynthesising original ambient bed (no external audio)...")
    write_wav(tmp / "music.wav", ambient_bed(vdur))

    # ---- mix + mux ---------------------------------------------------------------------
    out = args.out
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(args.video), "-i", str(tmp / "voice.wav"), "-i", str(tmp / "music.wav"),
        "-filter_complex",
        # music dropped well under the voice, then sidechain-ducked by the narration itself
        f"[2:a]volume={args.music_db}dB,afade=t=in:d=1.5,"
        f"afade=t=out:st={vdur-3:.2f}:d=3[bed];"
        "[1:a]asplit=2[v1][vkey];"
        "[bed][vkey]sidechaincompress=threshold=0.05:ratio=3:attack=20:release=300[duck];"
        "[v1][duck]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[mix];"
        "[mix]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart",
        str(out),
    ]
    run(cmd)
    print(f"wrote {out}")

    if args.in_place:
        shutil.move(str(out), str(args.video))
        print(f"moved into place: {args.video}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
