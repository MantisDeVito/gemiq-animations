"""
Bake the Anim 2 laser-arc composite into a single MP4 so the dashboard can
use a plain <video> (no iframe, no canvas) — fixes the iOS Safari iframe-
autoplay/loop flakiness for good.

Ports the JS drawLaserArc() in anim-2-composite.html to PIL. Uses the
"Standard" preset values (default on the page).
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).parent
SRC = ROOT / "01-gens" / "anim-2-shopify" / "_seamless-loop-v3.mp4"
FRAMES_DIR = ROOT / "_bake-tmp-laser-frames"
OUT = ROOT / "01-gens" / "anim-2-shopify" / "_composite-standard-v1.mp4"

# Standard preset (from BUILTIN_PRESETS.standard in anim-2-composite.html)
LASER_START = 1.755
LASER_DUR = 1.405
STROKE = 5
GLOW = 1.0
HEAD = 1200  # px in JS units (before scale)
BEAMS = 1
COLOUR = (28, 180, 192)  # #1cb4c0
GLOW_OPACITY = 0.5
CORE_OPACITY = 0.75

# Video specs (matches _seamless-loop-v3.mp4)
W, H = 1928, 1076
FPS = 24
DURATION = 6.083333
TOTAL_FRAMES = round(DURATION * FPS) + 1  # 147 — cover the last partial frame

# Bezier path in viewBox (0,0)-(1180,440): M 220 220 Q 590 -20 960 220
SCALE_X = W / 1180
SCALE_Y = H / 440
P0 = (220 * SCALE_X, 220 * SCALE_Y)
PC = (590 * SCALE_X, -20 * SCALE_Y)
P1 = (960 * SCALE_X, 220 * SCALE_Y)
AVG_SCALE = (SCALE_X + SCALE_Y) / 2

# Sample the quadratic Bezier and build cumulative arc-length table
N = 400
ts = np.linspace(0, 1, N + 1)
pts_x = (1 - ts) ** 2 * P0[0] + 2 * (1 - ts) * ts * PC[0] + ts ** 2 * P1[0]
pts_y = (1 - ts) ** 2 * P0[1] + 2 * (1 - ts) * ts * PC[1] + ts ** 2 * P1[1]
pts = list(zip(pts_x.tolist(), pts_y.tolist()))
seg_lens = np.hypot(np.diff(pts_x), np.diff(pts_y))
cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
ACTUAL_LEN = float(cum_len[-1])


def lerp_point(target_len: float) -> tuple[float, float]:
    """Return the (x, y) point at the given cumulative arc length."""
    if target_len <= 0:
        return pts[0]
    if target_len >= ACTUAL_LEN:
        return pts[-1]
    i = int(np.searchsorted(cum_len, target_len))
    if i == 0:
        return pts[0]
    seg = cum_len[i] - cum_len[i - 1]
    if seg <= 0:
        return pts[i - 1]
    u = (target_len - cum_len[i - 1]) / seg
    x = pts[i - 1][0] + u * (pts[i][0] - pts[i - 1][0])
    y = pts[i - 1][1] + u * (pts[i][1] - pts[i - 1][1])
    return (x, y)


def visible_subcurve(s_start: float, s_end: float) -> list[tuple[float, float]]:
    """Return the polyline (sampled points) that lies between s_start and s_end."""
    if s_end <= s_start:
        return []
    sub = [lerp_point(s_start)]
    mask = (cum_len > s_start) & (cum_len < s_end)
    for i in np.where(mask)[0]:
        sub.append(pts[int(i)])
    sub.append(lerp_point(s_end))
    return sub


def draw_stroked_line(size, points, colour_rgb, alpha_0_1, width, blur_px):
    """Draw a stroked polyline, then Gaussian-blur it — matches canvas
    `shadowBlur` + stroke."""
    if len(points) < 2 or alpha_0_1 <= 0:
        return None
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    a = max(0, min(255, int(round(alpha_0_1 * 255))))
    draw.line(points, fill=(*colour_rgb, a), width=max(1, int(round(width))), joint="curve")
    # Round caps — draw circles at the endpoints (PIL doesn't have lineCap=round).
    r = max(1, int(round(width / 2)))
    for (x, y) in (points[0], points[-1]):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*colour_rgb, a))
    if blur_px > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur_px))
    return layer


def render_laser_frame(t_sec: float) -> Image.Image:
    """Return an RGBA image of the laser overlay for time t_sec, or fully
    transparent if the laser isn't firing this frame."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for beam in range(BEAMS):
        start_offset = beam * 0.15  # cfg.stagger (Standard preset)
        lt = t_sec - LASER_START - start_offset
        if lt < 0 or lt > LASER_DUR:
            continue
        progress = lt / LASER_DUR

        head_len = HEAD * AVG_SCALE
        path_len_scaled = ACTUAL_LEN  # cum length already in canvas pixels
        head_pos = -head_len + (path_len_scaled + 2 * head_len) * progress
        seg_start = max(0.0, head_pos - head_len)
        seg_end = max(0.0, min(path_len_scaled, head_pos))
        if seg_end <= seg_start:
            continue

        alpha_mul = 1.0
        if progress > 0.88:
            alpha_mul = 1 - (progress - 0.88) / 0.12
        if progress < 0.02:
            alpha_mul = progress / 0.02
        alpha_mul = max(0.0, min(1.0, alpha_mul))

        sub = visible_subcurve(seg_start, seg_end)
        if len(sub) < 2:
            continue

        # Three layers, matching the JS:
        #   { blur: 24, alpha: 0.55, width: stroke*4.5 }  teal outer glow
        #   { blur: 12, alpha: 0.80, width: stroke*2.2 }  teal mid glow
        #   { blur:  5, alpha: 1.00, width: stroke*1.0 }  white hot core
        for blur_px, base_alpha, width, colour, alpha_chan in (
            (24 * GLOW, 0.55 * GLOW, STROKE * 4.5, COLOUR, GLOW_OPACITY),
            (12 * GLOW, 0.80 * GLOW, STROKE * 2.2, COLOUR, GLOW_OPACITY),
            (5 * GLOW,  1.00,         STROKE * 1.0, (255, 255, 255), CORE_OPACITY),
        ):
            layer = draw_stroked_line(
                (W, H), sub, colour,
                base_alpha * alpha_mul * alpha_chan,
                width, blur_px,
            )
            if layer is not None:
                img = Image.alpha_composite(img, layer)

    return img


def main():
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir()

    print(f"Rendering {TOTAL_FRAMES} laser-overlay frames at {W}x{H}...")
    for i in range(TOTAL_FRAMES):
        t = i / FPS
        frame = render_laser_frame(t)
        frame.save(FRAMES_DIR / f"laser_{i:04d}.png", optimize=False)
        if i % 24 == 0:
            print(f"  frame {i}/{TOTAL_FRAMES}  t={t:.2f}s")

    print("Compositing with ffmpeg...")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(SRC),
        "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "laser_%04d.png"),
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-movflags", "+faststart",
        "-an",
        str(OUT),
    ]
    subprocess.run(cmd, check=True)
    print(f"\nDone: {OUT}  ({OUT.stat().st_size / 1024 / 1024:.2f} MB)")
    shutil.rmtree(FRAMES_DIR)


if __name__ == "__main__":
    main()
