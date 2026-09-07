#!/usr/bin/env python3
"""Procedural smooth-stroke source domain (appearance-matched to upsampled MNIST).

The Canny control changed two things at once relative to digit strokes: content (natural
edges vs. handwriting) and appearance (binary 1-px maps vs. smooth ~10-px strokes), and
the binary maps let the autoencoder copy its input through the skip connections (stage-1
loss 0.0000). This source keeps MNIST's *appearance statistics* -- stroke width, edge
transition width, support, gray-level richness -- while carrying no digit semantics: random
cubic Bezier strokes, rendered with MNIST-like thickness and Gaussian-blurred to MNIST's
10-90% edge transition. If it transfers like MNIST, the digits are not special; if not,
they carry something beyond appearance.

Matched by construction (defaults: 2-4 strokes, width 5-9% of the side, sigma 4 px),
measured with the supplementary's own edge_stats.analyze on 150 images against 120
bicubic-upsampled MNIST digits measured by the same code:
                      MNIST      strokes
  mean intensity      0.128      0.128
  support (>0.05)     0.200      0.205
  10-90% transition   11.0+-4.0  10.9+-4.1 px
  Sobel |grad| @edge  0.096      0.095
  stroke width/side   0.096      0.098

Usage:
    python make_stroke_source.py --dst /kaggle/temp/stroke_maps --n 20000 --seed 2027
    python make_stroke_source.py --dst /tmp/probe --n 300 --seed 2027 --stats
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def bezier(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def render(rng, size=256, n_strokes=(2, 4), width_frac=(0.05, 0.09), sigma=4.0,
           margin=0.12):
    canvas = np.zeros((size, size), np.float32)
    lo, hi = int(size * margin), int(size * (1 - margin))
    for _ in range(rng.integers(n_strokes[0], n_strokes[1] + 1)):
        pts = rng.uniform(lo, hi, size=(4, 2))
        # keep strokes reasonably extended, like digit strokes, not tiny squiggles
        if np.linalg.norm(pts[3] - pts[0]) < size * 0.3:
            pts[3] = pts[0] + (pts[3] - pts[0]) / (np.linalg.norm(pts[3] - pts[0]) + 1e-6) * size * 0.3
        curve = bezier(*pts).astype(np.int32)
        thickness = int(round(size * rng.uniform(*width_frac)))
        cv2.polylines(canvas, [curve.reshape(-1, 1, 2)], False, 1.0, thickness, cv2.LINE_AA)
    if sigma > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigma)
    return np.clip(canvas, 0, 1)


def stats(imgs):
    """Same probe as the Canny/MNIST source comparison, plus a transition-width estimate:
    along the gradient direction at edge pixels, the distance over which intensity rises
    from 10% to 90% of the local stroke peak (approximated by 0.8 / max gradient, the
    standard slope-based estimate for a smooth ramp)."""
    X = np.stack(imgs)
    gx = np.stack([cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) / 8 for x in X])
    gy = np.stack([cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3) / 8 for x in X])
    g = np.sqrt(gx ** 2 + gy ** 2)
    edge = (X > 0.3) & (X < 0.7)                      # ramp region of the strokes
    gmax = np.array([gi[e].max() if e.any() else np.nan for gi, e in zip(g, edge)])
    return {"mean_intensity": float(X.mean()),
            "support_gt_0.05": float((X > 0.05).mean()),
            "frac_gt_0.5": float((X > 0.5).mean()),
            "gray_levels_per_image": float(np.mean([len(np.unique(np.round(x, 2))) for x in X[:50]])),
            "grad_mag_at_ramp": float(np.nanmean([gi[e].mean() for gi, e in zip(g, edge) if e.any()])),
            "transition_px_est": float(np.nanmean(0.8 / gmax))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--stats", action="store_true", help="print appearance statistics")
    a = ap.parse_args()
    a.dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    keep = []
    for i in range(a.n):
        img = render(rng, a.size, sigma=a.sigma)
        cv2.imwrite(str(a.dst / f"stroke_{i:05d}.png"), (img * 255).astype(np.uint8))
        if a.stats and len(keep) < 300:
            keep.append(img)
    print(f"wrote {a.n} stroke images to {a.dst}")
    if a.stats:
        for k, v in stats(keep).items():
            print(f"  {k:24s} {v:.4f}")


if __name__ == "__main__":
    main()
