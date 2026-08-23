#!/usr/bin/env python3
"""Qualitative comparison figure for the DeepCrack arm.

Image selection is deterministic and stated: the 237 test stems are sorted
alphabetically and rows are taken at evenly spaced indices (0, 59, 118, 177,
236). No image is chosen by looking at any method's output.

Every method is rendered through the identical display transform, which is
the scoring transform: raw map -> bilinear resize to ground-truth resolution
-> standardize with that method's own 50-image calibration statistics ->
display z-scores clipped to [0, 4] with one shared colormap. What the figure
shows is therefore exactly what the metrics scored.

Usage:
  python p3_qualitative/make_figure.py --runs <runs_root> --data <arranged>
      --out figs/qualitative_deepcrack.jpg
where <runs_root>/<method>/s0/{maps,calib} holds .npy score maps and
<arranged>/test/{images,masks} the committed-split test set.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROW_INDICES = (0, 59, 118, 177, 236)
METHODS = [  # (dir name, panel title)
    ("ours", "Ours"),
    ("padim", "PaDiM"),
    ("patchcore", "PatchCore"),
    ("rd4ad", "RD4AD"),
    ("efficientad", "EfficientAD"),
    ("draem", "DRAEM"),
]
Z_MAX = 4.0


def calib_stats(calib_dir):
    vals = [np.load(p).astype(np.float64) for p in sorted(Path(calib_dir).glob("*.npy"))]
    flat = np.concatenate([v.ravel() for v in vals])
    return float(flat.mean()), float(flat.std() + 1e-8)


def load_z(map_path, mu, delta, size_wh):
    m = np.load(map_path).astype(np.float32)
    if m.ndim == 3:
        m = m.squeeze()
    im = Image.fromarray(m, mode="F").resize(size_wh, Image.BILINEAR)
    return (np.asarray(im) - mu) / delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, data = Path(args.runs), Path(args.data)
    stems = sorted(p.stem for p in (data / "test/images").iterdir())
    rows = [stems[i] for i in ROW_INDICES]
    print("rows (deterministic indices", ROW_INDICES, "):", rows)

    methods = [(d, t) for d, t in METHODS if (runs / d / "s0/maps").is_dir()]
    missing = [t for d, t in METHODS if not (runs / d / "s0/maps").is_dir()]
    if missing:
        print("NOT FOUND (skipped):", missing)
    stats = {d: calib_stats(runs / d / "s0/calib") for d, _ in methods}

    ncol = 2 + len(methods)
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(1.62 * ncol, 1.22 * len(rows)))
    for r, stem in enumerate(rows):
        img = Image.open(data / f"test/images/{stem}.jpg").convert("RGB")
        gt = np.asarray(Image.open(data / f"test/masks/{stem}.png").convert("L")) > 127
        size_wh = (gt.shape[1], gt.shape[0])
        panels = [(np.asarray(img), "Input", None),
                  (gt, "Ground truth", "gray")]
        for d, title in methods:
            mp = runs / d / f"s0/maps/{stem}.npy"
            z = load_z(mp, *stats[d], size_wh)
            panels.append((np.clip(z, 0.0, Z_MAX), title, "magma"))
        for c, (arr, title, cmap) in enumerate(panels):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(arr)
            elif cmap == "gray":
                ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=Z_MAX)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005,
                        wspace=0.03, hspace=0.03)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print("wrote", out)


if __name__ == "__main__":
    main()
