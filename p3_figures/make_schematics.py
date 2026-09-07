#!/usr/bin/env python3
"""Regenerate the paper's schematic figures from the architecture as implemented.

  architecture : Fig. S1 -- AG-DSCAE as coded in p0_reproduce/agdscae_ref.py
                 (4 DS-conv blocks 32/64/128/256, gates at every scale, mirrored
                 decoder with concatenated skips, sigmoid head)
  pipeline     : Fig. S2 -- the three training stages
  teaser       : Fig. 1  -- crack strokes (DeepCrack masks) beside upsampled MNIST
                 strokes, rendered at print resolution with no manual annotations

Usage: python p3_figures/make_schematics.py --out <figs dir> [--raw <deepcrack_raw>]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CH = [32, 64, 128, 256]
RES = [256, 128, 64, 32]
INK, ACC, ACC2, GRAY = "#1f2d3d", "#c0392b", "#2e86ab", "#7f8c8d"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)


def arrow(ax, p, q, color=INK, lw=1.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, color=color, lw=lw, ls=ls))


def architecture(out):
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 11.6); ax.set_ylim(0, 4.7); ax.axis("off")
    ax.text(0.15, 4.45, "Input $x$ (3 x 256 x 256)", fontsize=8, color=INK)
    ax.text(0.15, 4.08, "source encoder $E_s$  (MNIST-pretrained, frozen after stage 1)", fontsize=8.5, color=INK, weight="bold")
    ax.text(0.15, 1.28, "target encoder $E_t$  (initialized from $E_s$, trained in stages 2-3)", fontsize=8.5, color=INK, weight="bold")
    for y0, fc in ((3.15, "#eef2f7"), (1.55, "white")):
        for i, (c, r) in enumerate(zip(CH, RES)):
            x = 0.15 + i * 1.32
            box(ax, x, y0, 1.12, 0.7, f"DS-conv\n{c} ch, {r}$^2$", fc=fc, fs=7)
            if i < 3:
                arrow(ax, (x + 1.12, y0 + 0.35), (x + 1.32, y0 + 0.35))
    for i in range(4):
        x = 0.15 + i * 1.32 + 0.56
        box(ax, x - 0.36, 2.45, 0.72, 0.5, f"gate $a_{i+1}$", fc="#fdecea", ec=ACC, fs=7)
        arrow(ax, (x, 3.15), (x, 2.95), color=ACC, lw=0.9)
        arrow(ax, (x, 2.25), (x, 2.45), color=ACC, lw=0.9)
    ax.text(5.5, 2.7, "$\\tilde h_i = a_i\\, h_i^s + (1-a_i)\\, h_i^t$   (Eq. 8)", fontsize=8, color=ACC, va="center")
    box(ax, 5.55, 1.55, 1.15, 0.7, "bottleneck\n256 ch, 16$^2$", fc="white", fs=7)
    arrow(ax, (0.15 + 3 * 1.32 + 1.12, 1.9), (5.55, 1.9))
    ax.text(5.55, 2.36, "OT loss on $z^s, z^t$ (stages 2-3)", fontsize=6.8, color=ACC2, style="italic")
    ax.text(8.2, 4.08, "decoder $D_t$", fontsize=8.5, color=INK, weight="bold")
    for j, (c, r) in enumerate(zip(CH[::-1], RES[::-1])):
        y = 3.35 - j * 0.7
        box(ax, 8.2, y, 1.75, 0.55, f"up 2x, DS-conv\n{c} ch, {r}$^2$", fc="#eefaf1", fs=6.6)
        if j < 3:
            arrow(ax, (9.075, y), (9.075, y - 0.15))
        arrow(ax, (9.95, y + 0.275), (10.08, y + 0.275), color=ACC, style="-", lw=0.7)
    arrow(ax, (6.7, 1.9), (8.2, 1.5))
    ax.text(10.12, 2.6, "concatenated skips:\nblended $\\tilde h_i$ at the\nmatching scale", fontsize=6.8, color=GRAY, va="center")
    box(ax, 8.2, 0.45, 1.75, 0.45, "1x1 conv + sigmoid", fc="white", fs=7)
    arrow(ax, (9.075, 1.25), (9.075, 0.9))
    ax.text(10.12, 0.67, "$\\hat x$ (3 x 256 x 256); score =\nsmoothed $\\|\\nabla x - \\nabla \\hat x\\|$ (Eq. 14)", fontsize=6.8, color=INK, va="center")
    ax.text(0.15, 0.55, "Every DS-conv block: depthwise 3x3 -> pointwise 1x1 -> BN -> ReLU -> dropout 0.1.   0.82 M parameters in total.",
            fontsize=7, color=GRAY)
    fig.savefig(out / "architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(9.6, 2.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.3); ax.axis("off")
    stages = [("Stage 1: source pretraining", "#eef2f7",
               "MNIST digits, 28$^2$ -> 256$^2$ bicubic\nMSE + 0.5 edge loss, 50 epochs\ntrains $E_s$, $D$; $E_s$ then frozen"),
              ("Stage 2: OT alignment", "#eefaf1",
               "unlabeled crack images\n$E_t$ init from $E_s$; MSE + 0.1 OT + 0.3 edge\nstructure-preserving cost (Eq. 4)"),
              ("Stage 3: gated adaptation", "#fdecea",
               "gates $a_1..a_4$ inserted\n(1-SSIM) + 0.05 OT + 0.01 entropy + 0.3 edge\ntrains $E_t$, $D$, gates")]
    for i, (title, fc, body) in enumerate(stages):
        x = 0.2 + i * 3.25
        box(ax, x, 0.35, 2.85, 1.6, "", fc=fc, fs=7)
        ax.text(x + 1.425, 1.72, title, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
        ax.text(x + 1.425, 0.95, body, ha="center", va="center", fontsize=7, color=INK)
        if i < 2:
            arrow(ax, (x + 2.85, 1.15), (x + 3.25, 1.15), lw=1.2)
    ax.text(5.0, 0.12, "Inference: crack map = standardized, Gaussian-smoothed gradient reconstruction error (Eq. 14); calibration from 50 unlabeled images.",
            ha="center", fontsize=7, color=GRAY)
    fig.savefig(out / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def teaser(out, raw):
    import torch, torch.nn.functional as F
    from PIL import Image
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p0_reproduce"))
    from agdscae_ref import load_mnist
    rng = np.random.default_rng(2027)
    masks = sorted((Path(raw) / "train_lab").glob("*.png"))
    # deterministic pick: masks with a mid-size crack fraction, first 6 by name after filtering
    picks = []
    for m in masks:
        a = np.asarray(Image.open(m).convert("L")) > 127
        if 0.02 < a.mean() < 0.08:
            picks.append((m, a))
        if len(picks) == 6:
            break
    digits = load_mnist(out.parent / ".mnist_cache", 2000, 2027)
    idx = rng.choice(len(digits), 6, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(9.0, 3.3))
    for k, (m, a) in enumerate(picks):
        h, w = a.shape; s = min(h, w)
        crop = a[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        crop = np.asarray(Image.fromarray(crop.astype(np.uint8) * 255).resize((256, 256), Image.NEAREST))
        axes[0, k].imshow(crop, cmap="gray", vmin=0, vmax=255)
    for k, i in enumerate(idx):
        x = torch.from_numpy(digits[i].astype(np.float32) / 255.0)
        x = F.interpolate(x[None, None], size=256, mode="bicubic", align_corners=False).clamp(0, 1)[0, 0].numpy()
        axes[1, k].imshow(x, cmap="gray", vmin=0, vmax=1)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, 0].set_ylabel("crack masks\n(DeepCrack)", fontsize=9, color=INK)
    axes[1, 0].set_ylabel("digit strokes\n(MNIST, 28$^2$ -> 256$^2$)", fontsize=9, color=INK)
    fig.subplots_adjust(left=0.09, right=0.995, top=0.98, bottom=0.02, wspace=0.05, hspace=0.06)
    fig.savefig(out / "teaser.pdf", bbox_inches="tight")
    fig.savefig(out / "teaser.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("teaser crack masks:", [m.name for m, _ in picks], "| digits idx:", idx.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None, help="DeepCrack raw root (needed for the teaser)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    architecture(out); pipeline(out)
    if a.raw:
        teaser(out, a.raw)
    print("wrote", sorted(p.name for p in out.iterdir()))
