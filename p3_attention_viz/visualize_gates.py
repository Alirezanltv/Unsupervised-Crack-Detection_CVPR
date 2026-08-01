#!/usr/bin/env python3
"""Visualize the cross-domain attention gates a_i -- the titular contribution.

The paper's title says "Attention-Guided"; reviewers will expect to SEE the
gates. This produces (a) qualitative gate heatmaps per scale and (b) a
quantitative test of the informal claim that gates prefer transferable edge
structure: mean gate value on edge pixels vs. background pixels, per scale.

WHAT YOU MUST ADAPT (one function): load_model_and_gates() below must return
your trained AG-DSCAE and a callable that, given a batch of images, returns
the list [a_1..a_4] of gate tensors (B x C x H_i x W_i, values in [0,1]).
Two common ways:
  - if your forward already returns the gates, just wrap it;
  - otherwise register forward hooks on the four attention modules and
    collect their sigmoid outputs.

Usage:
    python visualize_gates.py --checkpoint model.pt \
        --images data/concrete/test/images --n 6 --out gates_out/

Outputs in --out:
    gate_maps.png        one row per image: input | mean gate per scale (4) | overlay
    gate_stats.csv       per scale: mean gate on Canny-edge pixels vs background
    channel_gates.csv    per scale: channel-mean gate values, sorted

Selection rule (stated so nobody can accuse cherry-picking): the --n images
are the FIRST n of the sorted test directory, not hand-picked. If you swap in
curated exemplars for the paper figure, say so in the caption.

Requires a trained checkpoint;
syntax-checked only.
"""
import argparse
import csv
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# ADAPT THIS FUNCTION to your codebase.
# --------------------------------------------------------------------------
def load_model_and_gates(checkpoint: Path):
    """Return (model, get_gates) where get_gates(batch_tensor) -> [a_1..a_4].

    Reference implementation using forward hooks -- adjust module paths:

        import torch
        model = torch.load(checkpoint, map_location="cuda").eval()
        gates = []
        def hook(_m, _inp, out):
            gates.append(out.detach())          # out must be the sigmoid gate
        for att in [model.att1, model.att2, model.att3, model.att4]:
            att.gate_sigmoid.register_forward_hook(hook)
        def get_gates(x):
            gates.clear()
            with torch.no_grad():
                model(x.cuda())
            return [g.cpu() for g in gates]
        return model, get_gates
    """
    raise NotImplementedError("wire this to your AG-DSCAE implementation")


def load_image(path: Path, size: int = 256) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("gates_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from skimage.feature import canny
    from scipy.ndimage import zoom

    model, get_gates = load_model_and_gates(args.checkpoint)

    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})[:args.n]
    imgs = np.stack([load_image(p) for p in paths])
    batch = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    gate_list = get_gates(batch)          # [a_i] each B x C x H_i x W_i
    n_scales = len(gate_list)

    # ---- quantitative: edge vs background gate means, per scale ----------
    stats_rows, chan_rows = [], []
    for s, g in enumerate(gate_list, start=1):
        g = g.numpy()
        edge_means, bg_means = [], []
        for b in range(g.shape[0]):
            gray = imgs[b].mean(axis=2)
            edges = canny(gray, sigma=1.0, low_threshold=50 / 255,
                          high_threshold=150 / 255)
            # downsample the edge mask to this scale's resolution
            fy = g.shape[2] / edges.shape[0]
            fx = g.shape[3] / edges.shape[1]
            em = zoom(edges.astype(np.float32), (fy, fx), order=1) > 0.2
            gm = g[b].mean(axis=0)                    # channel-mean gate map
            if em.sum() and (~em).sum():
                edge_means.append(float(gm[em].mean()))
                bg_means.append(float(gm[~em].mean()))
        stats_rows.append({"scale": s,
                           "gate_on_edges": np.mean(edge_means),
                           "gate_on_background": np.mean(bg_means),
                           "difference": np.mean(edge_means) - np.mean(bg_means)})
        for ci, v in sorted(enumerate(g.mean(axis=(0, 2, 3))),
                            key=lambda t: -t[1]):
            chan_rows.append({"scale": s, "channel": ci, "mean_gate": float(v)})

    for name, rows in [("gate_stats.csv", stats_rows),
                       ("channel_gates.csv", chan_rows)]:
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("per-scale edge-vs-background gate means:")
    for r in stats_rows:
        print(f"  scale {r['scale']}: edges {r['gate_on_edges']:.3f} "
              f"vs background {r['gate_on_background']:.3f} "
              f"(diff {r['difference']:+.3f})")

    # ---- qualitative figure ---------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(paths)
    fig, axes = plt.subplots(n, n_scales + 2, figsize=(2.1 * (n_scales + 2), 2.1 * n))
    axes = np.atleast_2d(axes)
    for b in range(n):
        axes[b, 0].imshow(imgs[b])
        axes[b, 0].set_ylabel(paths[b].stem[:12], fontsize=6)
        for s, g in enumerate(gate_list, start=1):
            gm = g[b].numpy().mean(axis=0)
            axes[b, s].imshow(gm, cmap="viridis", vmin=0, vmax=1)
            if b == 0:
                axes[b, s].set_title(f"scale {s}", fontsize=8)
        gm_full = zoom(gate_list[0][b].numpy().mean(axis=0),
                       np.array(imgs[b].shape[:2]) / gate_list[0][b].shape[1:],
                       order=1)
        axes[b, -1].imshow(imgs[b])
        axes[b, -1].imshow(gm_full, cmap="magma", alpha=0.45, vmin=0, vmax=1)
        if b == 0:
            axes[b, 0].set_title("input", fontsize=8)
            axes[b, -1].set_title("overlay (scale 1)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out / "gate_maps.png", dpi=200)
    print(f"wrote {args.out}/gate_maps.png and CSVs")


if __name__ == "__main__":
    main()
