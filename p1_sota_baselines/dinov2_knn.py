#!/usr/bin/env python3
"""Frozen-foundation-model baseline: DINOv2 patch features + nearest-neighbour scoring.

The PatchCore recipe on DINOv2 tokens: build a memory bank of L2-normalised patch tokens
from the training images (no coreset subsampling -- the bank is small enough to keep
whole, which removes a tuning knob), score every test patch by its distance to the nearest
bank token, upsample the patch grid to the input size. No training, no seeds: the run is
deterministic given the backbone.

Maps are written in exactly the layout run_baselines.py produces
(<out>/<name>/s0/{maps,calib}/<stem>.npy), so common/eval_maps.py and
common/sweep_threshold.py score them unchanged.

Input resolution defaults to 448 so the patch grid is 32x32 (patch 14); at 256 it would be
18x18, too coarse for hairline cracks. Latency per image (feature extraction + scoring) is
recorded in <out>/<name>/s0/latency.json.

Usage:
    python dinov2_knn.py --data /kaggle/temp/data/deepcrack --out /kaggle/working/runs \
        --backbone vits14 [--size 448] [--k 1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)


def load_batch(paths, size, dev):
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
        xs.append(torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1))
    x = torch.stack(xs)
    return ((x - MEAN) / STD).to(dev)


@torch.no_grad()
def tokens(model, x):
    out = model.forward_features(x)
    t = out["x_norm_patchtokens"]                     # B x N x D
    return F.normalize(t, dim=-1)


@torch.no_grad()
def nn_dist(q, bank, k, chunk=65536):
    """q: N x D, bank: M x D (both unit-norm). Returns k-NN mean distance per query."""
    best = None
    for i in range(0, bank.shape[0], chunk):
        d = torch.cdist(q, bank[i:i + chunk])         # N x chunk
        top = d.topk(min(k, d.shape[1]), dim=1, largest=False).values
        best = top if best is None else torch.cat([best, top], dim=1).topk(k, dim=1, largest=False).values
    return best.mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14", "vitl14"])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default=None, help="run name (default dinov2_<backbone>)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name = a.name or f"dinov2_{a.backbone}"
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{a.backbone}").to(dev).eval()
    grid = a.size // 14

    # memory bank from the training images
    bank = []
    train = list_images(a.data / "train/good")
    for i in range(0, len(train), a.batch):
        bank.append(tokens(model, load_batch(train[i:i + a.batch], a.size, dev)).reshape(-1, model.embed_dim))
    bank = torch.cat(bank)
    print(f"{name}: bank {tuple(bank.shape)} from {len(train)} images on {dev}", flush=True)

    root = a.out / name / "s0"
    lat = []
    for split, sub in (("test/images", "maps"), ("calib", "calib")):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        paths = list_images(a.data / split)
        for i in range(0, len(paths), a.batch):
            ps = paths[i:i + a.batch]
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = load_batch(ps, a.size, dev)
            t = tokens(model, x)                      # B x N x D
            for j, p in enumerate(ps):
                s = nn_dist(t[j], bank, a.k).reshape(1, 1, grid, grid)
                s = F.interpolate(s, size=(a.size, a.size), mode="bilinear", align_corners=False)[0, 0]
                np.save(d / f"{p.stem}.npy", s.cpu().numpy().astype(np.float32))
            if dev == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) / len(ps))
        print(f"  {split}: {len(paths)} maps -> {d}", flush=True)
    (root / "latency.json").write_text(json.dumps({
        "name": name, "backbone": a.backbone, "input_size": a.size, "patch_grid": grid,
        "bank_tokens": int(bank.shape[0]), "embed_dim": int(bank.shape[1]), "k": a.k,
        "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "ms_per_image_incl_io": round(1000 * float(np.mean(lat[2:])), 2)}, indent=2))
    print("done", name)


if __name__ == "__main__":
    main()
