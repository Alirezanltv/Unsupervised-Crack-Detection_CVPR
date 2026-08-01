#!/usr/bin/env python3
"""Normalize benchmark ground-truth masks to binary 0/255 PNGs.

Handles: PNG/BMP masks with arbitrary positive labels, and CrackForest
`.mat` files (field 'groundTruth' -> 'Segmentation', where cracks are
label 2 in the original release).

Usage: python prepare_masks.py --src raw_masks/ --dst masks/
Not yet run on real data; syntax-checked only.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_any(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        m = loadmat(str(path))
        gt = m["groundTruth"][0, 0]
        seg = gt["Segmentation"] if "Segmentation" in gt.dtype.names else gt[0]
        return (np.asarray(seg) == 2).astype(np.uint8) * 255
    arr = np.asarray(Image.open(path).convert("L"))
    return ((arr > 127) if arr.max() > 1 else (arr > 0)).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(args.src.iterdir()):
        if p.suffix.lower() not in {".png", ".bmp", ".mat", ".jpg", ".tif"}:
            continue
        Image.fromarray(load_any(p)).save(args.dst / f"{p.stem}.png")
        n += 1
    print(f"wrote {n} masks to {args.dst}")


if __name__ == "__main__":
    main()
