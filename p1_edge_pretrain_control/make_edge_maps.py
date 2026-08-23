#!/usr/bin/env python3
"""Build the edge-detector pretraining control's source domain.

Reviewer 4's objection: maybe a hand-engineered edge prior does the same job
as MNIST strokes. This script turns any natural-image folder (e.g. BSDS500
images, or any photo collection) into a folder of edge maps that drops into
Stage-1 pretraining IN PLACE of MNIST -- everything downstream stays identical.

Two detectors:
  canny : classical Canny, thresholds 50/150 (the same values used elsewhere
          in the paper), via OpenCV.
  hed   : Holistically-Nested Edge Detection via OpenCV DNN. Download the two
          files below into --hed-dir first:
            deploy.prototxt   https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt
            hed.caffemodel    http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel

Usage:
    python make_edge_maps.py --src ~/data/bsds/images --dst ~/data/canny_maps \
        --detector canny --size 256
Then point your Stage-1 trainer's source-domain folder at --dst.

Not yet run on real data; syntax-checked only.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def canny_map(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)


class Hed:
    def __init__(self, hed_dir: Path):
        proto, weights = hed_dir / "deploy.prototxt", hed_dir / "hed.caffemodel"
        if not (proto.exists() and weights.exists()):
            raise SystemExit(f"put deploy.prototxt and hed.caffemodel in {hed_dir}")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, size=(w, h),
                                     mean=(104.00699, 116.66877, 122.67891),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0, 0]
        return (255 * np.clip(out, 0, 1)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--detector", choices=["canny", "hed"], default="canny")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--hed-dir", type=Path, default=Path("hed_model"))
    ap.add_argument("--crops", type=int, default=1,
                    help="seeded random crops per source image, so a small "
                         "corpus (e.g. BSDS500's 500 images x 40 crops) can "
                         "match the 20k-image MNIST source-set size")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    hed = Hed(args.hed_dir) if args.detector == "hed" else None
    rng = np.random.default_rng(args.seed)
    n = 0
    for p in sorted(args.src.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(h, w) < args.size:
            scale = args.size / min(h, w)
            img = cv2.resize(img, (max(args.size, int(w * scale)),
                                   max(args.size, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        for k in range(args.crops):
            if args.crops == 1:
                crop = cv2.resize(img, (args.size, args.size),
                                  interpolation=cv2.INTER_AREA)
            else:
                y = int(rng.integers(0, h - args.size + 1))
                x = int(rng.integers(0, w - args.size + 1))
                crop = img[y:y + args.size, x:x + args.size]
            if args.detector == "canny":
                edge = canny_map(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            else:
                edge = hed(crop)
            cv2.imwrite(str(args.dst / f"{p.stem}_{k:03d}.png"), edge)
            n += 1
    print(f"wrote {n} edge maps to {args.dst}")


if __name__ == "__main__":
    main()
