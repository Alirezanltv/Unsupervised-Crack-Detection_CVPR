#!/usr/bin/env python3
"""Shared metric engine for ALL methods (ours and every baseline).

Consumes dumped anomaly maps + ground-truth masks and reports, per image set:
  - MIoU and F1 at the paper's exact protocol: standardize the raw map with
    (mu, delta) estimated on held-out unlabeled images, sigmoid, threshold 0.5
  - threshold-free pixel AUROC and Average Precision from the raw map

Using one engine for every method is what makes the numbers like-for-like.

Expected layout (float maps, same stem as the image they belong to):
    maps/
      img_0001.npy   (float32 HxW, raw anomaly scores BEFORE sigmoid)
    masks/
      img_0001.png   (0/255 or 0/1 binary ground truth)
    calib/
      *.npy          (raw maps for held-out UNLABELED images; used only to
                      estimate mu and delta -- no masks involved)

Usage:
    python eval_maps.py --maps runs/ours/concrete/maps \
                        --masks data/concrete/test_masks \
                        --calib runs/ours/concrete/calib_maps \
                        --out results/ours_concrete.json

Numbers printed are per-run; aggregate over seeds with common/stats.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score


def load_map(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float64)
    return np.asarray(Image.open(path), dtype=np.float64)


def load_mask(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return (m > 127).astype(np.uint8) if m.max() > 1 else m.astype(np.uint8)


def resize_map(raw: np.ndarray, shape) -> np.ndarray:
    """Bilinear-upsample an anomaly map to ground-truth resolution (the
    standard convention when models emit maps at their working size)."""
    from scipy.ndimage import zoom
    return zoom(raw, (shape[0] / raw.shape[0], shape[1] / raw.shape[1]), order=1)


def standardize(raw: np.ndarray, mu: float, delta: float) -> np.ndarray:
    z = (raw - mu) / max(delta, 1e-12)
    return 1.0 / (1.0 + np.exp(-z))          # sigmoid, as in Eq. (13)


def find_pairs(maps_dir: Path, masks_dir: Path):
    pairs = []
    for mp in sorted(maps_dir.iterdir()):
        if mp.suffix.lower() not in {".npy", ".png", ".tif", ".tiff"}:
            continue
        for ext in (".png", ".bmp", ".tif", ".tiff", ".jpg"):
            gt = masks_dir / (mp.stem + ext)
            if gt.exists():
                pairs.append((mp, gt))
                break
    if not pairs:
        raise SystemExit(f"no (map, mask) pairs found in {maps_dir} / {masks_dir}")
    return pairs


def binary_metrics(pred: np.ndarray, gt: np.ndarray):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    iou_crack = tp / max(tp + fp + fn, 1)
    iou_bg = tn / max(tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"miou": 0.5 * (iou_crack + iou_bg), "iou_crack": iou_crack,
            "f1": f1, "precision": prec, "recall": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True,
                    help="raw maps of held-out unlabeled images for (mu, delta)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    calib = np.concatenate([load_map(p).ravel()
                            for p in sorted(args.calib.iterdir())
                            if p.suffix.lower() in {".npy", ".png"}])
    mu, delta = float(calib.mean()), float(calib.std())

    per_img, ys, ss = [], [], []
    for mp, gt_path in find_pairs(args.maps, args.masks):
        raw, gt = load_map(mp), load_mask(gt_path)
        if raw.shape != gt.shape:
            raw = resize_map(raw, gt.shape)
        prob = standardize(raw, mu, delta)
        per_img.append(binary_metrics((prob >= args.threshold).astype(np.uint8), gt))
        ys.append(gt.ravel())
        ss.append(prob.ravel())

    y, s = np.concatenate(ys), np.concatenate(ss)
    res = {k: float(np.mean([m[k] for m in per_img])) for k in per_img[0]}
    res.update(n_images=len(per_img), mu=mu, delta=delta,
               threshold=args.threshold,
               pixel_auroc=float(roc_auc_score(y, s)),
               pixel_ap=float(average_precision_score(y, s)))

    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
