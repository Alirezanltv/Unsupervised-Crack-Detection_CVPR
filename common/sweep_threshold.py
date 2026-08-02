#!/usr/bin/env python3
"""Threshold sweep over dumped anomaly maps: ODS-style operating points.

Fixed-0.5-after-standardization proved to over-fire for every method tested
(it thresholds at the calibration MEAN). This tool reports defensible
operating points instead, computed identically for every method:

  - ODS-F1  : the single dataset-wide threshold maximizing F1 (the edge-
              detection literature's standard), with its MIoU alongside
  - best-MIoU: the dataset-wide threshold maximizing MIoU
  - fixed-0.5: the old protocol, for reference

Thresholds are swept in standardized-sigmoid space (the same space the paper
defines), so the chosen operating point is stated explicitly in the paper,
identical convention for all methods.

Usage:
    python sweep_threshold.py --maps runs/ours/maps --masks data/test/masks \
        --calib runs/ours/calib [--steps 99] [--out sweep.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_maps import load_map, load_mask, resize_map, standardize, find_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=99)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    calib = np.concatenate([load_map(p).ravel()
                            for p in sorted(args.calib.iterdir())
                            if p.suffix.lower() in {".npy", ".png"}])
    mu, delta = float(calib.mean()), float(calib.std())

    probs, gts = [], []
    for mp, gt_path in find_pairs(args.maps, args.masks):
        raw, gt = load_map(mp), load_mask(gt_path)
        if raw.shape != gt.shape:
            raw = resize_map(raw, gt.shape)
        probs.append(standardize(raw, mu, delta).ravel())
        gts.append(gt.ravel())
    prob = np.concatenate(probs)
    gt = np.concatenate(gts).astype(bool)

    ths = np.linspace(0.01, 0.99, args.steps)
    rows = []
    for t in ths:
        pred = prob >= t
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        miou = 0.5 * (tp / max(tp + fp + fn, 1) + tn / max(tn + fp + fn, 1))
        rows.append({"t": round(float(t), 3), "f1": f1, "miou": miou,
                     "precision": prec, "recall": rec})

    ods = max(rows, key=lambda r: r["f1"])
    bmiou = max(rows, key=lambda r: r["miou"])
    fixed = min(rows, key=lambda r: abs(r["t"] - 0.5))
    res = {"mu": mu, "delta": delta, "n_thresholds": args.steps,
           "ods_f1": ods, "best_miou": bmiou, "fixed_0.5": fixed}
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps({"summary": res, "curve": rows}, indent=2))


if __name__ == "__main__":
    main()
