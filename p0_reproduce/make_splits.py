#!/usr/bin/env python3
"""Deterministic split generation. Run once per dataset, commit the output.

Writes splits/<name>_{train,calib,test}.txt (one relative path per line).
Splits are a pure function of (file list, seed), so anyone can regenerate
and verify them. No experiment should run on an uncommitted split.

Usage:
  # datasets with an official test split (e.g. DeepCrack): only carve calib
  python make_splits.py --name deepcrack --images /data/deepcrack_raw/train_img \
      --test-images /data/deepcrack_raw/test_img --calib 50

  # datasets without an official split (e.g. Ozgenel classification):
  python make_splits.py --name concrete --images /data/ozgenel/Positive \
      --extra-images /data/ozgenel/Negative --calib 200 --test-frac 0.15
"""
import argparse
import random
from pathlib import Path

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def listing(d: Path):
    return sorted(str(p.relative_to(d.parent)) for p in d.rglob("*")
                  if p.suffix.lower() in EXTS and not p.name.startswith("."))


def write(out: Path, name: str, split: str, items):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{name}_{split}.txt"
    f.write_text("\n".join(items) + "\n")
    print(f"{f}  ({len(items)} files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", type=Path, required=True,
                    help="pool of (unlabeled) images to split")
    ap.add_argument("--extra-images", type=Path, default=None,
                    help="optional second pool merged into --images")
    ap.add_argument("--test-images", type=Path, default=None,
                    help="official test set, if one exists (no carving)")
    ap.add_argument("--calib", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="fraction carved for test when no official split")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "splits")
    args = ap.parse_args()

    pool = listing(args.images)
    if args.extra_images:
        pool += listing(args.extra_images)
    pool = sorted(pool)
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    if args.test_images is not None:
        test = listing(args.test_images)
    else:
        if not 0 < args.test_frac < 1:
            raise SystemExit("--test-frac required when no --test-images")
        n_test = int(len(pool) * args.test_frac)
        test, pool = pool[:n_test], pool[n_test:]

    if args.calib >= len(pool):
        raise SystemExit(f"calib {args.calib} >= remaining pool {len(pool)}")
    calib, train = pool[:args.calib], pool[args.calib:]

    write(args.out, args.name, "train", sorted(train))
    write(args.out, args.name, "calib", sorted(calib))
    write(args.out, args.name, "test", sorted(test))
    print(f"seed={args.seed}; commit the three files before running anything.")


if __name__ == "__main__":
    main()
