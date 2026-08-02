#!/usr/bin/env python3
"""Build the harness data layout from committed split lists.

Usage (DeepCrack):
    python arrange_from_splits.py --raw-root /kaggle/temp/deepcrack_raw \
        --splits p0_reproduce/splits --name deepcrack \
        --dst /kaggle/temp/data/deepcrack \
        --masks-dir /kaggle/temp/deepcrack_raw/test_lab
"""
import argparse
import shutil
from pathlib import Path


def place(listfile: Path, raw: Path, dst: Path) -> int:
    n = 0
    for rel in listfile.read_text().split():
        src = raw / rel
        shutil.copy2(src, dst / src.name)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--masks-dir", type=Path, default=None,
                    help="directory of ground-truth masks for the test split")
    args = ap.parse_args()

    for sub in ("train/good", "test/images", "test/masks", "calib"):
        shutil.rmtree(args.dst / sub, ignore_errors=True)
        (args.dst / sub).mkdir(parents=True)

    counts = {
        "train/good": place(args.splits / f"{args.name}_train.txt",
                            args.raw_root, args.dst / "train/good"),
        "calib": place(args.splits / f"{args.name}_calib.txt",
                       args.raw_root, args.dst / "calib"),
        "test/images": place(args.splits / f"{args.name}_test.txt",
                             args.raw_root, args.dst / "test/images"),
    }
    if args.masks_dir:
        n = 0
        for p in sorted(args.masks_dir.iterdir()):
            if p.is_file():
                shutil.copy2(p, args.dst / "test/masks" / p.name)
                n += 1
        counts["test/masks"] = n
    print(counts)


if __name__ == "__main__":
    main()
