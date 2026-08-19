#!/usr/bin/env python3
"""Prepare the Concrete/Ozgenel clean-normal arm.

Sources (attach as Kaggle inputs):
  --cls-root  Ozgenel classification set (Negative/ = 20k crack-free images)
              e.g. /kaggle/input/surface-crack-detection
  --seg-root  Ozgenel segmentation set (458 hi-res images + BW masks)
              e.g. /kaggle/input/concrete-crack-segmentation-dataset

Layout produced (identical to the DeepCrack arrangement):
  <dst>/train/good   250 clean normals   (sampled from Negative, seed 2027)
  <dst>/calib        50 clean normals    (disjoint sample, unlabeled use)
  <dst>/test/images  458 crack images    (resized, max side --max-side)
  <dst>/test/masks   458 binary masks    (nearest-neighbor, >127 binarized)

Split lists are written to --splits-out as concrete_{train,calib,test}.txt
(train/calib lines relative to cls-root) so the sample is committed to git
and every later run is reproducible without re-sampling.

The segmentation set's folder names vary between mirrors, so the script
auto-discovers the image/mask pair: it looks for two directories holding the
same number (>350) of stem-matched files and takes as masks the directory
whose sampled files decode to single-channel/binary content. Override with
--seg-images/--seg-masks if the guess is ever wrong (the choice is printed).
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_seg_pair(root):
    by_dir = defaultdict(list)
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in EXTS and p.is_file():
            by_dir[p.parent].append(p)
    dirs = {d: sorted(fs) for d, fs in by_dir.items() if len(fs) > 350}
    if len(dirs) < 2:
        raise SystemExit(f"expected two dirs with >350 files under {root}, "
                         f"found {[(str(d), len(f)) for d, f in dirs.items()]}")
    cands = sorted(dirs)
    best = None
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            stems_a = {f.stem for f in dirs[a]}
            stems_b = {f.stem for f in dirs[b]}
            common = stems_a & stems_b
            if len(common) > 350 and (best is None or len(common) > best[2]):
                best = (a, b, len(common))
    if best is None:
        raise SystemExit("no stem-matched directory pair found")
    a, b, n = best

    def maskiness(d):
        score = 0
        for f in dirs[d][:5]:
            arr = np.asarray(Image.open(f).convert("L"))
            frac_extreme = float(((arr < 32) | (arr > 223)).mean())
            score += frac_extreme
        return score

    masks_dir, imgs_dir = (a, b) if maskiness(a) > maskiness(b) else (b, a)
    print(f"segmentation pair: images={imgs_dir}  masks={masks_dir}  ({n} stems)")
    common = sorted({f.stem for f in dirs[imgs_dir]} & {f.stem for f in dirs[masks_dir]})
    img_by_stem = {f.stem: f for f in dirs[imgs_dir]}
    msk_by_stem = {f.stem: f for f in dirs[masks_dir]}
    return [(img_by_stem[s], msk_by_stem[s]) for s in common]


def resize_max_side(im, max_side, resample):
    w, h = im.size
    if max(w, h) <= max_side:
        return im
    if w >= h:
        return im.resize((max_side, round(h * max_side / w)), resample)
    return im.resize((round(w * max_side / h), max_side), resample)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls-root", required=True)
    ap.add_argument("--seg-root", required=True)
    ap.add_argument("--seg-images", default=None, help="override auto-detection")
    ap.add_argument("--seg-masks", default=None, help="override auto-detection")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--splits-out", required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-calib", type=int, default=50)
    ap.add_argument("--max-side", type=int, default=512)
    args = ap.parse_args()

    cls_root = Path(args.cls_root)
    neg_dirs = [d for d in cls_root.rglob("*") if d.is_dir() and d.name.lower() == "negative"]
    if not neg_dirs:
        raise SystemExit(f"no Negative/ directory under {cls_root}")
    neg = sorted(p for p in neg_dirs[0].iterdir() if p.suffix.lower() in EXTS)
    print(f"clean normals available: {len(neg)} in {neg_dirs[0]}")
    if len(neg) < args.n_train + args.n_calib:
        raise SystemExit("not enough Negative images")

    rng = random.Random(args.seed)
    sample = rng.sample(neg, args.n_train + args.n_calib)
    train, calib = sample[:args.n_train], sample[args.n_train:]

    dst = Path(args.dst)
    for sub in ("train/good", "calib", "test/images", "test/masks"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for p in train:
        shutil.copy2(p, dst / "train/good" / p.name)
    for p in calib:
        shutil.copy2(p, dst / "calib" / p.name)

    if args.seg_images and args.seg_masks:
        imgs = sorted(p for p in Path(args.seg_images).iterdir() if p.suffix.lower() in EXTS)
        msks = {p.stem: p for p in Path(args.seg_masks).iterdir() if p.suffix.lower() in EXTS}
        pairs = [(i, msks[i.stem]) for i in imgs if i.stem in msks]
    else:
        pairs = find_seg_pair(args.seg_root)
    print(f"test pairs: {len(pairs)}")

    for img_p, msk_p in pairs:
        im = resize_max_side(Image.open(img_p).convert("RGB"), args.max_side, Image.BILINEAR)
        im.save(dst / "test/images" / (img_p.stem + ".jpg"), quality=95)
        mk = resize_max_side(Image.open(msk_p).convert("L"), args.max_side, Image.NEAREST)
        arr = (np.asarray(mk) > 127).astype(np.uint8) * 255
        Image.fromarray(arr).save(dst / "test/masks" / (img_p.stem + ".png"))

    so = Path(args.splits_out)
    so.mkdir(parents=True, exist_ok=True)
    (so / "concrete_train.txt").write_text(
        "\n".join(str(p.relative_to(cls_root)) for p in train) + "\n")
    (so / "concrete_calib.txt").write_text(
        "\n".join(str(p.relative_to(cls_root)) for p in calib) + "\n")
    (so / "concrete_test.txt").write_text(
        "\n".join(p.stem for p, _ in pairs) + "\n")

    counts = {s: len(list((dst / s).iterdir()))
              for s in ("train/good", "calib", "test/images", "test/masks")}
    print(counts)
    crack_frac = []
    for m in sorted((dst / "test/masks").iterdir())[:20]:
        crack_frac.append((np.asarray(Image.open(m)) > 127).mean())
    print(f"sanity: mean crack fraction over first 20 masks = {np.mean(crack_frac):.4f}")


if __name__ == "__main__":
    main()
