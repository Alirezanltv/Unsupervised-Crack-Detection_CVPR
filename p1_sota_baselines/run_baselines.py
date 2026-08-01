#!/usr/bin/env python3
"""Run modern industrial anomaly-detection baselines on the crack datasets.

Covers the commented rows of Table 1: PatchCore, PaDiM, DRAEM, RD4AD
(reverse distillation), EfficientAD, and (if your anomalib version ships it)
SimpleNet. Anomaly maps are dumped to disk and scored with the SAME engine
as our model (common/eval_maps.py), so numbers are like-for-like.

Not yet run on real data;
syntax-checked only. Pin anomalib and check its API if imports fail --
anomalib moves fast. Written against anomalib >= 1.0.

Dataset layout expected (per dataset, e.g. data/concrete):
    train/good/*.png        unlabeled training images (see PROTOCOL NOTE)
    test/images/*.png       test images
    test/masks/*.png        binary ground-truth masks (same stems)
    calib/*.png             held-out unlabeled images for (mu, delta)

PROTOCOL NOTE -- one-class methods assume anomaly-free training data, but our
unsupervised protocol trains on the same unlabeled split our model uses (which
contains cracks). Run BOTH settings if time permits:
  (a) train/good = the identical unlabeled split (fairest to our protocol);
  (b) train/good = a crack-free subset (fairest to the baseline).
Report (a) in Table 1 and (b) in the supplement, stating the choice.

Usage:
    python run_baselines.py --data data/concrete --out runs/concrete \
        --models patchcore padim draem rd4ad efficientad --seeds 0 1 2 3 4
Then for each run directory:
    python ../common/eval_maps.py --maps runs/concrete/patchcore/s0/maps \
        --masks data/concrete/test/masks --calib runs/concrete/patchcore/s0/calib
"""
import argparse
from pathlib import Path

import numpy as np


def build_model(name: str):
    import anomalib.models as M
    registry = {
        "patchcore": "Patchcore",
        "padim": "Padim",
        "draem": "Draem",
        "rd4ad": "ReverseDistillation",
        "efficientad": "EfficientAd",
        "simplenet": "Simplenet",
    }
    cls = getattr(M, registry[name], None)
    if cls is None:
        raise SystemExit(
            f"{name}: class {registry[name]} not in your anomalib version; "
            "use the official repo for this model and dump maps in the same "
            "layout, then score with common/eval_maps.py")
    return cls()


def run_one(model_name: str, data: Path, out: Path, seed: int, image_size: int):
    import torch
    from anomalib.data import Folder
    from anomalib.engine import Engine

    torch.manual_seed(seed)
    np.random.seed(seed)

    datamodule = Folder(
        name=data.name,
        root=data,
        normal_dir="train/good",
        abnormal_dir="test/images",
        mask_dir="test/masks",
        image_size=(image_size, image_size),
        num_workers=2,
    )
    model = build_model(model_name)
    engine = Engine(default_root_dir=out / model_name / f"s{seed}")
    engine.fit(model=model, datamodule=datamodule)

    maps_dir = out / model_name / f"s{seed}" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    preds = engine.predict(model=model, datamodule=datamodule)
    for batch in preds:
        for path, amap in zip(batch["image_path"], batch["anomaly_maps"]):
            stem = Path(path).stem
            np.save(maps_dir / f"{stem}.npy",
                    amap.squeeze().cpu().numpy().astype(np.float32))

    # calibration maps from the held-out unlabeled folder. Folder() cannot
    # represent a masks-free segmentation dir, so use PredictDataset instead.
    calib_dir = out / model_name / f"s{seed}" / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    try:
        from anomalib.data import PredictDataset
    except ImportError:
        from anomalib.data.predict import PredictDataset
    from torch.utils.data import DataLoader
    ds = PredictDataset(path=data / "calib",
                        image_size=(image_size, image_size))
    dl = DataLoader(ds, batch_size=8, num_workers=2,
                    collate_fn=getattr(ds, "collate_fn", None))
    for batch in engine.predict(model=model, dataloaders=dl):
        for path, amap in zip(batch["image_path"], batch["anomaly_maps"]):
            np.save(calib_dir / f"{Path(path).stem}.npy",
                    amap.squeeze().cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["patchcore", "padim",
                    "draem", "rd4ad", "efficientad"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    for m in args.models:
        for s in args.seeds:
            print(f"=== {m} seed {s} ===")
            run_one(m, args.data, args.out, s, args.image_size)


if __name__ == "__main__":
    main()
