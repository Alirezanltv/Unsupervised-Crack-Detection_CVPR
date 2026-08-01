#!/usr/bin/env python3
"""Dump AG-DSCAE anomaly maps in the layout common/eval_maps.py consumes.

This is the glue between a trained checkpoint and every paper number:
    checkpoint -> maps/*.npy + calib/*.npy -> eval_maps.py -> MIoU/F1/AUROC/AP

ADAPT ONE FUNCTION: load_model() below must return your trained model with a
callable that maps a float image tensor (B x 3 x H x W, [0,1]) to the RAW
anomaly map (B x H x W) BEFORE any sigmoid/thresholding — i.e. the smoothed
gradient-reconstruction error of Eq. (13) without mu/delta standardization
(standardization happens inside eval_maps from the calib maps).

Usage:
    python dump_maps.py --checkpoint best.pt \
        --images data/concrete/test/images --out runs/ours/concrete/maps
    python dump_maps.py --checkpoint best.pt \
        --images data/concrete/calib --out runs/ours/concrete/calib
"""
import argparse
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# ADAPT THIS to your AG-DSCAE code.
# --------------------------------------------------------------------------
def load_model(checkpoint: Path):
    """Return anomaly_fn(batch) -> raw map tensor (B x H x W).

    Reference pattern:

        import torch
        import torch.nn.functional as F
        from your_code import AGDSCAE, sobel_mag, gaussian_blur

        model = AGDSCAE()
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        model.cuda().eval()

        @torch.no_grad()
        def anomaly_fn(x):
            x = x.cuda()
            recon = model(x)
            err = (sobel_mag(x.mean(1)) - sobel_mag(recon.mean(1))).abs()
            return gaussian_blur(err, sigma=1.5).cpu()
        return anomaly_fn
    """
    raise NotImplementedError("wire this to your AG-DSCAE implementation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import torch
    from PIL import Image

    anomaly_fn = load_model(args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        raise SystemExit(f"no images in {args.images}")

    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i + args.batch]
        imgs = np.stack([
            np.asarray(Image.open(p).convert("RGB")
                       .resize((args.size, args.size), Image.BICUBIC),
                       dtype=np.float32) / 255.0
            for p in chunk])
        batch = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        maps = anomaly_fn(batch)
        for p, m in zip(chunk, maps):
            np.save(args.out / f"{p.stem}.npy",
                    m.squeeze().cpu().numpy().astype(np.float32))
    print(f"wrote {len(paths)} maps to {args.out}")


if __name__ == "__main__":
    main()
