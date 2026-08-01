#!/usr/bin/env python3
"""Smoke test for eval_maps.py on synthetic data with known answers."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "maps").mkdir()
        (root / "masks").mkdir()
        (root / "calib").mkdir()

        rng = np.random.default_rng(0)

        # PROTOCOL PROPERTY this test documents: thresholding the sigmoid of
        # the z-score at 0.5 is the same as thresholding the RAW map at the
        # calibration MEAN. That only yields high precision when normal-pixel
        # errors are sparse/heavy-tailed (most mass below the mean), which is
        # typical of smoothed gradient-reconstruction errors. With symmetric
        # (e.g. Gaussian) background scores, ~half the background exceeds the
        # mean and MIoU collapses -- verify your real error maps are skewed
        # enough that the fixed-0.5 protocol can actually produce the paper's
        # numbers.
        def background(shape):
            b = np.zeros(shape, np.float32)          # mostly zero error
            spikes = rng.random(shape) < 0.05        # sparse error bumps
            b[spikes] = rng.exponential(1.0, int(spikes.sum()))
            return b

        for i in range(4):
            np.save(root / "calib" / f"c{i}.npy", background((64, 64)))
        # test image: left half anomalous with huge scores -> prob ~ 1 there
        for i in range(3):
            m = background((64, 64))
            m[:, :32] += 50.0
            np.save(root / "maps" / f"img{i}.npy", m)
            gt = np.zeros((64, 64), np.uint8)
            gt[:, :32] = 255
            Image.fromarray(gt).save(root / "masks" / f"img{i}.png")

        out = root / "res.json"
        subprocess.run([sys.executable, str(HERE / "eval_maps.py"),
                        "--maps", str(root / "maps"),
                        "--masks", str(root / "masks"),
                        "--calib", str(root / "calib"),
                        "--out", str(out)],
                       check=True, capture_output=True)
        r = json.loads(out.read_text())
        # expected ~0.956: the ~5% background spikes above the calibration
        # mean become false positives, exactly as the protocol predicts
        assert 0.9 < r["miou"] < 0.99, r
        assert r["f1"] > 0.9, r
        assert r["pixel_auroc"] > 0.999, r
        assert r["pixel_ap"] > 0.999, r
        print("eval_maps smoke test ok:",
              {k: round(r[k], 4) for k in ("miou", "f1", "pixel_auroc", "pixel_ap")})


if __name__ == "__main__":
    main()
