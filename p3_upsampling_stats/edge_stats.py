#!/usr/bin/env python3
"""Quantify how 28x28 -> 256x256 bicubic upsampling changes MNIST edge statistics.

Addresses the reviewer critique that the source-domain preprocessing
"fundamentally alters its edge statistics" by measuring the alteration.

Method (all on public MNIST train images, fixed seed):
  * native  : 28x28 grayscale in [0,1]
  * upsampled: single-step PIL bicubic resize to 256x256, clipped to [0,1]
    (the paper's preprocessing; note PIL and torch bicubic kernels differ
    slightly -- rerun with your training transform to confirm)
  Statistics, computed identically on both versions:
  1. Edge transition width: for Canny edge pixels, the 10%-90% intensity
     rise distance along the gradient direction, in PIXELS (what a fixed
     conv kernel sees) and as a fraction of image side.
  2. Mean Sobel gradient magnitude at Canny edge pixels (intensity/pixel).
  3. Canny edge-pixel density (percent of pixels), thresholds 50/150 on
     uint8 -- the same thresholds used elsewhere in the paper.
  4. Stroke width from the medial axis of the Otsu-binarized digit, as a
     fraction of image side (scale-free sanity check).
  5. Edge-orientation histograms (16 bins over [0, pi)) at edge pixels:
     KL divergence and Pearson correlation between native and upsampled.

Outputs: results.json and upsampling_stats.png next to this script.

Usage:  python edge_stats.py [--n 1000] [--n-profile 200] [--data-dir .cache]
"""
import argparse
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"


def load_mnist(cache: Path) -> np.ndarray:
    cache.mkdir(parents=True, exist_ok=True)
    gz = cache / "train-images-idx3-ubyte.gz"
    if not gz.exists():
        print("downloading MNIST train images ...")
        urllib.request.urlretrieve(MNIST_URL, gz)
    with gzip.open(gz, "rb") as f:
        data = f.read()
    assert int.from_bytes(data[:4], "big") == 2051, "bad IDX magic"
    n = int.from_bytes(data[4:8], "big")
    return np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)


RESAMPLERS = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
              "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def upsample(img28: np.ndarray, scheme: str = "bicubic") -> np.ndarray:
    out = Image.fromarray(img28).resize((256, 256), RESAMPLERS[scheme])
    return np.asarray(out, dtype=np.float64).clip(0, 255) / 255.0


def sobel_mag_ori(img: np.ndarray):
    from scipy.ndimage import sobel
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    return np.hypot(gx, gy) / 8.0, np.mod(np.arctan2(gy, gx), np.pi)  # /8: Sobel gain


def canny_edges(img01: np.ndarray) -> np.ndarray:
    from skimage.feature import canny
    # 50/150 on uint8 == 50/255, 150/255 on [0,1]
    return canny(img01, sigma=1.0, low_threshold=50 / 255, high_threshold=150 / 255)


def transition_widths(img: np.ndarray, edges: np.ndarray, ori: np.ndarray,
                      half_len: int, rng, per_img: int = 30):
    """10-90% rise distance along the gradient direction, in pixels."""
    from scipy.ndimage import map_coordinates
    ys, xs = np.nonzero(edges)
    if len(ys) == 0:
        return []
    sel = rng.choice(len(ys), size=min(per_img, len(ys)), replace=False)
    t = np.linspace(-half_len, half_len, 4 * half_len + 1)
    widths = []
    for i in sel:
        y, x = ys[i], xs[i]
        th = ori[y, x]
        coords = np.vstack([y + t * np.sin(th), x + t * np.cos(th)])
        prof = map_coordinates(img, coords, order=1, mode="nearest")
        lo, hi = prof.min(), prof.max()
        if hi - lo < 0.2:               # no real edge along this cut
            continue
        p = (prof - lo) / (hi - lo)
        if p[0] > p[-1]:
            p = p[::-1]
        above10 = np.nonzero(p >= 0.1)[0]
        above90 = np.nonzero(p >= 0.9)[0]
        if len(above10) and len(above90):
            w = (above90[0] - above10[0]) * (t[1] - t[0])
            if w > 0:
                widths.append(w)
    return widths


def stroke_width_frac(img01: np.ndarray):
    from skimage.filters import threshold_otsu
    from skimage.morphology import medial_axis
    if img01.max() <= 0:
        return []
    binary = img01 > threshold_otsu(img01)
    if binary.sum() < 8:
        return []
    skel, dist = medial_axis(binary, return_distance=True)
    return (2.0 * dist[skel] / img01.shape[0]).tolist()


def kl_hist(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p / p.sum() + eps
    q = q / q.sum() + eps
    return float(np.sum(p * np.log(p / q)))


def analyze(images: np.ndarray, n_profile: int, rng):
    res = {"grad_mag_edge": [], "edge_density": [], "stroke_frac": [],
           "ori_hist": np.zeros(16), "trans_w": []}
    size = images.shape[1]
    half = 4 if size == 28 else 20
    for k, img in enumerate(images):
        img01 = img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img
        mag, ori = sobel_mag_ori(img01)
        edges = canny_edges(img01)
        if edges.sum() == 0:
            continue
        res["grad_mag_edge"].append(float(mag[edges].mean()))
        res["edge_density"].append(float(edges.mean() * 100))
        res["stroke_frac"].extend(stroke_width_frac(img01))
        h, _ = np.histogram(ori[edges], bins=16, range=(0, np.pi))
        res["ori_hist"] += h
        if k < n_profile:
            res["trans_w"].extend(transition_widths(img01, edges, ori, half, rng))
    return res, size


def compare_schemes(native: np.ndarray, n_profile: int):
    """Same estimators for every interpolation scheme; KL of orientation
    histograms is computed against the NATIVE distribution."""
    rn, _ = analyze(native, n_profile, np.random.default_rng(1))
    table = {}
    for scheme in RESAMPLERS:
        up = np.stack([upsample(im, scheme) for im in native])
        ru, _ = analyze(up, n_profile, np.random.default_rng(1))
        table[scheme] = {
            "transition_width_px_mean": float(np.mean(ru["trans_w"])),
            "transition_width_px_std": float(np.std(ru["trans_w"])),
            "grad_mag_at_edges": float(np.mean(ru["grad_mag_edge"])),
            "edge_density_pct": float(np.mean(ru["edge_density"])),
            "orientation_kl_vs_native": kl_hist(rn["ori_hist"], ru["ori_hist"]),
        }
    table["native_reference"] = {
        "transition_width_px_mean": float(np.mean(rn["trans_w"])),
        "grad_mag_at_edges": float(np.mean(rn["grad_mag_edge"])),
        "edge_density_pct": float(np.mean(rn["edge_density"])),
    }
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-profile", type=int, default=200)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).parent / ".cache")
    ap.add_argument("--schemes", action="store_true",
                    help="also compare nearest/bilinear/bicubic/lanczos; "
                         "writes results_schemes.json")
    args = ap.parse_args()
    here = Path(__file__).parent
    rng = np.random.default_rng(0)

    all_imgs = load_mnist(args.data_dir)
    idx = rng.choice(len(all_imgs), size=args.n, replace=False)
    native = all_imgs[idx]

    if args.schemes:
        print(f"comparing interpolation schemes on {args.n} digits ...")
        table = compare_schemes(native, args.n_profile)
        (here / "results_schemes.json").write_text(json.dumps(table, indent=2))
        print(json.dumps(table, indent=2))
        return

    up = np.stack([upsample(im) for im in native])

    print(f"analyzing {args.n} digits (native 28^2 and bicubic 256^2) ...")
    rn, _ = analyze(native, args.n_profile, np.random.default_rng(1))
    ru, _ = analyze(up, args.n_profile, np.random.default_rng(1))

    def ms(x):
        a = np.asarray(x, dtype=np.float64)
        return float(a.mean()), float(a.std())

    out = {}
    for key, label in [("trans_w", "transition_width_px"),
                       ("grad_mag_edge", "grad_mag_at_edges"),
                       ("edge_density", "edge_density_pct"),
                       ("stroke_frac", "stroke_width_frac")]:
        (mn, sn), (mu_, su) = ms(rn[key]), ms(ru[key])
        out[label] = {"native_mean": mn, "native_std": sn,
                      "up_mean": mu_, "up_std": su,
                      "ratio_up_over_native": mu_ / mn if mn else None}
    out["transition_width_frac"] = {
        "native": out["transition_width_px"]["native_mean"] / 28,
        "up": out["transition_width_px"]["up_mean"] / 256}
    out["orientation"] = {
        "kl_native_vs_up": kl_hist(rn["ori_hist"], ru["ori_hist"]),
        "pearson_r": float(np.corrcoef(rn["ori_hist"], ru["ori_hist"])[0, 1])}
    out["config"] = {"n_images": args.n, "n_profile_images": args.n_profile,
                     "interp": "PIL BICUBIC single-step 28->256",
                     "canny": "sigma=1.0, low=50/255, high=150/255"}

    (here / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ---- figure -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    ex = native[0].astype(np.float64) / 255.0
    exu = up[0]
    row = ex[ex.sum(axis=1).argmax()]
    rowu = exu[exu.sum(axis=1).argmax()]
    axes[0].plot(np.linspace(0, 1, 28), row, "o-", ms=3, label="native $28^2$")
    axes[0].plot(np.linspace(0, 1, 256), rowu, "-", label="bicubic $256^2$")
    axes[0].set_xlabel("position (fraction of image)")
    axes[0].set_ylabel("intensity")
    axes[0].set_title("intensity profile through one digit")
    axes[0].legend(fontsize=7)

    axes[1].hist(rn["trans_w"], bins=24, density=True, alpha=0.65,
                 label=f"native (mean {np.mean(rn['trans_w']):.1f} px)")
    axes[1].hist(ru["trans_w"], bins=24, density=True, alpha=0.65,
                 label=f"upsampled (mean {np.mean(ru['trans_w']):.1f} px)")
    axes[1].set_xlabel("10-90% edge transition width (px)")
    axes[1].set_title("edge sharpness in pixel units")
    axes[1].legend(fontsize=7)

    centers = np.linspace(0, np.pi, 17)[:-1] + np.pi / 32
    axes[2].plot(centers, rn["ori_hist"] / rn["ori_hist"].sum(), "o-", ms=3,
                 label="native")
    axes[2].plot(centers, ru["ori_hist"] / ru["ori_hist"].sum(), "s-", ms=3,
                 label="upsampled")
    axes[2].set_xlabel("edge orientation (rad)")
    axes[2].set_title("orientation distribution")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(here / "upsampling_stats.png", dpi=200)
    print("wrote results.json and upsampling_stats.png")


if __name__ == "__main__":
    main()
