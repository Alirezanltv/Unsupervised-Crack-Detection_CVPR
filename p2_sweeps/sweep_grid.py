#!/usr/bin/env python3
"""Hyperparameter sensitivity sweep orchestrator.

Sweeps the paper's hand-set values one axis at a time around the defaults
(cheaper than a full grid and answers the 'brittle?' question directly):

    lambda_ot   : 0.01 0.05 0.1 0.2 0.5      (default 0.1 stage 2 / 0.05 stage 3)
    lambda_edge : 0.1 0.3 0.5 1.0            (default 0.3)
    sinkhorn_k  : 10 25 50 100               (default 50)
    sinkhorn_eps: 0.01 0.05 0.1 0.5          (default 0.1)
    cost_beta   : 0.0 0.15 0.3 0.6           (orientation term; default 0.3)
    cost_gamma  : 0.0 0.1 0.2 0.4            (curvature term; default 0.2)

INTERFACE TO YOUR TRAINER -- adapt run_training() below: it must launch one
full 3-stage training on Concrete with the given overrides and return the
path to dumped test anomaly maps. Everything else (scoring via
common/eval_maps.py, CSV, plots) is provided.

Usage: python sweep_grid.py --trainer "python /path/to/your/train.py" \
                            --data /path/to/concrete --out sweeps/
Not yet run on real data; syntax-checked only.
"""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

AXES = {
    "lambda_ot": [0.01, 0.05, 0.1, 0.2, 0.5],
    "lambda_edge": [0.1, 0.3, 0.5, 1.0],
    "sinkhorn_k": [10, 25, 50, 100],
    "sinkhorn_eps": [0.01, 0.05, 0.1, 0.5],
    "cost_beta": [0.0, 0.15, 0.3, 0.6],
    "cost_gamma": [0.0, 0.1, 0.2, 0.4],
}
DEFAULTS = {"lambda_ot": 0.1, "lambda_edge": 0.3, "sinkhorn_k": 50,
            "sinkhorn_eps": 0.1, "cost_beta": 0.3, "cost_gamma": 0.2}


def run_training(trainer_cmd: str, data: Path, out: Path, overrides: dict) -> Path:
    """ADAPT THIS to your training entrypoint. It must:
    train the full pipeline with `overrides` applied, run test inference,
    and leave raw anomaly maps in <out>/maps plus calibration maps in
    <out>/calib. The example below passes overrides as --key value flags."""
    flags = " ".join(f"--{k} {v}" for k, v in overrides.items())
    cmd = f"{trainer_cmd} --data {data} --out {out} {flags}"
    print("::", cmd)
    subprocess.run(cmd, shell=True, check=True)
    return out


def score(maps_dir: Path, masks: Path, calib: Path) -> dict:
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "eval_maps", Path(__file__).parent.parent / "common" / "eval_maps.py")
    em = module_from_spec(spec)
    spec.loader.exec_module(em)
    import numpy as np
    calib_v = np.concatenate([em.load_map(p).ravel()
                              for p in sorted(calib.iterdir())])
    mu, delta = float(calib_v.mean()), float(calib_v.std())
    scores = []
    for mp, gt in em.find_pairs(maps_dir, masks):
        prob = em.standardize(em.load_map(mp), mu, delta)
        scores.append(em.binary_metrics((prob >= 0.5).astype("uint8"),
                                        em.load_mask(gt))["miou"])
    return {"miou": float(np.mean(scores))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("sweeps"))
    ap.add_argument("--axes", nargs="+", default=list(AXES))
    args = ap.parse_args()

    rows = []
    for axis in args.axes:
        for val in AXES[axis]:
            overrides = dict(DEFAULTS, **{axis: val})
            run_dir = args.out / f"{axis}_{val}"
            run_training(args.trainer, args.data, run_dir, overrides)
            r = score(run_dir / "maps", args.masks, run_dir / "calib")
            rows.append({"axis": axis, "value": val, **r})
            (args.out / "sweep_results.json").write_text(json.dumps(rows, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axes_list = sorted({r["axis"] for r in rows})
    fig, axs = plt.subplots(1, len(axes_list), figsize=(3 * len(axes_list), 2.6))
    for ax, axis in zip(np.atleast_1d(axs), axes_list):
        pts = sorted([(r["value"], r["miou"]) for r in rows if r["axis"] == axis])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-")
        ax.axvline(DEFAULTS[axis], ls="--", lw=0.8)
        ax.set_title(axis)
        ax.set_ylabel("MIoU")
    fig.tight_layout()
    fig.savefig(args.out / "sweep_curves.png", dpi=200)


if __name__ == "__main__":
    import numpy as np  # used in plotting section
    main()
