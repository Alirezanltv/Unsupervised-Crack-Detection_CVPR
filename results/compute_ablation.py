#!/usr/bin/env python3
"""Consolidate the matched-budget ablation runs into the paper's component
table, with stage-wise deltas and Welch t-tests where seed counts allow.

Variants (all trained with the same 3xE epoch budget as the full model):
  scratch  -- no source domain: target-only reconstruction + SSIM refinement
  pretrain -- + MNIST pretraining (no OT, no gates)
  ot       -- + optimal-transport alignment (no gates)
  full     -- + attention gates (= the released model)

Component attributions:
  source pretraining = pretrain - scratch
  OT alignment       = ot - pretrain
  attention gates    = full - ot

Run from the repo root:  python results/compute_ablation.py
Reads:  results/deepcrack_abl_{variant}_s{n}_{result,sweep}.json
Writes: results/deepcrack_ablation_final.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).parent

FULL = {
    # seed: (auroc, ap, ods_f1, ods_miou) -- from banked per-seed records
    0: (0.9307, 0.4883, 0.5005, 0.6432),
    1: (0.8925, 0.2640, 0.4053, 0.5890),
    2: (0.9099, 0.3683, 0.4528, 0.6158),
    3: (0.8889, 0.2878, 0.4380, 0.6063),
    4: (0.9029, 0.3075, 0.4379, 0.6047),
}
METRICS = ("auroc", "ap", "ods_f1", "ods_miou")
TARGET_SEEDS = 3  # per variant; table is PARTIAL until every variant has these


def load_variant(name):
    rows, seeds = [], []
    for s in range(5):
        rp = R / f"deepcrack_abl_{name}_s{s}_result.json"
        sp = R / f"deepcrack_abl_{name}_s{s}_sweep.json"
        if not (rp.exists() and sp.exists()):
            continue
        res = json.loads(rp.read_text())
        sw = json.loads(sp.read_text())
        sw = sw.get("summary", sw)
        rows.append((res["pixel_auroc"], res["pixel_ap"],
                     sw["ods_f1"]["f1"], sw["best_miou"]["miou"]))
        seeds.append(s)
    return np.array(rows), seeds


def summarize(arr, seeds):
    return {"seeds": seeds,
            **{m: {"mean": round(float(arr[:, i].mean()), 4),
                   "std": round(float(arr[:, i].std(ddof=1)), 4) if len(arr) > 1 else None}
               for i, m in enumerate(METRICS)}}


def delta(a, b, label):
    """Mean difference a-b per metric; Welch t-test when both sides have >=2 seeds."""
    out = {}
    for i, m in enumerate(METRICS):
        d = {"delta": round(float(a[:, i].mean() - b[:, i].mean()), 4)}
        if len(a) >= 2 and len(b) >= 2:
            t, p = stats.ttest_ind(a[:, i], b[:, i], equal_var=False)
            d.update(t=round(float(t), 3), p=round(float(p), 4))
        out[m] = d
    return {label: out}


def main():
    full = np.array([FULL[s] for s in sorted(FULL)])
    variants = {"full": (full, sorted(FULL))}
    for name in ("scratch", "pretrain", "ot"):
        variants[name] = load_variant(name)

    complete = all(len(variants[n][1]) >= TARGET_SEEDS
                   for n in ("scratch", "pretrain", "ot"))
    out = {
        "dataset": "DeepCrack official test split (237 images), committed splits seed 2027",
        "protocol": ("matched 3xE epoch budget for every variant; identical eval pipeline "
                     "as Table 1 (common/eval_maps.py + 99-point sweep)"),
        "status": "complete" if complete else
                  f"PARTIAL -- do not put in the paper until every variant has {TARGET_SEEDS} seeds",
        "models": {n: summarize(arr, seeds) for n, (arr, seeds) in variants.items()},
        "component_deltas": {},
    }
    out["component_deltas"].update(delta(variants["pretrain"][0], variants["scratch"][0],
                                         "source_pretraining (pretrain - scratch)"))
    out["component_deltas"].update(delta(variants["ot"][0], variants["pretrain"][0],
                                         "ot_alignment (ot - pretrain)"))
    out["component_deltas"].update(delta(variants["full"][0], variants["ot"][0],
                                         "attention_gates (full - ot)"))
    out["component_deltas"].update(delta(variants["full"][0], variants["scratch"][0],
                                         "whole_pipeline (full - scratch)"))

    (R / "deepcrack_ablation_final.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
