#!/usr/bin/env python3
"""Consolidate the Concrete/Ozgenel clean-normal arm into the paper's source
of truth, with Welch t-tests (Holm-corrected) for ours vs. each 5-seed
baseline. Same statistical conventions as compute_table1.py.

Run from the repo root:  python results/compute_concrete.py
Reads:  results/concrete_*_{result,sweep}.json
Writes: results/concrete_table_final.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).parent
METRICS = ("auroc", "ap", "ods_f1", "ods_miou")


def load(name, seeds):
    rows = []
    for s in seeds:
        res = json.loads((R / f"concrete_{name}_s{s}_result.json").read_text())
        sw = json.loads((R / f"concrete_{name}_s{s}_sweep.json").read_text())
        sw = sw.get("summary", sw)
        rows.append((res["pixel_auroc"], res["pixel_ap"],
                     sw["ods_f1"]["f1"], sw["best_miou"]["miou"]))
    return np.array(rows)


def summarize(arr):
    return {m: {"mean": round(float(arr[:, i].mean()), 4),
                "std": round(float(arr[:, i].std(ddof=1)), 4) if len(arr) > 1 else None,
                "n_seeds": len(arr)}
            for i, m in enumerate(METRICS)}


def welch_family(ours, other, label):
    raw = {}
    for i, m in enumerate(METRICS):
        t, p = stats.ttest_ind(ours[:, i], other[:, i], equal_var=False)
        raw[m] = {"t": round(float(t), 3), "p": round(float(p), 4),
                  "direction": "ours_higher" if ours[:, i].mean() > other[:, i].mean()
                  else "ours_lower"}
    order = sorted(METRICS, key=lambda m: raw[m]["p"])
    running = 0.0
    for rank, m in enumerate(order):
        running = max(running, (len(order) - rank) * raw[m]["p"])
        raw[m]["p_holm"] = round(min(running, 1.0), 4)
        raw[m]["significant_at_0.05"] = raw[m]["p_holm"] < 0.05
    return {label: raw}


def main():
    ours = load("ours", range(3))
    padim = load("padim", range(5))
    patchcore = load("patchcore", range(5))
    rd4ad = load("rd4ad", [0])
    eff = load("efficientad", [0])

    out = {
        "dataset": ("Ozgenel concrete: 250 clean-normal train / 50 calib images "
                    "(classification set Negative, seed 2027, committed lists); "
                    "446 stem-matched segmentation test pairs at max side 512"),
        "protocol": ("CLEAN-NORMAL regime: one-class baselines trained on genuinely "
                     "crack-free images (their intended assumption); ours trained on the "
                     "same unlabeled images; identical scoring engine and 99-point sweep "
                     "as the DeepCrack arm"),
        "models": {
            "ours": summarize(ours),
            "padim": summarize(padim),
            "patchcore": summarize(patchcore),
            "rd4ad_s0": summarize(rd4ad),
            "efficientad_s0": summarize(eff),
            "draem_dtd_s0": {"status": ("INTERRUPTED at epoch 132/300 by Kaggle "
                                        "infrastructure failure (filesystem went "
                                        "read-only; SIGBUS). No maps exported; no "
                                        "number enters the table. On-the-fly "
                                        "validation at interruption: pixel AUROC "
                                        "0.775, in the other methods' range -- "
                                        "directional only. Full record: "
                                        "results/logs/concrete/"
                                        "draem_dtd_interrupted_ep132.log")},
        },
        "welch_holm": {},
    }
    out["welch_holm"].update(welch_family(ours, padim, "ours_vs_padim"))
    out["welch_holm"].update(welch_family(ours, patchcore, "ours_vs_patchcore"))

    (R / "concrete_table_final.json").write_text(json.dumps(out, indent=2))
    for name in ("ours", "padim", "patchcore", "rd4ad_s0", "efficientad_s0"):
        s = out["models"][name]
        print(name.ljust(15), " ".join(
            f"{m}={s[m]['mean']}±{s[m]['std']}" if s[m]["std"] is not None
            else f"{m}={s[m]['mean']}" for m in METRICS))
    print(json.dumps(out["welch_holm"], indent=1))


if __name__ == "__main__":
    main()
