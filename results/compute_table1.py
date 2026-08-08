#!/usr/bin/env python3
"""Consolidate the DeepCrack per-seed JSONs into the paper's Table-1 source
of truth, with Welch t-tests (Holm-corrected) for ours vs. each 5-seed
baseline.

Welch (unequal-variance, unpaired) is the right test here: seeds are
independent random streams per method, so pairing by seed index would assert
a correlation that does not exist.

Run from the repo root:  python results/compute_table1.py
Reads:  results/deepcrack_*_result.json / _sweep.json (+ ours' banked seeds)
Writes: results/deepcrack_table1_final.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).parent

OURS = {
    # seed: (auroc, ap, ods_f1, ods_miou) -- from banked per-seed records
    0: (0.9307, 0.4883, 0.5005, 0.6432),
    1: (0.8925, 0.2640, 0.4053, 0.5890),
    2: (0.9099, 0.3683, 0.4528, 0.6158),
    3: (0.8889, 0.2878, 0.4380, 0.6063),
    4: (0.9029, 0.3075, 0.4379, 0.6047),
}
METRICS = ("auroc", "ap", "ods_f1", "ods_miou")


def load_model(name, seeds):
    rows = []
    for s in seeds:
        res = json.loads((R / f"deepcrack_{name}_s{s}_result.json").read_text())
        sw = json.loads((R / f"deepcrack_{name}_s{s}_sweep.json").read_text())
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
    """All four metrics tested; Holm correction within this comparison family."""
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
    ours = np.array([OURS[s] for s in sorted(OURS)])
    padim = load_model("padim", range(5))
    patchcore = load_model("patchcore", range(5))

    out = {
        "dataset": "DeepCrack official test split (237 images), committed splits seed 2027",
        "protocol": ("all methods trained on the identical crack-containing train split; "
                     "maps scored at GT resolution by common/eval_maps.py; AUROC/AP threshold-free; "
                     "ODS-F1 and best-MIoU at dataset-level thresholds swept on the standardized maps"),
        "models": {
            "agdscae": summarize(ours),
            "padim": summarize(padim),
            "patchcore": summarize(patchcore),
            "rd4ad_s0": {"auroc": 0.8723, "ap": 0.2193, "ods_f1": 0.3197,
                         "ods_miou": 0.5552, "n_seeds": 1},
            "efficientad_s0": {"auroc": 0.7563, "ap": 0.1358, "ods_f1": 0.2575,
                               "ods_miou": 0.5199, "n_seeds": 1},
            "draem_s0": {"auroc": 0.4996, "ap": 0.0432, "n_seeds": 1,
                         "status": "collapse under contaminated normals; report only with explanation"},
        },
        "welch_holm": {},
    }
    out["welch_holm"].update(welch_family(ours, padim, "agdscae_vs_padim"))
    out["welch_holm"].update(welch_family(ours, patchcore, "agdscae_vs_patchcore"))

    (R / "deepcrack_table1_final.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["welch_holm"], indent=1))
    for name in ("agdscae", "padim", "patchcore"):
        print(name, out["models"][name])


if __name__ == "__main__":
    main()
