#!/usr/bin/env python3
"""Variance-homogeneity tests behind the ablation's seed-spread remark (main paper Sec. 4.6).

The paper observes that seed spread changes across the four ablation variants but states
that three seeds per variant cannot establish it. This script records the tests that
statement rests on, in both common conventions, so the quoted p-value has a provenance:

  * Levene (mean-centred)            -- scipy.stats.levene(center="mean")
  * Brown-Forsythe (median-centred)  -- scipy.stats.levene(center="median"), scipy default

for the seed-matched (0-2) ODS-F1 and AUROC values of target-only, +pretraining, +OT, and
the full model, on the pairs the text discusses and across all four variants.

Run from the repo root:  python results/compute_levene.py
Reads:  results/deepcrack_abl_{scratch,pretrain,ot}_s{0,1,2}_{result,sweep}.json and the
        banked full-model seeds in compute_ablation.FULL
Writes: results/deepcrack_ablation_levene.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

R = Path(__file__).parent
sys.path.insert(0, str(R))
from compute_ablation import FULL, METRICS, load_variant  # noqa: E402

VARIANTS = {"target_only": "scratch", "pretraining": "pretrain", "ot": "ot"}
PAIRS = [("target_only", "full"), ("target_only", "pretraining"), ("ot", "full")]


def main():
    data = {}
    for key, name in VARIANTS.items():
        arr, seeds = load_variant(name)
        assert seeds == [0, 1, 2], (name, seeds)
        data[key] = arr
    data["full"] = np.array([FULL[s] for s in (0, 1, 2)])
    out = {"seeds": [0, 1, 2], "note": __doc__.strip().splitlines()[0], "tests": {}}
    for i, m in enumerate(METRICS):
        block = {}
        for a, b in PAIRS:
            for center in ("mean", "median"):
                w, p = stats.levene(data[a][:, i], data[b][:, i], center=center)
                block[f"{a}_vs_{b}/{center}"] = {"W": round(float(w), 3), "p": round(float(p), 4),
                                                 "std_a": round(float(data[a][:, i].std(ddof=1)), 4),
                                                 "std_b": round(float(data[b][:, i].std(ddof=1)), 4)}
        for center in ("mean", "median"):
            w, p = stats.levene(*[data[k][:, i] for k in ("target_only", "pretraining", "ot", "full")], center=center)
            block[f"all_four/{center}"] = {"W": round(float(w), 3), "p": round(float(p), 4)}
        out["tests"][m] = block
    (R / "deepcrack_ablation_levene.json").write_text(json.dumps(out, indent=2))
    for m in ("ods_f1", "auroc"):
        print(m)
        for k, v in out["tests"][m].items():
            print(f"  {k:32s} W={v['W']:6.3f}  p={v['p']:.4f}")


if __name__ == "__main__":
    main()
