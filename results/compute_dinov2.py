#!/usr/bin/env python3
"""Frozen DINOv2 k-NN baseline vs. ours, both datasets. The DINOv2 runs are
deterministic (single run each); ours has 5 seeds (DeepCrack) / 3 (concrete), so the
comparison is a one-sample test of our seeds against the DINOv2 value. Missing files
are reported as pending. Writes dinov2_table.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).parent
M = ("auroc", "ap", "ods_f1", "ods_miou")
OURS_DC = np.array([(0.9307, 0.4883, 0.5005, 0.6432), (0.8925, 0.2640, 0.4053, 0.5890),
                    (0.9099, 0.3683, 0.4528, 0.6158), (0.8889, 0.2878, 0.4380, 0.6063),
                    (0.9029, 0.3075, 0.4379, 0.6047)])


def one(prefix):
    r, w = R / f"{prefix}_result.json", R / f"{prefix}_sweep.json"
    if not (r.exists() and w.exists()):
        return None
    r = json.loads(r.read_text()); w = json.loads(w.read_text()); w = w.get("summary", w)
    return np.array((r["pixel_auroc"], r["pixel_ap"], w["ods_f1"]["f1"], w["best_miou"]["miou"]))


def ours_concrete():
    rows = []
    for s in range(3):
        r = json.loads((R / f"concrete_ours_s{s}_result.json").read_text())
        w = json.loads((R / f"concrete_ours_s{s}_sweep.json").read_text()); w = w.get("summary", w)
        rows.append((r["pixel_auroc"], r["pixel_ap"], w["ods_f1"]["f1"], w["best_miou"]["miou"]))
    return np.array(rows)


out = {}
for ds, ours in (("deepcrack", OURS_DC), ("concrete", ours_concrete())):
    out[ds] = {"ours": {m: {"mean": round(float(ours[:, i].mean()), 4), "std": round(float(ours[:, i].std(ddof=1)), 4)} for i, m in enumerate(M)}}
    for bb in ("vits14", "vitb14"):
        v = one(f"{ds}_dinov2_{bb}_s0")
        if v is None:
            out[ds][f"dinov2_{bb}"] = "PENDING"; continue
        rec = {m: round(float(v[i]), 4) for i, m in enumerate(M)}
        rec["one_sample_t_vs_ours"] = {m: {"ours_minus_dinov2": round(float(ours[:, i].mean() - v[i]), 4),
                                          "p": round(float(stats.ttest_1samp(ours[:, i], v[i]).pvalue), 4)} for i, m in enumerate(M)}
        out[ds][f"dinov2_{bb}"] = rec
(R / "dinov2_table.json").write_text(json.dumps(out, indent=2))
for ds in out:
    for k, v in out[ds].items():
        print(ds, k.ljust(14), v if isinstance(v, str) else {m: (v[m]["mean"] if isinstance(v[m], dict) else v[m]) for m in M})
