#!/usr/bin/env python3
"""Source-domain study: what does the stage-1 source contribute?
Rows: no source (target-only) / Canny maps (binary, sharp) / procedural strokes
(appearance-matched to MNIST, no digit content) / MNIST digits. All on DeepCrack,
matched seeds 0-2, Welch tests against the MNIST model. Missing rows are reported
as pending rather than failing. Writes deepcrack_source_study.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).parent
M = ("auroc", "ap", "ods_f1", "ods_miou")
OURS = {0: (0.9307, 0.4883, 0.5005, 0.6432), 1: (0.8925, 0.2640, 0.4053, 0.5890),
        2: (0.9099, 0.3683, 0.4528, 0.6158)}


def load(prefix):
    rows = []
    for s in range(3):
        r, w = R / f"{prefix}_s{s}_result.json", R / f"{prefix}_s{s}_sweep.json"
        if not (r.exists() and w.exists()):
            return None
        r = json.loads(r.read_text()); w = json.loads(w.read_text()); w = w.get("summary", w)
        rows.append((r["pixel_auroc"], r["pixel_ap"], w["ods_f1"]["f1"], w["best_miou"]["miou"]))
    return np.array(rows)


def summ(a):
    return {m: {"mean": round(float(a[:, i].mean()), 4), "std": round(float(a[:, i].std(ddof=1)), 4)} for i, m in enumerate(M)}


def welch(a, b):
    return {m: {"delta": round(float(a[:, i].mean() - b[:, i].mean()), 4),
                "p": round(float(stats.ttest_ind(a[:, i], b[:, i], equal_var=False).pvalue), 4),
                "seed_matched": [round(float(x), 4) for x in a[:, i] - b[:, i]]} for i, m in enumerate(M)}


mnist = np.array([OURS[s] for s in range(3)])
srcs = {"target_only": load("deepcrack_abl_scratch"), "canny_maps": load("deepcrack_canny"),
        "strokes": load("deepcrack_strokes"), "mnist": mnist}
out = {"protocol": "DeepCrack, full pipeline, seeds 0-2; only the stage-1 source differs",
       "sources": {k: (summ(v) if v is not None else "PENDING") for k, v in srcs.items()},
       "welch_vs_mnist": {k: welch(v, mnist) for k, v in srcs.items() if v is not None and k != "mnist"}}
if srcs["strokes"] is not None:
    out["welch_strokes_vs_target_only"] = welch(srcs["strokes"], srcs["target_only"])
(R / "deepcrack_source_study.json").write_text(json.dumps(out, indent=2))
for k, v in out["sources"].items():
    print(k.ljust(12), v if isinstance(v, str) else {m: v[m]["mean"] for m in M})
