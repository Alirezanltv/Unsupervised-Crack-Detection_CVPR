#!/usr/bin/env python3
"""Source-domain control: stage 1 pretrained on Canny edge maps of BSDS500 crops
instead of MNIST, everything downstream identical. Compares against the full
MNIST model on matched seeds 0-2 (Welch, unequal variance) and against the
target-only variant. Run from the repo root; writes deepcrack_source_control.json."""
import json
from pathlib import Path
import numpy as np
from scipy import stats

R = Path(__file__).parent
OURS = {0: (0.9307, 0.4883, 0.5005, 0.6432), 1: (0.8925, 0.2640, 0.4053, 0.5890),
        2: (0.9099, 0.3683, 0.4528, 0.6158)}
M = ("auroc", "ap", "ods_f1", "ods_miou")


def load(prefix):
    rows = []
    for s in range(3):
        r = json.loads((R / f"{prefix}_s{s}_result.json").read_text())
        w = json.loads((R / f"{prefix}_s{s}_sweep.json").read_text()); w = w.get("summary", w)
        rows.append((r["pixel_auroc"], r["pixel_ap"], w["ods_f1"]["f1"], w["best_miou"]["miou"]))
    return np.array(rows)


def summ(a):
    return {m: {"mean": round(float(a[:, i].mean()), 4), "std": round(float(a[:, i].std(ddof=1)), 4),
                "n_seeds": len(a)} for i, m in enumerate(M)}


def welch(a, b):
    out = {}
    for i, m in enumerate(M):
        t, p = stats.ttest_ind(a[:, i], b[:, i], equal_var=False)
        out[m] = {"delta": round(float(a[:, i].mean() - b[:, i].mean()), 4), "t": round(float(t), 3),
                  "p": round(float(p), 4), "seed_matched_diffs": [round(float(x), 4) for x in a[:, i] - b[:, i]]}
    return out


canny = load("deepcrack_canny"); scratch = load("deepcrack_abl_scratch")
full3 = np.array([OURS[s] for s in range(3)])
out = {"protocol": ("stage 1 on 20,000 Canny (50/150) maps of 256^2 seeded crops of BSDS500 images "
                    "(500 images x 40 crops, seed 2027) in place of the 20,000-digit MNIST subset; "
                    "no upsampling (maps born at 256^2); stages 2-3, splits, scoring, sweeps identical; seeds 0-2"),
       "source_statistics": {"canny_maps": "binary by construction: 2 gray levels, 1-px edge transitions",
                             "mnist_upsampled_stage1_input": {"mean_intensity": 0.1337, "frac_pixels_gt_0.5": 0.1331,
                                                              "frac_pixels_gt_0.05": 0.2035, "distinct_gray_levels_per_image": 101,
                                                              "edge_transition_px": 10.8, "n_digits_measured": 300}},
       "models": {"canny_source": summ(canny), "mnist_full_seeds_0_2": summ(full3), "target_only": summ(scratch)},
       "welch": {"canny_vs_mnist_full": welch(canny, full3), "canny_vs_target_only": welch(canny, scratch)}}
(R / "deepcrack_source_control.json").write_text(json.dumps(out, indent=2))
for k, v in out["models"].items():
    print(k.ljust(22), {m: v[m]["mean"] for m in M})
print("canny vs mnist p:", {m: out["welch"]["canny_vs_mnist_full"][m]["p"] for m in M})
