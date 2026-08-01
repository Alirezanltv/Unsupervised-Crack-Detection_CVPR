#!/usr/bin/env python3
"""Seed-level statistics: paired t-tests, Cohen's d, Holm-Bonferroni.

Input CSV: one row per seed, one column per method, values = the metric
(e.g. MIoU) for that seed. Example:

    seed,ours,fastflow,patchcore
    0,84.9,81.5,82.0
    1,84.1,81.0,81.7
    ...

Usage:
    python stats.py results/concrete_seeds.csv --ours ours

Reports, for every other column vs. --ours: mean +/- std, paired two-tailed
t-test, Cohen's d (paired), and Holm-Bonferroni adjusted p-values across all
comparisons in the file. Never report a comparison as significant unless it
survives the adjustment.

Self-test (no files needed):
    python stats.py --selftest
verifies the t/d formulas against scipy on random data.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps


def paired_report(a: np.ndarray, b: np.ndarray):
    d = a - b
    t, p = sps.ttest_rel(a, b)
    cohen = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.inf
    return {"mean_a": a.mean(), "std_a": a.std(ddof=1),
            "mean_b": b.mean(), "std_b": b.std(ddof=1),
            "delta": d.mean(), "t": float(t), "p": float(p),
            "cohens_d": float(cohen), "df": len(a) - 1}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def selftest():
    rng = np.random.default_rng(0)
    a, b = rng.normal(84, 1, 5), rng.normal(81, 1, 5)
    rep = paired_report(a, b)
    t, p = sps.ttest_rel(a, b)
    assert abs(rep["t"] - t) < 1e-12 and abs(rep["p"] - p) < 1e-12
    adj = holm(np.array([0.0024, 0.0242, 0.0941]))
    # Holm on the paper's three p-values: 0.0072, 0.0484, 0.0941
    assert np.allclose(adj, [0.0072, 0.0484, 0.0941]), adj
    print("selftest ok:", np.round(adj, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=None)
    ap.add_argument("--ours", default="ours")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.csv:
        ap.print_help()
        sys.exit(1)

    df = pd.read_csv(args.csv)
    ours = df[args.ours].to_numpy(dtype=float)
    others = [c for c in df.columns if c not in {args.ours, "seed"}]
    reports = {c: paired_report(ours, df[c].to_numpy(dtype=float)) for c in others}
    adj = holm(np.array([reports[c]["p"] for c in others]))
    for c, p_adj in zip(others, adj):
        r = reports[c]
        print(f"{args.ours} vs {c}: {r['mean_a']:.1f}+/-{r['std_a']:.1f} vs "
              f"{r['mean_b']:.1f}+/-{r['std_b']:.1f}  delta={r['delta']:+.1f}  "
              f"t({r['df']})={r['t']:.2f}  p={r['p']:.4f}  "
              f"p_holm={p_adj:.4f}  d={r['cohens_d']:.2f}")


if __name__ == "__main__":
    main()
