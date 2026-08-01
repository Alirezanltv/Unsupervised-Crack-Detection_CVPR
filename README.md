# Unsupervised Crack Detection — Experiments

Evaluation and baseline tooling for our work on cross-domain edge transfer
for unsupervised crack detection (AG-DSCAE: MNIST-pretrained autoencoder,
optimal-transport alignment, cross-domain attention gates).

This repo holds the *experiment harness*: baseline runners, a shared metric
engine, and analysis scripts. The model training code lives separately.

## Layout

```
common/                  shared tooling used by every experiment
  eval_maps.py           metric engine: MIoU/F1 at fixed 0.5 + pixel AUROC/AP
  stats.py               paired t-tests, Cohen's d, Holm correction
  smoke_test.py          synthetic-data test for eval_maps
p1_sota_baselines/       anomalib runner: PatchCore, PaDiM, DRAEM, RD4AD,
                         EfficientAD, SimpleNet
p1_edge_pretrain_control/  Canny/HED edge-map source-domain generators
p1_naive_finetune/       recipe for the matched-compute fine-tuning control
p2_benchmarks/           CrackForest / DeepCrack test-only evaluation
p2_sweeps/               hyperparameter sensitivity sweep orchestrator
p3_upsampling_stats/     MNIST upsampling edge-statistics analysis (+ results)
p3_attention_viz/        attention-gate visualization
colab_run.ipynb          run everything on Google Colab
```

## Status

- **The project is in a rebuild phase: see `p0_reproduce/RETRAIN_PLAN.md`
  first.** Original checkpoints and splits were lost; all model-derived
  results are being regenerated with committed deterministic splits.
- Measured so far: `p3_upsampling_stats/` (MNIST analysis, results
  committed) and `results/deepcrack_seed0.json` (PatchCore + PaDiM on the
  DeepCrack test split, run on Colab).
- Every method's anomaly maps are scored by the same engine
  (`common/eval_maps.py`), so numbers are directly comparable.

## Setup

```
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

Quick sanity check (no data needed):

```
python common/stats.py --selftest
python common/smoke_test.py
```

## Dataset layout

Each dataset directory should look like:

```
<dataset>/
  train/good/     unlabeled training images
  test/images/    test images
  test/masks/     binary ground-truth masks (same file stems)
  calib/          held-out unlabeled images (for map standardization)
```

## Run order

1. `p1_sota_baselines/run_baselines.py` — modern one-class baselines,
   5 seeds; read the protocol note in the script header first.
2. `p1_edge_pretrain_control/` + `p1_naive_finetune/` — the two controls.
3. `p2_benchmarks/` — CrackForest and DeepCrack, test-only.
4. `p2_sweeps/` — sensitivity around the default hyperparameters.
5. `p3_attention_viz/` — gate heatmaps and edge-vs-background gate statistics
   (adapt `load_model_and_gates()` to the trainer, then point it at a
   checkpoint).

Score any run with:

```
python common/eval_maps.py --maps <run>/maps --masks <dataset>/test/masks --calib <run>/calib
```

and aggregate seeds with `common/stats.py`.

## Ground rules

- Only measured numbers go into the paper. No projections, no placeholders.
- If a baseline wins, it goes in the table anyway.
- Keep each run's config, seed, and commit hash.
