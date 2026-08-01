# Rebuild plan — regenerating the paper's evidence from code

Status: the original trained checkpoints, split lists, and result artifacts
from the earlier submission are lost. Every model-derived number in the
current draft is therefore unsupported, and the paper must not be submitted
until this plan replaces them with freshly measured ones. The text, method,
and hypothesis survive unchanged; the numbers get regenerated and swapped in
wholesale — whatever they turn out to be.

Non-negotiables:
- Splits are generated once by `make_splits.py` (fixed seed) and the
  resulting `splits/*.txt` files are committed. No experiment runs on an
  uncommitted split.
- Every training run ends with `dump_maps.py` -> `common/eval_maps.py`, so
  each run immediately yields MIoU/F1 (at the real protocol) + AUROC/AP.
- Checkpoint after every stage (Colab sessions die); save checkpoints and
  maps to Drive, never to /content only.
- Log per run: seed, split files, git hash, command line.

## Dataset redesign (verifiable provenance only)

| Dataset | Train (unlabeled) | Pixel test | Source |
|---|---|---|---|
| Concrete | Özgenel classification set (Kaggle mirror) | Özgenel segmentation set (Mendeley, browser download) | both public |
| DeepCrack | official train_img | official test_img + test_lab | repo zip |
| CrackForest | none (test-only) or half-split | CFD images + converted masks | GitHub |

The Bridge Crack dataset is **dropped**: the old 1,842-image subset cannot be
reconstructed and SDNET2018 carries no pixel masks. A smaller paper about
datasets we can defend beats a bigger one about datasets we cannot.

## Tier 1 — minimum submittable core (start immediately)

1. AG-DSCAE full pipeline (150 epochs, 3 stages) on Concrete, DeepCrack:
   3 seeds each = 6 trainings. CrackForest is evaluated test-only with the
   Concrete-trained model (cross-dataset, no extra training).
   T4 estimate: 16-24 h per training with checkpoint/resume; budget
   ~2 weeks part-time, or days with Colab Pro / a rented GPU.
2. Baselines under the identical protocol: PatchCore + PaDiM 5 seeds
   (minutes each; DeepCrack seed 0 already measured), DRAEM / RD4AD /
   EfficientAD at >= 1 seed each (hours each).
3. Protocol determination, decided by measurement and then written into the
   paper: with fresh AG-DSCAE maps, evaluate fixed-0.5, percentile, and
   validation-optimal thresholds. Whatever protocol the paper reports, it is
   the one the code actually runs, stated plainly, identical for all methods
   (plus threshold-free AUROC/AP everywhere).

Deliverable: new Table 1 (2 datasets + cross-dataset CFD column, mean +/- std
over 3 seeds, AUROC/AP columns), new efficiency numbers from the same runs.

## Tier 2 — reduced ablations (after Tier 1 works end to end)

4. Component ablation on Concrete, 3 seeds: (a) no transfer (target-only),
   (b) + MNIST pretraining, (c) + OT alignment, (d) full (+attention).
   Stages are shareable across configs (one Stage-1 pretrain reused
   everywhere), so this is ~8 additional trainings, most shorter than full.
5. Source ablation, reduced: MNIST vs BSD500-edges vs none (BSD500 already
   downloaded; edge maps via p1_edge_pretrain_control). Reuses (a)/(b) from
   the component ablation; +2 trainings.

Deliverable: new (smaller) ablation table replacing old Tables 4-7.

## Tier 3 — if time allows before the deadline

6. Naive fine-tuning control (p1_naive_finetune recipe): +1 training.
7. Attention-gate visualization (p3_attention_viz): free once a Tier-1
   checkpoint exists.
8. More seeds, more baselines, upsampling-response retrain.

## Honest accounting rules

- If rebuilt AG-DSCAE lands below the old numbers, the paper reports the
  new numbers. If a baseline wins, the table says so and the framing adapts.
- The statistics section is recomputed from the new seeds via
  `common/stats.py` (Holm-corrected); claims follow the numbers.
- Timeline check: CVPR 2027 (projected ~Nov 13) is feasible only if Tier 1
  starts this week and scope stays cut. WACV 2027 is the fallback with the
  same format and less time pressure.
