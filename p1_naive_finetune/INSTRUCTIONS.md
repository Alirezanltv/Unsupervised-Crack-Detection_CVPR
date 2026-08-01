# Naive fine-tuning control (fills the commented row of paper Table 2)

Reviewer 3's demand: compare the three-stage protocol against the simplest
alternative at the SAME compute budget. This needs your original AG-DSCAE
code; the recipe below is a config change, not new code.

## Recipe (5 seeds: 0-4)

1. **Stage 1, unchanged**: pretrain encoder+decoder on MNIST exactly as in
   the paper — 50 epochs, AdamW lr 1e-3, L = MSE + 0.5·edge loss, cosine
   annealing, batch 32.
2. **Then fine-tune directly on the target dataset for 100 epochs** with
   ONLY the reconstruction objective the model already uses:
   - loss: MSE + 0.3·edge loss  (NO L_OT, NO attention modules, NO staging)
   - nothing frozen; the whole encoder+decoder trains
   - lr 1e-4 for epochs 51–100, then 5e-5 for 101–150 (mirrors the paper's
     schedule so the only difference is the removed machinery)
   - batch 16, same augmentation, early stopping patience 10
3. **Inference identical to the paper**: gradient-reconstruction error map,
   standardize with (mu, delta) from held-out unlabeled images, fixed 0.5.
4. Dump raw anomaly maps as `.npy` (same stems as images) into
   `runs/naive_ft/s{seed}/maps` plus calibration maps into `.../calib`,
   then score:

   ```
   python ../common/eval_maps.py --maps runs/naive_ft/s0/maps \
       --masks <dataset>/test/masks --calib runs/naive_ft/s0/calib
   ```

5. Aggregate the 5 seeds with `../common/stats.py` and put the mean±std into
   the commented row of Table 2 in `main.tex`. Whatever the number is, it
   goes in — if naive fine-tuning matches the full model, that is a result
   the paper must state, not hide.

Total budget: 150 epochs — identical to the full pipeline.
