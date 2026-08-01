# CrackForest + DeepCrack evaluation (test-only, Concrete-trained model)

Standard public benchmarks with pixel masks; evaluation is test-only with
your Concrete-trained model, so no training is needed — just inference.

## Get the data
- **CrackForest (CFD)**: https://github.com/cuilimeng/CrackForest-dataset
  (118 road images with pixel annotations; masks are in `.mat`/`.png`
  depending on mirror — `prepare_masks.py` normalizes them to 0/255 PNG).
- **DeepCrack**: https://github.com/yhlleo/DeepCrack (537 images with masks,
  official train/test split — use the TEST split only).

## Run
1. Run your model's inference on each benchmark's images; dump raw anomaly
   maps as `.npy` (same stems) into `runs/<bench>/maps`, and maps for ~100
   held-out unlabeled Concrete images into `runs/<bench>/calib` (the mu/delta
   calibration must come from the TRAINING domain — do not recalibrate on the
   benchmark, that would leak).
2. `python prepare_masks.py --src <raw mask dir> --dst <bench>/masks`
3. `python ../common/eval_maps.py --maps runs/<bench>/maps --masks <bench>/masks --calib runs/<bench>/calib`

Report MIoU/F1 (fixed 0.5) **and** pixel AUROC/AP (threshold-free) for both
benchmarks. AUROC/AP make the results commensurable with the anomaly-
detection literature and are immune to the binarization critique entirely.
Run the strongest baselines (../p1_sota_baselines) on the same benchmarks if
time allows — a table with only our method on a new benchmark convinces no one.
