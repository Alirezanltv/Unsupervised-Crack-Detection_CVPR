# CrackForest (CFD) cross-dataset evaluation

DeepCrack-trained AG-DSCAE seed 0, test-only on all 118 CFD images.
Calibration (mu/delta) from DeepCrack calib split - training-domain only, no
target-domain information used. Masks converted from the CFD .mat ground
truth (label==2). Run locally on CPU, 2026-08-03.

  in-domain (DeepCrack): AUROC 0.931 / AP 0.488 / ODS-F1 0.500
  cross    (CFD):        AUROC 0.851 / AP 0.111 / ODS-F1 0.213

Reading: ranking ability transfers moderately (8-point AUROC drop with zero
adaptation); fine localization does not (AP collapses on CFD's hairline
cracks - consistent with the thin-crack failure mode). Honest framing for
the paper: cross-dataset generalization is partial and metric-dependent;
report both numbers, claim only what they support.
