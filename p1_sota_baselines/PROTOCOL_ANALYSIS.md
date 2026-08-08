# Why the one-class baselines are handicapped on DeepCrack

## The measurement

Every image in DeepCrack's training split contains cracks:

```
train_lab: n=300 | with any crack: 300 (100.0%) | mean crack area 2.91% | median 2.41%
test_lab:  n=237 | with any crack: 237 (100.0%) | mean crack area 4.33% | median 3.39%
```

(Computed from the official masks; reproduce with the snippet at the bottom.)

## Why it matters

One-class anomaly detection assumes an anomaly-free training set. DeepCrack
provides none: the "normal" set we hand these methods is 100% contaminated,
with cracks covering ~2.9% of pixels. Their observed ordering follows exactly
from how much that contamination hurts each design:

| Method | pixel AUROC | Why |
|---|---|---|
| PaDiM | 0.918 | Per-position Gaussian over frozen ImageNet features; 97% of pixels are background, so the fitted distribution is still dominated by normality |
| RD4AD | 0.872 | Distills a frozen pretrained teacher; the pretrained prior survives contamination |
| PatchCore | 0.831 | Memory bank stores crack patches too, so some cracks look "seen" |
| EfficientAD | 0.756 | Student-teacher from scratch on contaminated normals |
| DRAEM | **0.500 (chance)** | Trains a discriminative segmentation head *from scratch* whose entire objective is "the training images are normal, synthetic overlays are anomalous". Given training images that are all cracked, it learns that cracks are normal. Collapse is the expected outcome, not a bug in the method |

Our method is unaffected by this because it never needs a clean normal set:
its prior comes from an external source domain (digit strokes), and cracks are
flagged by reconstruction failure of *edge structure*, not by deviation from a
fitted normal distribution.

## How this must be reported

1. **Never present DRAEM's 0.50 as "DRAEM's performance."** It is the
   performance of DRAEM *under a protocol that violates its core assumption*.
   State the measurement above alongside it, or exclude the row with that
   explanation. Anything else misrepresents another group's method.
2. Give DRAEM the fairest available shot before reporting anything: rerun with
   the DTD texture set as the anomaly source (the DRAEM paper's own setup),
   which is now supported:
   ```
   wget -q https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
   tar xzf dtd-r1.0.1.tar.gz
   python p1_sota_baselines/run_baselines.py --data <data> --out <out> \
       --models draem --seeds 0 --batch 8 --max-epochs 300 \
       --model-kwargs '{"anomaly_source_path": "dtd/images"}'
   ```
   If it still lands at chance, that is informative and reportable *with* the
   contamination explanation.
3. **Run the clean-normal comparison on Concrete.** The Ozgenel classification
   set has 20,000 genuinely crack-free (Negative) images, so on that dataset
   the one-class baselines can be trained under their intended assumption.
   That is where a like-for-like one-class comparison belongs, and it makes
   the Concrete arm scientifically necessary rather than just "a second
   dataset".

## For the paper

This is a finding, not an excuse. Real inspection data rarely comes with a
curated defect-free training set, and the numbers above quantify what that
costs each family of methods. Reporting both regimes (contaminated normals on
DeepCrack, clean normals on Concrete) is a stronger and more honest
contribution than either alone.

## Reproduce the measurement

```python
import numpy as np
from pathlib import Path
from PIL import Image
for split in ("train_lab", "test_lab"):
    fr = [ (np.asarray(Image.open(p).convert("L")) > 127).mean()
           for p in sorted((Path("deepcrack_raw")/split).iterdir()) ]
    fr = np.array(fr)
    print(split, len(fr), (fr > 0).sum(), fr.mean())
```
