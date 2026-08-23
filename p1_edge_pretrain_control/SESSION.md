# Edge-source control session (Reviewer-4 control): Canny maps vs. MNIST

Purpose: answer "why MNIST and not a hand-engineered edge prior?" with data. Stage-1
pretrains on Canny edge maps of natural images (BSDS500) instead of digits; stages 2-3,
splits, scoring and sweeps stay byte-identical to the main runs. 3 seeds, DeepCrack arm.

Budget: ~6 h on T4x2 (GPU0: seeds 0-1 back-to-back, GPU1: seed 2). One session.
Everything below was smoke-tested end-to-end on CPU (generator -> folder source ->
3-stage training) before commit; the MNIST default path is regression-checked unchanged.

Kaggle notebook: T4 x2, Persistence "Variables and Files", run interactively as usual.

## Cell 1 — deps (light; no anomalib needed this session)

```python
!pip install -q scikit-learn scikit-image
import torch; print("cuda", torch.cuda.is_available(), "| gpus", torch.cuda.device_count())
```

## Cell 2 — repo + data + edge-map generation

```python
import os
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
!git fetch origin && git reset --hard origin/main
assert "--source-dir" in open("p0_reproduce/agdscae_ref.py").read(), "pull did not bring the edge-source support"

# DeepCrack (identical arrangement to every prior session)
!mkdir -p /kaggle/temp && wget -q https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip -O /kaggle/temp/dc.zip && unzip -q -o /kaggle/temp/dc.zip -d /kaggle/temp/deepcrack_raw
!python p0_reproduce/arrange_from_splits.py --raw-root /kaggle/temp/deepcrack_raw --splits p0_reproduce/splits --name deepcrack --dst /kaggle/temp/data/deepcrack --masks-dir /kaggle/temp/deepcrack_raw/test_lab

# BSDS500 -> 20,000 Canny maps (500 images x 40 seeded crops, matching MNIST's 20k)
!wget -q https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -O /kaggle/temp/bsr.tgz && tar xzf /kaggle/temp/bsr.tgz -C /kaggle/temp
!python p1_edge_pretrain_control/make_edge_maps.py --src /kaggle/temp/BSR/BSDS500/data/images --dst /kaggle/temp/canny_maps --detector canny --size 256 --crops 40 --seed 2027
!ls /kaggle/temp/canny_maps | wc -l
```

Expect the DeepCrack counts line, then `wrote 20000 edge maps`, then `20000`.
(If the Berkeley server is ever down, attach the Kaggle dataset
`balraj98/berkeley-segmentation-dataset-500-bsds500` via kagglehub instead and point
`--src` at its images directory.)

## Cell 3 — launcher

```python
import subprocess, os, time
DATA, RAW = "/kaggle/temp/data/deepcrack", "/kaggle/temp/deepcrack_raw"

def chain(out, train_cmd):
    return (f"{train_cmd} && "
        f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DATA}/test/images --out {out}/maps && "
        f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DATA}/calib --out {out}/calib && "
        f"python -u common/eval_maps.py --maps {out}/maps --masks {DATA}/test/masks --calib {out}/calib --out {out}/result.json && "
        f"python -u common/sweep_threshold.py --maps {out}/maps --masks {DATA}/test/masks --calib {out}/calib --out {out}/sweep.json")

def ejob(seed):
    out = f"/kaggle/working/edgectl/canny_s{seed}"
    return (f"canny_s{seed}", chain(out,
        f"python -u p0_reproduce/agdscae_ref.py train --raw-root {RAW} --splits p0_reproduce/splits "
        f"--name deepcrack --out {out} --seed {seed} --stage-epochs 50 --source-subset 20000 "
        f"--source-dir /kaggle/temp/canny_maps"))

JOBS = {0: [ejob(0), ejob(1)],
        1: [ejob(2)]}

def launch(gpu, name, cmd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return name, subprocess.Popen(cmd, shell=True, env=env,
        stdout=open(f"/kaggle/working/{name}.log", "w"), stderr=subprocess.STDOUT)

def tail(p):
    try:
        L = open(p, errors="replace").read().splitlines()
        return L[-1][-110:] if L else ""
    except FileNotFoundError:
        return "(no log)"

state = {g: {"q": list(j), "cur": None} for g, j in JOBS.items()}
while any(s["q"] or s["cur"] for s in state.values()):
    for g, s in state.items():
        if s["cur"] is None and s["q"]:
            s["cur"] = launch(g, *s["q"].pop(0))
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} START {s['cur'][0]}", flush=True)
        elif s["cur"] and s["cur"][1].poll() is not None:
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {s['cur'][0]} EXIT {s['cur'][1].returncode}", flush=True)
            s["cur"] = None
    time.sleep(60)
    for g, s in state.items():
        if s["cur"]:
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {s['cur'][0]:12s} | {tail('/kaggle/working/' + s['cur'][0] + '.log')}", flush=True)
print("SESSION DONE")
```

Stage-1 losses will NOT match the MNIST runs' familiar 0.006x (different source content —
Canny maps are sparser than digit strokes, so expect a different loss scale). That is
correct behavior, not a bug. Stages 2-3 operate on the same crack data as always.

## Cell 4 — bundle

```python
!cd /kaggle/working && find . -path "./edgectl/*" \( -name "result.json" -o -name "sweep.json" \) | zip -q edgectl_results.zip -@
!cd /kaggle/working && zip -qr edgectl_ckpts.zip edgectl -i "*stage3.pt" 2>/dev/null; ls -la /kaggle/working/edgectl_*.zip
```

Download `edgectl_results.zip` -> git directory, as always.

## What happens with the numbers (pre-committed, either way)

Compare canny (3 seeds) against the full MNIST model (matched seeds 0-2) on all four
metrics, Welch tests. Both outcomes are reportable:
- MNIST >= Canny: the digit prior is at least as good as a hand-engineered edge prior —
  a striking, counterintuitive support for the paper's premise. Goes into Sec. 4 as a
  new control row + 2-3 sentences; limitation 5's "untested" clause updates.
- Canny > MNIST: the honest finding becomes "a cross-domain edge prior is what matters,
  and engineered edge maps are a stronger choice than digits"; the framing shifts and the
  paper says so. Also publishable; arguably more useful.

Optional extension if quota allows later: HED maps (needs the caffemodel; check that
http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel still exists before planning on it).
