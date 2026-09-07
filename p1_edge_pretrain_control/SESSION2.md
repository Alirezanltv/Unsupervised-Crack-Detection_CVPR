# Final compute session: stroke-source control + DINOv2 baseline + T4 latency

Three review objections answered in one T4x2 session (~6.5 h):
1. "A natural-edge prior should do the digits' job" / "the Canny result is just a
   skip-connection shortcut" -> procedural smooth strokes, appearance-matched to MNIST
   (same transition width, gradient magnitude, support, stroke width; no digit semantics;
   cannot be copied through the skips). 3 seeds, DeepCrack, full pipeline.
2. "Where is the foundation-model baseline?" -> frozen DINOv2 ViT-S/14 and ViT-B/14 patch
   features + nearest-neighbour scoring (the PatchCore recipe on DINOv2), 448 input =
   32x32 patch grid, on DeepCrack AND the concrete clean-normal arm. Deterministic.
3. "No latency numbers" -> forward latency of every baseline on the T4 (1-epoch fits;
   latency does not depend on weights), of AG-DSCAE's full scoring path, and of DINOv2.

Notebook: T4 x2, Persistence "Variables and Files", interactive. The two concrete
datasets must still be attached (or kagglehub mounts them). No pip installs beyond what
Cell 1 does for anomalib (needed only for the latency job).

Fail-fast: Cell 1 asserts the repo commit; Cell 3 smoke-tests the full stroke chain on
GPU (~5 min) before launching anything long. Logs open in APPEND mode (resume-safe).
If disconnected: re-run Cells 1-3; training resumes from checkpoints, finished jobs skip.

---------------------------------------------------------------------------------------
## Cell 1 -- preflight + anomalib (latency job only)

```python
import os, shutil, subprocess, importlib, sys
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
g = subprocess.run("git fetch origin && git reset --hard origin/main", shell=True, capture_output=True, text=True)
assert g.returncode == 0, "GIT SYNC FAILED:\n" + g.stderr[-600:]
for f, tok in (("p1_edge_pretrain_control/make_stroke_source.py", "def render"),
               ("p1_sota_baselines/dinov2_knn.py", "x_norm_patchtokens"),
               ("p1_sota_baselines/run_baselines.py", "--time-only")):
    assert tok in open(f).read(), f"{f} is stale: push the latest commits, then re-run"
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, f"need 2 GPUs, found {torch.cuda.device_count()}"
pin = subprocess.run([sys.executable, "-c", "import torch;print(f'torch=={torch.__version__.split(\"+\")[0]}')"], capture_output=True, text=True).stdout.strip()
open("/kaggle/working/constraints.txt", "w").write(pin + "\n")
subprocess.run("pip install -q -c /kaggle/working/constraints.txt anomalib==1.1.1 lightning imgaug FrEIA einops timm kornia open_clip_torch scikit-image scikit-learn opencv-python-headless 'matplotlib<3.10' 'pandas<3' 'rich==13.7.1'", shell=True)
from pathlib import Path
Path("/kaggle/working/pyshim").mkdir(exist_ok=True)
Path("/kaggle/working/pyshim/sitecustomize.py").write_text('import numpy as np\nif not hasattr(np, "sctypes"):\n    np.sctypes = {"float":[np.float16,np.float32,np.float64],"int":[np.int8,np.int16,np.int32,np.int64],"uint":[np.uint8,np.uint16,np.uint32,np.uint64],"complex":[np.complex64,np.complex128],"others":[bool,object,bytes,str,np.void]}\nfor _n,_t in (("bool",bool),("float",float),("int",int),("object",object),("str",str)):\n    if not hasattr(np,_n): setattr(np,_n,_t)\n')
for m in ("sklearn", "scipy", "cv2", "PIL", "anomalib"):
    importlib.import_module(m)
free_gb = shutil.disk_usage("/kaggle/working").free / 1e9
assert free_gb > 5, f"only {free_gb:.1f} GB free"
print(f"PREFLIGHT OK | commit {subprocess.run('git rev-parse --short HEAD', shell=True, capture_output=True, text=True).stdout.strip()} | {torch.cuda.device_count()} GPUs | {free_gb:.0f} GB free")
```

## Cell 2 -- data: DeepCrack, concrete, 20,000 stroke images (~8 min)

```python
import os, subprocess, glob, kagglehub
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {cmd}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout
sh("mkdir -p /kaggle/temp")
if not os.path.isdir("/kaggle/temp/data/deepcrack/test/masks"):
    sh("wget -q https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip -O /kaggle/temp/dc.zip && unzip -q -o /kaggle/temp/dc.zip -d /kaggle/temp/deepcrack_raw")
    print(sh("python p0_reproduce/arrange_from_splits.py --raw-root /kaggle/temp/deepcrack_raw --splits p0_reproduce/splits --name deepcrack --dst /kaggle/temp/data/deepcrack --masks-dir /kaggle/temp/deepcrack_raw/test_lab").strip())
dc = {s: len(os.listdir(f"/kaggle/temp/data/deepcrack/{s}")) for s in ("train/good", "calib", "test/images", "test/masks")}
assert dc == {"train/good": 250, "calib": 50, "test/images": 237, "test/masks": 237}, dc
if not os.path.isdir("/kaggle/temp/data/concrete/test/masks"):
    cls_path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
    seg_path = kagglehub.dataset_download("motono0223/concrete-crack-segmentation-dataset")
    print(sh(f'python p0_reproduce/concrete_prep.py --cls-root "{cls_path}" --seg-root "{seg_path}" --dst /kaggle/temp/data/concrete --splits-out /kaggle/temp/splits_check').strip()[-300:])
cc = {s: len(os.listdir(f"/kaggle/temp/data/concrete/{s}")) for s in ("train/good", "calib", "test/images", "test/masks")}
assert cc == {"train/good": 250, "calib": 50, "test/images": 446, "test/masks": 446}, cc
if len(glob.glob("/kaggle/temp/stroke_maps/*.png")) != 20000:
    print(sh("python p1_edge_pretrain_control/make_stroke_source.py --dst /kaggle/temp/stroke_maps --n 20000 --seed 2027 --stats").strip())
n = len(glob.glob("/kaggle/temp/stroke_maps/*.png")); assert n == 20000, n
print("DATA OK |", dc, "|", cc, "| strokes:", n)
```

Expect the stroke statistics block (mean ~0.128, support ~0.20, gray levels ~101).

## Cell 3 -- smoke test, then all jobs; auto-zip; printed table

```python
import subprocess, os, time, json, glob, shutil
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
DC, RAW, CON, MAPS = "/kaggle/temp/data/deepcrack", "/kaggle/temp/deepcrack_raw", "/kaggle/temp/data/concrete", "/kaggle/temp/stroke_maps"
OUT, RUNS = "/kaggle/working/final", "/kaggle/working/runs_final"
PY = "PYTHONPATH=/kaggle/working/pyshim "

def score(maps, calib, data, o):
    return (f"mkdir -p {o} && python -u common/eval_maps.py --maps {maps} --masks {data}/test/masks --calib {calib} --out {o}/result.json && "
            f"python -u common/sweep_threshold.py --maps {maps} --masks {data}/test/masks --calib {calib} --out {o}/sweep.json")

def stroke_chain(out, epochs, subset):
    return (f"python -u p0_reproduce/agdscae_ref.py train --raw-root {RAW} --splits p0_reproduce/splits --name deepcrack "
            f"--out {out} --seed {out[-1]} --stage-epochs {epochs} --source-subset {subset} --source-dir {MAPS} && "
            f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DC}/test/images --out {out}/maps && "
            f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DC}/calib --out {out}/calib && "
            + score(f"{out}/maps", f"{out}/calib", DC, out))

def dino_job():
    cmds = []
    for data, tag in ((DC, "deepcrack"), (CON, "concrete")):
        for bb in ("vits14", "vitb14"):
            r = f"{RUNS}/{tag}/dinov2_{bb}/s0"
            cmds.append(f"python -u p1_sota_baselines/dinov2_knn.py --data {data} --out {RUNS}/{tag} --backbone {bb} --size 448 && "
                        + score(f"{r}/maps", f"{r}/calib", data, f"{OUT}/{tag}_dinov2_{bb}") + f" && cp {r}/latency.json {OUT}/{tag}_dinov2_{bb}/")
    return " && ".join(cmds)

def latency_job():
    return (PY + f"python -u p1_sota_baselines/run_baselines.py --data {DC} --out {RUNS}/latency --models padim patchcore --seeds 0 --time-only && "
            + PY + f"python -u p1_sota_baselines/run_baselines.py --data {DC} --out {RUNS}/latency --models rd4ad efficientad draem --seeds 0 --max-epochs 1 --batch 8 --time-only && "
            f"python -u p2_benchmarks/measure_cost.py --ckpt {OUT}/strokes_s2/ckpt_stage3.pt --out {OUT}/latency_ours_t4.json --iters 50 && "
            f"mkdir -p {OUT}/latency_baselines && cp {RUNS}/latency/*/s0/latency.json {OUT}/latency_baselines/ 2>/dev/null; "
            f"for m in padim patchcore rd4ad efficientad draem; do cp {RUNS}/latency/$m/s0/latency.json {OUT}/latency_baselines/$m.json; done")

def env_for(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

# ---- smoke test of the stroke chain on GPU0 (~5 min)
smoke = f"{OUT}/_smoke_s0"; shutil.rmtree(smoke, ignore_errors=True)
r = subprocess.run(stroke_chain(smoke, 1, 64), shell=True, env=env_for(0), capture_output=True, text=True)
if r.returncode != 0 or not os.path.exists(f"{smoke}/sweep.json"):
    print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit("SMOKE TEST FAILED -- nothing launched.")
print("SMOKE TEST OK. Launching.", flush=True)

JOBS = {0: [("strokes_s0", stroke_chain(f"{OUT}/strokes_s0", 50, 20000)), ("strokes_s1", stroke_chain(f"{OUT}/strokes_s1", 50, 20000))],
        1: [("strokes_s2", stroke_chain(f"{OUT}/strokes_s2", 50, 20000)), ("dinov2", dino_job()), ("latency", latency_job())]}

def zip_results():
    subprocess.run(f"cd /kaggle/working && rm -f final_results.zip && find final \\( -name result.json -o -name sweep.json -o -name '*.json' \\) -not -path '*_smoke*' | zip -q final_results.zip -@", shell=True)

def launch(gpu, name, cmd):
    return name, subprocess.Popen(cmd, shell=True, env=env_for(gpu), stdout=open(f"/kaggle/working/{name}.log", "a"), stderr=subprocess.STDOUT)

def tail(name, n=1):
    try:
        L = open(f"/kaggle/working/{name}.log", errors="replace").read().splitlines(); return "\n".join(L[-n:]) if L else ""
    except FileNotFoundError:
        return "(no log)"

state = {g: {"q": list(j), "cur": None} for g, j in JOBS.items()}
failed, t0, last = [], time.time(), 0
while any(s["q"] or s["cur"] for s in state.values()):
    for g, s in state.items():
        if s["cur"] is None and s["q"]:
            s["cur"] = launch(g, *s["q"].pop(0)); print(f"[{time.strftime('%H:%M:%S')}] GPU{g} START {s['cur'][0]}", flush=True)
        elif s["cur"] and s["cur"][1].poll() is not None:
            name, rc = s["cur"][0], s["cur"][1].returncode
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {name} EXIT {rc}", flush=True)
            if rc != 0:
                failed.append(name); print("---- last 25 log lines of", name); print(tail(name, 25)); print("----", flush=True)
            zip_results(); s["cur"] = None
    time.sleep(60)
    if time.time() - last > 300:
        last = time.time()
        for g, s in state.items():
            if s["cur"]: print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {s['cur'][0]:10s} | {tail(s['cur'][0])[-100:]}", flush=True)
zip_results()
print(f"\nSESSION DONE in {(time.time()-t0)/3600:.1f} h | failed: {failed or 'none'}")

print("\n==== RESULTS -- paste this block ====")
for d in sorted(glob.glob(f"{OUT}/*")):
    if os.path.isdir(d) and os.path.exists(f"{d}/result.json"):
        res = json.load(open(f"{d}/result.json")); sw = json.load(open(f"{d}/sweep.json")); sw = sw.get("summary", sw)
        print(f"{os.path.basename(d):22s} AUROC {res['pixel_auroc']:.4f}  AP {res['pixel_ap']:.4f}  ODS-F1 {sw['ods_f1']['f1']:.4f}  MIoU {sw['best_miou']['miou']:.4f}")
for f in sorted(glob.glob(f"{OUT}/latency_baselines/*.json")) + sorted(glob.glob(f"{OUT}/*/latency.json")) + [f"{OUT}/latency_ours_t4.json"]:
    if os.path.exists(f):
        j = json.load(open(f)); print(f"latency {os.path.basename(os.path.dirname(f)) if f.endswith('latency.json') else os.path.basename(f):24s} {j.get('ms_per_image', j.get('ms_per_image_incl_io', j.get('latency_ms_full_score_path')))} ms")
print("=" * 60); print("zip: /kaggle/working/final_results.zip")
```

## Cell 4 -- re-zip + reprint (any time)

```python
import subprocess, json, glob, os
OUT = "/kaggle/working/final"
subprocess.run("cd /kaggle/working && rm -f final_results.zip && find final -name '*.json' -not -path '*_smoke*' | zip -q final_results.zip -@", shell=True)
for d in sorted(glob.glob(f"{OUT}/*")):
    if os.path.isdir(d) and os.path.exists(f"{d}/result.json"):
        res = json.load(open(f"{d}/result.json")); sw = json.load(open(f"{d}/sweep.json")); sw = sw.get("summary", sw)
        print(f"{os.path.basename(d):22s} AUROC {res['pixel_auroc']:.4f}  AP {res['pixel_ap']:.4f}  ODS-F1 {sw['ods_f1']['f1']:.4f}  MIoU {sw['best_miou']['miou']:.4f}")
print(subprocess.run("ls -la /kaggle/working/final_results.zip", shell=True, capture_output=True, text=True).stdout)
```

Queue timing: GPU0 strokes_s0 -> strokes_s1 (~6 h). GPU1 strokes_s2 (~3 h) -> dinov2 (4 runs,
~30 min) -> latency (~20 min). Download final_results.zip -> git directory.
