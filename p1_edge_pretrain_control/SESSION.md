# Edge-source control session (Canny maps vs. MNIST) -- paste-ready Kaggle cells

Purpose: answer "why MNIST and not a hand-engineered edge prior?" with data. Stage 1
pretrains on Canny edge maps of BSDS500 natural images instead of digits; stages 2-3,
splits, scoring and sweeps stay identical. 3 seeds, DeepCrack arm, ~6.5 h on T4x2.

Notebook settings: accelerator T4 x2, Persistence "Variables and Files", run interactively
(the persisted clone carries the private-repo credentials; a clean commit run would not).
No pip installs: everything needed is preinstalled on Kaggle.

Cell 1 fails fast on any environment problem. Cell 3 runs a ~5-minute full-chain smoke test
on the GPU BEFORE launching the 6-hour jobs, so a broken pipeline errors out in minutes,
not hours. Results are zipped automatically after every finished job and the metrics table
is printed at the end, so nothing needs to be run by hand afterwards.

If the session disconnects: re-run Cells 1-3. Training resumes from the last epoch
checkpoint; finished jobs are skipped; the smoke test re-runs (5 min).

---------------------------------------------------------------------------------------
## Cell 1 -- preflight (fails loudly, costs 30 seconds)

```python
import os, shutil, subprocess, importlib
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
g = subprocess.run("git fetch origin && git reset --hard origin/main", shell=True, capture_output=True, text=True)
assert g.returncode == 0, "GIT SYNC FAILED (network or credentials):\n" + g.stderr[-600:]
print(g.stdout.strip()[-200:])
assert "--source-dir" in open("p0_reproduce/agdscae_ref.py").read(), "trainer lacks --source-dir: push the latest commits, then re-run"
assert "--crops" in open("p1_edge_pretrain_control/make_edge_maps.py").read(), "generator lacks --crops: push the latest commits, then re-run"
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, f"need 2 GPUs, found {torch.cuda.device_count()}"
for m in ("sklearn", "scipy", "cv2", "PIL", "numpy"):
    importlib.import_module(m)
free_gb = shutil.disk_usage("/kaggle/working").free / 1e9
assert free_gb > 5, f"only {free_gb:.1f} GB free in /kaggle/working"
print(f"PREFLIGHT OK | commit {subprocess.run('git rev-parse --short HEAD', shell=True, capture_output=True, text=True).stdout.strip()} | {torch.cuda.device_count()} GPUs | {free_gb:.0f} GB free")
```

## Cell 2 -- data (DeepCrack + 20,000 Canny maps; ~5 minutes)

```python
import os, subprocess, glob
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
counts = {s: len(os.listdir(f"/kaggle/temp/data/deepcrack/{s}")) for s in ("train/good", "calib", "test/images", "test/masks")}
assert counts == {"train/good": 250, "calib": 50, "test/images": 237, "test/masks": 237}, counts

if len(glob.glob("/kaggle/temp/canny_maps/*.png")) != 20000:
    try:
        sh("wget -q https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz -O /kaggle/temp/bsr.tgz && tar xzf /kaggle/temp/bsr.tgz -C /kaggle/temp")
        src = "/kaggle/temp/BSR/BSDS500/data/images"
    except RuntimeError:
        import kagglehub
        src = kagglehub.dataset_download("balraj98/berkeley-segmentation-dataset-500-bsds500")
    print(sh(f"python p1_edge_pretrain_control/make_edge_maps.py --src {src} --dst /kaggle/temp/canny_maps --detector canny --size 256 --crops 40 --seed 2027").strip())
n_maps = len(glob.glob("/kaggle/temp/canny_maps/*.png"))
assert n_maps == 20000, f"expected 20000 Canny maps, got {n_maps}"
print("DATA OK |", counts, "| canny maps:", n_maps)
```

## Cell 3 -- smoke test, then the real jobs, with automatic zipping and a printed table

```python
import subprocess, os, time, json, glob, shutil
os.chdir("/kaggle/working/Unsupervised-Crack-Detection_CVPR")
DATA, RAW, MAPS = "/kaggle/temp/data/deepcrack", "/kaggle/temp/deepcrack_raw", "/kaggle/temp/canny_maps"
OUT = "/kaggle/working/edgectl"

def chain(out, epochs, subset):
    return (f"python -u p0_reproduce/agdscae_ref.py train --raw-root {RAW} --splits p0_reproduce/splits "
            f"--name deepcrack --out {out} --seed {out[-1]} --stage-epochs {epochs} --source-subset {subset} "
            f"--source-dir {MAPS} && "
            f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DATA}/test/images --out {out}/maps && "
            f"python -u p0_reproduce/agdscae_ref.py dump --ckpt {out}/ckpt_stage3.pt --images {DATA}/calib --out {out}/calib && "
            f"python -u common/eval_maps.py --maps {out}/maps --masks {DATA}/test/masks --calib {out}/calib --out {out}/result.json && "
            f"python -u common/sweep_threshold.py --maps {out}/maps --masks {DATA}/test/masks --calib {out}/calib --out {out}/sweep.json")

def env_for(gpu):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

# ---- 1. smoke test: the entire chain at toy size on GPU0 (~5 min). Fails here, not at hour 6.
smoke = f"{OUT}/_smoke_s0"
shutil.rmtree(smoke, ignore_errors=True)
r = subprocess.run(chain(smoke, 1, 64), shell=True, env=env_for(0), capture_output=True, text=True)
if r.returncode != 0 or not os.path.exists(f"{smoke}/sweep.json"):
    print(r.stdout[-1500:]); print(r.stderr[-1500:])
    raise SystemExit("SMOKE TEST FAILED -- nothing launched. Paste the output above.")
print("SMOKE TEST OK: full chain (train/dump/eval/sweep) runs with the Canny source. Launching real jobs.", flush=True)

# ---- 2. real jobs
JOBS = {0: [f"{OUT}/canny_s0", f"{OUT}/canny_s1"],
        1: [f"{OUT}/canny_s2"]}

def zip_results():
    subprocess.run(f"cd /kaggle/working && rm -f edgectl_results.zip && find edgectl -path '*canny_s*' \\( -name result.json -o -name sweep.json \\) | zip -q edgectl_results.zip -@", shell=True)

def launch(gpu, out):
    name = os.path.basename(out)
    return name, subprocess.Popen(chain(out, 50, 20000), shell=True, env=env_for(gpu),
                                  stdout=open(f"/kaggle/working/{name}.log", "w"), stderr=subprocess.STDOUT)

def tail(name, n=1):
    try:
        L = open(f"/kaggle/working/{name}.log", errors="replace").read().splitlines()
        return "\n".join(L[-n:]) if L else ""
    except FileNotFoundError:
        return "(no log)"

state = {g: {"q": list(j), "cur": None} for g, j in JOBS.items()}
failed, t0, last_print = [], time.time(), 0
while any(s["q"] or s["cur"] for s in state.values()):
    for g, s in state.items():
        if s["cur"] is None and s["q"]:
            s["cur"] = launch(g, s["q"].pop(0))
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} START {s['cur'][0]}", flush=True)
        elif s["cur"] and s["cur"][1].poll() is not None:
            name, rc = s["cur"][0], s["cur"][1].returncode
            print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {name} EXIT {rc}", flush=True)
            if rc != 0:
                failed.append(name); print("---- last 25 log lines of", name); print(tail(name, 25)); print("----", flush=True)
            zip_results()
            s["cur"] = None
    time.sleep(60)
    if time.time() - last_print > 300:
        last_print = time.time()
        for g, s in state.items():
            if s["cur"]:
                print(f"[{time.strftime('%H:%M:%S')}] GPU{g} {s['cur'][0]:9s} | {tail(s['cur'][0])[-100:]}", flush=True)

zip_results()
print(f"\nSESSION DONE in {(time.time()-t0)/3600:.1f} h | failed jobs: {failed or 'none'}")

# ---- 3. print the numbers so they can be pasted directly
rows = []
for d in sorted(glob.glob(f"{OUT}/canny_s*")):
    try:
        res = json.load(open(f"{d}/result.json")); sw = json.load(open(f"{d}/sweep.json")); sw = sw.get("summary", sw)
        rows.append((os.path.basename(d), res["pixel_auroc"], res["pixel_ap"], sw["ods_f1"]["f1"], sw["best_miou"]["miou"]))
    except Exception as e:
        rows.append((os.path.basename(d), None, None, None, None))
print("\n==== EDGE-SOURCE CONTROL (Canny maps as source) -- paste this block ====")
print(f"{'run':10s} {'AUROC':>8s} {'AP':>8s} {'ODS-F1':>8s} {'MIoU':>8s}")
for r in rows:
    print(f"{r[0]:10s} " + " ".join(f"{v:8.4f}" if v is not None else "   FAILED" for v in r[1:]))
print("=" * 70)
print("zip ready: /kaggle/working/edgectl_results.zip")
```

## Cell 4 -- re-zip + reprint (safe any time, e.g. after a resume)

```python
import subprocess, json, glob, os
subprocess.run("cd /kaggle/working && rm -f edgectl_results.zip && find edgectl -path '*canny_s*' \\( -name result.json -o -name sweep.json \\) | zip -q edgectl_results.zip -@", shell=True)
for d in sorted(glob.glob("/kaggle/working/edgectl/canny_s*")):
    try:
        res = json.load(open(f"{d}/result.json")); sw = json.load(open(f"{d}/sweep.json")); sw = sw.get("summary", sw)
        print(f"{os.path.basename(d):10s} AUROC {res['pixel_auroc']:.4f}  AP {res['pixel_ap']:.4f}  ODS-F1 {sw['ods_f1']['f1']:.4f}  MIoU {sw['best_miou']['miou']:.4f}")
    except Exception:
        print(f"{os.path.basename(d):10s} not finished")
print(subprocess.run("ls -la /kaggle/working/edgectl_results.zip", shell=True, capture_output=True, text=True).stdout)
```

Stage-1 losses will NOT look like the MNIST runs' 0.006x: Canny maps are sparser than digit
strokes, so the loss scale differs. That is expected. Stages 2-3 use the usual crack data.

## What happens with the numbers (pre-committed, either way)
Compare canny (3 seeds) to the full MNIST model on matched seeds 0-2, all four metrics,
Welch tests. MNIST >= Canny: the digit prior matches or beats a hand-engineered edge prior --
strong support for the premise; new control row + 2-3 sentences in Sec. 4, limitation 5
updated. Canny > MNIST: the paper says so and reframes to "a cross-domain edge prior is what
matters, and engineered edges are the stronger choice" -- still publishable.
