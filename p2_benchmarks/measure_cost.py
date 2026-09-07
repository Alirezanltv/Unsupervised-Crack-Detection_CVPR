#!/usr/bin/env python3
"""Inference cost of AG-DSCAE: parameters, multiply-accumulates for one 256x256
image (hardware-independent, counted with forward hooks on every conv/linear),
and wall-clock latency of the full scoring path (forward + gradient error +
Gaussian smoothing) on whatever device is available. Writes a JSON artifact.

Usage: python p2_benchmarks/measure_cost.py --ckpt <ckpt_stage3.pt> --out results/model_cost.json
"""
import argparse, json, time, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p0_reproduce"))
from agdscae_ref import AGDSCAE, grad_mag, gaussian_kernel


def count_macs(model, x):
    macs = [0]
    def hook(m, i, o):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            k = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels // m.groups)
            macs[0] += k * o.numel()
    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    with torch.no_grad():
        model(x)
    for h in hs: h.remove()
    return macs[0]


def score(model, x, g):
    xh, _ = model(x)
    e = (grad_mag(x.mean(1, keepdim=True)) - grad_mag(xh.mean(1, keepdim=True))).abs()
    return F.conv2d(e, g, padding=g.shape[-1] // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    model = AGDSCAE(with_attention=True).to(dev).eval()
    model.load_state_dict(ck["state"] if "state" in ck else ck, strict=False)
    x = torch.rand(1, 3, 256, 256, device=dev)
    g = gaussian_kernel(1.5, 11).to(dev)
    params = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, x)
    with torch.no_grad():
        for _ in range(3): score(model, x, g)
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.iters): score(model, x, g)
        if dev == "cuda": torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / a.iters * 1000
    out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "input": "1x3x256x256",
           "params": params, "params_M": round(params / 1e6, 3), "macs": macs, "gmacs": round(macs / 1e9, 3),
           "gflops_2x": round(2 * macs / 1e9, 3), "latency_ms_full_score_path": round(ms, 2), "iters": a.iters}
    Path(a.out).write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
