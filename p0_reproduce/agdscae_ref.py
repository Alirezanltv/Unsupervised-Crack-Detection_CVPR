#!/usr/bin/env python3
"""AG-DSCAE reference implementation, reconstructed from the paper's spec.

STATUS: this is a faithful re-implementation of the method as the paper
describes it (architecture Sec. 3.5, losses Eqs. 4-13, hyperparameters
supp. Sec. B). It was written because the original training code was not
available at rebuild time. If the original surfaces, reconcile the two and
record every difference. Numbers produced by this file are numbers of THIS
file; the released code and the paper's method section must stay in sync.

Reconstruction decisions the paper under-specifies (recorded here on purpose):
  - orientation/curvature probes for the OT cost: fixed Sobel / Laplacian
    responses on the channel-mean feature map (the paper says "small
    convolutional probes" without weights).
  - MNIST upsampling uses torch bicubic (single-step 28->256), the
    supplementary's stated pipeline.
  - SSIM: standard single-scale, 11x11 Gaussian window.

Subcommands:
  train  3-stage training with per-epoch checkpointing and auto-resume
  dump   checkpoint -> raw anomaly maps for common/eval_maps.py

Examples (DeepCrack, splits from p0_reproduce/splits):
  python p0_reproduce/agdscae_ref.py train \
      --raw-root /content/deepcrack_raw --splits p0_reproduce/splits \
      --name deepcrack --out /content/drive/MyDrive/agdscae/deepcrack_s0 \
      --seed 0 [--stage-epochs 2 for a smoke run]

  python p0_reproduce/agdscae_ref.py dump \
      --ckpt /content/drive/MyDrive/agdscae/deepcrack_s0/ckpt_stage3.pt \
      --images /content/data/deepcrack/test/images --out .../maps
"""
import argparse
import gzip
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------- model ----
class DSConv(nn.Module):
    """Depthwise separable conv block: DW 3x3 -> PW 1x1 -> BN -> ReLU -> drop."""

    def __init__(self, cin, cout, p_drop=0.1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout, momentum=0.9, eps=1e-5)
        self.drop = nn.Dropout2d(p_drop)

    def forward(self, x):
        return self.drop(F.relu(self.bn(self.pw(self.dw(x)))))


class Encoder(nn.Module):
    CH = [32, 64, 128, 256]

    def __init__(self):
        super().__init__()
        cin = 3
        self.blocks = nn.ModuleList()
        for c in self.CH:
            self.blocks.append(DSConv(cin, c))
            cin = c

    def forward(self, x):
        feats = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
            x = F.max_pool2d(x, 2)
        return feats, x            # skips at 256..32, bottleneck 16^2x256


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = Encoder.CH[::-1]                       # 256,128,64,32
        self.ups, self.blocks = nn.ModuleList(), nn.ModuleList()
        cin = ch[0]
        for c in ch:
            self.ups.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.blocks.append(DSConv(2 * c, c))    # concat skip
            cin = c
        self.head = nn.Conv2d(cin, 3, 1)

    def forward(self, bott, skips):
        x = bott
        for up, blk, skip in zip(self.ups, self.blocks, skips[::-1]):
            x = up(x)
            x = blk(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.head(x))


class AttentionGate(nn.Module):
    """Eq. 7-8: per-location gate between frozen source and target features."""

    def __init__(self, c):
        super().__init__()
        cp = max(c // 4, 8)
        self.w1 = nn.Conv2d(4 * c, cp, 1)
        self.w2 = nn.Conv2d(cp, c, 1)

    def forward(self, hs, ht):
        z = torch.cat([hs, ht, hs * ht, (hs - ht).abs()], dim=1)
        a = torch.sigmoid(self.w2(F.relu(self.w1(z))))
        return a * hs + (1 - a) * ht, a


class AGDSCAE(nn.Module):
    def __init__(self, with_attention=False):
        super().__init__()
        self.enc_s = Encoder()                      # frozen source encoder
        self.enc_t = Encoder()                      # adapting target encoder
        self.dec = Decoder()
        self.gates = (nn.ModuleList([AttentionGate(c) for c in Encoder.CH])
                      if with_attention else None)

    def forward(self, x):
        if self.gates is None:
            skips, bott = self.enc_t(x)
            return self.dec(bott, skips), None
        with torch.no_grad():
            skips_s, _ = self.enc_s(x)
        skips_t, bott = self.enc_t(x)
        blended, amaps = [], []
        for g, hs, ht in zip(self.gates, skips_s, skips_t):
            h, a = g(hs, ht)
            blended.append(h)
            amaps.append(a)
        return self.dec(bott, blended), amaps


# --------------------------------------------------------------- losses ----
_SOBX = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)


def grad_mag(gray):                                 # B x 1 x H x W
    kx = _SOBX.to(gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def edge_l1(x, xh):
    return (grad_mag(x.mean(1, keepdim=True)) -
            grad_mag(xh.mean(1, keepdim=True))).abs().mean()


def gaussian_kernel(sigma, ksize):
    ax = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2
    k = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).outer(k / k.sum())
    return k.view(1, 1, ksize, ksize)


def ssim(x, y, window=None):
    if window is None:
        window = gaussian_kernel(1.5, 11).to(x.device)
    c = x.shape[1]
    w = window.expand(c, 1, -1, -1)
    mu_x = F.conv2d(x, w, padding=5, groups=c)
    mu_y = F.conv2d(y, w, padding=5, groups=c)
    sx = F.conv2d(x * x, w, padding=5, groups=c) - mu_x ** 2
    sy = F.conv2d(y * y, w, padding=5, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / \
        ((mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2))
    return s.mean()


def feature_geometry(feat):
    """Orientation angle + curvature response of the channel-mean map."""
    g = feat.mean(1, keepdim=True)
    kx = _SOBX.to(feat.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, kx.transpose(2, 3), padding=1)
    theta = torch.atan2(gy, gx)
    curv = F.conv2d(g, _LAP.to(feat.device).view(1, 1, 3, 3), padding=1)
    return theta.flatten(1), curv.flatten(1)


def sinkhorn_ot(zs, zt, alpha=1.0, beta=0.3, gamma=0.2, eps=0.1, iters=50):
    """Entropic OT loss with the structure-preserving cost (Eq. 4-6)."""
    fs = F.normalize(zs.flatten(1), dim=1)
    ft = F.normalize(zt.flatten(1), dim=1)
    c_feat = torch.cdist(fs, ft) ** 2
    ths, cvs = feature_geometry(zs)
    tht, cvt = feature_geometry(zt)
    c_or = 1 - torch.cos(ths.mean(1)[:, None] - tht.mean(1)[None, :]).abs()
    c_cv = (cvs.abs().mean(1)[:, None] - cvt.abs().mean(1)[None, :]).abs()
    C = alpha * c_feat + beta * c_or + gamma * c_cv
    C = C - C.median()
    K = torch.exp(-C / eps)
    a = torch.full((C.shape[0],), 1.0 / C.shape[0], device=C.device)
    b = torch.full((C.shape[1],), 1.0 / C.shape[1], device=C.device)
    u = torch.ones_like(a)
    for _ in range(iters):
        u = a / (K @ (b / (K.t() @ u + 1e-12)) + 1e-12)
    v = b / (K.t() @ u + 1e-12)
    P = u[:, None] * K * v[None, :]
    return (P * C).sum()


def gate_entropy(amaps):
    e = 0.0
    for a in amaps:
        a = a.clamp(1e-6, 1 - 1e-6)
        e = e + (-(a * a.log() + (1 - a) * (1 - a).log())).mean()
    return -e / len(amaps)          # bonus (negative loss) keeps gates soft


# ----------------------------------------------------------------- data ----
def load_mnist(cache: Path, subset: int, seed: int):
    cache.mkdir(parents=True, exist_ok=True)
    gz = cache / "train-images-idx3-ubyte.gz"
    if not gz.exists():
        urllib.request.urlretrieve(MNIST_URL, gz)
    with gzip.open(gz, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[4:8], "big")
    imgs = np.frombuffer(data, np.uint8, offset=16).reshape(n, 28, 28)
    if subset and subset < n:
        idx = np.random.default_rng(seed).choice(n, subset, replace=False)
        imgs = imgs[idx]
    return imgs


class MnistSource(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = torch.from_numpy(self.imgs[i].astype(np.float32) / 255.0)
        x = F.interpolate(x[None, None], size=256, mode="bicubic",
                          align_corners=False).clamp(0, 1)[0]
        return x.repeat(3, 1, 1)


class CrackSet(Dataset):
    def __init__(self, raw_root: Path, listfile: Path, augment: bool):
        from PIL import Image
        self.Image = Image
        self.paths = [raw_root / rel for rel in listfile.read_text().split()]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.Image.open(self.paths[i]).convert("RGB").resize(
            (256, 256), self.Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        if self.augment:
            if random.random() < 0.5:
                x = x.flip(-1)
            ang = random.uniform(-10, 10)
            x = torchvision_rotate(x, ang)
            x = (x * random.uniform(0.9, 1.1) +
                 random.uniform(-0.1, 0.1)).clamp(0, 1)
            if random.random() < 0.3:
                x = (x + 0.02 * torch.randn_like(x)).clamp(0, 1)
        return x


def torchvision_rotate(x, ang):
    try:
        from torchvision.transforms.functional import rotate
        return rotate(x, ang)
    except ImportError:
        return x


# ---------------------------------------------------------------- train ----
def anomaly_map(model, x):
    with torch.no_grad():
        xh, _ = model(x)
    err = (grad_mag(x.mean(1, keepdim=True)) -
           grad_mag(xh.mean(1, keepdim=True))).abs()
    k = gaussian_kernel(1.5, 7).to(x.device)
    return F.conv2d(err, k, padding=3).squeeze(1)


def save_ckpt(path, model, stage, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stage": stage, "epoch": epoch,
                "state": model.state_dict(),
                "with_attention": model.gates is not None}, path)


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    E = args.stage_epochs

    src = DataLoader(MnistSource(load_mnist(out / "mnist", args.source_subset,
                                            args.seed)),
                     batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    tgt = DataLoader(CrackSet(Path(args.raw_root),
                              Path(args.splits) / f"{args.name}_train.txt", True),
                     batch_size=16, shuffle=True, num_workers=2, drop_last=True)

    model = AGDSCAE(with_attention=False).to(dev)

    def stage_done(k):
        return (out / f"ckpt_stage{k}.pt").exists()

    # ---- Stage 1: source pretraining -----------------------------------
    if not stage_done(1):
        opt = torch.optim.AdamW(list(model.enc_t.parameters()) +
                                list(model.dec.parameters()),
                                lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
        for ep in range(E):
            for x in src:
                x = x.to(dev)
                xh, _ = model(x)
                loss = F.mse_loss(xh, x) + 0.5 * edge_l1(x, xh)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            print(f"[stage1 ep{ep + 1}/{E}] loss={loss.item():.4f}", flush=True)
            save_ckpt(out / "ckpt_stage1.pt", model, 1, ep)
    else:
        model.load_state_dict(torch.load(out / "ckpt_stage1.pt",
                                         map_location=dev)["state"])
        print("stage 1 checkpoint found, skipping")

    # freeze source encoder = copy of the pretrained encoder
    model.enc_s.load_state_dict(model.enc_t.state_dict())
    for p in model.enc_s.parameters():
        p.requires_grad_(False)

    # ---- Stage 2: OT alignment ------------------------------------------
    if not stage_done(2):
        opt = torch.optim.AdamW(list(model.enc_t.parameters()) +
                                list(model.dec.parameters()),
                                lr=1e-4, weight_decay=1e-4)
        src_it = iter(src)
        for ep in range(E):
            for bi, x in enumerate(tgt):
                x = x.to(dev)
                xh, _ = model(x)
                loss = F.mse_loss(xh, x) + 0.3 * edge_l1(x, xh)
                if bi % 5 == 0:
                    try:
                        xs = next(src_it)
                    except StopIteration:
                        src_it = iter(src)
                        xs = next(src_it)
                    _, zs = model.enc_s(xs.to(dev))
                    _, zt = model.enc_t(x)
                    loss = loss + 0.1 * sinkhorn_ot(zs, zt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            print(f"[stage2 ep{ep + 1}/{E}] loss={loss.item():.4f}", flush=True)
            save_ckpt(out / "ckpt_stage2.pt", model, 2, ep)
    else:
        model.load_state_dict(torch.load(out / "ckpt_stage2.pt",
                                         map_location=dev)["state"])
        print("stage 2 checkpoint found, skipping")

    # ---- Stage 3: attention-guided adaptation ---------------------------
    state = model.state_dict()
    model = AGDSCAE(with_attention=True).to(dev)
    model.load_state_dict(state, strict=False)
    for p in model.enc_s.parameters():
        p.requires_grad_(False)
    if not stage_done(3):
        opt = torch.optim.AdamW([p for p in model.parameters()
                                 if p.requires_grad], lr=5e-5, weight_decay=1e-4)
        src_it = iter(src)
        for ep in range(E):
            for bi, x in enumerate(tgt):
                x = x.to(dev)
                xh, amaps = model(x)
                loss = (1 - ssim(x, xh)) + 0.3 * edge_l1(x, xh) \
                    + 0.01 * gate_entropy(amaps)
                if bi % 5 == 0:
                    try:
                        xs = next(src_it)
                    except StopIteration:
                        src_it = iter(src)
                        xs = next(src_it)
                    _, zs = model.enc_s(xs.to(dev))
                    _, zt = model.enc_t(x)
                    loss = loss + 0.05 * sinkhorn_ot(zs, zt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            print(f"[stage3 ep{ep + 1}/{E}] loss={loss.item():.4f}", flush=True)
            save_ckpt(out / "ckpt_stage3.pt", model, 3, ep)
    print("training complete:", out / "ckpt_stage3.pt")


# ----------------------------------------------------------------- dump ----
def dump(args):
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev)
    model = AGDSCAE(with_attention=ck.get("with_attention", True)).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in EXTS)
    for i in range(0, len(paths), 16):
        chunk = paths[i:i + 16]
        x = torch.stack([
            torch.from_numpy(np.asarray(
                Image.open(p).convert("RGB").resize((256, 256), Image.BICUBIC),
                np.float32) / 255.0).permute(2, 0, 1)
            for p in chunk]).to(dev)
        maps = anomaly_map(model, x)
        for p, m in zip(chunk, maps):
            np.save(out / f"{p.stem}.npy", m.cpu().numpy().astype(np.float32))
    print(f"wrote {len(paths)} maps to {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--raw-root", required=True)
    t.add_argument("--splits", required=True)
    t.add_argument("--name", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--stage-epochs", type=int, default=50)
    t.add_argument("--source-subset", type=int, default=70000,
                   help="reduce for smoke runs / slow GPUs; paper uses 70000")
    d = sub.add_parser("dump")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--images", required=True)
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    (train if args.cmd == "train" else dump)(args)


if __name__ == "__main__":
    main()
