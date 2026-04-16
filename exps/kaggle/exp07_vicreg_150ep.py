# =============================================================================
# exp07_vicreg_150ep.py
# VICReg + Student-t KSD — 150 SSL epochs (scale-up of exp02)
#
# GOAL: Isolate the effect of epoch budget.
#   exp02 (74.11%) was bottlenecked by underfitting, not the objective.
#   This is identical config, 3× the SSL epochs: 50 → 150.
#   Expect +8–15% improvement from extra training signal alone.
#
# Everything else identical to exp02:
#   batch=128, LR=5e-4, proj_dim=512, VICReg weights, KSD λ=0.05, ν=3
#
# TARGET: >85% in 150 SSL + 50 LP epochs
# =============================================================================

import subprocess, sys

def _install(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ["timm", "tqdm"]:
    try: __import__(_p)
    except ImportError: _install(_p)

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

DATA_PATH = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
SAVE_DIR  = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

SSL_EPOCHS  = 150       # ← 3× exp02
LP_EPOCHS   = 50
BATCH_SIZE  = 128
LR_SSL      = 5e-4
WD          = 0.05
PROJ_DIM    = 512
W_INV, W_VAR, W_COV = 25.0, 25.0, 1.0
W_KSD       = 0.05
NU          = 3.0
WARMUP_EP   = 10        # scale warmup with epochs (was 5 at 50ep)


# ── Student-t KSD ─────────────────────────────────────────────────────────────
class StudentT_KSD(nn.Module):
    def __init__(self, sigma=1.0, nu=NU, beta=0.5):
        super().__init__()
        self.sigma, self.nu, self.beta = sigma, nu, beta

    def forward(self, z):
        n, d = z.shape
        with torch.no_grad():
            dist_sq = torch.cdist(z, z).pow(2)
            alpha   = 1.0 / (dist_sq.median() + 1e-6)
        norm_sq = z.pow(2).sum(-1, keepdim=True)
        s       = -((self.nu + d) / (self.nu * self.sigma**2 + norm_sq)) * z
        K       = (1 + alpha * dist_sq) ** (-self.beta)
        diff    = z.unsqueeze(1) - z.unsqueeze(0)
        gc      = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)
        grad_k  = gc.unsqueeze(-1) * diff
        h       = ((s @ s.T) * K
                   + (s.unsqueeze(1) * (-grad_k)).sum(-1)
                   + (grad_k * s.unsqueeze(0)).sum(-1)
                   + gc * (d - 2 * alpha * (self.beta + 1) * dist_sq
                           / (1 + alpha * dist_sq)))
        return (h.sum() - h.trace()) / (n * (n - 1))


def vicreg_loss(z1, z2):
    n, d = z1.shape
    inv  = F.mse_loss(z1, z2)
    def var_t(z): return F.relu(1.0 - z.std(dim=0)).mean()
    def cov_t(z):
        zc = z - z.mean(0)
        C  = (zc.T @ zc) / (n - 1)
        return (C.pow(2).sum() - C.diag().pow(2).sum()) / d
    return (W_INV * inv + W_VAR * (var_t(z1) + var_t(z2))
            + W_COV * (cov_t(z1) + cov_t(z2))), inv.item()


class VICReg_ST(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model('vit_small_patch8_224', pretrained=False,
                                           num_classes=512, drop_path_rate=0.1, img_size=128)
        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim),
        )
    def forward(self, x):
        feat = self.backbone(x)
        return feat, self.projector(feat)


class TwoViewFolder(datasets.ImageFolder):
    def __init__(self, root, aug):
        super().__init__(root, None)
        self.aug = aug
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        return self.aug(img), self.aug(img), label


def build_aug():
    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply(
            [transforms.Lambda(lambda img: __import__('PIL').ImageOps.solarize(img, 128))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def run_lp(backbone, device):
    print("\n" + "─" * 60)
    print("  STAGE 2: LINEAR PROBE  (50 ep, frozen backbone)")
    print("─" * 60)
    tf = transforms.Compose([
        transforms.Resize(146), transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr = DataLoader(datasets.ImageFolder(f"{DATA_PATH}/train", tf),
                    batch_size=256, shuffle=True, drop_last=True,
                    num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    vl = DataLoader(datasets.ImageFolder(f"{DATA_PATH}/val", tf),
                    batch_size=256, shuffle=False,
                    num_workers=4, pin_memory=True, prefetch_factor=4)
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.eval()
    head = nn.Linear(512, 10).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0)
    best = 0.0
    for ep in range(LP_EPOCHS):
        head.train()
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(head(backbone(x)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); c = t = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                c += (head(backbone(x)).argmax(1) == y).sum(); t += y.size(0)
        acc = 100. * c / t; best = max(best, acc)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  LP {ep+1:>3}/{LP_EPOCHS} | Acc: {acc:.2f}% | Best: {best:.2f}%")
    return best


def main():
    print("\n" + "=" * 60)
    print("  EXP-07: VICReg + KSD-ST  |  150 SSL + 50 LP  |  H100")
    print(f"  batch={BATCH_SIZE}  LR={LR_SSL}  proj={PROJ_DIM}  λ_ksd={W_KSD}")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug    = build_aug()
    loader = DataLoader(TwoViewFolder(f"{DATA_PATH}/train", aug),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    total_steps  = SSL_EPOCHS * len(loader)
    warmup_steps = WARMUP_EP  * len(loader)

    raw_model = VICReg_ST().to(device)
    model     = torch.compile(raw_model, mode='max-autotune')
    ksd_fn    = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(SSL_EPOCHS):
        raw_model.train()
        pbar = tqdm(loader, desc=f"SSL {epoch+1:>3}/{SSL_EPOCHS}")
        for v1, v2, _ in pbar:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, z1 = model(v1)
                _, z2 = model(v2)
                vic, inv = vicreg_loss(z1, z2)
                ksd      = ksd_fn(torch.cat([z1, z2], dim=0))
                loss     = vic + W_KSD * ksd
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
            if (epoch + 1) % 50 == 0:
                pbar.set_postfix({"inv": f"{inv:.3f}", "ksd": f"{ksd.item():.3f}"})

    ckpt = f"{SAVE_DIR}/exp07_vicreg_150ep.pth"
    torch.save(raw_model.state_dict(), ckpt)
    print(f"\n  Checkpoint: {ckpt}")

    best_acc = run_lp(raw_model.backbone, device)

    report = {"method": "EXP07_VICReg_KSD_ST_150ep", "ssl_epochs": SSL_EPOCHS,
              "batch_size": BATCH_SIZE, "lr": LR_SSL, "w_ksd": W_KSD, "nu": NU,
              "linear_probe_acc": best_acc, "paper_sota": 91.90,
              "delta": round(best_acc - 91.90, 2)}
    with open("/kaggle/working/results_exp07.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (Δ={best_acc-91.90:+.2f}% vs paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
