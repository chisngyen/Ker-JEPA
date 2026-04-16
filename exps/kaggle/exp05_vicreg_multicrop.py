# =============================================================================
# exp05_vicreg_multicrop.py
# VICReg + Multi-Crop (2 global + 4 local) + Student-t KSD
#
# WHY THIS BEATS exp02 (74.11%) AT THE SAME 50 EPOCHS:
#   VICReg was the Phase 2 winner but only sees 1 view pair per step.
#   Multi-crop gives 1 global-global pair (full VICReg) + 8 global-local
#   invariance terms per step — 9× more training signal per image
#   with the same batch size and epoch count.
#
#   Method (SwAV / DINO-inspired multi-crop recipe for VICReg):
#     · Global-global: full VICReg (inv + var + cov + KSD-ST)
#     · Global-local:  invariance only (no var/cov on small crops)
#       → prevents the local branch from collapsing while still
#         enforcing semantic consistency across scales.
#
#   λ_mc  = 0.5  (global-local inv scale)
#   λ_ksd = 0.05 (Student-t KSD on global projections)
#
# TARGET: >80% in 50 SSL + 50 LP epochs
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

SSL_EPOCHS  = 50
LP_EPOCHS   = 50
BATCH_SIZE  = 128
LR_SSL      = 5e-4
WD          = 0.05
PROJ_DIM    = 512       # same as exp02 (VICReg expander)
N_LOCAL     = 4         # local crops in addition to 2 global
W_INV       = 25.0
W_VAR       = 25.0
W_COV       = 1.0
W_MC        = 0.5       # global→local invariance weight
W_KSD       = 0.05
NU          = 3.0
WARMUP_EP   = 5


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
        norm_sq   = z.pow(2).sum(-1, keepdim=True)
        s         = -((self.nu + d) / (self.nu * self.sigma**2 + norm_sq)) * z
        K         = (1 + alpha * dist_sq) ** (-self.beta)
        diff      = z.unsqueeze(1) - z.unsqueeze(0)
        gc        = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)
        grad_k    = gc.unsqueeze(-1) * diff
        h         = ((s @ s.T) * K
                     + (s.unsqueeze(1) * (-grad_k)).sum(-1)
                     + (grad_k * s.unsqueeze(0)).sum(-1)
                     + gc * (d - 2 * alpha * (self.beta + 1) * dist_sq
                             / (1 + alpha * dist_sq)))
        return (h.sum() - h.trace()) / (n * (n - 1))


# ── VICReg helpers ────────────────────────────────────────────────────────────
def vicreg_loss(z1, z2):
    n, d = z1.shape
    inv = F.mse_loss(z1, z2)

    def var_t(z):
        return F.relu(1.0 - z.std(dim=0)).mean()

    def cov_t(z):
        zc = z - z.mean(0)
        C  = (zc.T @ zc) / (n - 1)
        return (C.pow(2).sum() - C.diag().pow(2).sum()) / d

    var = var_t(z1) + var_t(z2)
    cov = cov_t(z1) + cov_t(z2)
    return W_INV * inv + W_VAR * var + W_COV * cov, inv.item(), var.item(), cov.item()


# ── Model ─────────────────────────────────────────────────────────────────────
class VICReg_MC(nn.Module):
    """Same expander as exp02."""
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


# ── Multi-Crop Dataset ────────────────────────────────────────────────────────
class MultiCropFolder(datasets.ImageFolder):
    """2 global crops + n_local local crops per image."""
    def __init__(self, root, global_aug, local_aug, n_local=N_LOCAL):
        super().__init__(root, None)
        self.g_aug   = global_aug
        self.l_aug   = local_aug
        self.n_local = n_local

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        g1  = self.g_aug(img)
        g2  = self.g_aug(img)
        lcs = [self.l_aug(img) for _ in range(self.n_local)]
        return g1, g2, lcs, label


def mc_collate(batch):
    g1  = torch.stack([b[0] for b in batch])
    g2  = torch.stack([b[1] for b in batch])
    lcs = [torch.stack([b[2][i] for b in batch]) for i in range(len(batch[0][2]))]
    lbl = torch.tensor([b[3] for b in batch])
    return g1, g2, lcs, lbl


def build_aug():
    base = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply(
            [transforms.Lambda(lambda img: __import__('PIL').ImageOps.solarize(img, 128))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    global_aug = transforms.Compose(
        [transforms.RandomResizedCrop(128, scale=(0.4, 1.0))] + base)
    local_aug  = transforms.Compose(
        [transforms.RandomResizedCrop(128, scale=(0.05, 0.4))] + base)
    return global_aug, local_aug


# ── Linear Probe ──────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  EXP-05: VICReg + MULTICROP + KSD-ST  |  50+50  |  H100")
    print(f"  2 global + {N_LOCAL} local crops  |  λ_mc={W_MC}  λ_ksd={W_KSD}")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    g_aug, l_aug = build_aug()
    loader = DataLoader(
        MultiCropFolder(f"{DATA_PATH}/train", g_aug, l_aug),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, collate_fn=mc_collate,
    )
    total_steps  = SSL_EPOCHS * len(loader)
    warmup_steps = WARMUP_EP  * len(loader)

    raw_model = VICReg_MC().to(device)
    model     = torch.compile(raw_model, mode='max-autotune')
    ksd_fn    = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(SSL_EPOCHS):
        raw_model.train()
        pbar = tqdm(loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for g1, g2, lcs, _ in pbar:
            g1 = g1.to(device, non_blocking=True)
            g2 = g2.to(device, non_blocking=True)
            lcs = [lc.to(device, non_blocking=True) for lc in lcs]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, zg1 = model(g1)
                _, zg2 = model(g2)

                # Full VICReg on global-global pair
                vic, inv, var, cov = vicreg_loss(zg1, zg2)

                # Invariance only on global-local pairs
                mc_loss = sum(
                    F.mse_loss(zg1, model(lc)[1]) + F.mse_loss(zg2, model(lc)[1])
                    for lc in lcs
                ) / len(lcs)

                # Student-t KSD on global projections
                ksd = ksd_fn(torch.cat([zg1, zg2], dim=0))

                loss = vic + W_MC * mc_loss + W_KSD * ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()

            pbar.set_postfix({"inv": f"{inv:.3f}", "var": f"{var:.3f}",
                               "mc": f"{mc_loss.item():.3f}", "ksd": f"{ksd.item():.3f}"})

    ckpt = f"{SAVE_DIR}/exp05_vicreg_multicrop.pth"
    torch.save(raw_model.state_dict(), ckpt)
    print(f"\n  Checkpoint: {ckpt}")

    best_acc = run_lp(raw_model.backbone, device)

    report = {"method": "EXP05_VICReg_MultiCrop_StudentT", "ssl_epochs": SSL_EPOCHS,
              "n_local": N_LOCAL, "w_mc": W_MC, "w_ksd": W_KSD, "nu": NU,
              "linear_probe_acc": best_acc, "paper_sota": 91.90,
              "delta": round(best_acc - 91.90, 2)}
    with open("/kaggle/working/results_exp05_vicreg_multicrop.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (Δ={best_acc-91.90:+.2f}% vs paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
