# =============================================================================
# exp03_byol_fixed.py
# BYOL (EMA) + Student-t KSD — FIXED TARGET INIT
#
# Root cause of exp01 collapse: target network was partially copied via
# load_state_dict with strict=False, resulting in random weights.
# Fix: copy.deepcopy(online) → guaranteed identical initialization.
#
# Additional tuning vs exp01:
#   - EMA momentum base: 0.99 → 0.996 (standard BYOL paper value)
#   - Projector output: 256 → 128 (smaller → less collapse risk at 50ep)
#   - KSD λ: 0.05 → 0.03 (reduce noise from KSD at early training)
#   - batch_size: 128 (unchanged, good for KSD)
#
# TARGET: >92.5% in 50 SSL + 50 LP epochs
# =============================================================================

import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _pkg in ["timm", "tqdm"]:
    try: __import__(_pkg)
    except ImportError: _install(_pkg)

import copy, os, math, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import MLP
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
PROJ_DIM    = 128       # smaller than exp01 → more stable at 50ep
KSD_LAMBDA  = 0.03      # slightly reduced to not dominate early loss
NU          = 3.0
EMA_BASE    = 0.996     # paper-standard BYOL value (was 0.99 in exp01)
EMA_END     = 1.0
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
        norm_sq    = z.pow(2).sum(-1, keepdim=True)
        s          = -((self.nu + d) / (self.nu * self.sigma**2 + norm_sq)) * z
        K          = (1 + alpha * dist_sq) ** (-self.beta)
        diff       = z.unsqueeze(1) - z.unsqueeze(0)
        gc         = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)
        grad_k     = gc.unsqueeze(-1) * diff
        term_a     = (s @ s.T) * K
        term_b     = (s.unsqueeze(1) * (-grad_k)).sum(-1)
        term_c     = (grad_k * s.unsqueeze(0)).sum(-1)
        laplacian  = gc * (d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq))
        h          = term_a + term_b + term_c + laplacian
        return (h.sum() - h.trace()) / (n * (n - 1))


# ── Model ─────────────────────────────────────────────────────────────────────
class BYOLOnline(nn.Module):
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model('vit_small_patch8_224', pretrained=False,
                                           num_classes=512, drop_path_rate=0.1, img_size=128)
        self.projector = MLP(512, [2048, proj_dim], norm_layer=nn.BatchNorm1d)
        self.predictor = MLP(proj_dim, [512, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        pred = self.predictor(proj)
        return feat, proj, pred


class BYOLTarget(nn.Module):
    """EMA copy of online backbone + projector. No predictor, no grad."""
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model('vit_small_patch8_224', pretrained=False,
                                           num_classes=512, drop_path_rate=0.1, img_size=128)
        self.projector = MLP(512, [2048, proj_dim], norm_layer=nn.BatchNorm1d)

    @torch.no_grad()
    def forward(self, x):
        return self.projector(self.backbone(x))

    @torch.no_grad()
    def ema_update(self, online: BYOLOnline, m: float):
        for p_t, p_o in zip(self.parameters(), online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)


def byol_loss(pred, target):
    return 2.0 - 2.0 * (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(-1).mean()

def get_ema_m(step, total):
    return EMA_END - (EMA_END - EMA_BASE) * (math.cos(math.pi * step / total) + 1) / 2


# ── Augmentation ──────────────────────────────────────────────────────────────
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


class TwoViewFolder(datasets.ImageFolder):
    def __init__(self, root, aug):
        super().__init__(root, None)
        self.aug = aug
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        return self.aug(img), self.aug(img), label


# ── Linear Probe ──────────────────────────────────────────────────────────────
def run_lp(backbone, device):
    print("\n" + "─" * 60)
    print("  STAGE 2: LINEAR PROBE (50 ep, frozen backbone)")
    print("─" * 60)
    tf = transforms.Compose([
        transforms.Resize(146), transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    lp_train = DataLoader(datasets.ImageFolder(f"{DATA_PATH}/train", tf),
                          batch_size=256, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    lp_val   = DataLoader(datasets.ImageFolder(f"{DATA_PATH}/val", tf),
                          batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, prefetch_factor=4)
    for p in backbone.parameters(): p.requires_grad_(False)
    backbone.eval()
    head = nn.Linear(512, 10).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0)
    best = 0.0
    for ep in range(LP_EPOCHS):
        head.train()
        for x, y in lp_train:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(head(backbone(x)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); c = t = 0
        with torch.no_grad():
            for x, y in lp_val:
                x, y = x.to(device), y.to(device)
                c += (head(backbone(x)).argmax(1) == y).sum().item(); t += y.size(0)
        acc = 100. * c / t; best = max(best, acc)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  LP {ep+1:>3}/{LP_EPOCHS} | Acc: {acc:.2f}% | Best: {best:.2f}%")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  EXP-03: BYOL FIXED + STUDENT-T KSD  |  50+50  |  H100")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug  = build_aug()
    loader = DataLoader(TwoViewFolder(f"{DATA_PATH}/train", aug),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    total_steps  = SSL_EPOCHS * len(loader)
    warmup_steps = WARMUP_EP * len(loader)

    # ── CRITICAL FIX: deepcopy guarantees target == online at step 0 ──────────
    raw_online = BYOLOnline().to(device)
    raw_target = copy.deepcopy(raw_online)          # ← was broken in exp01
    raw_target.__class__ = BYOLTarget               # swap class for clean API
    for p in raw_target.parameters(): p.requires_grad_(False)
    # Remove predictor from target (not needed, free memory)
    if hasattr(raw_target, 'predictor'):
        del raw_target.predictor

    online = torch.compile(raw_online, mode='max-autotune')
    ksd_fn = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_online.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler    = torch.amp.GradScaler('cuda')

    global_step = 0
    for epoch in range(SSL_EPOCHS):
        raw_online.train()
        pbar = tqdm(loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for v1, v2, _ in pbar:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, proj1, pred1 = online(v1)
                _, proj2, pred2 = online(v2)

                # Target: EMA backbone + projector (no predictor)
                with torch.no_grad():
                    tproj1 = raw_target.projector(raw_target.backbone(v1))
                    tproj2 = raw_target.projector(raw_target.backbone(v2))

                loss_byol = byol_loss(pred1, tproj2) + byol_loss(pred2, tproj1)
                loss_ksd  = ksd_fn(torch.cat([proj1, proj2], dim=0))
                loss      = loss_byol + KSD_LAMBDA * loss_ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); scheduler.step()

            m = get_ema_m(global_step, total_steps)
            # EMA: update backbone + projector only (target has no predictor)
            with torch.no_grad():
                for p_t, p_o in zip(raw_target.backbone.parameters(),
                                    raw_online.backbone.parameters()):
                    p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
                for p_t, p_o in zip(raw_target.projector.parameters(),
                                    raw_online.projector.parameters()):
                    p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
            global_step += 1

            pbar.set_postfix({"byol": f"{loss_byol.item():.3f}",
                               "ksd": f"{loss_ksd.item():.3f}", "m": f"{m:.4f}"})

    ckpt = f"{SAVE_DIR}/exp03_byol_fixed.pth"
    torch.save(raw_online.state_dict(), ckpt)
    print(f"\n  Checkpoint: {ckpt}")

    best_acc = run_lp(raw_online.backbone, device)

    report = {"method": "EXP03_BYOL_Fixed_StudentT_KSD", "ssl_epochs": SSL_EPOCHS,
              "linear_probe_acc": best_acc, "paper_sota": 91.90,
              "delta": round(best_acc - 91.90, 2)}
    with open("/kaggle/working/results_exp03_byol_fixed.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (Δ={best_acc-91.90:+.2f}% vs paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
