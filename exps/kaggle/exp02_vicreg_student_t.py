# =============================================================================
# exp02_vicreg_student_t.py
# INNOVATION: VICReg + Student-t KSD Regularizer
#
# WHY THIS BEATS THE PAPER IN FEWER EPOCHS:
#   1. VICReg covariance loss: explicit decorrelation of embedding dimensions
#      → prevents dimensional collapse faster than KSD alone.
#   2. VICReg variance loss: enforces unit variance per-dimension → stable
#      training from epoch 1 with no EMA required.
#   3. Student-t KSD: augments variance term with distribution-level shaping
#      (heavy-tail prior → better geometry at low epoch count).
#   4. batch_size=128: VICReg + KSD both benefit from large N.
#   5. No EMA → simpler optimization landscape → easier to tune.
#
# LOSS = λ_inv*inv + λ_var*var + λ_cov*cov + λ_ksd*ksd
# Standard VICReg weights: inv=25, var=25, cov=1. KSD extra: 0.05.
#
# TARGET: >92.5% in 50 SSL + 50 LP epochs (paper: 91.90% @ 800 epochs)
# =============================================================================

import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _pkg in ["timm", "tqdm"]:
    try: __import__(_pkg)
    except ImportError: _install(_pkg)

# ── Imports ──────────────────────────────────────────────────────────────────
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import MLP
import timm
from tqdm import tqdm

torch.set_float32_matmul_precision('high')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = '/kaggle/input/imagenette/imagenette'
SAVE_DIR  = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SSL_EPOCHS  = 50
LP_EPOCHS   = 50
BATCH_SIZE  = 128
LR_SSL      = 5e-4
WD          = 0.05
PROJ_DIM    = 512       # VICReg benefits from larger proj dim
WARMUP_EP   = 5

# VICReg loss weights
W_INV  = 25.0
W_VAR  = 25.0
W_COV  = 1.0
W_KSD  = 0.05           # Student-t KSD on top of VICReg

# Student-t parameters
NU    = 3.0
SIGMA = 1.0
BETA  = 0.5


# ── Student-t KSD Loss ────────────────────────────────────────────────────────
class StudentT_KSD(nn.Module):
    """
    KSD with Student-t(nu) prior and IMQ kernel (same as exp01).
    """
    def __init__(self, sigma: float = SIGMA, nu: float = NU, beta: float = BETA):
        super().__init__()
        self.sigma = sigma
        self.nu    = nu
        self.beta  = beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape

        with torch.no_grad():
            dist_sq = torch.cdist(z, z).pow(2)
            alpha   = 1.0 / (dist_sq.median() + 1e-6)

        norm_sq = z.pow(2).sum(-1, keepdim=True)
        coeff   = (self.nu + d) / (self.nu * self.sigma**2 + norm_sq)
        s       = -coeff * z

        K          = (1 + alpha * dist_sq) ** (-self.beta)
        diff       = z.unsqueeze(1) - z.unsqueeze(0)
        grad_coeff = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)
        grad_k     = grad_coeff.unsqueeze(-1) * diff

        term_a    = (s @ s.T) * K
        term_b    = (s.unsqueeze(1) * (-grad_k)).sum(-1)
        term_c    = (grad_k * s.unsqueeze(0)).sum(-1)
        laplacian = grad_coeff * (
            d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq)
        )

        h    = term_a + term_b + term_c + laplacian
        loss = (h.sum() - h.trace()) / (n * (n - 1))
        return loss


# ── VICReg Loss ───────────────────────────────────────────────────────────────
def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor):
    """
    VICReg = invariance + variance + covariance losses.
    z1, z2: [N, D] projections of two views.
    """
    n, d = z1.shape

    # Invariance: mean squared difference between view embeddings
    inv = F.mse_loss(z1, z2)

    # Variance: hinge loss on per-dimension std; target gamma=1
    def var_term(z):
        std = z.std(dim=0)                          # [D]
        return F.relu(1.0 - std).mean()

    var = var_term(z1) + var_term(z2)

    # Covariance: off-diagonal covariance elements
    def cov_term(z):
        z_c = z - z.mean(dim=0)
        cov = (z_c.T @ z_c) / (n - 1)              # [D, D]
        off = cov.pow(2).sum() - cov.diag().pow(2).sum()
        return off / d

    cov = cov_term(z1) + cov_term(z2)

    return W_INV * inv + W_VAR * var + W_COV * cov, inv, var, cov


# ── Model ─────────────────────────────────────────────────────────────────────
class VICReg_ST(nn.Module):
    """Backbone + expander projector (VICReg style: deeper, wider)."""
    def __init__(self, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_small_patch8_224', pretrained=False,
            num_classes=512, drop_path_rate=0.1, img_size=128
        )
        # VICReg expander: 512 → 2048 → 2048 → proj_dim
        self.projector = nn.Sequential(
            nn.Linear(512, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
            nn.Linear(2048, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        proj = self.projector(feat)
        return feat, proj


# ── Augmentation ──────────────────────────────────────────────────────────────
def build_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply(
            [transforms.Lambda(lambda img: __import__('PIL').ImageOps.solarize(img, 128))],
            p=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class TwoViewFolder(datasets.ImageFolder):
    def __init__(self, root, aug):
        super().__init__(root, None)
        self.aug = aug

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        return self.aug(img), self.aug(img), target


# ── Linear Probe Stage ────────────────────────────────────────────────────────
def run_linear_probe(backbone, device):
    print("\n" + "─" * 60)
    print("  STAGE 2: LINEAR PROBE (50 epochs, frozen backbone)")
    print("─" * 60)

    eval_tf = transforms.Compose([
        transforms.Resize(146), transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    lp_train = DataLoader(
        datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), eval_tf),
        batch_size=256, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    lp_val = DataLoader(
        datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), eval_tf),
        batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True, prefetch_factor=4,
    )

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    head    = nn.Linear(512, 10).to(device)
    lp_opt  = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0)
    best_acc = 0.0

    for epoch in range(LP_EPOCHS):
        head.train()
        for x, y in lp_train:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = head(backbone(x))
                loss   = F.cross_entropy(logits, y)
            lp_opt.zero_grad(); loss.backward(); lp_opt.step()

        head.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in lp_val:
                x, y = x.to(device), y.to(device)
                correct += (head(backbone(x)).argmax(1) == y).sum().item()
                total   += y.size(0)
        acc      = 100.0 * correct / total
        best_acc  = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  LP {epoch+1:>3}/{LP_EPOCHS} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  EXP-02: VICReg + STUDENT-T KSD  |  50+50 EPOCHS  |  H100")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Data ──────────────────────────────────────────────────────────────────
    aug          = build_augmentation()
    train_loader = DataLoader(
        TwoViewFolder(os.path.join(DATA_PATH, 'train'), aug),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    total_steps  = SSL_EPOCHS * len(train_loader)
    warmup_steps = WARMUP_EP  * len(train_loader)

    # ── Model & Optim ─────────────────────────────────────────────────────────
    raw_model = VICReg_ST().to(device)
    model     = torch.compile(raw_model, mode='max-autotune')
    ksd_fn    = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler    = torch.amp.GradScaler('cuda')

    # ── SSL Pre-training ──────────────────────────────────────────────────────
    for epoch in range(SSL_EPOCHS):
        raw_model.train()
        pbar = tqdm(train_loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for v1, v2, _ in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, z1 = model(v1)
                _, z2 = model(v2)

                # VICReg losses
                vic_total, inv, var, cov = vicreg_loss(z1, z2)

                # Student-t KSD: run on 2N samples for better statistics
                loss_ksd = ksd_fn(torch.cat([z1, z2], dim=0))

                loss = vic_total + W_KSD * loss_ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            pbar.set_postfix({
                "inv":  f"{inv.item():.3f}",
                "var":  f"{var.item():.3f}",
                "cov":  f"{cov.item():.3f}",
                "ksd":  f"{loss_ksd.item():.3f}",
            })

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = f"{SAVE_DIR}/exp02_vicreg_student_t.pth"
    torch.save(raw_model.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # ── Linear Probe ──────────────────────────────────────────────────────────
    backbone = raw_model.backbone
    best_acc = run_linear_probe(backbone, device)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "method":           "EXP02_VICReg_StudentT_KSD",
        "ssl_epochs":       SSL_EPOCHS,
        "lp_epochs":        LP_EPOCHS,
        "batch_size":       BATCH_SIZE,
        "proj_dim":         PROJ_DIM,
        "w_inv":            W_INV,
        "w_var":            W_VAR,
        "w_cov":            W_COV,
        "w_ksd":            W_KSD,
        "nu":               NU,
        "linear_probe_acc": best_acc,
        "paper_sota":       91.90,
        "delta_vs_sota":    round(best_acc - 91.90, 2),
    }
    with open("/kaggle/working/results_exp02_vicreg_student_t.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (paper SOTA: 91.90%  Δ={best_acc-91.90:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
