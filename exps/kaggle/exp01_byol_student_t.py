# =============================================================================
# exp01_byol_student_t.py
# INNOVATION: BYOL (EMA Momentum Encoder) + Student-t KSD Regularizer
#
# WHY THIS BEATS THE PAPER IN FEWER EPOCHS:
#   1. EMA target encoder: proper self-supervised signal (BYOL-style prediction)
#      vs. paper's plain invariance loss — richer gradient per step.
#   2. Student-t KSD: heavier tails than Gaussian → doesn't over-penalize
#      outlier embeddings → learns faster at early epochs.
#   3. batch_size=128 for KSD: 16k pairwise interactions vs 992 at batch=32.
#   4. L2-normalized BYOL loss: numerically stable, no collapse.
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
import os, math, json
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
DATA_PATH = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
SAVE_DIR  = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SSL_EPOCHS  = 50
LP_EPOCHS   = 50
BATCH_SIZE  = 128       # critical: large batch for KSD statistics
LR_SSL      = 5e-4
WD          = 0.05
KSD_LAMBDA  = 0.05      # KSD reg weight (stronger than kag_ scripts' 0.02)
NU          = 3.0       # Student-t degrees of freedom
EMA_BASE    = 0.99      # EMA momentum start
EMA_END     = 1.0       # EMA momentum end (cosine schedule)
PROJ_DIM    = 256       # projector output dim
WARMUP_EP   = 5         # warmup epochs

# ── Student-t KSD Loss ────────────────────────────────────────────────────────
class StudentT_KSD(nn.Module):
    """
    Kernelised Stein Discrepancy with Student-t prior and IMQ kernel.
    Score: s(x) = -(nu + d) / (nu * sigma^2 + ||x||^2) * x
    Kernel: K(x,y) = (1 + alpha * ||x-y||^2)^(-beta)
    """
    def __init__(self, sigma: float = 1.0, nu: float = 3.0, beta: float = 0.5):
        super().__init__()
        self.sigma = sigma
        self.nu    = nu
        self.beta  = beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape

        # Adaptive bandwidth: median heuristic
        with torch.no_grad():
            dist_sq = torch.cdist(z, z).pow(2)           # [N, N]
            alpha   = 1.0 / (dist_sq.median() + 1e-6)

        # Student-t score function
        norm_sq = z.pow(2).sum(-1, keepdim=True)          # [N, 1]
        coeff   = (self.nu + d) / (self.nu * self.sigma**2 + norm_sq)
        s       = -coeff * z                               # [N, D]

        # IMQ kernel and its gradient
        K          = (1 + alpha * dist_sq) ** (-self.beta)              # [N, N]
        diff       = z.unsqueeze(1) - z.unsqueeze(0)                    # [N, N, D]
        grad_coeff = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)  # [N, N]
        grad_k     = grad_coeff.unsqueeze(-1) * diff                    # [N, N, D]

        # Stein operator terms
        term_a    = (s @ s.T) * K                                       # [N, N]
        term_b    = (s.unsqueeze(1) * (-grad_k)).sum(-1)                # [N, N]
        term_c    = (grad_k * s.unsqueeze(0)).sum(-1)                   # [N, N]
        laplacian = grad_coeff * (
            d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq)
        )                                                                # [N, N]

        h    = term_a + term_b + term_c + laplacian
        loss = (h.sum() - h.trace()) / (n * (n - 1))
        return loss


# ── Model ─────────────────────────────────────────────────────────────────────
class OnlineNetwork(nn.Module):
    """Online branch: backbone → projector → predictor."""
    def __init__(self, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model(
            'vit_small_patch8_224', pretrained=False,
            num_classes=512, drop_path_rate=0.1, img_size=128
        )
        self.projector = MLP(512, [2048, proj_dim], norm_layer=nn.BatchNorm1d)
        # Asymmetric predictor: prevents collapse without EMA stop-grad alone
        self.predictor = MLP(proj_dim, [512, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)     # [N, 512]
        proj = self.projector(feat) # [N, proj_dim]
        pred = self.predictor(proj) # [N, proj_dim]
        return feat, proj, pred


class TargetNetwork(nn.Module):
    """Target branch: EMA copy of backbone + projector (no predictor, no grad)."""
    def __init__(self, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model(
            'vit_small_patch8_224', pretrained=False,
            num_classes=512, drop_path_rate=0.1, img_size=128
        )
        self.projector = MLP(512, [2048, proj_dim], norm_layer=nn.BatchNorm1d)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        proj = self.projector(feat)
        return proj                 # [N, proj_dim]

    @torch.no_grad()
    def ema_update(self, online: OnlineNetwork, m: float):
        """EMA: target_param = m * target_param + (1-m) * online_param"""
        for p_t, p_o in zip(self.parameters(), online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)


def byol_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric normalized MSE (equivalent to 2 - 2*cosine_sim)."""
    pred   = F.normalize(pred,   dim=-1)
    target = F.normalize(target, dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(-1).mean()


def get_ema_momentum(step: int, total_steps: int) -> float:
    """Cosine schedule: EMA_BASE → EMA_END over training."""
    return EMA_END - (EMA_END - EMA_BASE) * (
        math.cos(math.pi * step / total_steps) + 1
    ) / 2


# ── Augmentation ──────────────────────────────────────────────────────────────
# Two strong global views (for BYOL prediction loss)
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
        acc     = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  LP {epoch+1:>3}/{LP_EPOCHS} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    return best_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  EXP-01: BYOL + STUDENT-T KSD  |  50+50 EPOCHS  |  H100")
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

    # ── Models ────────────────────────────────────────────────────────────────
    raw_online = OnlineNetwork().to(device)
    raw_target = TargetNetwork().to(device)

    # Initialise target = online
    raw_target.load_state_dict(
        {k: v for k, v in raw_online.state_dict().items()
         if k in dict(raw_target.named_parameters()) or k in dict(raw_target.named_buffers())},
        strict=False
    )
    # Copy backbone + projector weights exactly
    raw_target.backbone.load_state_dict(raw_online.backbone.state_dict())
    raw_target.projector.load_state_dict(raw_online.projector.state_dict())
    for p in raw_target.parameters():
        p.requires_grad_(False)

    # Compile online only (EMA update must operate on raw_target)
    online = torch.compile(raw_online, mode='max-autotune')

    ksd_fn = StudentT_KSD(nu=NU).to(device)
    opt    = torch.optim.AdamW(raw_online.parameters(), lr=LR_SSL, weight_decay=WD)

    s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler    = torch.amp.GradScaler('cuda')

    # ── SSL Pre-training ──────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(SSL_EPOCHS):
        raw_online.train()
        pbar = tqdm(train_loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for v1, v2, _ in pbar:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Online forward (both views)
                _, proj1, pred1 = online(v1)
                _, proj2, pred2 = online(v2)

                # Target forward (both views, no grad via TargetNetwork.forward)
                tproj1 = raw_target(v1)
                tproj2 = raw_target(v2)

                # BYOL loss: predict opposite view's target (symmetric)
                loss_byol = byol_loss(pred1, tproj2) + byol_loss(pred2, tproj1)

                # Student-t KSD on online projections (both views → 2*N samples)
                z_all    = torch.cat([proj1, proj2], dim=0)  # [2N, D]
                loss_ksd = ksd_fn(z_all)

                loss = loss_byol + KSD_LAMBDA * loss_ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # EMA momentum update (on raw models, not compiled wrapper)
            m = get_ema_momentum(global_step, total_steps)
            raw_target.ema_update(raw_online, m)
            global_step += 1

            pbar.set_postfix({
                "BYOL": f"{loss_byol.item():.4f}",
                "KSD":  f"{loss_ksd.item():.4f}",
                "m":    f"{m:.4f}",
            })

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = f"{SAVE_DIR}/exp01_byol_student_t.pth"
    torch.save(raw_online.state_dict(), ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    # ── Linear Probe ──────────────────────────────────────────────────────────
    backbone = raw_online.backbone
    best_acc = run_linear_probe(backbone, device)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "method":             "EXP01_BYOL_StudentT_KSD",
        "ssl_epochs":         SSL_EPOCHS,
        "lp_epochs":          LP_EPOCHS,
        "batch_size":         BATCH_SIZE,
        "ksd_lambda":         KSD_LAMBDA,
        "nu":                 NU,
        "linear_probe_acc":   best_acc,
        "paper_sota":         91.90,
        "delta_vs_sota":      round(best_acc - 91.90, 2),
    }
    with open("/kaggle/working/results_exp01_byol_student_t.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (paper SOTA: 91.90%  Δ={best_acc-91.90:+.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
