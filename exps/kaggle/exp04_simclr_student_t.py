# =============================================================================
# exp04_simclr_student_t.py
# SimCLR (NT-Xent contrastive) + Student-t KSD
#
# WHY SIMCLR WINS AT 50 EPOCHS vs non-contrastive methods:
#   - Each step SimCLR sees N*(N-1) negative pairs — gradient signal is
#     O(N²) richer than BYOL/VICReg which compare only 2 views.
#   - Hard negatives: within-batch, no memory bank needed.
#   - Converges in 50-100 epochs for small datasets (well-documented).
#   - Student-t KSD regularizer adds distribution shaping on top,
#     preventing embedding space from degenerating to a hypersphere
#     shell with low entropy.
#
# Key hyperparams:
#   - batch=256 (more negatives = better SimCLR; 255 negatives per sample)
#   - temperature τ=0.07 (standard SimCLR; lower = harder negatives)
#   - proj head: 512→2048→128 (standard SimCLR design)
#   - KSD λ=0.05 on the L2-normalized projections
#
# TARGET: >92.5% in 50 SSL + 50 LP epochs
# =============================================================================

import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _pkg in ["timm", "tqdm"]:
    try: __import__(_pkg)
    except ImportError: _install(_pkg)

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

DATA_PATH = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
SAVE_DIR  = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

SSL_EPOCHS  = 50
LP_EPOCHS   = 50
BATCH_SIZE  = 256       # bigger batch = more negatives for SimCLR
LR_SSL      = 5e-4
WD          = 0.05
TEMPERATURE = 0.07      # standard SimCLR; lower = harder negatives
PROJ_DIM    = 128       # standard SimCLR proj head output
KSD_LAMBDA  = 0.05
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
        term_a    = (s @ s.T) * K
        term_b    = (s.unsqueeze(1) * (-grad_k)).sum(-1)
        term_c    = (grad_k * s.unsqueeze(0)).sum(-1)
        laplacian = gc * (d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq))
        h         = term_a + term_b + term_c + laplacian
        return (h.sum() - h.trace()) / (n * (n - 1))


# ── NT-Xent (SimCLR) Loss ─────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    NT-Xent: InfoNCE over 2N samples (N positive pairs, 2N-2 negatives each).
    z1, z2: [N, D] — already L2 normalized.
    """
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.tau = temperature

    def forward(self, z1, z2):
        n = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                 # [2N, D]
        sim = (z @ z.T) / self.tau                      # [2N, 2N]

        # Mask out self-similarity on diagonal
        mask = torch.eye(2 * n, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(n, 2 * n, device=z.device),
            torch.arange(0, n,     device=z.device),
        ])                                              # [2N]
        return F.cross_entropy(sim, labels)


# ── Model ─────────────────────────────────────────────────────────────────────
class SimCLR_ST(nn.Module):
    """Backbone + SimCLR projection head (no predictor, no EMA)."""
    def __init__(self, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone  = timm.create_model('vit_small_patch8_224', pretrained=False,
                                           num_classes=512, drop_path_rate=0.1, img_size=128)
        # Standard SimCLR head: linear → BN → ReLU → linear
        self.projector = MLP(512, [2048, proj_dim], norm_layer=nn.BatchNorm1d,
                             activation_layer=nn.ReLU)

    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        z    = F.normalize(proj, dim=-1)    # L2 norm for cosine similarity
        return feat, z


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
    print("  EXP-04: SimCLR + STUDENT-T KSD  |  50+50  |  H100")
    print(f"  batch={BATCH_SIZE}  τ={TEMPERATURE}  λ_ksd={KSD_LAMBDA}  ν={NU}")
    print("=" * 60)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug    = build_aug()
    loader = DataLoader(TwoViewFolder(f"{DATA_PATH}/train", aug),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    total_steps  = SSL_EPOCHS * len(loader)
    warmup_steps = WARMUP_EP * len(loader)

    raw_model = SimCLR_ST().to(device)
    model     = torch.compile(raw_model, mode='max-autotune')
    ntxent    = NTXentLoss(temperature=TEMPERATURE)
    ksd_fn    = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_model.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler    = torch.amp.GradScaler('cuda')

    for epoch in range(SSL_EPOCHS):
        raw_model.train()
        pbar = tqdm(loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for v1, v2, _ in pbar:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, z1 = model(v1)
                _, z2 = model(v2)

                # NT-Xent: (N=256) → 510 negatives per sample
                loss_cl  = ntxent(z1, z2)

                # KSD on 2N normalized embeddings: shapes hypersphere distribution
                loss_ksd = ksd_fn(torch.cat([z1, z2], dim=0))

                loss = loss_cl + KSD_LAMBDA * loss_ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); scheduler.step()

            pbar.set_postfix({"NT-Xent": f"{loss_cl.item():.3f}",
                               "KSD":     f"{loss_ksd.item():.3f}"})

    ckpt = f"{SAVE_DIR}/exp04_simclr_student_t.pth"
    torch.save(raw_model.state_dict(), ckpt)
    print(f"\n  Checkpoint: {ckpt}")

    best_acc = run_lp(raw_model.backbone, device)

    report = {"method": "EXP04_SimCLR_StudentT_KSD", "ssl_epochs": SSL_EPOCHS,
              "batch_size": BATCH_SIZE, "temperature": TEMPERATURE,
              "ksd_lambda": KSD_LAMBDA, "nu": NU,
              "linear_probe_acc": best_acc, "paper_sota": 91.90,
              "delta": round(best_acc - 91.90, 2)}
    with open("/kaggle/working/results_exp04_simclr_student_t.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (Δ={best_acc-91.90:+.2f}% vs paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
