# =============================================================================
# exp06_dino_student_t.py
# DINO (Self-Distillation) + Student-t KSD
#
# WHY DINO IS THE STRONGEST CANDIDATE TO BREAK THE 91.9% BARRIER:
#   1. ViT-native: DINO was designed for ViT backbones — attention maps
#      collapse to meaningful object segments even at 50 epochs.
#   2. Knowledge distillation: teacher sharpens its output (τ_t=0.04),
#      student learns from the confident teacher → faster convergence
#      than variance-based methods.
#   3. Multi-crop: student sees 2 global + 4 local; teacher only global.
#      Local-to-global predictions give 8× more cross-scale signal/step.
#   4. Centering: running mean subtracted from teacher output prevents
#      trivial all-same-class collapse without batch statistics.
#   5. Student-t KSD on student global embeddings shapes the backbone's
#      representation space toward a heavy-tailed prior → better
#      geometry for linear probing.
#
# ARCHITECTURE:
#   Student: ViT-S/8 → DINOHead (MLP + L2 norm, out_dim=256)
#   Teacher: deepcopy(student), EMA momentum 0.996→1.0, + center buffer
#   Loss: CE(student_logit/τ_s, softmax(teacher_logit/τ_t)) summed over
#         all (student_view, teacher_global_view) cross-pairs
#       + λ_ksd * StudentT_KSD(student_global_backbone_feats)
#
# TARGET: >85% in 50 SSL + 50 LP epochs
# =============================================================================

import subprocess, sys

def _install(p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ["timm", "tqdm"]:
    try: __import__(_p)
    except ImportError: _install(_p)

import copy, os, math, json
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
N_LOCAL     = 4
OUT_DIM     = 256       # DINO prototype dimension
TAU_S       = 0.1       # student temperature (soft)
TAU_T       = 0.04      # teacher temperature (sharp)
CENTER_M    = 0.9       # center EMA momentum
EMA_BASE    = 0.996
EMA_END     = 1.0
W_KSD       = 0.03      # KSD on backbone features (not on DINO head output)
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


# ── DINO Head ─────────────────────────────────────────────────────────────────
class DINOHead(nn.Module):
    """
    3-layer MLP with L2-normalized output.
    Output is NOT a probability — temperatures applied in loss function.
    Note: weight_norm REMOVED — it creates non-leaf tensors that break
    copy.deepcopy. Replaced with orthogonal init (equivalent effect).
    """
    def __init__(self, in_dim=512, hidden=2048, out_dim=OUT_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # Plain linear — input is already L2-normalized so scale is controlled
        self.last_layer = nn.Linear(out_dim, out_dim, bias=False)
        nn.init.orthogonal_(self.last_layer.weight)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


# ── Full Student/Teacher Network ───────────────────────────────────────────────
class DINONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False,
                                          num_classes=512, drop_path_rate=0.1, img_size=128)
        self.head     = DINOHead(in_dim=512)

    def forward(self, x):
        feat = self.backbone(x)
        logit = self.head(feat)
        return feat, logit


# ── DINO Loss with Centering ───────────────────────────────────────────────────
class DINOLoss(nn.Module):
    """
    CE loss between student (all views) and teacher (global views only).
    Center is an EMA of teacher global outputs — prevents collapse.
    """
    def __init__(self, out_dim=OUT_DIM, tau_s=TAU_S, tau_t=TAU_T, center_m=CENTER_M):
        super().__init__()
        self.tau_s    = tau_s
        self.tau_t    = tau_t
        self.center_m = center_m
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_views, teacher_globals):
        """
        student_views:   list of [N, D] — all views (global + local)
        teacher_globals: list of [N, D] — global views only (no grad)
        Returns: scalar loss
        """
        # Teacher: center → sharpen
        teacher_probs = [
            F.softmax((t - self.center) / self.tau_t, dim=-1).detach()
            for t in teacher_globals
        ]

        loss  = 0.0
        count = 0
        for i, s in enumerate(student_views):
            s_log = F.log_softmax(s / self.tau_s, dim=-1)
            for j, tp in enumerate(teacher_probs):
                # Skip same-view pairs (only for global views)
                if i < len(teacher_globals) and i == j:
                    continue
                loss  += -(tp * s_log).sum(-1).mean()
                count += 1
        loss = loss / count

        # Update center: EMA of teacher global output (before softmax/centering)
        with torch.no_grad():
            batch_center = torch.cat(teacher_globals, dim=0).mean(0, keepdim=True)
            self.center  = self.center * self.center_m + batch_center * (1 - self.center_m)

        return loss


# ── Augmentations ─────────────────────────────────────────────────────────────
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
    global_aug = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.4, 1.0))] + base)
    local_aug  = transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.05, 0.4))] + base)
    return global_aug, local_aug


class MultiCropFolder(datasets.ImageFolder):
    def __init__(self, root, global_aug, local_aug, n_local=N_LOCAL):
        super().__init__(root, None)
        self.g_aug, self.l_aug, self.n_local = global_aug, local_aug, n_local

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        return (self.g_aug(img), self.g_aug(img),
                [self.l_aug(img) for _ in range(self.n_local)], label)


def mc_collate(batch):
    g1  = torch.stack([b[0] for b in batch])
    g2  = torch.stack([b[1] for b in batch])
    lcs = [torch.stack([b[2][i] for b in batch]) for i in range(len(batch[0][2]))]
    lbl = torch.tensor([b[3] for b in batch])
    return g1, g2, lcs, lbl


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


# ── EMA momentum schedule ─────────────────────────────────────────────────────
def get_ema_m(step, total):
    return EMA_END - (EMA_END - EMA_BASE) * (math.cos(math.pi * step / total) + 1) / 2


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  EXP-06: DINO + STUDENT-T KSD  |  50+50  |  H100")
    print(f"  2 global + {N_LOCAL} local crops  |  τ_s={TAU_S}  τ_t={TAU_T}  λ_ksd={W_KSD}")
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

    # ── DINO: deepcopy for correct teacher init ────────────────────────────────
    raw_student = DINONet().to(device)
    raw_teacher = copy.deepcopy(raw_student).to(device)
    for p in raw_teacher.parameters():
        p.requires_grad_(False)

    student = torch.compile(raw_student, mode='max-autotune')
    dino_criterion = DINOLoss().to(device)
    ksd_fn         = StudentT_KSD().to(device)

    opt = torch.optim.AdamW(raw_student.parameters(), lr=LR_SSL, weight_decay=WD)
    s1  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, [s1, s2], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    for epoch in range(SSL_EPOCHS):
        raw_student.train()
        pbar = tqdm(loader, desc=f"SSL {epoch+1:>2}/{SSL_EPOCHS}")
        for g1, g2, lcs, _ in pbar:
            g1  = g1.to(device, non_blocking=True)
            g2  = g2.to(device, non_blocking=True)
            lcs = [lc.to(device, non_blocking=True) for lc in lcs]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Student: all views
                sg1_feat, sg1_logit = student(g1)
                sg2_feat, sg2_logit = student(g2)
                sl_logits = [student(lc)[1] for lc in lcs]

                # Teacher: global views only, no grad
                with torch.no_grad():
                    _, tg1_logit = raw_teacher(g1)
                    _, tg2_logit = raw_teacher(g2)

                # DINO loss: student(all) vs teacher(global)
                all_student  = [sg1_logit, sg2_logit] + sl_logits
                all_teacher  = [tg1_logit, tg2_logit]
                loss_dino    = dino_criterion(all_student, all_teacher)

                # Student-t KSD on backbone features of global views
                loss_ksd = ksd_fn(torch.cat([sg1_feat, sg2_feat], dim=0))

                loss = loss_dino + W_KSD * loss_ksd

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()

            # EMA teacher update (backbone + head)
            m = get_ema_m(global_step, total_steps)
            with torch.no_grad():
                for p_t, p_s in zip(raw_teacher.parameters(), raw_student.parameters()):
                    p_t.data.mul_(m).add_(p_s.data, alpha=1 - m)
            global_step += 1

            pbar.set_postfix({"dino": f"{loss_dino.item():.4f}",
                               "ksd":  f"{loss_ksd.item():.4f}",
                               "m":    f"{m:.4f}"})

    ckpt = f"{SAVE_DIR}/exp06_dino_student_t.pth"
    torch.save(raw_student.state_dict(), ckpt)
    print(f"\n  Checkpoint: {ckpt}")

    best_acc = run_lp(raw_student.backbone, device)

    report = {"method": "EXP06_DINO_StudentT_KSD", "ssl_epochs": SSL_EPOCHS,
              "n_local": N_LOCAL, "tau_s": TAU_S, "tau_t": TAU_T,
              "w_ksd": W_KSD, "nu": NU,
              "linear_probe_acc": best_acc, "paper_sota": 91.90,
              "delta": round(best_acc - 91.90, 2)}
    with open("/kaggle/working/results_exp06_dino_student_t.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  FINAL: {best_acc:.2f}%  (Δ={best_acc-91.90:+.2f}% vs paper)")
    print("=" * 60)


if __name__ == "__main__":
    main()
