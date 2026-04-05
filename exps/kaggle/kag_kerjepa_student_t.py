# === SETUP (Auto-install dependencies) ===
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages = ["timm", "tqdm"]
for pkg in packages:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# === IMPORTS ===
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import MLP
import timm
from tqdm import tqdm

# === H100 OPTIMIZATIONS ===
torch.set_float32_matmul_precision('high')

# === CONFIG ===
DATA_PATH = '/kaggle/input/imagenette/imagenette'
SAVE_DIR = '/kaggle/working/models'
os.makedirs(SAVE_DIR, exist_ok=True)

# === LOSS: STUDENT-T KSD (Your Innovation) ===
class StudentT_KSDLoss(nn.Module):
    def __init__(self, sigma=1.0, nu=3.0, beta=0.5):
        super().__init__()
        self.sigma = sigma
        self.nu = nu # Degrees of freedom
        self.beta = beta # IMQ parameter

    def forward(self, z):
        n, d = z.shape
        with torch.no_grad():
            dist_sq = torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=-1)
            median_dist = torch.median(dist_sq)
            alpha = 1.0 / (median_dist + 1e-6)

        # 1. Student-t Score Function: - (nu + d) * z / (nu * sigma^2 + ||z||^2)
        norm_sq = torch.sum(z**2, dim=-1, keepdim=True)
        coeff_s = (self.nu + d) / (self.nu * (self.sigma**2) + norm_sq)
        s = -coeff_s * z 

        # 2. IMQ Kernel & Gradients
        K = (1 + alpha * dist_sq)**(-self.beta)
        diff = z.unsqueeze(1) - z.unsqueeze(0) 
        grad_coeff = -2 * alpha * self.beta * (1 + alpha * dist_sq)**(-self.beta - 1)
        grad_k = grad_coeff.unsqueeze(-1) * diff 

        # 3. KSD Terms
        term_a = (s @ s.T) * K
        term_b = torch.sum(s.unsqueeze(1) * (-grad_k), dim=-1)
        term_c = torch.sum(grad_k * s.unsqueeze(0), dim=-1)
        laplacian = grad_coeff * (d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq))

        k_stein = term_a + term_b + term_c + laplacian
        loss = (torch.sum(k_stein) - torch.trace(k_stein)) / (n * (n - 1))
        return loss

# === MODEL CORE ===
class KerJEPA_Kaggle(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=512, drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.proj(emb)

# === DATA LOADER (8 VIEWS) ===
class MultiCropFolder(datasets.ImageFolder):
    def __init__(self, root, global_trans, local_trans, n_local=6):
        super().__init__(root, None)
        self.global_trans = global_trans
        self.local_trans = local_trans
        self.n_local = n_local

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        views = [self.global_trans(img) for _ in range(2)] + [self.local_trans(img) for _ in range(self.n_local)]
        return torch.stack(views), target

# === MAIN LOGIC ===
def main():
    print("\n" + "🚀 KAG_B3: KERJEPA STUDENT-T INNOVATION | 50+50 EPOCHS | H100".center(60, "="))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. SETTINGS
    ssl_epochs = 50
    lp_epochs = 50
    batch_size = 32
    lr_ssl = 5e-4
    lamb = 0.02
    
    # 2. DATA
    global_trans = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    local_trans = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_loader = DataLoader(MultiCropFolder(os.path.join(DATA_PATH, 'train'), global_trans, local_trans), 
                              batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    # 3. STAGE 1: PRE-TRAINING
    model = KerJEPA_Kaggle().to(device)
    loss_fn = StudentT_KSDLoss(nu=3.0).to(device)
    
    # H100 Compile
    model = torch.compile(model)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr_ssl, weight_decay=0.05)
    
    # Schedulers
    warmup_steps = 5 * len(train_loader)
    total_steps = ssl_epochs * len(train_loader)
    s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(ssl_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"SSL Ep {epoch+1}/{ssl_epochs}")
        for views, _ in pbar:
            n, v, c, h, w = views.shape
            views = views.view(-1, c, h, w).to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, proj = model(views)
                proj = proj.reshape(n, v, -1).transpose(0, 1) # [V, N, D]
                inv_loss = (proj - proj.mean(0, keepdim=True)).square().mean()
                reg_loss = loss_fn(proj.mean(0))
                ssl_loss = reg_loss * lamb + inv_loss * (1 - lamb)
            
            opt.zero_grad(); scaler.scale(ssl_loss).backward(); scaler.step(opt); scaler.update()
            scheduler.step()
            pbar.set_postfix({"Loss": f"{ssl_loss.item():.4f}"})

    torch.save(model.state_dict(), f"{SAVE_DIR}/kag_kerjepa_student_t.pth")
    
    # 4. STAGE 2: FINAL LINEAR PROBE
    print("\n" + "🧪 STAGE 2: FINAL LINEAR PROBE (50 EPOCHS)".center(60, "-"))
    eval_trans = transforms.Compose([
        transforms.Resize(146), transforms.CenterCrop(128), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    lp_train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), eval_trans), 
                                 batch_size=256, shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), eval_trans), 
                            batch_size=128, shuffle=False, num_workers=8)
    
    backbone = model._orig_mod.backbone if hasattr(model, '_orig_mod') else model.backbone
    for p in backbone.parameters(): p.requires_grad = False
    backbone.eval()
    
    final_head = nn.Linear(512, 10).to(device)
    lp_opt = torch.optim.AdamW(final_head.parameters(), lr=1e-2, weight_decay=0)
    
    best_acc = 0.0
    for epoch in range(lp_epochs):
        final_head.train()
        for x, y in lp_train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                feat = backbone(x)
                logits = final_head(feat)
                loss = F.cross_entropy(logits, y)
            lp_opt.zero_grad(); loss.backward(); lp_opt.step()
            
        final_head.eval(); correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (final_head(backbone(x)).argmax(1) == y).sum().item(); total += y.size(0)
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch+1) % 10 == 0 or epoch == 0: print(f"  LP Ep {epoch+1}/{lp_epochs} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    report = {"method": "KerJEPA_StudentT_Innovation", "pretraining_epochs": ssl_epochs, "linear_probe_acc": best_acc}
    with open("/kaggle/working/results_kerjepa_student_t.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "═"*60)
    print(f"🎯 FINAL STUDENT-T REPORT: {best_acc:.2f}%")
    print("═"*60)

if __name__ == "__main__":
    main()
