import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import MLP
import timm
from tqdm import tqdm

# --- Optimization ---
torch.set_float32_matmul_precision('high')

class SIGReg(nn.Module):
    def __init__(self, knots=17, proj_dim=128):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class LeJEPA_Official(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=512, drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.proj(emb)

class MultiCropFolder(datasets.ImageFolder):
    def __init__(self, root, global_trans, local_trans, n_local=6):
        super().__init__(root)
        self.global_trans = global_trans
        self.local_trans = local_trans
        self.n_local = n_local

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        views = [self.global_trans(img) for _ in range(2)] + [self.local_trans(img) for _ in range(self.n_local)]
        return torch.stack(views), target

def main():
    print("\n" + "🚀 REPRODUCE: LEJEPA (GAUSSIAN BASELINE) | 50+50 EPOCHS".center(60, "="))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = 'd:/MachineLearning/Ker-JEPA/imagenette'
    
    # --- 1. SETTINGS ---
    ssl_epochs = 50
    lp_epochs = 50
    batch_size = 32
    lr_ssl = 5e-4
    lamb = 0.02
    
    # --- 2. DATA AUGMENTATION ---
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
    
    train_loader = DataLoader(MultiCropFolder(os.path.join(data_path, 'train'), global_trans, local_trans), 
                              batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # --- STAGE 1: PRE-TRAINING ---
    model = LeJEPA_Official().to(device)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to(device) # Online probe
    sigreg = SIGReg().to(device)
    
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr_ssl, "weight_decay": 0.05},
        {"params": probe.parameters(), "lr": 1e-3}
    ])
    
    # Warmup + Cosine Scheduler
    warmup_steps = 5 * len(train_loader)
    total_steps = ssl_epochs * len(train_loader)
    s1 = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    scaler = torch.amp.GradScaler('cuda')
    
    for epoch in range(ssl_epochs):
        model.train(); probe.train()
        pbar = tqdm(train_loader, desc=f"SSL Ep {epoch+1}/{ssl_epochs}")
        for views, y in pbar:
            n, v, c, h, w = views.shape
            views = views.view(-1, c, h, w).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                emb, proj = model(views)
                proj = proj.reshape(n, v, -1).transpose(0, 1) # [V, N, D]
                inv_loss = (proj - proj.mean(0, keepdim=True)).square().mean()
                reg_loss = sigreg(proj)
                ssl_loss = reg_loss * lamb + inv_loss * (1 - lamb)
                
                # Online Probe
                y_rep = y.repeat_interleave(v)
                logits = probe(emb.detach())
                probe_loss = F.cross_entropy(logits, y_rep)
                loss = ssl_loss + probe_loss
            
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            scheduler.step()
            acc = (logits.argmax(1) == y_rep).float().mean() * 100
            pbar.set_postfix({"Loss": f"{ssl_loss.item():.4f}", "Acc": f"{acc.item():.2f}%"})

    save_path = "models/reproduce_lejepa_latest.pth"
    torch.save(model.state_dict(), save_path)
    
    # --- STAGE 2: LINEAR PROBE (TRANSFER) ---
    print("\n" + "🧪 STAGE 2: FINAL LINEAR PROBE (50 EPOCHS)".center(60, "-"))
    # Standard Eval Augmentation
    eval_trans = transforms.Compose([
        transforms.Resize(146), transforms.CenterCrop(128), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(data_path, 'val'), eval_trans), 
                            batch_size=128, shuffle=False, num_workers=4)
    lp_train_loader = DataLoader(datasets.ImageFolder(os.path.join(data_path, 'train'), eval_trans), 
                                 batch_size=256, shuffle=True, num_workers=4)
    
    # Freeze backbone
    for p in model.backbone.parameters(): p.requires_grad = False
    model.backbone.eval()
    
    final_head = nn.Linear(512, 10).to(device)
    lp_opt = torch.optim.AdamW(final_head.parameters(), lr=1e-2, weight_decay=0)
    
    best_acc = 0.0
    for epoch in range(lp_epochs):
        final_head.train()
        for x, y in lp_train_loader:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad(): feat = model.backbone(x)
                logits = final_head(feat)
                loss = F.cross_entropy(logits, y)
            lp_opt.zero_grad(); loss.backward(); lp_opt.step()
            
        final_head.eval(); correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (final_head(model.backbone(x)).argmax(1) == y).sum().item(); total += y.size(0)
        acc = 100 * correct / total
        best_acc = max(best_acc, acc)
        if (epoch+1) % 10 == 0 or epoch == 0: print(f"  LP Ep {epoch+1}/{lp_epochs} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    print("\n" + "═"*60)
    print(f"🎯 FINAL REPORT: {best_acc:.2f}%")
    print("═"*60)

if __name__ == "__main__":
    main()
