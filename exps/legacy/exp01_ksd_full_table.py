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

class KSDLoss(nn.Module):
    """Unified KSD Loss for Table reproduction"""
    def __init__(self, kernel_type='imq', prior_type='gaussian', sigma=1.0, beta=0.5):
        super().__init__()
        self.kernel_type = kernel_type
        self.prior_type = prior_type
        self.sigma = sigma
        self.beta = beta

    def forward(self, z):
        n, d = z.shape
        with torch.no_grad():
            dist_sq = torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=-1)
            median_sq = torch.median(dist_sq)
            alpha = 1.0 / (median_sq + 1e-6)

        # Score function
        s = -z / (self.sigma ** 2) if self.prior_type == 'gaussian' else -torch.sign(z)
        
        # IMQ Kernel
        K = (1 + alpha * dist_sq)**(-self.beta)
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        grad_coeff = -2 * alpha * self.beta * (1 + alpha * dist_sq)**(-self.beta - 1)
        grad_k = grad_coeff.unsqueeze(-1) * diff
        
        term_a = (s @ s.T) * K
        term_b = torch.sum(s.unsqueeze(1) * (-grad_k), dim=-1)
        term_c = torch.sum(grad_k * s.unsqueeze(0), dim=-1)
        laplacian = grad_coeff * (d - 2 * alpha * (self.beta + 1) * dist_sq / (1 + alpha * dist_sq))

        k_stein = term_a + term_b + term_c + laplacian
        return (torch.sum(k_stein) - torch.trace(k_stein)) / (n * (n - 1))

class LeJEPA_Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=512, drop_path_rate=0.1, img_size=128)
        self.proj = MLP(512, [2048, 2048, 128], norm_layer=nn.BatchNorm1d)
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
        return torch.stack(views) , target

def main():
    print("\n" + "🚀 REPRODUCE: KERJEPA (GAUSSIAN KSD TABLE 1) | 50+50 EPOCHS".center(60, "="))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = 'd:/MachineLearning/Ker-JEPA/imagenette'
    
    # --- DATA ---
    global_trans = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.4, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    local_trans = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.05, 0.4)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(MultiCropFolder(os.path.join(data_path, 'train'), global_trans, local_trans), 
                              batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # --- STAGE 1 ---
    model = LeJEPA_Base().to(device)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to(device)
    loss_fn = KSDLoss().to(device)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": 5e-4, "weight_decay": 0.05}, {"params": probe.parameters(), "lr": 1e-3}])
    
    warmup = 5 * len(train_loader); total = 50 * len(train_loader)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [torch.optim.lr_scheduler.LinearLR(opt, 0.01, total_iters=warmup), 
                                                           torch.optim.lr_scheduler.CosineAnnealingLR(opt, total-warmup)], [warmup])
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(50):
        model.train(); probe.train()
        pbar = tqdm(train_loader, desc=f"SSL Ep {epoch+1}/50")
        for views, y in pbar:
            n, v, c, h, w = views.shape
            views = views.view(-1, c, h, w).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                emb, proj = model(views)
                proj = proj.reshape(n, v, -1).transpose(0, 1)
                ssl_loss = loss_fn(proj.mean(0)) * 0.02 + (proj - proj.mean(0, keepdim=True)).square().mean() * 0.98
                y_rep = y.repeat_interleave(v).to(device)
                probe_loss = F.cross_entropy(probe(emb.detach()), y_rep)
                loss = ssl_loss + probe_loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); scheduler.step()
            pbar.set_postfix({"Loss": f"{ssl_loss.item():.4f}", "Acc": f"{(probe(emb.detach()).argmax(1)==y_rep).float().mean()*100:.1f}%"})

    save_path = "models/reproduce_ksd_table_latest.pth"
    torch.save(model.state_dict(), save_path)

    # --- STAGE 2 ---
    print("\n" + "🧪 STAGE 2: FINAL LINEAR PROBE".center(60, "-"))
    eval_trans = transforms.Compose([transforms.Resize(146), transforms.CenterCrop(128), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    lp_train = DataLoader(datasets.ImageFolder(os.path.join(data_path, 'train'), eval_trans), batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(data_path, 'val'), eval_trans), batch_size=128, shuffle=False, num_workers=4)
    
    for p in model.backbone.parameters(): p.requires_grad = False
    model.backbone.eval(); final_head = nn.Linear(512, 10).to(device)
    lp_opt = torch.optim.AdamW(final_head.parameters(), lr=1e-2)
    
    best_acc = 0.0
    for epoch in range(50):
        final_head.train()
        for x, y in lp_train:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(final_head(model.backbone(x)), y)
            lp_opt.zero_grad(); loss.backward(); lp_opt.step()
        final_head.eval(); correct = 0; total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (final_head(model.backbone(x)).argmax(1) == y).sum().item(); total += y.size(0)
        acc = 100 * correct / total; best_acc = max(best_acc, acc)
        if (epoch+1) % 10 == 0: print(f"  LP Ep {epoch+1}/50 | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
    print(f"\n🎯 FINAL KSD TABLE REPORT: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
