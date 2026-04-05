import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# --- H100 Optimization Strategy ---
torch.set_float32_matmul_precision('high')

# Try to load HF Kernels for H100
try:
    from kernels import get_kernel
    flash_attn_kernel = get_kernel("kernels-community/flash-attn", version=1)
    HAS_HF_KERNELS = True
    print("[INFO] H100 Optimized Kernels loaded successfully.")
except ImportError:
    HAS_HF_KERNELS = False
    print("[WARN] HF Kernels not found. Falling back to native Flash Attention.")

# --- 1. Core Mathematical Implementation (KSD + Student-t) ---

class KerJEPALoss(nn.Module):
    def __init__(self, prior_type='gaussian', sigma=1.0, nu=3.0, beta=0.5):
        super().__init__()
        self.prior_type = prior_type
        self.sigma = sigma
        self.nu = nu # Degrees of freedom for Student-t
        self.beta = beta # IMQ Kernel parameter

    def score_function(self, x):
        """Calculates s_Q(x) = grad_x log Q(x)"""
        d = x.shape[-1]
        if self.prior_type == 'gaussian':
            # sQ(x) = -x / sigma^2
            return -x / (self.sigma ** 2)
        elif self.prior_type == 'student-t':
            # sQ(x) = -((nu + d) / (nu * sigma^2 + ||x||^2)) * x
            norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
            coeff = (self.nu + d) / (self.nu * (self.sigma**2) + norm_sq)
            return -coeff * x
        else:
            raise ValueError(f"Unknown prior: {self.prior_type}")

    def imq_kernel(self, x, y, alpha):
        """k(x, y) = (1 + alpha * ||x-y||^2)^-beta"""
        dist_sq = torch.sum((x.unsqueeze(1) - y.unsqueeze(0))**2, dim=-1)
        return (1 + alpha * dist_sq)**(-self.beta)

    def forward(self, z):
        """Unsliced KSD for a batch of embeddings z [N, D]"""
        n, d = z.shape
        
        # 1. Median trick for bandwidth alpha
        with torch.no_grad():
            full_dist_sq = torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=-1)
            median_sq = torch.median(full_dist_sq)
            alpha = 1.0 / (median_sq + 1e-6)

        # 2. Compute KSD Terms (U-statistic estimator)
        # kstein(x, y) = sQ(x)T k(x,y) sQ(y) + sQ(x)T grad_y k(x,y) + grad_x k(x,y)T sQ(y) + Laplacian k(x,y)
        
        s = self.score_function(z) # [N, D]
        K = self.imq_kernel(z, z, alpha) # [N, N]
        
        # Term A: sQ(x)T k(x,y) sQ(y)
        term_a = (s @ s.T) * K
        
        # Term B & C: Gradients (Simplified for radial kernel)
        diff = z.unsqueeze(1) - z.unsqueeze(0) # [N, N, D]
        # grad_x k(x,y) = -2 * alpha * beta * (x-y) * (1 + alpha||x-y||^2)^(-beta-1)
        grad_coeff = -2 * alpha * self.beta * (1 + alpha * full_dist_sq)**(-self.beta - 1)
        grad_k = grad_coeff.unsqueeze(-1) * diff # [N, N, D]
        
        term_b = torch.sum(s.unsqueeze(1) * (-grad_k), dim=-1) # s(x)T grad_y k
        term_c = torch.sum(grad_k * s.unsqueeze(0), dim=-1) # grad_x k T s(y)
        
        # Term D: Trace of Hessian (Laplacian)
        # Derived for IMQ: 2*alpha*beta*(1+alpha||r||^2)^(-beta-1) * [2*alpha*(beta+1)||r||^2/(1+alpha||r||^2) - d]
        laplacian = -2 * alpha * self.beta * (1 + alpha * full_dist_sq)**(-self.beta - 1) * (
            d - 2 * alpha * (self.beta + 1) * full_dist_sq / (1 + alpha * full_dist_sq)
        )
        
        k_stein = term_a + term_b + term_c + laplacian
        
        # Average over off-diagonal elements for U-statistic
        loss = (torch.sum(k_stein) - torch.trace(k_stein)) / (n * (n - 1))
        return loss

# --- 2. Model Architecture ---

class KerJEPA(nn.Module):
    def __init__(self, model_name='vit_small_patch8_224', output_dim=128):
        super().__init__()
        # Encoder
        self.encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=128)
        embed_dim = self.encoder.embed_dim
        
        # Predictor MLP
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, output_dim)
        )
        
        # Target Encoder (EMA)
        self.target_encoder = timm.create_model(model_name, pretrained=False, num_classes=0, img_size=128)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self, m=0.996):
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(m).add_(p_online.data, alpha=1 - m)

    def forward(self, views):
        # views: [4, Batch, 3, H, W] for KerJEPA 4-view setup
        v_online = views[0]
        v_targets = views[1:]
        
        # Online prediction
        z_online = self.encoder(v_online)
        pred = self.predictor(z_online)
        
        # Target representations
        with torch.no_grad():
            targets = [self.target_encoder(v) for v in v_targets]
            targets = torch.stack(targets).mean(dim=0) # Average target views
            
        return pred, targets

# --- 3. Dataset & Loading (4-Views) ---

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, n_views=4):
        self.base_dataset = base_dataset
        self.n_views = n_views
        
    def __getitem__(self, index):
        img, label = self.base_dataset[index]
        # Assuming base_dataset returns a single transformed image
        # In a real setup, we apply transformation n times here
        # For simplicity in this self-contained script, we rely on the loader's transform
        # but to get DIFFERENT views, we need to call the transform multiple times.
        # This implementation assumes img is already the pilot, we need others.
        return img, label

    def __len__(self):
        return len(self.base_dataset)

def get_transforms(img_size=128):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class FourViewFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        views = [self.transform(sample) for _ in range(4)]
        return torch.stack(views), target

# --- 4. Training Engine ---

def train():
    # Hyperparams from REPRODUCTION_CONFIG.md
    IMG_SIZE = 128
    BATCH_SIZE = 256
    EPOCHS = 800
    LR = 0.0005
    WD = 0.05
    PRIOR = 'gaussian' # Set to 'student-t' to climb!
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    data_path = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
    if not os.path.exists(data_path):
        print(f"[ERROR] Data path {data_path} not found. Please download ImageNette first.")
        return

    transform = get_transforms(IMG_SIZE)
    train_dataset = FourViewFolder(os.path.join(data_path, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=2)

    # Initialize Model & Loss
    model = KerJEPA(output_dim=128).to(device)
    ksd_loss_fn = KerJEPALoss(prior_type=PRIOR).to(device)
    
    # torch.compile for H100
    raw_model = model
    model = torch.compile(model, mode='max-autotune')

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    scaler = torch.amp.GradScaler('cuda', enabled=True) # BF16/Mixed Precision

    print(f"Starting Training: {PRIOR} Prior on H100...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for views, _ in pbar:
            views = views.transpose(0, 1).to(device, non_blocking=True) # [4, B, 3, H, W]
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred, target = model(views)
                
                # JEPA Alignment Loss (MSE)
                align_loss = F.mse_loss(pred, target)
                
                # KerJEPA Regularization (KSD)
                reg_loss = ksd_loss_fn(pred)
                
                loss = align_loss + 1.0 * reg_loss # Lambda = 1.0
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            raw_model.update_target(m=0.996)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "align": f"{align_loss.item():.4f}"})
            
        scheduler.step()
        
        # Simple checkpoint
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"kerjepa_{PRIOR}_epoch{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    train()
