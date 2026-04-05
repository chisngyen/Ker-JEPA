import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from scipy.special import roots_hermite

# --- H100 Optimization ---
torch.set_float32_matmul_precision('high')

# Try to load HF Kernels for H100
try:
    from kernels import get_kernel
    flash_attn_kernel = get_kernel("kernels-community/flash-attn", version=1)
    HAS_HF_KERNELS = True
except ImportError:
    HAS_HF_KERNELS = False

class MMDLoss(nn.Module):
    """Unsliced MMD Loss (Gaussian)"""
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, z):
        n, d = z.shape
        dist_sq = torch.sum((z.unsqueeze(1) - z.unsqueeze(0))**2, dim=-1)
        gamma = 1.0 / (2 * self.sigma**2)
        
        # MMD^2(P, N) = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
        # Term A: E[exp(-gamma ||x-x'||^2)]
        term_a = torch.exp(-gamma * dist_sq).mean()
        
        # Term B: Expected kernel under Gaussian Prior (Prop 10)
        coeff_b = 1.0 / (1 + 2 * gamma * self.sigma**2)**(d/2)
        term_b = coeff_b * torch.exp(-gamma / (1 + 2 * gamma * self.sigma**2) * torch.sum(z**2, dim=-1)).mean()
        
        # Term C: Constant (Prop 10)
        term_c = 1.0 / (1 + 4 * gamma * self.sigma**2)**(d/2)
        
        loss = term_a - 2 * term_b + term_c
        return loss

class SlicedMMDLoss(nn.Module):
    """Sliced MMD based on Figure 1 of Paper"""
    def __init__(self, mode='finite', prior_type='gaussian', n_slices=1024, n_knots=21, sigma=1.0):
        super().__init__()
        self.mode = mode
        self.prior_type = prior_type
        self.n_slices = n_slices
        self.sigma = sigma
        
        knots, weights = roots_hermite(n_knots)
        self.register_buffer('knots', torch.tensor(knots * math.sqrt(2)).view(1, 1, -1).float())
        self.register_buffer('weights', torch.tensor(weights / math.sqrt(math.pi)).view(1, 1, -1).float())

    def target_cf(self, w):
        if self.prior_type == 'gaussian':
            return torch.exp(-0.5 * (self.sigma**2) * w**2)
        elif self.prior_type == 'laplace':
            return 1.0 / (1.0 + (self.sigma**2) * w**2)
        return torch.ones_like(w)

    def forward(self, z):
        if self.mode == 'analytic':
            return MMDLoss(sigma=self.sigma).forward(z)
            
        n, d = z.shape
        noise = torch.randn(self.n_slices, d, device=z.device)
        theta = F.normalize(noise, dim=1)
        proj = (z @ theta.T).unsqueeze(-1) # [Batch, Slices, 1]
        
        args = proj * self.knots
        emp_cf = torch.mean(torch.cos(args), dim=0) # [Slices, Knots]
        tgt_cf = self.target_cf(self.knots).squeeze() # [Knots]
        
        err = (emp_cf - tgt_cf).pow(2)
        smmd = torch.sum(err * self.weights, dim=-1).mean()
        return smmd

# --- Boilerplate Model/Data/Train (Self-Contained) ---

class KerJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0, img_size=128)
        self.predictor = nn.Sequential(nn.Linear(self.encoder.embed_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(), nn.Linear(2048, self.encoder.embed_dim))
        self.target_encoder = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0, img_size=128)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
    def update_target(self, epoch, max_epochs, m_start=0.996, m_end=1.0):
        # Cosine schedule for target momentum
        m = m_end - (m_end - m_start) * (math.cos(math.pi * epoch / max_epochs) + 1) / 2
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(m).add_(p_online.data, alpha=1 - m)

    def forward(self, views):
        v_online = views[0]
        v_targets = views[1:]
        z_online = self.encoder(v_online)
        pred = self.predictor(z_online)
        with torch.no_grad():
            targets = torch.stack([self.target_encoder(v) for v in v_targets]).mean(dim=0)
        return pred, targets

class FourViewFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        views = [self.transform(sample) for _ in range(4)]
        return torch.stack(views), target

def run_experiment(config):
    print(f"\n>>> Running MMD Experiment: {config['name']}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = '/kaggle/input/datasets/aniladepu/imagenette/imagenette'
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.RandomApply([transforms.Lambda(lambda img: __import__('PIL').ImageOps.solarize(img, 128))], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = DataLoader(FourViewFolder(os.path.join(data_path, 'train'), transform=transform), 
                              batch_size=256, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    
    raw_model = KerJEPA().to(device)
    model = torch.compile(raw_model)
    loss_fn = SlicedMMDLoss(mode=config['mode'], prior_type=config['prior']).to(device) if config['type'] == 'sliced' else MMDLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(30):
        model.train()
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        for views, _ in pbar:
            views = views.transpose(0, 1).to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred, target = model(views)
                # Lambda = 0.1 for fair comparison with KSD table (Table 1/7)
                loss = F.mse_loss(pred, target) + 0.1 * loss_fn(pred)
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            pbar.set_postfix({"L": f"{loss.item():.3f}"})
        
        # Save Model at end
        os.makedirs("models", exist_ok=True)
        save_path = f"models/reproduce_{config['name']}_ep{epoch+1}.pth"
        torch.save(raw_model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    configs = [
        {"name": "MMD_Unsliced_Gaussian", "type": "unsliced", "prior": "gaussian"},
        {"name": "MMD_Sliced_Analytic_Gaussian", "type": "sliced", "mode": "analytic", "prior": "gaussian"},
        {"name": "MMD_Sliced_Finite_Laplace", "type": "sliced", "mode": "finite", "prior": "laplace"},
    ]
    for cfg in configs: run_experiment(cfg)
