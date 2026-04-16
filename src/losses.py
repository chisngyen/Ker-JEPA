import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentT_KSD(nn.Module):
    """
    Kernel Stein Discrepancy (KSD) with Student-t Kernel.
    Used to impose a heavy-tailed prior on the latent space.
    """
    def __init__(self, sigma=1.0, nu=3.0, beta=0.5):
        super().__init__()
        self.sigma = sigma
        self.nu = nu
        self.beta = beta

    def forward(self, z):
        n, d = z.shape
        with torch.no_grad():
            dist_sq = torch.cdist(z, z).pow(2)
            alpha = 1.0 / (dist_sq.median() + 1e-6)
            
        norm_sq = z.pow(2).sum(-1, keepdim=True)
        # Score function for Student-t prior
        s = -((self.nu + d) / (self.nu * self.sigma**2 + norm_sq)) * z
        
        K = (1 + alpha * dist_sq) ** (-self.beta)
        diff = z.unsqueeze(1) - z.unsqueeze(0)
        gc = -2 * alpha * self.beta * (1 + alpha * dist_sq) ** (-self.beta - 1)
        grad_k = gc.unsqueeze(-1) * diff
        
        # KSD Calculation
        h = ((s @ s.T) * K
             + (s.unsqueeze(1) * (-grad_k)).sum(-1)
             + (grad_k * s.unsqueeze(0)).sum(-1)
             + gc * (d - 2 * alpha * (self.beta + 1) * dist_sq
                     / (1 + alpha * dist_sq)))
        
        return (h.sum() - h.trace()) / (n * (n - 1))

def vicreg_loss(z1, z2, w_inv=25.0, w_var=25.0, w_cov=1.0):
    """
    Standard VICReg loss: Invariance, Variance, and Covariance.
    """
    n, d = z1.shape
    
    # SimCLR-style Invariance
    inv = F.mse_loss(z1, z2)
    
    # Variance regularization
    def var_t(z):
        return F.relu(1.0 - z.std(dim=0)).mean()
    
    # Covariance regularization
    def cov_t(z):
        zc = z - z.mean(0)
        C = (zc.T @ zc) / (n - 1)
        return (C.pow(2).sum() - C.diag().pow(2).sum()) / d
    
    loss = (w_inv * inv 
            + w_var * (var_t(z1) + var_t(z2))
            + w_cov * (cov_t(z1) + cov_t(z2)))
    
    return loss, inv.item()
