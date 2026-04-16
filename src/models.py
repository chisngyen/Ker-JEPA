import torch
import torch.nn as nn
import timm

class VICReg_ST(nn.Module):
    """
    Ker-JEPA architecture using a ViT backbone and a projector head.
    Standardized for VICReg + Student-t KSD training.
    """
    def __init__(self, model_name='vit_small_patch8_224', proj_dim=512, img_size=128):
        super().__init__()
        # Backbone: Vision Transformer (ViT-S/8)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=False,
            num_classes=512, 
            drop_path_rate=0.1, 
            img_size=img_size
        )
        self.embed_dim = 512 # ViT-S/8 embedding dimension
        
        # Projector: 3-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, proj_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projector(feat)
        return feat, z
