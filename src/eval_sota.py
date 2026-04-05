import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import timm
# Add src to path for loader import
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
from loader import get_imagenette_loaders

# --- H100 Optimization ---
torch.set_float32_matmul_precision('high')

class LinearProbe(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.classifier(feat)

def evaluate_checkpoint(ckpt_path, device="cuda"):
    print(f"\n🔍 EVALUATING: {os.path.basename(ckpt_path)}")
    
    # 1. Setup Data
    data_path = r"D:\MachineLearning\Ker-JEPA\imagenette"
    train_loader, val_loader = get_imagenette_loaders(data_path, batch_size=256, img_size=128, num_workers=4)
    
    # 2. Setup Model
    backbone = timm.create_model('vit_small_patch8_224', pretrained=False, num_classes=0, img_size=128)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Handle both raw encoder and KerJEPA wrapper (including torch.compile prefixes)
    new_state_dict = {}
    for k, v in state_dict.items():
        # Strip torch.compile prefix _orig_mod.
        clean_k = k.replace('_orig_mod.', '')
        
        if clean_k.startswith('encoder.'):
            new_state_dict[clean_k.replace('encoder.', '')] = v
        elif clean_k.startswith('backbone.'):
            new_state_dict[clean_k.replace('backbone.', '')] = v
        elif not any(clean_k.startswith(p) for p in ['predictor.', 'target_encoder.', 'proj.']):
            new_state_dict[clean_k] = v
            
    backbone.load_state_dict(new_state_dict, strict=False)
    model = LinearProbe(backbone, num_classes=10).to(device)
    # model = torch.compile(model) # Disabled to avoid Triton Mismatch on local env
    
    # 3. Standard Linear Probe Protocol (100 Epochs — matching exp code)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=0.01, weight_decay=0)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(100):
        # Training loop (only linear head)
        model.classifier.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 20 == 0:
            print(f"  > Ep {epoch+1}/100 | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
            
    return best_acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print(f"❌ Error: {models_dir} directory not found.")
        return

    ckpt_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not ckpt_files:
        print("❌ No .pth checkpoints found in models/")
        return

    results = []
    print(f"🚀 Found {len(ckpt_files)} checkpoints. Starting Master Evaluation...")

    for ckpt in ckpt_files:
        full_path = os.path.join(models_dir, ckpt)
        acc = evaluate_checkpoint(full_path, device)
        results.append((ckpt, acc))

    # --- Sort results by accuracy (descending) ---
    results.sort(key=lambda x: x[1], reverse=True)

    # --- Print Final Report Table ---
    print("\n" + "="*60)
    print("📋 KER-JEPA MASTER EVALUATION REPORT (STAGE 2)")
    print("="*60)
    print(f"{'Checkpoint Name':<45} | {'Top-1 Acc (%)':<12}")
    print("-" * 60)
    
    paper_sota = 91.90
    for ckpt, acc in results:
        delta = acc - paper_sota
        delta_str = f"({'+' if delta >=0 else ''}{delta:.2f})"
        print(f"{ckpt:<45} | {acc:>11.2f}% {delta_str}")
    
    print("="*60)
    print(f"Target Baseline (Paper): {paper_sota}%")
    print("="*60)

    # --- Save to Markdown ---
    from datetime import datetime
    md_path = os.path.join(os.path.dirname(__file__), "..", "docs", "eval_results.md")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    
    best_name, best_acc = results[0]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📋 Ker-JEPA Master Evaluation Report\n\n")
        f.write(f"> **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"> **Protocol**: Linear Probe (frozen backbone, 100 epochs, AdamW lr=0.01)  \n")
        f.write(f"> **Dataset**: Imagenette (128×128)  \n")
        f.write(f"> **Backbone**: ViT-S/8  \n\n")
        f.write("## Results (Sorted by Top-1 Accuracy)\n\n")
        f.write("| Rank | Checkpoint | Top-1 Acc (%) | Δ vs SOTA |\n")
        f.write("|:----:|:-----------|:-------------:|:---------:|\n")
        for i, (ckpt, acc) in enumerate(results):
            delta = acc - paper_sota
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            name = ckpt.replace('.pth', '').replace('reproduce_', '')
            f.write(f"| {i+1} | `{name}` | **{acc:.2f}** | {delta_str} |\n")
        f.write(f"\n## Baseline\n\n")
        f.write(f"- **Paper SOTA Target**: {paper_sota}%\n")
        f.write(f"- **Best Result**: `{best_name}` → **{best_acc:.2f}%**\n")
    
    print(f"\n✅ Results saved to: {os.path.abspath(md_path)}")

if __name__ == "__main__":
    main()
