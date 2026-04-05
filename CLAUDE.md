# KerJEPA Project — CLAUDE.md

## Project Overview

Course project: reproduce **KerJEPA** (Microsoft Research, arXiv:2512.19605) and surpass SOTA on ImageNette by swapping the Gaussian prior with a **Student-t prior** in the KSD regularizer.

- **Baseline target**: KerJEPA IMQ + Gaussian = 91.90% (reproduce first)
- **Innovation target**: KerJEPA IMQ + Student-t (ν=3) > 92.5%
- **Robustness target**: ImageNet-100 > 86.0%

Paper + references are in `./papers/`.

---

## File Structure

```
Ker-JEPA/
├── src/                                 # Pre-training and utility scripts
│   ├── train_h100.py                    # Main training script (H100, full config)
│   └── loader.py                        # Data loading utilities
├── docs/                                # Project documentation & plans
│   ├── plan.md                          # Master experiment plan
│   ├── REPRODUCTION_CONFIG.md           # Exact hyperparams from paper
│   ├── MEGA_BENCHMARKS.md               # Dataset links & evaluation targets
│   ├── datasets.md                      # Dataset handling details
│   └── trakers.md                       # Experiment tracking
├── exps/                                # Specialized experiment scripts
│   ├── reproduce/                       # Reproduction of paper results
│   │   ├── exp01_ksd_full_table.py      # Reproduces Table 1: all KSD variants
│   │   ├── exp02_mmd_variants.py        # Reproduces MMD variants
│   │   ├── exp03_lejepa_official.py     # Reproduces LeJEPA baseline (91.13%)
│   │   └── run_reproduce_combo.py       # Runs all 3 reproduction scripts sequentially
│   └── my-methods/                      # Novel research variants
│       └── exp01_student_t_ksd.py       # Innovation: Student-t prior KSD
└── papers/                              # Research papers
    ├── KerJEPA_2512.19605.pdf
    ├── LeJEPA.pdf
    ├── I-JEPA.pdf
    ├── DINOv2.pdf / DINOv3.pdf
    ├── V-JEPA_2_1.pdf
    └── RESOURCES.md
```

---

## Architecture

- **Backbone**: `vit_small_patch8_224` (via timm), input 128×128
- **Predictor**: MLP [embed_dim → 2048 → BN → GELU → 128]
- **Target encoder**: EMA copy of backbone, momentum cosine schedule 0.996→1.0 over 800 epochs
- **Views**: 4 views per image (FourViewFolder)

## Training Config (from paper)

| Param | Value |
|-------|-------|
| Epochs | 800 |
| Batch size | 256 |
| Optimizer | AdamW (lr=5e-4, wd=0.05) |
| Precision | BF16 |
| KSD λ | 0.1 |
| IMQ β | 0.5 |
| Bandwidth | Median trick |

## Augmentation Pipeline

```python
RandomResizedCrop(128, scale=(0.2, 1.0))
RandomHorizontalFlip()
ColorJitter(0.4, 0.4, 0.4, 0.1)
RandomGrayscale(p=0.2)
GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
RandomApply([Lambda(solarize)], p=0.1)   # use PIL.ImageOps.solarize, NOT transforms.Solarize
ToTensor()
Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
```

---

## Known Bugs Fixed

1. **`transforms.Solarize` does not exist on torchvision < 0.11** — use `PIL.ImageOps.solarize` via Lambda:
   ```python
   transforms.Lambda(lambda img: __import__('PIL').ImageOps.solarize(img, 128))
   ```
   Fixed in all 4 training files.

2. **EMA update on compiled model** — `torch.compile` wraps the model; always call `.update_target()` on the raw (unwrapped) model:
   ```python
   raw_model = KerJEPA().to(device)
   model = torch.compile(raw_model, ...)
   # ...
   raw_model.update_target(...)   # NOT model.update_target()
   ```
   Fixed in `exp01`, `exp02`, `exp03`, and `train_h100.py`.

3. **Sliced KSD imaginary ECF term** (`exp01_ksd_full_table.py`) — second term of `imag` must use `cos`, not `sin`:
   ```python
   # Correct:
   imag = s.unsqueeze(-1) * torch.sin(args) + w * torch.cos(args)
   ```

---

## Evaluation Protocol

**Linear Probe** (strictly frozen backbone):
- Freeze encoder after pre-training
- Train a linear head for 100 epochs with AdamW (lr=0.01, batch=256)
- Report Top-1 Accuracy on validation set
- Do NOT fine-tune the backbone during eval (unfair comparison)

---

## Student-t Score Function (Innovation)

$$s_Q(x) = -\frac{\nu + d}{\nu \sigma^2 + \|x\|_2^2} \cdot x$$

Start with ν=3. This gives heavier tails than Gaussian → better representation of outlier embeddings.

---

## Environment Notes

- Platform: H100 GPU (Kaggle / remote)
- `torch.set_float32_matmul_precision('high')` — set at top of every script
- `torch.compile(model, mode='max-autotune')` — used for H100 speed
- `torch.cuda.amp.GradScaler` + `autocast(dtype=torch.bfloat16)`
