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
├── src/
│   ├── eval_sota.py                     # Master linear-probe evaluator (loads any .pth)
│   └── loader.py                        # Data loading utilities
├── exps/
│   ├── kaggle/                          # All Kaggle H100 scripts (self-contained, pretrain+LP)
│   │   │
│   │   ├── ── Phase 1: Reproduction (kag_ scripts) ──────────────────────────
│   │   ├── kag_lejepa_baseline.py       # B1: LeJEPA SIGReg baseline
│   │   ├── kag_kerjepa_gauss.py         # B2: KerJEPA Gaussian KSD (paper spirit)
│   │   ├── kag_kerjepa_student_t.py     # B3: Student-t KSD (ablation, no EMA)
│   │   ├── kag_kerjepa_laplace.py       # B4: Laplace prior ablation
│   │   ├── kag_kerjepa_rbf.py           # B5: RBF kernel ablation
│   │   ├── kag_kerjepa_sliced.py        # B6: Sliced KSD finite
│   │   ├── kag_kerjepa_analytic.py      # B7: Sliced KSD analytic
│   │   │
│   │   └── ── Phase 2: Innovation (exp_ scripts, aim >92.5%) ────────────────
│   │       ├── exp01_byol_student_t.py  # E1: BYOL EMA + Student-t KSD [MAIN]
│   │       └── exp02_vicreg_student_t.py# E2: VICReg + Student-t KSD
│   │
│   ├── legacy/                          # Old scripts (pre-Kaggle refactor)
│   └── tracker.md                       # Experiment results tracker
└── papers/
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

## Innovation Experiment Architecture (exp_ scripts)

### exp01 — BYOL + Student-t KSD

```
Online: backbone → projector(512→2048→256) → predictor(256→512→256)
Target: backbone (EMA) → projector(512→2048→256)   [no grad, no predictor]

Loss = BYOL_cosine(pred1, target2.detach()) + BYOL_cosine(pred2, target1.detach())
     + 0.05 * StudentT_KSD(cat[proj1, proj2])

EMA momentum: cosine schedule 0.99 → 1.0 over total SSL steps
Key: EMA update on raw_target (NOT compiled wrapper)
```

### exp02 — VICReg + Student-t KSD

```
Model: backbone → expander(512→2048→2048→512)

Loss = 25*inv + 25*var + 1*cov          ← standard VICReg
     + 0.05 * StudentT_KSD(cat[z1, z2]) ← distribution shaping

No EMA — variance/covariance terms prevent collapse directly.
```

### Shared Protocol (exp_ scripts)

| Param | Value |
|-------|-------|
| batch_size | 128 (vs kag_ 32) |
| Augmentation | Full paper pipeline (blur + solarize) |
| LR warmup | 5 epochs linear |
| LR decay | Cosine |
| BF16 | Yes |
| compile | max-autotune |

---

## Environment Notes

- Platform: H100 GPU (Kaggle / remote)
- `torch.set_float32_matmul_precision('high')` — set at top of every script
- `torch.compile(model, mode='max-autotune')` — used for H100 speed
- `torch.cuda.amp.GradScaler` + `autocast(dtype=torch.bfloat16)`
