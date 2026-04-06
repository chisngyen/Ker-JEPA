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
│   │       ├── exp01_byol_student_t.py  # E1: BYOL EMA + KSD-ST [FAILED 47%]
│   │       ├── exp02_vicreg_student_t.py# E2: VICReg + KSD-ST [74.11% ★ best]
│   │       ├── exp03_byol_fixed.py      # E3: BYOL deepcopy fix [41% dead end]
│   │       ├── exp04_simclr_student_t.py# E4: SimCLR + KSD-ST [71.67%]
│   │       ├── exp05_vicreg_multicrop.py# E5: VICReg + multicrop + KSD-ST
│   │       └── exp06_dino_student_t.py  # E6: DINO + KSD-ST [high priority]
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

## Experiment Insights (Running Log)

From Phase 2 runs so far:

| # | Insight | Source |
|:--|:--------|:-------|
| 1 | **`copy.deepcopy` mandatory for EMA init** — `load_state_dict(strict=False)` leaves target random → BYOL collapse (47%) | exp01 |
| 2 | **Non-contrastive (BYOL, VICReg) underfit at 50 epochs** — gradient signal/step too sparse | exp01, exp02 |
| 3 | **VICReg does not collapse** — LP ep1=69.5% proves features are reasonable, just undertrained | exp02 |
| 4 | **At 50ep budget: contrastive > non-contrastive** — SimCLR sees O(N²) pairs/step; batch=256 → 510 negatives | theory + exp01/02 |

---

## Known Bugs Fixed

1. **EMA target init with `load_state_dict(strict=False)`** — does NOT copy all weights correctly. Always use `copy.deepcopy(online_model)` for target init. Fixed in `exp03_byol_fixed.py`.

2. **`transforms.Solarize` does not exist on torchvision < 0.11** — use `PIL.ImageOps.solarize` via Lambda:
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

### exp01 — BYOL + Student-t KSD [FAILED — target init bug, 47.21%]
### exp02 — VICReg + Student-t KSD [74.11% — underfitting at 50ep]
### exp03 — BYOL Fixed + Student-t KSD [FAILED — 41.04%]
### exp04 — SimCLR + Student-t KSD [71.67%]
### exp05 — VICReg + Multi-Crop + Student-t KSD [PENDING]

```
Build on exp02 (best 74.11%): add 2 global + 4 local crops
Global-global: full VICReg + KSD-ST
Global-local: invariance only (9× more signal, same epoch count)
λ_mc=0.5, λ_ksd=0.05
```

### exp06 — DINO + Student-t KSD [PENDING — high priority]

```
ViT-native self-distillation: teacher (EMA deepcopy) sharpens output τ_t=0.04,
student learns τ_s=0.1. Multi-crop: student(all) vs teacher(global).
Centering prevents trivial collapse. KSD-ST on backbone feats λ=0.03.
Expected: highest accuracy of all Phase 2 experiments.
```

### Shared Protocol (exp_ scripts)

| Param | Value |
|-------|-------|
| SSL epochs | 50 |
| LP epochs | 50 |
| batch_size | 128 (exp01-03) / 256 (exp04 SimCLR) |
| Augmentation | Full paper pipeline (blur + solarize) |
| LR warmup | 5 epochs linear |
| LR decay | Cosine |
| BF16 | Yes |
| compile | max-autotune |
| Data path | `/kaggle/input/datasets/aniladepu/imagenette/imagenette` |

---

## Environment Notes

- Platform: H100 GPU (Kaggle / remote)
- `torch.set_float32_matmul_precision('high')` — set at top of every script
- `torch.compile(model, mode='max-autotune')` — used for H100 speed
- `torch.cuda.amp.GradScaler` + `autocast(dtype=torch.bfloat16)`
