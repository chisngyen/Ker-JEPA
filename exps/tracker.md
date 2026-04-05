# KerJEPA Experiment Tracker

> Protocol: ViT-S/8 @ 128×128, Kaggle H100, BF16, AdamW.
> Linear Probe: frozen backbone, 50 epochs, AdamW lr=1e-2.

---

## Phase 1 — Reproduction (kag_ scripts, 50+50 epochs)

Baseline runs using simplified architecture (no EMA target encoder).

| ID | File | Description | Prior | Kernel | SSL Loss | LP Acc (%) | Status |
|:--:|:-----|:------------|:------|:-------|:---------|:----------:|:------:|
| B1 | `kag_lejepa_baseline.py`   | LeJEPA official baseline        | Gaussian   | —   | SIGReg          | —  | Pending |
| B2 | `kag_kerjepa_gauss.py`     | KerJEPA Gaussian KSD (paper)    | Gaussian   | IMQ | KSD-Gauss       | —  | Pending |
| B3 | `kag_kerjepa_student_t.py` | Student-t KSD (ablation)        | Student-t  | IMQ | KSD-ST          | —  | Pending |
| B4 | `kag_kerjepa_laplace.py`   | Laplace prior ablation          | Laplace    | IMQ | KSD-Laplace     | —  | Pending |
| B5 | `kag_kerjepa_rbf.py`       | RBF kernel ablation             | Gaussian   | RBF | KSD-RBF         | —  | Pending |
| B6 | `kag_kerjepa_sliced.py`    | Sliced KSD (finite)             | Gaussian   | IMQ | SKSD-Finite     | —  | Pending |
| B7 | `kag_kerjepa_analytic.py`  | Sliced KSD (analytic)           | Gaussian   | IMQ | SKSD-Analytic   | —  | Pending |

**Paper SOTA (800 epochs):** 91.90%

---

## Phase 2 — Innovation Experiments (exp_ scripts, aim to beat SOTA in ≤50 epochs)

Key improvements over Phase 1:
- `batch_size=128` (vs 32): 16k pairwise KSD interactions vs 992.
- Full augmentation pipeline (Gaussian blur + solarize).
- `LR warmup = 5 epochs` + cosine decay.

### exp01 — BYOL + Student-t KSD

**File:** `exp01_byol_student_t.py`

**Why it can win:**
- EMA momentum target encoder (BYOL-style): proper asymmetric self-supervised
  prediction signal → richer gradient per step vs. plain invariance loss.
- Student-t score function ν=3: heavy tails → no dead gradients at outlier
  embeddings → faster convergence in early epochs.
- Symmetric BYOL loss over 2 views + Student-t KSD on 2N samples.
- EMA schedule: 0.99 → 1.0 (cosine over total steps).

| Param | Value |
|:------|:------|
| SSL epochs | 50 |
| LP epochs | 50 |
| Batch size | 128 |
| Projector | 512→2048→256 (online), 512→2048→256 (target) |
| Predictor | 256→512→256 (asymmetric) |
| KSD λ | 0.05 |
| ν (Student-t) | 3.0 |
| EMA momentum | 0.99→1.0 cosine |
| Loss | BYOL_cos + 0.05·KSD_ST |

| Run | LP Acc (%) | Δ vs SOTA | Notes |
|:----|:----------:|:---------:|:------|
| — | — | — | Pending first run |

---

### exp02 — VICReg + Student-t KSD

**File:** `exp02_vicreg_student_t.py`

**Why it can win:**
- VICReg covariance loss: explicit dimensional decorrelation → prevents collapse
  from epoch 1 (no EMA needed).
- VICReg variance loss: enforces unit std per-dimension → stable training.
- Student-t KSD augments the variance term with distribution-level shaping.
- No EMA → simpler optimization → easier to get right at 50 epochs.

| Param | Value |
|:------|:------|
| SSL epochs | 50 |
| LP epochs | 50 |
| Batch size | 128 |
| Projector | 512→2048→2048→512 (expander) |
| KSD λ | 0.05 |
| VICReg weights | inv=25, var=25, cov=1 |
| ν (Student-t) | 3.0 |
| Loss | VICReg + 0.05·KSD_ST |

| Run | LP Acc (%) | Δ vs SOTA | Notes |
|:----|:----------:|:---------:|:------|
| — | — | — | Pending first run |

---

## Summary Table (fill as results come in)

| ID | Method | LP Acc (%) | Δ vs 91.90% | Epochs |
|:--:|:-------|:----------:|:-----------:|:------:|
| B2 | KerJEPA Gauss (paper spirit) | — | — | 50 |
| B3 | KerJEPA Student-t (kag_) | — | — | 50 |
| E1 | **BYOL + Student-t KSD** | — | — | 50 |
| E2 | **VICReg + Student-t KSD** | — | — | 50 |
| — | **Paper SOTA** | **91.90** | **0.00** | **800** |
| — | **Innovation Target** | **>92.50** | **>+0.60** | **50** |
