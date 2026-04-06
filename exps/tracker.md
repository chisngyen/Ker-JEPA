# KerJEPA Experiment Tracker

> Protocol: ViT-S/8 @ 128×128, Kaggle H100, BF16, AdamW.
> Linear Probe: frozen backbone, 50 epochs, AdamW lr=1e-2.
> **Target: >92.5% in ≤50 SSL epochs** (paper: 91.90% @ 800 epochs)

---

## Phase 1 — Reproduction (kag_ scripts)

| ID | File | SSL Loss | LP Acc (%) | Status |
|:--:|:-----|:---------|:----------:|:------:|
| B1 | `kag_lejepa_baseline.py`   | SIGReg              | —  | Pending |
| B2 | `kag_kerjepa_gauss.py`     | KSD-Gauss           | —  | Pending |
| B3 | `kag_kerjepa_student_t.py` | KSD-ST (no EMA)     | —  | Pending |
| B4 | `kag_kerjepa_laplace.py`   | KSD-Laplace         | —  | Pending |
| B5 | `kag_kerjepa_rbf.py`       | KSD-RBF             | —  | Pending |
| B6 | `kag_kerjepa_sliced.py`    | SKSD-Finite         | —  | Pending |
| B7 | `kag_kerjepa_analytic.py`  | SKSD-Analytic       | —  | Pending |

**Paper SOTA (800 epochs):** 91.90%

---

## Phase 2 — Innovation Experiments (exp_ scripts, 50 SSL epochs)

### Lessons learned (running log)

| # | Insight | Source |
|:--|:--------|:-------|
| 1 | **`copy.deepcopy(online)` for EMA target init** — `load_state_dict(strict=False)` can leave the teacher partly random; always deep-copy the online net for BYOL-style targets | exp01 → exp03 fix |
| 2 | **BYOL-style objectives are the wrong tool at 50 SSL epochs here** — even with correct EMA (**41.04%**), far behind VICReg/SimCLR; pairwise teacher signal is too weak per step for this budget | exp01, exp03 |
| 3 | **VICReg does NOT collapse** — LP ep1 ≈ 69.5% shows usable features immediately; bottleneck is **training length**, not representation death | exp02 curve |
| 4 | **Student-t KSD is compatible with VICReg at λ used** — no collapse; final LP **74.11%** sets the Phase 2 bar (ablation vs λ=0 not run) | exp02 |
| 5 | **Contrastive (SimCLR) ≫ BYOL, but does not beat VICReg at 50ep** — SimCLR **71.67%** vs BYOL fixed **41%** (many negatives vs one EMA pair); still **−2.44 pp** vs VICReg — invariance + variance/covariance (VICReg) fit this ViT-S/8 + aug stack better than NT-Xent + KSD here | exp03, exp04 |
| 6 | **Broken EMA init can look “better” than correct BYOL** — random teacher exp01 **47.21%** > fixed exp03 **41.04%**; not a reason to keep the bug — metrics are not monotonic in “correctness” under severe undertraining | exp01 vs exp03 |
| 7 | **Innovation target (>92.5% @ 50 SSL) not met** — best **74.11%** is **−17.79 pp** vs paper 800-ep KerJEPA (91.90%); gap is mostly **epoch budget + objective** (not Student-t KSD alone) | E2 vs paper |

---

### Phase 2 synthesis (E1–E4, 50 SSL + 50 LP)

**Best checkpoint (leader):** `/kaggle/working/models/exp02_vicreg_student_t.pth` — **74.11%** LP (VICReg + Student-t KSD).

| Rank | ID | Method | LP (%) | Δ vs 91.90% |
|:---:|:--:|:-------|:------:|:-----------:|
| 1 | E2 | VICReg + KSD-ST | **74.11** | −17.79 |
| 2 | E4 | SimCLR + KSD-ST | 71.67 | −20.23 |
| 3 | E1 | BYOL + KSD-ST (bug) | 47.21 | −44.69 |
| 4 | E3 | BYOL fixed + KSD-ST | 41.04 | −50.86 |

**Takeaways**

1. **For the next run, start from exp02** (weights + recipe), not SimCLR/BYOL — highest LP with frozen probe under the shared protocol.
2. **To approach paper numbers**, extend SSL epochs (800-scale), use the **full KerJEPA / predictor–target** setup from Phase 1 `kag_*` scripts, and treat Phase 2 as a **quick ablation lane**, not a fair SOTA match.
3. **BYOL at 50ep is a dead end** in this setup; fixing EMA did not help — prioritize **more steps** or **richer loss** (contrastive or VICReg-style), not BYOL tweaks alone.
4. **SimCLR + KSD** is viable but **not automatically better than VICReg**; if the course story is “Student-t KSD + best SSL,” **VICReg + KSD-ST** is the current evidence-backed choice.

---

### exp01 — BYOL + Student-t KSD (broken target init)

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| Run 1 | **47.21%** | -44.69 | Collapse — `load_state_dict(strict=False)` left target random |

**Status:** FAILED (bug). Replaced by exp03.

---

### exp02 — VICReg + Student-t KSD

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| Run 1 | **74.11%** | -17.79 | Working, not collapsed. Underfitting — 50ep too few for VICReg. Checkpoint: `/kaggle/working/models/exp02_vicreg_student_t.pth` |

**Status:** **Phase 2 leader.** Contrastive (exp04) did not beat this at 50ep; >92.5% target unmet — scale SSL epochs and/or align with full KerJEPA pipeline.

---

### exp03 — BYOL Fixed + Student-t KSD

**File:** `exp03_byol_fixed.py`

Key fix: `copy.deepcopy(online)` for target. EMA base: 0.996. Proj dim: 128.

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| Run 1 | **41.04%** | -50.86 | Best during LP (final ep 39.52%). Checkpoint: `/kaggle/working/models/exp03_byol_fixed.pth` |

**Status:** Complete. Correct EMA init; still worst among serious runs — BYOL + 50ep underfits vs VICReg/SimCLR.

---

### exp04 — SimCLR + Student-t KSD

**File:** `exp04_simclr_student_t.py`

**Hypothesis (outcome):** NT-Xent + large batch beats BYOL at 50ep but **does not beat VICReg** (71.67% vs 74.11%) — hypothesis partially supported.

| Param | Value |
|:------|:------|
| batch | 256 (510 negatives per sample) |
| temperature τ | 0.07 |
| KSD λ | 0.05 |
| ν (Student-t) | 3.0 |
| Proj head | 512→2048→128 (SimCLR standard) |

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| Run 1 | **71.67%** | -20.23 | Best during LP (final ep 70.55%). Checkpoint: `/kaggle/working/models/exp04_simclr_student_t.pth` |

**Status:** Complete @ 50 SSL + 50 LP.

---

---

### exp05 — VICReg + Multi-Crop + Student-t KSD

**File:** `exp05_vicreg_multicrop.py`

**Why:** exp02 (VICReg) is Phase 2 leader at 74.11%. Multi-crop (2 global + 4 local)
gives 9× more cross-scale training signal per step without adding epochs.

| Param | Value |
|:------|:------|
| Crops | 2 global (scale 0.4–1.0) + 4 local (scale 0.05–0.4) |
| Loss | VICReg(global-global) + 0.5·inv(global-local) + 0.05·KSD-ST |
| Proj dim | 512 (same as exp02) |

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| — | — | — | Pending |

---

### exp06 — DINO + Student-t KSD

**File:** `exp06_dino_student_t.py`

**Why:** DINO is ViT-native — attention maps localize objects even at 50 epochs.
Teacher sharpening (τ_t=0.04) + centering prevents collapse differently from
VICReg. Multi-crop local→global distillation is the most signal-rich
SSL objective available at this budget.

| Param | Value |
|:------|:------|
| Crops | 2 global + 4 local |
| τ_student | 0.1 |
| τ_teacher | 0.04 |
| EMA momentum | 0.996→1.0 cosine |
| KSD λ | 0.03 (on backbone feats, not DINO head) |
| DINO out_dim | 256 |

| Run | LP Acc | Δ | Notes |
|:----|:------:|:-:|:------|
| — | — | — | Pending |

---

## Master Summary

| ID | Method | LP Acc (%) | Δ vs 91.90% | Epochs |
|:--:|:-------|:----------:|:-----------:|:------:|
| E1 | BYOL + KSD-ST (exp01, broken) | 47.21 | -44.69 | 50 |
| E2 | **VICReg + KSD-ST (exp02) ★ best** | **74.11** | **-17.79** | 50 |
| E3 | BYOL Fixed + KSD-ST (exp03) | 41.04 | -50.86 | 50 |
| E4 | SimCLR + KSD-ST (exp04) | 71.67 | -20.23 | 50 |
| E5 | **VICReg + MultiCrop + KSD-ST (exp05)** | — | — | 50 |
| E6 | **DINO + KSD-ST (exp06)** | — | — | 50 |
| —  | **Paper SOTA** | **91.90** | **0.00** | **800** |
| —  | **Innovation Target** | **>92.50** | **>+0.60** | **50** |
