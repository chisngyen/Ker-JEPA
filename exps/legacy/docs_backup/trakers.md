# KerJEPA Experiment Tracker (2026)

This file tracks the real-time progress and results of all experiments conducted to surpass the KerJEPA SOTA.

## 🚀 Live Experiments

| Date | ID | Algorithm | Prior | Dataset | Epochs | Final Loss | HW | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04-05 | **S1-A** | KerJEPA (Unsliced) | Student-t (ν=3) | ImageNette (128) | 30/30 | **2.0856** | H100 | ✅ Completed |
| 2026-04-05 | **B1-R** | KerJEPA (Official) | Gaussian | ImageNette (128) | - | - | H100 | ⏳ Pending |

---

## 🔬 Experiment Details: S1-A (Student-t Innovation)

**Configuration:**
- **Script**: `exps/my-methods/exp01_student_t_ksd.py`
- **Backbone**: ViT-Small (patch8)
- **Input Size**: 128x128
- **KSD Kernel**: IMQ (β=0.5)
- **Optimization**: BF16 + Flash Attention 2 + `torch.compile(mode='max-autotune')`

**Training Metrics:**
- **Progression**:
    - Epoch 01: 4.4555
    - Epoch 10: 2.7307
    - Epoch 20: 2.2983
    - Epoch 30: **2.0856**
- **Speed**: ~2.5s / iteration (Batch Size: 256)
- **Total Samples**: 1.2M (estimated)

**Inductor Autotune Stats (H100):**
- **Best Convolution Kernel**: `triton_convolution2d_1` (0.2811 ms)
- **Best AddMM Kernel**: `bias_addmm` (0.1075 ms)
- **Best MM Kernel**: `mm` (0.0441 ms)

---

## 📈 Benchmarking Results (Linear Probe)

*Linear probing is required to verify the ACC (%) against the target >92.5%.*

| Experiment ID | Algorithm | Prior | ACC (%) | Target (%) | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B2 (Paper)** | KerJEPA | Gaussian | 91.90 | 91.90 | 0.00 |
| **S1-A (Current)** | KerJEPA | Student-t | *TBD* | **92.50** | *TBD* |

---
> [!TIP]
> The Student-t loss is decreasing steadily. Loss < 2.1 at Epoch 30 is a strong indicator of convergence for a 128x128 training run. Next step: Run linear evaluation.
