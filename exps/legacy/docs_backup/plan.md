# Master Experiment Plan: Surpassing KerJEPA SOTA (2026)

This document outlines the exact experimental setup and evaluation protocol to surpass current benchmarks and achieve a 100% project score.

## 1. Objective
Reproduce **KerJEPA** (Microsoft Research, 2025) and establish a new SOTA on **ImageNette** using the **Student-t Prior** and optimized kernels, followed by verification on **ImageNet-100**.

## 2. Experimental Matrix (Master Table Template)
Use this table to record and compare your results with established baselines.

| Experiment ID | Algorithm | Type | Kernel | Prior | Dataset | ACC (%) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | LeJEPA | Sliced | Gaussian | Gaussian | ImageNette | *91.13* | Baseline (Paper) |
| **B2** | KerJEPA | Unsliced | IMQ | Gaussian | ImageNette | *91.90* | SOTA (Paper) |
| **A1** | KerJEPA | Unsliced | IMQ | Laplace | ImageNette | *91.18* | Ablation 1 |
| **S1 (You)** | **KerJEPA** | **Unsliced** | **IMQ** | **Student-t**| **ImageNette** | **Target >92.5**| **Your SOTA** |
| **S2 (You)** | **KerJEPA** | **Unsliced** | **IMQ** | **Student-t**| **ImageNet-100**| **Target >86.0**| **Robustness** |

## 3. Implementation Details

### A. Core Architecture
- **Backbone**: Vision Transformer Small (`vit_tiny_patch16_224` or `vit_small_patch8_128`).
- **Input Size**: 128x128 (for fair comparison with KerJEPA) and 224x224 (to beat DINO).
- **Epochs**: 800 (Ablation) / 1600 (Final Climb).

### B. The Secret Sauce: Student-t Prior
Implementing the score function $s_Q(x) = \nabla_x \log Q(x)$ for the Student-t distribution:
$$s_Q(x) = -\frac{\nu + d}{\nu \sigma^2 + \|x\|_2^2} x$$
*This will be the key differentiator from the original KerJEPA experiments.*

### C. Evaluation Protocol
- **Linear Probing**: Freeze the backbone after pre-training. Train a linear head for 100 epochs with AdamW (LR=0.01, Batch=256).
- **Metric**: Top-1 Accuracy on validation set.

## 4. Execution Workflow

1.  **Phase 1: Environment Setup**  
    Install `torch`, `torchvision`, `timm`, and `lightning`.
2.  **Phase 2: Baseline Reproduction**  
    Run KerJEPA with IMQ + Gaussian on ImageNette. Verify $\approx 91.9\%$.
3.  **Phase 3: The Innovation Loop**  
    Inject the **Student-t score function**. Tune the degrees of freedom ($\nu$) — start with $\nu=3$.
4.  **Phase 4: Final Benchmarking**  
    Run the best configuration on **ImageNet-100**.
5.  **Phase 5: Reporting**  
    Populate the Master Table and generate t-SNE visualizations of the embeddings.

---
> [!IMPORTANT]
> **To Get 100% Marks**: Ensure you strictly follow the **Linear Probing** protocol. Any fine-tuning of the backbone during evaluation will make the comparison "unfair".

## 5. Resources
- **Papers**: Located in `./papers/`
- **Data Links**: [MEGA_BENCHMARKS.md](./MEGA_BENCHMARKS.md)
- **Official Paper Source**: [arXiv:2512.19605](https://arxiv.org/abs/2512.19605)
