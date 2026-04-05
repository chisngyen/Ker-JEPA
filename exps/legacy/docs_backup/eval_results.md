# 📋 Ker-JEPA Master Evaluation Report

> **Generated**: 2026-04-05 20:34:58  
> **Protocol**: Linear Probe (frozen backbone, 25 epochs, AdamW lr=0.01)  
> **Dataset**: Imagenette (128×128)  
> **Backbone**: ViT-S/8  

## Results (Sorted by Top-1 Accuracy)

| Rank | Checkpoint | Top-1 Acc (%) | Δ vs SOTA |
|:----:|:-----------|:-------------:|:---------:|
| 1 | `KSD_Sliced_Analytic_Gaussian_ep30` | **42.37** | -49.53 |
| 2 | `KSD_Unsliced_Gaussian_Gaussian_ep30` | **41.15** | -50.75 |
| 3 | `KSD_Unsliced_Gaussian_Laplace_ep30` | **33.07** | -58.83 |
| 4 | `KSD_Unsliced_IMQ_Laplace_ep30` | **30.83** | -61.07 |
| 5 | `KSD_Sliced_Finite_Laplace_ep30` | **29.17** | -62.73 |
| 6 | `lejepa_ep30` | **28.64** | -63.26 |
| 7 | `KSD_Sliced_Finite_Gaussian_ep30` | **27.77** | -64.13 |
| 8 | `sota_climb_ep30` | **25.32** | -66.58 |
| 9 | `KSD_Unsliced_IMQ_Gaussian_ep30` | **14.88** | -77.02 |

## Baseline

- **Paper SOTA Target**: 91.9%
- **Best Result**: `reproduce_KSD_Sliced_Analytic_Gaussian_ep30.pth` → **42.37%**
