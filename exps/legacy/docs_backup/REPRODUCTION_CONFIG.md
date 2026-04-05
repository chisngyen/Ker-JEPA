# KerJEPA Reproduction Configuration (Microsoft Research)

To ensure a fair 100% score, start by reproducing the baseline exactly as stated in the paper.

## 1. Architecture
- **Backbone**: Vision Transformer Small (Patch 8, `vit_small_patch8_224` tailored for 128px).
- **Projector**: 3-layer MLP.
  - Hidden sizes: [2048, 2048, 128].
  - Output dimension: 128.
- **Target Encoder**: EMA update of backbone weights.
  - Momentum: **0.996 → 1.0 (Cosine Schedule over 800 epochs)**.

## 2. Dataset & Augmentation
- **Target**: ImageNette (10 classes).
- **Input Resolution**: 128 x 128 px.
- **Views**: 4 views per instance (standard SSL often uses 2, KerJEPA uses 4).
- **Augmentation Pipeline**:
  - `RandomResizedCrop(128, scale=(0.2, 1.0))`
  - `RandomHorizontalFlip()`
  - `ColorJitter(0.4, 0.4, 0.4, 0.1)`
  - `RandomGrayscale(p=0.2)`
  - `GaussianBlur(p=0.1)`
  - `Solarization(p=0.1)`

## 3. Training Hyperparameters
- **Epochs**: 800.
- **Batch Size**: 256.
- **Optimizer**: AdamW.
- **Learning Rate**: 
  - Peak LR: 0.0005.
  - Warmup: 40 epochs.
  - Schedule: Cosine Annealing.
- **Weight Decay**: 0.05.
- **Precision**: `torch.bfloat16` (BF16).

## 4. Regularizer (The 91.90% SOTA Config)
- **Method**: Unsliced KSD.
- **Base Kernel**: Inverse Multiquadric (IMQ).
  - Formula: $k(x, y) = (1 + \frac{\|x-y\|^2}{\beta^2})^{-0.5}$
  - $\beta$: Median of pairwise distances (Median Trick).
- **Regularization Weight ($\lambda$):** **0.1** (Verified from Appendix F, Table 7).
- **Target Prior**: Isotropic Gaussian ($\mathcal{N}(0, \sigma^2 I_d)$).
  - $\sigma$: 1.0.

## 5. Evaluation Protocol
- **Method**: Online Linear Probe (Instance Normalized).
- **Final Validation**: Freeze backbone, train 100-epoch linear head.
- **Expected Accuracy**: **91.90% ± 0.44**.

---
**Next Step**: Swap Gaussian Prior for **Student-t** ($\nu=3$) to surpass 92%.
