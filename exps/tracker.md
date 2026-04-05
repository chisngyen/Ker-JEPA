# KerJEPA Performance Tracker: 50+50 Unified Protocol

This tracker logs the results of the 7 core experiments designed to reproduce the KerJEPA paper and validate the **Student-t Innovation**.

| Config ID | Description | Prior | Kernel | SSL Loss | Linear Probe Acc (%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | LeJEPA Official | Gaussian | - | SIGReg | | [ ] Pending |
| **B2** | KerJEPA SOTA (2025) | Gaussian | IMQ | KSD-Gauss | | [ ] Pending |
| **B3** | **Student-t (Innovation)**| **Student-t**| **IMQ** | **KSD-ST**| | [ ] Pending |
| **B4** | Laplace Ablation | Laplace | IMQ | KSD-Lap | | [ ] Pending |
| **B5** | Kernel Ablation | Gaussian | RBF | KSD-RBF | | [ ] Pending |
| **B6** | Sliced (Finite) | Gaussian | IMQ | SKSD-Fin | | [ ] Pending |
| **B7** | Sliced (Analytic) | Gaussian | IMQ | SKSD-Ana | | [ ] Pending |

## Protocol Summary
- **SSL**: ViT-Small/8, 50 Epochs, 8-view Multi-crop, AdamW 5e-4.
- **LP**: 50 Epochs, Frozen Backbone, 1-view 128x128.
- **Batch Size**: 32 (SSL), 256 (LP).
- **Target Platform**: Kaggle H100 (Linux).
