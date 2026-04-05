# Dataset Configuration & Metadata

This document summarizes the understanding of the datasets to be used for the "Climbing the Bench" project.

## 1. ImageNet-100 (`wilyzh/imagenet100`)
- **Structure**: 
  - `ImageNet100/train/`: Class-specific directories (`nxxxx`).
  - `ImageNet100/val/`: Class-specific directories (`nxxxx`).
- **Nature**: A subset of ImageNet-1K with 100 classes. Standard for verifying robustness and scaling.
- **Loading Strategy**: `torchvision.datasets.ImageFolder` pointed at the `train/` and `val/` sub-directories.

## 2. ImageNette (`aniladepu/imagenette`)
- **Structure**: 
  - `imagenette/imagenette/train/`: Class-specific directories (`n01440764`, etc.).
  - `imagenette/imagenette/val/`: Class-specific directories (`n01440764`, etc.).
- **Noisy Label Data**: 
  - `train_noisy_imagenette.csv` & `val_noisy_imagenette.csv`.
  - **Columns**: `path`, `noisy_labels_0` (clean), `noisy_labels_1`, `noisy_labels_5`, `noisy_labels_25`, `noisy_labels_50`, `is_valid`.
- **Nature**: 10 distinct classes of ImageNet. Used for rapid algorithmic validation (KerJEPA baseline).
- **Loading Strategy**:
  - **Standard**: `ImageFolder` using the `imagenette/imagenette/` path.
  - **Robustness Testing**: `pandas` + custom Dataset class to utilize the `noisy_labels_X` columns from the CSV files.

## 3. Pre-processing Requirements
- **Normalization**: Standard ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
- **Resolution**: 
  - Baseline (Fair): 128x128.
  - SOTA-Beating: 224x224.
- **Augmentation**: RandomResizedCrop (scale 0.2-1.0) and HorizontalFlip.

---
**Status**: All dataset structures verified. Ready for model implementation.
