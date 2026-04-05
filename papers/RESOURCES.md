# Official Research Resources & Paper Links

Below are the **100% verified** official paper links and benchmark data sources for your research.

## 1. Research Papers (2025-2026 SOTA)

| Method | Paper Title | arXiv Link |
| :--- | :--- | :--- |
| **DINOv3** | DINOv3: Universal Vision Encoders at Scale | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104) |
| **V-JEPA 2.1** | V-JEPA 2.1: Unlocking Dense Features in Video SSL | [arXiv:2603.14482](https://arxiv.org/abs/2603.14482) |
| **LeJEPA** | LeJEPA: Provable & Scalable SSL Without Heuristics | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) |
| **KerJEPA** | KerJEPA: Kernel Discrepancies for Euclidean SSL | [arXiv:2512.19605](https://arxiv.org/abs/2512.19605) |
| **I-JEPA** | SSL from Images with a Joint-Embedding Architecture | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| **DINOv2** | DINOv2: Robust Visual Features without Supervision | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |

## 2. Benchmark Datasets (Direct Download Links)

### **ImageNette (The Main Competition)**
Used by KerJEPA (91.9%) and LeJEPA (91.13%).
- **Official Repo**: [fastai/imagenette](https://github.com/fastai/imagenette)
- **Direct Download (160px)**: [imagenette2-160.tgz](https://s3.amazonaws.com/fast-ai-imageclassification/imagenette2-160.tgz)
- **Direct Download (Full Resolution)**: [imagenette2.tgz](https://s3.amazonaws.com/fast-ai-imageclassification/imagenette2.tgz)

### **ImageNet-1K (Gold Standard)**
- **Official**: [image-net.org/download](http://www.image-net.org/download) (Requires registration)
- **Kaggle Alternative**: [Kaggle ImageNet Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)
- **DINO/JEPA Script**: Most papers use the `timm` or `torchvision` loaders for ImageNet-1K.

### **VTAB (Visual Task Adaptation Benchmark)**
Used for transfer learning evaluation.
- **Official Repo**: [google-research/task_adaptation](https://github.com/google-research/task_adaptation)

---
*Note: Use these links to ensure you are comparing against the most recent and official versions of both data and literature.*
