# Mega Benchmarks Reference List (SSL 2026)

This list contains all major benchmarks you need to evaluate your SSL model comprehensively to "surpass every bench".

## 1. Core Classification (Semantics)

| Dataset                | Description                 | Purpose               | Official Link                                               |
| :--------------------- | :-------------------------- | :-------------------- | :---------------------------------------------------------- |
| **ImageNet-1K**  | 1,000 classes, 1.2M images  | World Standard        | [Image-Net](http://www.image-net.org/)                         |
| **ImageNette**   | 10 classes, subset of IN-1K | Rapid Prototyping     | [fast.ai](https://github.com/fastai/imagenette)                |
| **ImageNet-100** | 100 classes subset          | SSL Proxy             | [Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100) |
| **CIFAR-10/100** | 32x32 images                | Tiny Dataset Baseline | [UToronto](https://www.cs.toronto.edu/~kriz/cifar.html)        |
| **Places365**    | 365 scene categories        | Scene Understanding   | [MIT](http://places2.csail.mit.edu/)                           |

2. Dense Prediction (Detection & Segmentation)

| Dataset                   | Description              | Purpose                | Official Link                                           |
| :------------------------ | :----------------------- | :--------------------- | :------------------------------------------------------ |
| **COCO**            | 80 object categories     | Detection/Instance Seg | [COCO](https://cocodataset.org/)                           |
| **ADE20K**          | 150 semantic categories  | Semantic Segmentation  | [MIT](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |
| **Cityscapes**      | Urban street scenes      | Semantic/Instance Seg  | [Cityscapes](https://www.cityscapes-dataset.com/)          |
| **Pascal VOC 2012** | 20 basic categories      | Detection/Segmentation | [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/)          |
| **LVIS**            | 1,200+ object categories | Long-tailed Detection  | [LVIS](https://www.lvisdataset.org/)                       |

## 3. Robustness & Out-of-Distribution (OOD)

| Dataset               | Description                  | Purpose               | Official Link                                                    |
| :-------------------- | :--------------------------- | :-------------------- | :--------------------------------------------------------------- |
| **ImageNet-V2** | New test set for IN-1K       | Generalization Check  | [GitHub](https://github.com/modestyachts/ImageNetV2)                |
| **ObjectNet**   | Real-world poses/backgrounds | Geometric Robustness  | [ObjectNet.dev](https://objectnet.dev/)                             |
| **ImageNet-A**  | Naturally adversarial images | Classifier Robustness | [GitHub](https://github.com/hendrycks/natural-adversarial-examples) |
| **ImageNet-R**  | Art, cartoons, sketches      | Distribution Shift    | [GitHub](https://github.com/hendrycks/imagenet-r)                   |
| **ImageNet-C**  | 15 types of corruption       | Corruption Robustness | [GitHub](https://github.com/hendrycks/robustness)                   |

## 4. Fine-Grained & Specialized

| Dataset                    | Description                 | Purpose             | Official Link                                                     |
| :------------------------- | :-------------------------- | :------------------ | :---------------------------------------------------------------- |
| **iNaturalist 2021** | 10,000 species              | Scale/Fine-grained  | [iNat](https://github.com/visipedia/inat_comp)                       |
| **CUB-200-2011**     | 200 bird species            | Part-based semantic | [Caltech](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |
| **Oxford-IIIT Pet**  | Dogs and Cats breeds        | Fine-grained        | [Oxford](http://www.robots.ox.ac.uk/~vgg/data/pets/)                 |
| **Food-101**         | 101 food categories         | Fine-grained        | [ETHZ](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)       |
| **EuroSAT**          | Sentinel-2 Satellite images | Remote Sensing      | [GitHub](https://github.com/phelber/EuroSAT)                         |

## 5. Transfer Learning Benchmarks

| Dataset           | Description                     | Purpose                 | Official Link                                             |
| :---------------- | :------------------------------ | :---------------------- | :-------------------------------------------------------- |
| **VTAB-1k** | 19 tasks (natural, specialized) | General Foundation Eval | [Google](https://github.com/google-research/task_adaptation) |
| **Wilds**   | 10 datasets with shifts         | Real-world reliability  | [Wilds](https://wilds.stanford.edu/)                         |

## 6. Video (Temporal Learning)

| Dataset                | Description                     | Purpose              | Official Link                                               |
| :--------------------- | :------------------------------ | :------------------- | :---------------------------------------------------------- |
| **Kinetics-400** | 400 action classes              | SOTA Video Benchmark | [DeepMind](https://deepmind.com/research/open-source/kinetics) |
| **Something-v2** | Action interacting with objects | Physics/Reasoning    | [Qualcomm](https://20bn.com/datasets/something-something)      |
| **UCF101**       | 101 action classes              | Legacy Video         | [UCF](https://www.crcv.ucf.edu/data/UCF101.php)                |

---

> [!TIP]
> If you want to impress your teacher with a "Universal" model, you should show results on **ImageNette (Linear Probe)**, **ImageNet-V2 (Robustness)**, and **COCO (Transfer)**. This proves your model is not just guessing but has "understanding".
