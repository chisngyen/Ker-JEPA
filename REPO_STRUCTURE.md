# KerJEPA Project Structure (Cleaned)

This repository is optimized for **Remote-Training (Kaggle H100)**. No local datasets are required.

## 📁 `exps/` - Experiments & Benchmarks
- **`exps/kaggle/`**: **Primary Suite.** 7 standalone scripts optimized for Kaggle H100. This is your source of truth for the paper.
- **`exps/legacy/`**: Archived scripts and documentation from previous iterations.
- **`exps/tracker.md`**: Master status table for the 7 core experiments (B1-B7).

## 📁 `src/` - Core Utilities
- `eval_sota.py`: Standardized evaluation logic for linear probing.
- `loader.py`: Data loading and Multi-crop utilities.

## 📁 `docs/` - Results & Reports
- Primary folder for new reports and benchmark visualizations generated from Kaggle results.

---
**Data Strategy**: To maintain a lightweight repo, datasets (ImageNette) are handled remotely on Kaggle at `/kaggle/input/imagenette/imagenette`.
