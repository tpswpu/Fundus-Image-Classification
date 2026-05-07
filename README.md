# Robust Clinical Image Classification

A modular, publication-ready deep learning framework based on the Swin Transformer architecture, specifically designed for transparent and rigorous clinical image classification.

## 📁 Project Structure

```text
SwinEntropy-Project/
├── configs/                   # YAML configuration files
├── data/                      # Data loaders and augmentation pipelines
├── models/                    # Network architectures (SwinEntropyClassifier)
├── core/                      # Training and Evaluation engines
├── utils/                     # Statistical metrics, CAM, and plotting tools
├── tools/                     # Executable scripts for train/eval
└── README.md

## 📊 Data Preparation
Organize your dataset in the following directory structure. The dataloader will automatically infer class names from the subfolders.

Plaintext
Dataset_Root/
├── Train/
│   ├── CFP-0/
│   ├── CFP-1/
│   └── ...
└── Val/
    ├── CFP-0/
    ├── CFP-1/
    └── ...
Update the data_root and input_dir paths in configs/swin_entropy_cfp.yaml accordingly.

## 🚀 Quick Start
1. Training
Configure your hyperparameters in configs/swin_entropy_cfp.yaml, then run:

Bash
python tools/train.py
Best weights and training logs will be saved in the runs/train/ directory.

2. Evaluation & Interpretability Analysis
To run full metrics evaluation, generate high-DPI charts, and execute the Grad-CAM occlusion experiments:

Bash
python tools/predict.py
All outputs, including .csv source data for charts, metrics reports, and CAM overlay images, will be stored in runs/eval/.
```
