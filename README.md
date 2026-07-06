# Four-Class Chest X-ray Classification for COVID-19 and Lung Conditions

**Course:** AI for Medicine – University of Bologna  
**Supervisor:** Prof. Stefano Diciotti  
**Academic Year:** 2025/2026  
**Student:** Mina Hosseina – Electronic Engineering  

---

## Overview

This project develops a reproducible deep learning pipeline for automatic four-class classification of chest X-ray images into:

- **COVID-19**
- **Lung Opacity**
- **Normal**
- **Viral Pneumonia**

Two pretrained CNN architectures are compared under different transfer learning strategies using the publicly available [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

## Repository Structure

```
├── 01_mobilenetv2_twostage_final.ipynb     # MobileNetV2 two-stage fine-tuning (main model)
├── 02_resnet18_twostage_final.ipynb        # ResNet18 two-stage fine-tuning (comparison)
├── 03_mobilenetv2_onestage_final.ipynb     # MobileNetV2 single-stage baseline
└── README.md
```

---

## Dataset

**COVID-19 Radiography Database** — publicly available on Kaggle.

| Class | Images |
|---|---|
| COVID-19 | 3,616 |
| Lung Opacity | 6,012 |
| Normal | 10,192 |
| Viral Pneumonia | 1,345 |
| **Total** | **21,165** |

54 exact duplicate images were removed using SHA-256 hashing before splitting, reducing the dataset to 21,111 images.

The cleaned dataset was split using a stratified 70/15/15 strategy:

| Split | Images |
|---|---|
| Train | 14,777 |
| Validation | 3,167 |
| Test | 3,167 |

---

## Methods

### Preprocessing
- SHA-256 duplicate removal before splitting
- Stratified 70/15/15 train/validation/test split with fixed random seed
- Resize to 224 × 224 pixels
- Convert to 3-channel format (required by pretrained architectures)
- Model-specific normalization:
  - MobileNetV2: rescaled to [-1, 1] (Keras standard)
  - ResNet18: ImageNet mean and std (Torchvision standard)
- Training-only augmentation: horizontal flip, small rotation, mild zoom/crop

### Models

| Model | Strategy | Framework |
|---|---|---|
| Majority-class baseline | Always predicts Normal | scikit-learn |
| MobileNetV2 single-stage | Frozen backbone, trained head only | TensorFlow/Keras |
| MobileNetV2 two-stage | Stage 1: frozen backbone → Stage 2: fine-tune last 30 layers | TensorFlow/Keras |
| ResNet18 two-stage | Stage 1: frozen backbone → Stage 2: fine-tune final residual block | PyTorch |

### Training Details
- Loss: class-weighted cross-entropy (weights computed from training set only)
- Early stopping based on validation loss with best weight restoration
- MobileNetV2: Adam optimizer, lr=1e-3 (Stage 1), lr=1e-5 (Stage 2)
- ResNet18: AdamW optimizer, lr=1e-4 (Stage 1), lr=1e-5 (Stage 2)
- Batch normalization layers kept frozen during fine-tuning (MobileNetV2)

### Evaluation Metrics
- Accuracy, Balanced Accuracy, Macro F1, Weighted F1
- Per-class sensitivity and specificity
- AUROC and AUPRC (one-vs-rest, for final two-stage models)
- Confusion matrices
- Grad-CAM visualizations (MobileNetV2)

---

## Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| Majority-class baseline | 0.4828 | 0.2500 | 0.1628 | 0.3144 |
| MobileNetV2 single-stage | 0.8184 | 0.8567 | 0.8101 | 0.8215 |
| MobileNetV2 two-stage | 0.9170 | 0.9253 | 0.9229 | 0.9167 |
| ResNet18 two-stage | 0.9252 | 0.9339 | **0.9350** | 0.9253 |

All models evaluated on the same held-out test set of 3,167 images.

Key findings:
- Two-stage fine-tuning improved MobileNetV2 macro F1 from 0.810 to 0.923
- ResNet18 achieved the best overall performance but with a small margin over MobileNetV2
- Most errors occurred between visually similar classes: COVID-19, Lung Opacity, and Normal
- Grad-CAM showed the model focusing on lung regions in correct predictions

---

## How to Reproduce

1. Clone this repository
2. Open the notebooks in [Google Colab](https://colab.research.google.com/)
3. The dataset is downloaded automatically via `kagglehub` — a Kaggle API key is required
4. Run notebooks in order: `01` → `03` → `02`
   - Notebook `01` saves the train/validation/test split files
   - Notebooks `02` and `03` reload the same splits to ensure identical evaluation subsets
5. A fixed random seed is set at the start of each notebook

### Dependencies
```
TensorFlow / Keras
PyTorch
torchvision
scikit-learn
NumPy
pandas
Matplotlib
Pillow (PIL)
OpenCV
kagglehub
```

> **Note:** The full image dataset and trained model weight files are not included in this repository due to size constraints.

---

## Limitations

- Single random seed — result stability across initializations was not evaluated
- Checkpoint selection based on validation loss rather than macro F1
- SHA-256 hashing detects only exact duplicates, not near-duplicates or reprocessed images
- No patient-level split (patient identifiers not available in the dataset)
- Evaluated on a single public benchmark — not validated on independent clinical data

---

## Ethics

The COVID-19 Radiography Database is publicly available under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. No private patient data were used. This project is an academic study and is not intended as a clinical diagnostic tool.

---

## References

1. T. Rahman, *COVID-19 Radiography Database*, Kaggle. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. M. E. H. Chowdhury et al., "Can AI help in screening viral and COVID-19 pneumonia?", *IEEE Access*, 2020.
3. T. Rahman et al., "Exploring the effect of image enhancement techniques on COVID-19 detection", *Computers in Biology and Medicine*, 2021.
4. M. Sandler et al., "MobileNetV2: Inverted residuals and linear bottlenecks", *CVPR*, 2018.
5. K. He et al., "Deep residual learning for image recognition", *CVPR*, 2016.
6. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks", *ICCV*, 2017.
