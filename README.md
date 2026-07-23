# Breast Cancer Classification with Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-ANN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Dataset-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-2EA44F?style=for-the-badge)](#disclaimer)

A beginner-friendly **B.Tech AI mini project** that classifies breast tumors as **Malignant** or **Benign** using a compact feed-forward Artificial Neural Network (ANN) built with TensorFlow/Keras on the Breast Cancer Wisconsin dataset.

> **Test Accuracy:** ~**97.37%** &nbsp;|&nbsp; **Parameters:** 662 &nbsp;|&nbsp; **Epochs:** 10

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Pipeline](#data-pipeline)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Automated Build Pipeline](#automated-build-pipeline)
- [Tech Stack](#tech-stack)
- [Disclaimer](#disclaimer)

---

## Overview

This project demonstrates an end-to-end machine learning workflow:

1. Load and explore the Breast Cancer Wisconsin (Diagnostic) dataset
2. Split and standardize features (no data leakage)
3. Train a simple ANN for binary classification
4. Evaluate with accuracy/loss curves, confusion matrix, and classification report
5. Run a predictive system that returns diagnosis + confidence scores
6. Auto-generate screenshots, metrics, LaTeX report, and Beamer presentation

**Class labels**

| Label | Diagnosis   |
|------:|-------------|
| `0`   | Malignant   |
| `1`   | Benign      |

---

## Features

- Clean, commented Jupyter notebook for beginners
- Stratified train/test split (80% / 20%)
- Feature scaling with `StandardScaler`
- Keras Sequential ANN with ReLU + Softmax
- Accuracy & loss visualization
- Confusion matrix and classification report
- Single-sample prediction with class probabilities
- One-command build script for screenshots, PDF report, and slides

---

## Project Workflow

```mermaid
flowchart TD
    A([Start]) --> B[Import Libraries]
    B --> C[Load Breast Cancer Dataset<br/>569 samples · 30 features]
    C --> D[Explore Data<br/>shapes · stats · class balance]
    D --> E[Train / Test Split<br/>80% / 20% stratified]
    E --> F[Standardize Features<br/>StandardScaler on train only]
    F --> G[Build ANN<br/>Input 30 → Dense 20 → Dense 2]
    G --> H[Compile<br/>Adam + Sparse Categorical Crossentropy]
    H --> I[Train<br/>10 epochs · 10% validation split]
    I --> J[Plot Accuracy & Loss]
    J --> K[Evaluate on Test Set]
    K --> L[Confusion Matrix & Classification Report]
    L --> M[Predict Diagnosis + Confidence]
    M --> N([End])

    style A fill:#1B4F72,stroke:#0D2B3E,color:#fff
    style N fill:#1B4F72,stroke:#0D2B3E,color:#fff
    style G fill:#117A65,stroke:#0B5345,color:#fff
    style I fill:#B9770E,stroke:#7E5109,color:#fff
    style K fill:#6C3483,stroke:#4A235A,color:#fff
    style M fill:#1A5276,stroke:#154360,color:#fff
```

---

## Dataset

**Source:** Breast Cancer Wisconsin (Diagnostic) — available via `sklearn.datasets.load_breast_cancer()`

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Samples               | 569                                        |
| Features              | 30 numeric tumor measurements              |
| Classes               | Malignant (`0`), Benign (`1`)              |
| Train / Test          | 80% / 20% (stratified)                     |
| Preprocessing         | `StandardScaler` (fit on training data)    |

Features describe cell-nucleus characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, and related statistics.

---

## Model Architecture

```mermaid
flowchart LR
    subgraph INPUT["Input Layer"]
        I["30 Features<br/>Tumor Measurements"]
    end

    subgraph HIDDEN["Hidden Layer"]
        H["Dense 20<br/>ReLU"]
    end

    subgraph OUTPUT["Output Layer"]
        O["Dense 2<br/>Softmax<br/>Malignant / Benign"]
    end

    I --> H --> O

    style I fill:#2874A6,stroke:#1A5276,color:#fff
    style H fill:#B9770E,stroke:#7E5109,color:#fff
    style O fill:#6C3483,stroke:#4A235A,color:#fff
```

| Layer          | Units | Activation | Role                          |
|----------------|------:|------------|-------------------------------|
| Input          | 30    | —          | Standardized feature vector   |
| Dense (hidden) | 20    | ReLU       | Non-linear feature learning   |
| Dense (output) | 2     | Softmax    | Class probabilities           |

**Training setup**

- Optimizer: `Adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`
- Epochs: `10`
- Validation split: `10%` of training data
- Trainable parameters: **662**

---

## Data Pipeline

```mermaid
flowchart LR
    A[Raw Features X<br/>569 × 30] --> B[train_test_split<br/>stratify=Y]
    B --> C[X_train / Y_train]
    B --> D[X_test / Y_test]
    C --> E[StandardScaler.fit_transform]
    D --> F[StandardScaler.transform]
    E --> G[ANN Training]
    F --> H[ANN Evaluation]
    G --> H
    H --> I[Metrics · CM · Report · Prediction]

    style A fill:#D6EAF8,stroke:#2874A6
    style G fill:#117A65,stroke:#0B5345,color:#fff
    style H fill:#6C3483,stroke:#4A235A,color:#fff
    style I fill:#1A5276,stroke:#154360,color:#fff
```

---

## Results

Metrics from the automated build (`metrics.txt`):

| Metric                    | Value    |
|---------------------------|----------|
| Test Accuracy             | **97.37%** |
| Test Loss                 | 0.1077   |
| Final Training Accuracy   | 96.33%   |
| Final Validation Accuracy | 95.65%   |
| Trainable Parameters      | 662      |
| Test Samples              | 114      |

### Evaluation Artifacts

Generated under `screenshots/`:

| File | Description |
|------|-------------|
| `05_accuracy_curve.png` | Training vs validation accuracy |
| `06_loss_curve.png` | Training vs validation loss |
| `07_test_accuracy.png` | Held-out test metrics |
| `08_confusion_matrix.png` | Malignant vs Benign confusion matrix |
| `09_classification_report.png` | Precision / Recall / F1 |
| `10_prediction.png` | Sample prediction with confidence |

---

## Project Structure

```text
Breast-Cancer-Project/
├── Breast_Cancer_Classification_with_Neural_Network.ipynb  # Main notebook
├── build_project.py                                        # Full automation pipeline
├── metrics.txt                                             # Latest build metrics
├── screenshots/                                            # 300 DPI figures
├── report/
│   ├── main.tex / main.pdf                                 # Project report
│   └── references.bib
└── presentation/
    ├── presentation.tex / presentation.pdf                 # Beamer slides
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- (Optional) MiKTeX / TeX Live for PDF report & presentation
- (Optional) Node.js `npx` for Mermaid CLI diagrams

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nbformat nbclient ipykernel
```

### Run the Notebook

```bash
jupyter notebook Breast_Cancer_Classification_with_Neural_Network.ipynb
```

Or in VS Code / Cursor: open the notebook and use **Run All**.

---

## Automated Build Pipeline

One command runs the full project build:

```bash
python build_project.py
```

```mermaid
flowchart TD
    S1[1 · Install missing packages] --> S2[2 · Auto-detect & execute notebook]
    S2 --> S3[3 · Generate 300 DPI screenshots + metrics.txt]
    S3 --> S4[4 · Compile LaTeX report]
    S4 --> S5[5 · Compile Beamer presentation]
    S5 --> S6[6 · Print build summary]

    style S1 fill:#1B4F72,color:#fff
    style S2 fill:#117A65,color:#fff
    style S3 fill:#B9770E,color:#fff
    style S4 fill:#6C3483,color:#fff
    style S5 fill:#1A5276,color:#fff
    style S6 fill:#1B4F72,color:#fff
```

**Outputs**

- `screenshots/` — dataset info, curves, confusion matrix, prediction, diagrams
- `metrics.txt` — numeric build metrics
- `report/main.pdf` — project report
- `presentation/presentation.pdf` — presentation slides

---

## Tech Stack

| Category        | Tools                                      |
|-----------------|--------------------------------------------|
| Language        | Python                                     |
| Deep Learning   | TensorFlow / Keras                         |
| ML Utilities    | scikit-learn, NumPy, Pandas                |
| Visualization   | Matplotlib, Seaborn                        |
| Notebook        | Jupyter                                    |
| Documentation   | LaTeX (report + Beamer)                    |
| Automation      | `build_project.py`, Mermaid CLI (optional) |

---

## Disclaimer

This project is for **educational / academic demonstration only**.  
It is **not** a medical device and must **not** be used for real clinical diagnosis or treatment decisions.

---

## Author

**prasansha20** — B.Tech Artificial Intelligence Mini Project

Repository: [github.com/prasansha20/Breast-Cancer-Project](https://github.com/prasansha20/Breast-Cancer-Project)
