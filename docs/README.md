# Documentation

Welcome to the documentation for the **Breast Cancer Classification with ANN** project.

This folder explains the theory, setup, dataset, model, usage, and deliverables in a clear academic style.

## Contents

| Document | Description |
|----------|-------------|
| [01 — Setup & Installation](01_setup.md) | Environment setup and dependencies |
| [02 — Dataset Guide](02_dataset.md) | Wisconsin Breast Cancer dataset details |
| [03 — Model Design](03_model.md) | ANN architecture, loss, and training |
| [04 — How to Run](04_usage.md) | Step-by-step notebook usage |
| [05 — Results & Evaluation](05_results.md) | Metrics, confusion matrix, interpretation |
| [06 — Report & Presentation](06_report_presentation.md) | LaTeX report and Beamer slides |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook Breast_Cancer_Classification_with_Neural_Network.ipynb
```

Or open the notebook in VS Code / Cursor and click **Run All**.

## Project Goal

Build a beginner-friendly neural network that classifies breast tumors as:

- **0 → Malignant**
- **1 → Benign**

using 30 numeric clinical features from the Breast Cancer Wisconsin dataset.

## Recommended Reading Order

```mermaid
flowchart LR
    A[Setup] --> B[Dataset]
    B --> C[Model Design]
    C --> D[How to Run]
    D --> E[Results]
    E --> F[Report & Slides]

    style A fill:#1B4F72,color:#fff
    style C fill:#117A65,color:#fff
    style E fill:#6C3483,color:#fff
    style F fill:#B9770E,color:#fff
```

---

**Back to:** [Project README](../README.md)
