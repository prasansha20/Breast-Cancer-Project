# 02 — Dataset Guide

## Source

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, loaded through scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer()
```

Original source: UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic).

## Summary

| Property | Value |
|----------|-------|
| Samples | 569 |
| Features | 30 numeric |
| Task | Binary classification |
| Classes | Malignant (`0`), Benign (`1`) |
| Missing values | None |

## Class Labels

| Label | Meaning | Clinical note |
|------:|---------|---------------|
| `0` | Malignant | Cancerous tumor |
| `1` | Benign | Non-cancerous tumor |

## Feature Groups

Each sample describes a cell nucleus using measurements such as:

- radius, texture, perimeter, area
- smoothness, compactness, concavity, concave points
- symmetry, fractal dimension

For each base measurement, the dataset typically includes related statistics (mean / error / worst-style summaries), giving **30 features** total.

## Train / Test Split Used in This Project

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=2,
    stratify=Y,
)
```

- **80%** training
- **20%** testing
- **`stratify=Y`** keeps similar class balance in both sets

## Why Standardization?

Features have different scales (e.g., area vs smoothness). Neural networks train more stably when inputs are standardized:

```python
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)   # fit on train only
X_test_std = scaler.transform(X_test)         # transform test with same scaler
```

Fitting the scaler only on training data prevents **data leakage**.

## Exploration Checklist (Notebook)

1. Convert features to a pandas `DataFrame`
2. Check `shape`, `isnull()`, `describe()`
3. Inspect class counts with `value_counts()`
4. Compare class-wise feature means with `groupby("label").mean()`

## Next

Continue to [03 — Model Design](03_model.md).
