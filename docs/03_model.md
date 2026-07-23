# 03 — Model Design

## Goal

Map a vector of **30 standardized features** to one of two classes:

- Class `0` → Malignant
- Class `1` → Benign

## Architecture

```mermaid
flowchart LR
    I["Input<br/>30 features"] --> H["Dense 20<br/>ReLU"]
    H --> O["Dense 2<br/>Softmax"]

    style I fill:#2874A6,color:#fff
    style H fill:#B9770E,color:#fff
    style O fill:#6C3483,color:#fff
```

| Layer | Type | Units | Activation | Purpose |
|-------|------|------:|------------|---------|
| Input | `Input(shape=(30,))` | 30 | — | Accept feature vector |
| Hidden | `Dense` | 20 | ReLU | Learn non-linear patterns |
| Output | `Dense` | 2 | Softmax | Class probabilities |

**Total trainable parameters:** 662

Parameter count:

- Hidden weights: `30 × 20 + 20 = 620`
- Output weights: `20 × 2 + 2 = 42`
- Total: `620 + 42 = 662`

## Keras Code

```python
model = keras.Sequential(
    [
        keras.Input(shape=(30,)),
        keras.layers.Dense(20, activation="relu", name="hidden_layer"),
        keras.layers.Dense(2, activation="softmax", name="output_layer"),
    ],
    name="breast_cancer_ann",
)
```

## Compilation Settings

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

| Setting | Choice | Why |
|---------|--------|-----|
| Optimizer | Adam | Fast, stable default for small networks |
| Loss | Sparse categorical cross-entropy | Labels are integers `0` / `1` |
| Output activation | Softmax | Produces a valid probability distribution |
| Metric | Accuracy | Easy to interpret for binary classification |

## Training Configuration

| Hyperparameter | Value |
|----------------|------:|
| Epochs | 10 |
| Validation split | 10% of training data |
| Batch size | Keras default |
| Random seed | `3` (for reproducibility) |

```python
history = model.fit(
    X_train_std,
    Y_train,
    validation_split=0.1,
    epochs=10,
)
```

## Prediction Logic

`model.predict()` returns probabilities for both classes:

```text
[P(Malignant=0), P(Benign=1)]
```

Convert to a label with `argmax`:

```python
Y_pred_labels = np.argmax(Y_pred_probs, axis=1)
```

Example:

- `[0.08, 0.92]` → predicted class **1 (Benign)**
- `[0.81, 0.19]` → predicted class **0 (Malignant)**

## Design Notes for Beginners

- A small network is enough for this tabular dataset
- Softmax + sparse categorical loss is the standard multi-class pattern (here used for 2 classes)
- Always scale features before training neural networks on numeric tabular data

## Next

Continue to [04 — How to Run](04_usage.md).
