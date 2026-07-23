# 01 — Setup & Installation

## Requirements

- **Python** 3.10 or newer
- **pip** package manager
- (Optional) **Jupyter Notebook** / VS Code / Cursor for interactive runs
- (Optional) **MiKTeX** or **TeX Live** if you want to recompile the PDF report/slides

## Create a Virtual Environment (Recommended)

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

Or install packages manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

## Verify Installation

```python
import numpy
import pandas
import matplotlib
import sklearn
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Setup OK")
```

## Open the Notebook

```bash
jupyter notebook Breast_Cancer_Classification_with_Neural_Network.ipynb
```

In Cursor / VS Code:

1. Open `Breast_Cancer_Classification_with_Neural_Network.ipynb`
2. Select the Python kernel with the installed packages
3. Click **Run All**

## Common Issues

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Activate venv and reinstall `requirements.txt` |
| TensorFlow install fails on Windows | Use Python 3.10–3.12 and upgrade pip: `python -m pip install -U pip` |
| Kernel not found | Install `ipykernel` and restart the editor |
| Plots not showing | Ensure notebook cells run in order from the top |

## Next

Continue to [02 — Dataset Guide](02_dataset.md).
