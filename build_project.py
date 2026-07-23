#!/usr/bin/env python3
"""
build_project.py
================
Fully automated build pipeline for the Breast Cancer ANN B.Tech AI Mini Project.

Usage:
    python build_project.py

What it does:
  1. Installs missing Python packages
  2. Auto-detects and executes the Jupyter Notebook end-to-end
  3. Generates high-resolution screenshots (300 DPI)
  4. Saves metrics.txt
  5. Compiles LaTeX report (pdflatex + bibtex)
  6. Compiles LaTeX Beamer presentation
  7. Prints a clean success/failure summary
"""

from __future__ import annotations

import io
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

# =============================================================================
# Paths & constants
# =============================================================================
ROOT = Path(__file__).resolve().parent
SCREENSHOTS = ROOT / "screenshots"
REPORT_DIR = ROOT / "report"
PRESENTATION_DIR = ROOT / "presentation"
METRICS_FILE = ROOT / "metrics.txt"
DPI = 300
RANDOM_SEED = 3
EPOCHS = 10
TEST_SIZE = 0.2
TRAIN_RANDOM_STATE = 2

REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    # (import_name, pip_name)
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("tensorflow", "tensorflow"),
    ("nbformat", "nbformat"),
    ("nbclient", "nbclient"),
    ("ipykernel", "ipykernel"),
]

WORKFLOW_MERMAID = """flowchart TD
    A[Start] --> B[Import Libraries<br/>TensorFlow, NumPy, Pandas, etc.]
    B --> C[Load Breast Cancer Dataset<br/>569 samples, 30 features]
    C --> D[Explore Dataset<br/>Shapes, labels, statistics]
    D --> E[Train-Test Split<br/>80% train / 20% test]
    E --> F[Standardize Features<br/>StandardScaler]
    F --> G[Build ANN Model<br/>Input 30 → Dense 20 → Dense 2]
    G --> H[Compile Model<br/>Adam + Sparse Categorical Crossentropy]
    H --> I[Train Model<br/>10 Epochs with Validation]
    I --> J[Plot Accuracy & Loss Curves]
    J --> K[Evaluate on Test Set]
    K --> L[Confusion Matrix & Classification Report]
    L --> M[Predict Diagnosis + Confidence]
    M --> N[End]

    style A fill:#1B4F72,stroke:#0D2B3E,color:#FFFFFF
    style N fill:#1B4F72,stroke:#0D2B3E,color:#FFFFFF
    style G fill:#117A65,stroke:#0B5345,color:#FFFFFF
    style I fill:#B9770E,stroke:#7E5109,color:#FFFFFF
    style K fill:#6C3483,stroke:#4A235A,color:#FFFFFF
    style M fill:#1A5276,stroke:#154360,color:#FFFFFF
"""

ARCHITECTURE_MERMAID = """flowchart LR
    subgraph INPUT["Input Layer"]
        I["30 Features<br/>Tumor Measurements"]
    end

    subgraph H1["Hidden Layer"]
        D1["Dense 20<br/>ReLU"]
    end

    subgraph OUT["Output Layer"]
        O["Dense 2<br/>Softmax<br/>Malignant / Benign"]
    end

    I --> D1 --> O

    style I fill:#2874A6,stroke:#1A5276,color:#FFFFFF
    style D1 fill:#B9770E,stroke:#7E5109,color:#FFFFFF
    style O fill:#6C3483,stroke:#4A235A,color:#FFFFFF
"""

# Sample used in the notebook predictive system
SAMPLE_INPUT = (
    11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115,
    0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758,
    0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72, 82.98, 516.5,
    0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563,
)

CLASS_NAMES = ["Malignant (0)", "Benign (1)"]


# =============================================================================
# Colored logging
# =============================================================================
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"


def _enable_windows_ansi() -> None:
    if platform.system() != "Windows":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


def _configure_utf8_stdout() -> None:
    """Avoid Windows cp1252 UnicodeEncodeError on symbols like checkmarks."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


_enable_windows_ansi()
_configure_utf8_stdout()


def safe_print(msg: str = "") -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def log_info(msg: str) -> None:
    safe_print(f"{Colors.CYAN}[INFO]{Colors.RESET}  {msg}")


def log_ok(msg: str) -> None:
    safe_print(f"{Colors.GREEN}[ OK ]{Colors.RESET}  {msg}")


def log_warn(msg: str) -> None:
    safe_print(f"{Colors.YELLOW}[WARN]{Colors.RESET}  {msg}")


def log_err(msg: str) -> None:
    safe_print(f"{Colors.RED}[FAIL]{Colors.RESET}  {msg}")


def log_stage(title: str) -> None:
    bar = "=" * 62
    safe_print(f"\n{Colors.BOLD}{Colors.BLUE}{bar}{Colors.RESET}")
    safe_print(f"{Colors.BOLD}{Colors.BLUE}  {title}{Colors.RESET}")
    safe_print(f"{Colors.BOLD}{Colors.BLUE}{bar}{Colors.RESET}\n")


MARK_OK = "[OK]"
MARK_FAIL = "[X]"


# =============================================================================
# Build status
# =============================================================================
@dataclass
class BuildStatus:
    notebook_ok: bool = False
    screenshots_ok: bool = False
    metrics_ok: bool = False
    report_ok: bool = False
    presentation_ok: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(
            [
                self.notebook_ok,
                self.screenshots_ok,
                self.metrics_ok,
                self.report_ok,
                self.presentation_ok,
            ]
        )


# =============================================================================
# Utilities
# =============================================================================
def ensure_dirs() -> None:
    for path in (SCREENSHOTS, REPORT_DIR, PRESENTATION_DIR, REPORT_DIR / "sections"):
        path.mkdir(parents=True, exist_ok=True)
    log_ok(f"Directories ready under {ROOT}")


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture: bool = True,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    log_info("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=capture,
        text=True,
        env=env,
    )


def which(name: str) -> Optional[str]:
    return shutil.which(name)


def detect_notebook() -> Path:
    """
    Automatically find the project notebook.
    Preference order:
      1) Breast_Cancer*.ipynb / *breast*cancer*.ipynb
      2) Any single *.ipynb in the project root
    """
    preferred_patterns = (
        "Breast_Cancer*.ipynb",
        "*Breast*Cancer*.ipynb",
        "*breast*cancer*.ipynb",
    )
    candidates: List[Path] = []
    for pattern in preferred_patterns:
        candidates.extend(sorted(ROOT.glob(pattern)))

    # Deduplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in candidates:
        key = path.resolve()
        if key not in seen and path.is_file():
            seen.add(key)
            unique.append(path)

    if unique:
        chosen = unique[0]
        if len(unique) > 1:
            log_warn(f"Multiple matching notebooks; using {chosen.name}")
        return chosen

    notebooks = sorted(p for p in ROOT.glob("*.ipynb") if p.is_file())
    if not notebooks:
        raise FileNotFoundError(f"No .ipynb notebook found in {ROOT}")
    if len(notebooks) > 1:
        log_warn(
            "Multiple notebooks found; using "
            f"{notebooks[0].name}. Prefer naming it Breast_Cancer_*.ipynb"
        )
    return notebooks[0]


# =============================================================================
# 1) Dependency installation
# =============================================================================
def package_importable(import_name: str) -> bool:
    try:
        __import__(import_name)
        return True
    except Exception:
        return False


def install_missing_packages() -> None:
    log_stage("STAGE 1/6 - Installing dependencies")
    missing = [(imp, pip) for imp, pip in REQUIRED_PACKAGES if not package_importable(imp)]
    if not missing:
        log_ok("All required Python packages are already installed")
        return

    pip_names = sorted({pip for _, pip in missing})
    log_warn(f"Missing packages: {', '.join(pip_names)}")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *pip_names]
    try:
        result = run_cmd(cmd, check=False, capture=True)
        if result.returncode != 0:
            log_err(result.stderr[-2000:] if result.stderr else "pip install failed")
            raise RuntimeError("Failed to install required Python packages")
        log_ok("Packages installed successfully")
    except Exception as exc:
        raise RuntimeError(f"Dependency installation failed: {exc}") from exc

    still_missing = [imp for imp, _ in REQUIRED_PACKAGES if not package_importable(imp)]
    if still_missing:
        raise RuntimeError(
            "Packages installed but still not importable: " + ", ".join(still_missing)
        )


# =============================================================================
# 2) Notebook execution
# =============================================================================
def execute_notebook(notebook: Path) -> None:
    log_stage("STAGE 2/6 - Executing Jupyter Notebook")
    if not notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    log_info(f"Loading {notebook.name}")
    nb = nbformat.read(notebook, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )

    try:
        log_info("Running all cells (this may take several minutes)...")
        client.execute()
    except CellExecutionError as exc:
        msg = str(exc)
        log_err("Notebook execution failed")
        print(f"\n{Colors.RED}{msg}{Colors.RESET}\n")
        raise RuntimeError(f"Notebook execution error:\n{msg}") from exc
    except Exception as exc:
        log_err(f"Unexpected notebook error: {exc}")
        raise

    nbformat.write(nb, notebook)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = []
    for idx, cell in enumerate(code_cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append(f"Cell {idx}: {out.get('ename')}: {out.get('evalue')}")
    if errors:
        raise RuntimeError("Notebook finished with cell errors:\n" + "\n".join(errors))

    log_ok(f"Notebook executed successfully ({len(code_cells)} code cells)")


# =============================================================================
# 3) Screenshot generation (300 DPI)
# =============================================================================
def save_text_image(text: str, filename: str, title: Optional[str] = None, fontsize: int = 11) -> Path:
    import matplotlib.pyplot as plt

    lines = text.strip("\n").splitlines() or [""]
    max_len = max(len(line) for line in lines)
    width = max(8, min(16, max_len * 0.085))
    height = max(3, min(22, 0.35 * len(lines) + 1.2))

    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12, loc="left")
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=fontsize,
        fontfamily="monospace",
        verticalalignment="top",
        horizontalalignment="left",
    )
    fig.patch.set_facecolor("white")
    out = SCREENSHOTS / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    log_ok(f"Saved {out.name}")
    return out


def save_fig(fig, filename: str) -> Path:
    out = SCREENSHOTS / filename
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    import matplotlib.pyplot as plt

    plt.close(fig)
    log_ok(f"Saved {out.name}")
    return out


def render_mermaid_or_fallback(source: str, png_name: str, fallback: Callable[[Path], None]) -> None:
    """Try Mermaid CLI; fall back to matplotlib drawing if unavailable."""
    stem = Path(png_name).stem
    mmd_path = SCREENSHOTS / f"{stem}.mmd"
    png_path = SCREENSHOTS / png_name
    mmd_path.write_text(source.strip() + "\n", encoding="utf-8")

    npx = which("npx")
    if npx:
        cmd = [
            npx,
            "--yes",
            "@mermaid-js/mermaid-cli@11.4.2",
            "-i",
            str(mmd_path),
            "-o",
            str(png_path),
            "-b",
            "white",
            "-s",
            "3",
        ]
        try:
            result = run_cmd(cmd, cwd=ROOT, check=False)
            if result.returncode == 0 and png_path.exists():
                log_ok(f"Saved {png_name} via Mermaid CLI")
                return
            log_warn(f"Mermaid CLI failed for {png_name}; using matplotlib fallback")
        except Exception as exc:
            log_warn(f"Mermaid CLI error ({exc}); using matplotlib fallback")
    else:
        log_warn("npx not found; using matplotlib fallback for diagrams")

    fallback(png_path)
    log_ok(f"Saved {png_name} via matplotlib fallback")


def draw_workflow_fallback(png_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    steps = [
        "Start",
        "Import Libraries",
        "Load Breast Cancer Dataset",
        "Explore Dataset",
        "Train-Test Split",
        "Standardize Features",
        "Build ANN Model",
        "Compile (Adam + SCE)",
        "Train (10 Epochs)",
        "Plot Acc / Loss",
        "Evaluate Test Set",
        "Confusion Matrix & Report",
        "Predict + Confidence",
        "End",
    ]
    colors = {
        0: "#1B4F72",
        6: "#117A65",
        8: "#B9770E",
        10: "#6C3483",
        12: "#1A5276",
        13: "#1B4F72",
    }

    fig, ax = plt.subplots(figsize=(7, 15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.2 + 0.5)
    ax.axis("off")
    ax.set_title("Breast Cancer ANN Project Workflow", fontsize=14, fontweight="bold", pad=12)

    y = len(steps) * 1.2
    centers = []
    for i, step in enumerate(steps):
        y -= 1.15
        color = colors.get(i, "#D6EAF8")
        text_color = "white" if i in colors else "#1B2631"
        box = FancyBboxPatch(
            (1.5, y - 0.35),
            7,
            0.7,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.2,
            edgecolor="#34495E",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(5, y, step, ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")
        centers.append(y)

    for i in range(len(centers) - 1):
        ax.annotate(
            "",
            xy=(5, centers[i + 1] + 0.35),
            xytext=(5, centers[i] - 0.35),
            arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.4),
        )

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_architecture_fallback(png_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    layers = [
        ("Input Layer", "30 Features\nTumor Stats", "#2874A6"),
        ("Hidden Layer", "Dense 20\nReLU", "#B9770E"),
        ("Output Layer", "Dense 2\nSoftmax\nMalig. / Benign", "#6C3483"),
    ]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("ANN Architecture — Breast Cancer Classifier", fontsize=14, fontweight="bold")

    xs = [1.5, 5.0, 8.5]
    for x, (header, body, color) in zip(xs, layers):
        outer = FancyBboxPatch(
            (x, 1.2),
            2.5,
            2.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.0,
            edgecolor="#7D6608",
            facecolor="#FCF3CF",
        )
        ax.add_patch(outer)
        ax.text(x + 1.25, 3.5, header, ha="center", va="center", fontsize=9, fontweight="bold")
        inner = FancyBboxPatch(
            (x + 0.25, 1.5),
            2.0,
            1.6,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.0,
            edgecolor="#1C2833",
            facecolor=color,
        )
        ax.add_patch(inner)
        ax.text(x + 1.25, 2.3, body, ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], 2.5),
            xytext=(xs[i] + 2.5, 2.5),
            arrowprops=dict(arrowstyle="->", color="#1C2833", lw=1.6),
        )

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_screenshots_and_metrics() -> dict:
    log_stage("STAGE 3/6 - Generating screenshots (300 DPI) & metrics")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import tensorflow as tf
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    render_mermaid_or_fallback(WORKFLOW_MERMAID, "11_workflow.png", draw_workflow_fallback)
    render_mermaid_or_fallback(
        ARCHITECTURE_MERMAID, "12_architecture.png", draw_architecture_fallback
    )

    log_info("Loading Breast Cancer Wisconsin dataset...")
    dataset = load_breast_cancer()
    X = dataset.data
    Y = dataset.target
    feature_names = list(dataset.feature_names)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=TRAIN_RANDOM_STATE,
        stratify=Y,
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    dataset_info = (
        f"Breast Cancer Wisconsin Dataset Information\n"
        f"{'=' * 48}\n"
        f"Total samples           : {X.shape[0]}\n"
        f"Number of features      : {X.shape[1]}\n"
        f"Feature type            : Numeric tumor measurements\n"
        f"Classes                 : 0 = Malignant, 1 = Benign\n"
        f"Class counts (full)     : Malignant={int((Y == 0).sum())}, "
        f"Benign={int((Y == 1).sum())}\n"
        f"\n"
        f"Training samples        : {X_train.shape[0]}\n"
        f"Test samples            : {X_test.shape[0]}\n"
        f"Train labels shape      : {Y_train.shape}\n"
        f"Test labels shape       : {Y_test.shape}\n"
        f"\n"
        f"Preprocessing           : StandardScaler (fit on train only)\n"
        f"Train/Test split        : {int((1 - TEST_SIZE) * 100)}% / "
        f"{int(TEST_SIZE * 100)}% (stratified)\n"
        f"TensorFlow version      : {tf.__version__}"
    )
    save_text_image(dataset_info, "01_dataset_info.png", title="Dataset Information")

    # Class distribution + mean of first few features (tabular data overview)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    counts = pd.Series(Y).value_counts().sort_index()
    axes[0].bar(
        ["Malignant (0)", "Benign (1)"],
        [counts.get(0, 0), counts.get(1, 0)],
        color=["#C0392B", "#27AE60"],
        edgecolor="#1C2833",
    )
    axes[0].set_title("Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Number of Samples")
    axes[0].grid(True, axis="y", alpha=0.3)

    df_preview = pd.DataFrame(X[:, :6], columns=[n.replace(" ", "\n") for n in feature_names[:6]])
    df_preview["label"] = Y
    means = df_preview.groupby("label").mean()
    x_pos = np.arange(means.shape[1])
    width = 0.35
    axes[1].bar(x_pos - width / 2, means.loc[0], width, label="Malignant", color="#C0392B")
    axes[1].bar(x_pos + width / 2, means.loc[1], width, label="Benign", color="#27AE60")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(means.columns, fontsize=7)
    axes[1].set_title("Mean of First 6 Features by Class", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Breast Cancer Dataset Overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_dataset_overview.png")

    log_info("Building and training ANN...")
    model = keras.Sequential(
        [
            keras.Input(shape=(30,)),
            keras.layers.Dense(20, activation="relu", name="hidden_layer"),
            keras.layers.Dense(2, activation="softmax", name="output_layer"),
        ],
        name="breast_cancer_ann",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stream = io.StringIO()
    model.summary(print_fn=lambda line: stream.write(line + "\n"))
    save_text_image(stream.getvalue(), "03_model_summary.png", title="Model Summary")

    history = model.fit(
        X_train_std,
        Y_train,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,
    )

    progress_lines = [
        "Epoch | Train Acc | Val Acc  | Train Loss | Val Loss",
        "-" * 54,
    ]
    for epoch in range(len(history.history["accuracy"])):
        progress_lines.append(
            f"{epoch + 1:5d} | "
            f"{history.history['accuracy'][epoch] * 100:8.2f}% | "
            f"{history.history['val_accuracy'][epoch] * 100:7.2f}% | "
            f"{history.history['loss'][epoch]:10.4f} | "
            f"{history.history['val_loss'][epoch]:8.4f}"
        )
    progress_lines.append("-" * 54)
    progress_lines.append(
        f"Final Training Accuracy   : {history.history['accuracy'][-1] * 100:.2f}%"
    )
    progress_lines.append(
        f"Final Validation Accuracy : {history.history['val_accuracy'][-1] * 100:.2f}%"
    )
    save_text_image(
        "\n".join(progress_lines),
        "04_training_progress.png",
        title="Training Progress (10 Epochs)",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    ax.set_title("Model Accuracy over Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "05_accuracy_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="Training Loss", marker="o")
    ax.plot(history.history["val_loss"], label="Validation Loss", marker="o")
    ax.set_title("Model Loss over Epochs", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "06_loss_curve.png")

    test_loss, test_accuracy = model.evaluate(X_test_std, Y_test, verbose=0)
    correct = int(round(test_accuracy * len(Y_test)))
    test_text = (
        f"Test Set Evaluation\n"
        f"{'=' * 40}\n"
        f"Test Loss     : {test_loss:.4f}\n"
        f"Test Accuracy : {test_accuracy * 100:.2f}%\n"
        f"\n"
        f"Correct predictions : {correct} / {len(Y_test)}\n"
        f"Misclassifications  : {len(Y_test) - correct}"
    )
    save_text_image(test_text, "07_test_accuracy.png", title="Test Accuracy")

    Y_pred = model.predict(X_test_std, verbose=0)
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    conf_mat = confusion_matrix(Y_test, Y_pred_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(
        "Confusion Matrix – Breast Cancer Classification",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "08_confusion_matrix.png")

    report = classification_report(
        Y_test,
        Y_pred_labels,
        target_names=CLASS_NAMES,
        digits=4,
    )
    save_text_image(report, "09_classification_report.png", title="Classification Report")

    # Single-sample prediction (same sample as notebook)
    sample = np.asarray(SAMPLE_INPUT, dtype=float).reshape(1, -1)
    sample_std = scaler.transform(sample)
    sample_probs = model.predict(sample_std, verbose=0)[0]
    sample_label = int(np.argmax(sample_probs))
    sample_name = "Malignant" if sample_label == 0 else "Benign"
    sample_conf = float(sample_probs[sample_label] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].axis("off")
    pred_text = (
        "BREAST CANCER PREDICTION RESULT\n"
        f"{'=' * 34}\n"
        f"Predicted class index : {sample_label}\n"
        f"Predicted diagnosis   : {sample_name}\n"
        f"Confidence            : {sample_conf:.2f}%\n"
        f"{'-' * 34}\n"
        f"P(Malignant = 0)      : {sample_probs[0] * 100:.2f}%\n"
        f"P(Benign = 1)         : {sample_probs[1] * 100:.2f}%\n"
        f"{'=' * 34}\n"
        f"\nResult: The tumor is predicted to be {sample_name.upper()}."
    )
    axes[0].text(
        0.02,
        0.95,
        pred_text,
        transform=axes[0].transAxes,
        fontsize=11,
        fontfamily="monospace",
        va="top",
        color="#1C2833",
    )
    axes[0].set_title("Predictive System Output", fontsize=12, fontweight="bold")

    axes[1].bar(
        ["Malignant (0)", "Benign (1)"],
        [sample_probs[0] * 100, sample_probs[1] * 100],
        color=["#C0392B", "#27AE60"],
        edgecolor="#1C2833",
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Probability (%)")
    axes[1].set_title("Class Probabilities", fontsize=12, fontweight="bold")
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, val in enumerate(sample_probs * 100):
        axes[1].text(i, val + 2, f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Prediction Output with Confidence Scores", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "10_prediction.png")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "epochs": EPOCHS,
        "trainable_params": int(model.count_params()),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_test": int(len(Y_test)),
    }

    metrics_text = (
        f"Breast Cancer ANN Build Metrics\n"
        f"{'=' * 40}\n"
        f"test_loss={metrics['test_loss']:.6f}\n"
        f"test_accuracy={metrics['test_accuracy']:.6f}\n"
        f"final_train_acc={metrics['final_train_acc']:.6f}\n"
        f"final_val_acc={metrics['final_val_acc']:.6f}\n"
        f"final_train_loss={metrics['final_train_loss']:.6f}\n"
        f"final_val_loss={metrics['final_val_loss']:.6f}\n"
        f"epochs={metrics['epochs']}\n"
        f"trainable_params={metrics['trainable_params']}\n"
        f"n_samples={metrics['n_samples']}\n"
        f"n_features={metrics['n_features']}\n"
        f"n_test={metrics['n_test']}\n"
        f"test_accuracy_percent={metrics['test_accuracy'] * 100:.2f}\n"
    )
    METRICS_FILE.write_text(metrics_text, encoding="utf-8")
    (SCREENSHOTS / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    log_ok(f"Saved metrics -> {METRICS_FILE.name} and screenshots/metrics.txt")

    for dest in (REPORT_DIR, PRESENTATION_DIR):
        for png in SCREENSHOTS.glob("*.png"):
            shutil.copy2(png, dest / png.name)
    log_ok("Synced screenshots into report/ and presentation/")

    required = [
        "01_dataset_info.png",
        "02_dataset_overview.png",
        "03_model_summary.png",
        "04_training_progress.png",
        "05_accuracy_curve.png",
        "06_loss_curve.png",
        "07_test_accuracy.png",
        "08_confusion_matrix.png",
        "09_classification_report.png",
        "10_prediction.png",
        "11_workflow.png",
        "12_architecture.png",
    ]
    missing = [name for name in required if not (SCREENSHOTS / name).exists()]
    if missing:
        raise RuntimeError("Missing screenshot files: " + ", ".join(missing))

    log_ok(f"All {len(required)} screenshots generated at {DPI} DPI")
    return metrics


# =============================================================================
# 4–5) LaTeX compilation with auto-fix retries
# =============================================================================
def find_latex_bins() -> Tuple[Optional[str], Optional[str]]:
    return which("pdflatex"), which("bibtex")


def common_latex_autofix(tex_path: Path, log_text: str) -> bool:
    """Attempt automatic fixes for common LaTeX issues."""
    if not tex_path.exists():
        return False

    original = tex_path.read_text(encoding="utf-8", errors="ignore")
    updated = original
    changed = False

    if "Extra }, or forgotten \\endgroup" in log_text:
        pattern = re.compile(
            r"\{\\Huge\s*\\textbf\{([^}]*)\\\\\[.*?cm\]\s*([^}]*)\}\}",
            re.DOTALL,
        )
        new_updated, n = pattern.subn(r"{\\LARGE \\textbf{\1\\\\[0.2cm]\n\2}", updated)
        if n:
            updated = new_updated
            changed = True
            log_warn(f"Auto-fixed extra brace pattern in {tex_path.name}")

    if "File ended while scanning use of" in log_text or "not found" in log_text.lower():
        if "\\graphicspath" not in updated:
            updated = updated.replace(
                "\\begin{document}",
                "\\graphicspath{{./}{../screenshots/}}\n\\begin{document}",
            )
            changed = True
            log_warn(f"Injected \\graphicspath into {tex_path.name}")

    missing_imgs = re.findall(r"File `([^']+\.png)' not found", log_text)
    for img in missing_imgs:
        src = SCREENSHOTS / Path(img).name
        dst = tex_path.parent / Path(img).name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            changed = True
            log_warn(f"Copied missing image {src.name} -> {dst.parent.name}/")

    if changed and updated != original:
        tex_path.write_text(updated, encoding="utf-8")
        return True

    return changed


def compile_latex_document(
    workdir: Path,
    tex_name: str,
    *,
    use_bibtex: bool,
    pdf_name: str,
    max_attempts: int = 3,
) -> None:
    pdflatex, bibtex = find_latex_bins()
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found on PATH. Install MiKTeX (Windows) or TeX Live (Linux/macOS)."
        )

    tex_path = workdir / tex_name
    if not tex_path.exists():
        raise FileNotFoundError(f"LaTeX source not found: {tex_path}")

    for png in SCREENSHOTS.glob("*.png"):
        shutil.copy2(png, workdir / png.name)

    for attempt in range(1, max_attempts + 1):
        log_info(f"LaTeX build attempt {attempt}/{max_attempts} for {tex_name}")
        logs: List[str] = []

        def _pdflatex() -> subprocess.CompletedProcess:
            return run_cmd(
                [pdflatex, "-interaction=nonstopmode", tex_name],
                cwd=workdir,
                check=False,
            )

        r1 = _pdflatex()
        logs.append(r1.stdout or "")
        logs.append(r1.stderr or "")

        if use_bibtex and bibtex and (workdir / tex_name.replace(".tex", ".aux")).exists():
            rb = run_cmd([bibtex, tex_name.replace(".tex", "")], cwd=workdir, check=False)
            logs.append(rb.stdout or "")
            logs.append(rb.stderr or "")

        r2 = _pdflatex()
        logs.append(r2.stdout or "")
        r3 = _pdflatex()
        logs.append(r3.stdout or "")
        logs.append(r3.stderr or "")

        pdf_path = workdir / pdf_name
        combined = "\n".join(logs)
        fatal = bool(re.search(r"^!", combined, re.MULTILINE)) and not pdf_path.exists()

        if pdf_path.exists() and pdf_path.stat().st_size > 1000:
            if fatal:
                log_warn("LaTeX reported errors but PDF was produced; continuing")
            log_ok(
                f"Compiled {pdf_path.relative_to(ROOT)} "
                f"({pdf_path.stat().st_size / 1024:.1f} KB)"
            )
            return

        log_warn(f"Build incomplete for {tex_name}")
        fixed = common_latex_autofix(tex_path, combined)
        if not fixed and attempt == max_attempts:
            tail = "\n".join(combined.splitlines()[-40:])
            raise RuntimeError(
                f"Failed to compile {tex_name} after {max_attempts} attempts.\n"
                f"Last log excerpt:\n{tail}"
            )
        if fixed:
            log_info("Applied auto-fix; retrying...")


def ensure_latex_sources() -> None:
    """Create report/presentation LaTeX sources if they are missing."""
    main_tex = REPORT_DIR / "main.tex"
    bib_file = REPORT_DIR / "references.bib"
    presentation_tex = PRESENTATION_DIR / "presentation.tex"

    if not main_tex.exists():
        log_info("Creating report/main.tex")
        main_tex.write_text(REPORT_MAIN_TEX, encoding="utf-8")
    if not bib_file.exists():
        log_info("Creating report/references.bib")
        bib_file.write_text(REPORT_BIB, encoding="utf-8")
    if not presentation_tex.exists():
        log_info("Creating presentation/presentation.tex")
        presentation_tex.write_text(PRESENTATION_TEX, encoding="utf-8")


def compile_report() -> None:
    log_stage("STAGE 4/6 - Compiling LaTeX report")
    ensure_latex_sources()
    compile_latex_document(
        REPORT_DIR,
        "main.tex",
        use_bibtex=True,
        pdf_name="main.pdf",
    )


def compile_presentation() -> None:
    log_stage("STAGE 5/6 - Compiling Beamer presentation")
    ensure_latex_sources()
    compile_latex_document(
        PRESENTATION_DIR,
        "presentation.tex",
        use_bibtex=False,
        pdf_name="presentation.pdf",
    )


# =============================================================================
# Embedded LaTeX templates (created on first run if missing)
# =============================================================================
REPORT_MAIN_TEX = r"""\documentclass[12pt,a4paper]{article}

\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{amsmath}
\usepackage{setspace}
\usepackage{fancyhdr}

\graphicspath{{./}{../screenshots/}}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}
\pagestyle{fancy}
\fancyhf{}
\rhead{Breast Cancer Classification with ANN}
\lhead{B.Tech AI Mini Project}
\cfoot{\thepage}
\onehalfspacing

\title{
  \vspace{-1cm}
  {\LARGE\textbf{Breast Cancer Classification using\\[0.2cm]
  Artificial Neural Networks}}\\[0.8cm]
  {\large B.Tech Artificial Intelligence -- Mini Project Report}
}
\author{Student Name\\Department of Artificial Intelligence}
\date{\today}

\begin{document}
\maketitle
\begin{abstract}
This report presents a beginner-friendly artificial neural network (ANN) for
binary classification of breast tumors as malignant or benign using the
Breast Cancer Wisconsin dataset. Features are standardized, the model is trained
with the Adam optimizer, and performance is evaluated using accuracy, loss curves,
a confusion matrix, and a classification report. A simple predictive system
returns the diagnosis with class probabilities and confidence.
\end{abstract}

\tableofcontents
\newpage

\section{Introduction}
Breast cancer diagnosis from clinical measurements is a classic machine-learning
problem. In this project we build a compact feed-forward neural network that
maps 30 numeric tumor features to one of two classes:
\textbf{0 = Malignant} and \textbf{1 = Benign}. The implementation uses
TensorFlow/Keras and scikit-learn, and is designed to be easy for beginners to
follow end-to-end.

\section{Dataset}
We use the Breast Cancer Wisconsin (Diagnostic) dataset available in
scikit-learn~\cite{sklearn}. It contains 569 samples and 30 real-valued features
describing cell nuclei characteristics derived from digitized images of fine
needle aspirates.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{01_dataset_info.png}
  \caption{Dataset information summary generated by the build pipeline.}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{02_dataset_overview.png}
  \caption{Class distribution and mean feature comparison for selected attributes.}
\end{figure}

\section{Methodology}
\subsection{Preprocessing}
Data are split into training (80\%) and test (20\%) sets with stratification to
preserve class balance. Features are scaled with \texttt{StandardScaler} fitted
only on the training set to avoid data leakage.

\subsection{Model Architecture}
The network is a sequential ANN:
\begin{itemize}
  \item Input layer: 30 features
  \item Hidden layer: 20 neurons with ReLU activation
  \item Output layer: 2 neurons with Softmax activation
\end{itemize}
The model is compiled with the Adam optimizer and sparse categorical
cross-entropy loss.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{12_architecture.png}
  \caption{ANN architecture used for breast cancer classification.}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{03_model_summary.png}
  \caption{Keras model summary showing layer shapes and parameter counts.}
\end{figure}

\section{Training}
The model is trained for 10 epochs with a 10\% validation split from the
training data. Accuracy and loss are monitored on both training and validation
sets.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{04_training_progress.png}
  \caption{Per-epoch training and validation metrics.}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{05_accuracy_curve.png}
  \caption{Training and validation accuracy over epochs.}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.85\textwidth]{06_loss_curve.png}
  \caption{Training and validation loss over epochs.}
\end{figure}

\section{Results}
\subsection{Test Performance}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{07_test_accuracy.png}
  \caption{Final evaluation on the held-out test set.}
\end{figure}

\subsection{Confusion Matrix and Classification Report}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.75\textwidth]{08_confusion_matrix.png}
  \caption{Confusion matrix for malignant vs benign predictions.}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{09_classification_report.png}
  \caption{Precision, recall, and F1-score for each class.}
\end{figure}

\subsection{Predictive System}
A single patient sample is standardized with the fitted scaler and passed to the
trained model. The system prints the predicted diagnosis, confidence, and class
probabilities.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\textwidth]{10_prediction.png}
  \caption{Example prediction output with confidence scores.}
\end{figure}

\section{Project Workflow}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.55\textwidth]{11_workflow.png}
  \caption{End-to-end project workflow.}
\end{figure}

\section{Conclusion}
A simple ANN achieves strong test accuracy on the Breast Cancer Wisconsin
dataset and provides interpretable probability outputs for each diagnosis.
This pipeline demonstrates data preprocessing, neural network training,
evaluation, and a practical predictive interface suitable for academic
mini-project demonstration.

\section*{Disclaimer}
This work is for educational purposes only and is \textbf{not} a medical
diagnostic tool.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

REPORT_BIB = r"""@article{sklearn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A.
          and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}

@misc{tensorflow,
  title={TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems},
  author={{TensorFlow Developers}},
  year={2024},
  howpublished={\url{https://www.tensorflow.org/}}
}

@misc{wdbc,
  title={Breast Cancer Wisconsin (Diagnostic) Data Set},
  author={{UCI Machine Learning Repository}},
  year={1995},
  howpublished={\url{https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)}}
}
"""

PRESENTATION_TEX = r"""\documentclass[aspectratio=169]{beamer}

\usetheme{Madrid}
\usecolortheme{default}
\usepackage{graphicx}
\usepackage{booktabs}

\graphicspath{{./}{../screenshots/}}

\title{Breast Cancer Classification using ANN}
\subtitle{B.Tech AI Mini Project}
\author{Student Name}
\institute{Department of Artificial Intelligence}
\date{\today}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Outline}
  \tableofcontents
\end{frame}

\section{Introduction}
\begin{frame}{Problem Statement}
  \begin{itemize}
    \item Classify breast tumors as \textbf{Malignant (0)} or \textbf{Benign (1)}
    \item Use a simple Artificial Neural Network (ANN)
    \item Dataset: Breast Cancer Wisconsin (569 samples, 30 features)
    \item Tools: Python, TensorFlow/Keras, scikit-learn
  \end{itemize}
\end{frame}

\section{Dataset}
\begin{frame}{Dataset Overview}
  \begin{center}
    \includegraphics[width=0.92\textwidth,height=0.72\textheight,keepaspectratio]{01_dataset_info.png}
  \end{center}
\end{frame}

\begin{frame}{Class Distribution}
  \begin{center}
    \includegraphics[width=0.92\textwidth,height=0.72\textheight,keepaspectratio]{02_dataset_overview.png}
  \end{center}
\end{frame}

\section{Model}
\begin{frame}{ANN Architecture}
  \begin{center}
    \includegraphics[width=0.9\textwidth,height=0.7\textheight,keepaspectratio]{12_architecture.png}
  \end{center}
\end{frame}

\begin{frame}{Model Summary}
  \begin{center}
    \includegraphics[width=0.85\textwidth,height=0.7\textheight,keepaspectratio]{03_model_summary.png}
  \end{center}
\end{frame}

\section{Training}
\begin{frame}{Training Progress}
  \begin{center}
    \includegraphics[width=0.9\textwidth,height=0.7\textheight,keepaspectratio]{04_training_progress.png}
  \end{center}
\end{frame}

\begin{frame}{Accuracy Curve}
  \begin{center}
    \includegraphics[width=0.85\textwidth,height=0.7\textheight,keepaspectratio]{05_accuracy_curve.png}
  \end{center}
\end{frame}

\begin{frame}{Loss Curve}
  \begin{center}
    \includegraphics[width=0.85\textwidth,height=0.7\textheight,keepaspectratio]{06_loss_curve.png}
  \end{center}
\end{frame}

\section{Results}
\begin{frame}{Test Accuracy}
  \begin{center}
    \includegraphics[width=0.8\textwidth,height=0.7\textheight,keepaspectratio]{07_test_accuracy.png}
  \end{center}
\end{frame}

\begin{frame}{Confusion Matrix}
  \begin{center}
    \includegraphics[width=0.65\textwidth,height=0.7\textheight,keepaspectratio]{08_confusion_matrix.png}
  \end{center}
\end{frame}

\begin{frame}{Classification Report}
  \begin{center}
    \includegraphics[width=0.9\textwidth,height=0.7\textheight,keepaspectratio]{09_classification_report.png}
  \end{center}
\end{frame}

\begin{frame}{Prediction Demo}
  \begin{center}
    \includegraphics[width=0.92\textwidth,height=0.7\textheight,keepaspectratio]{10_prediction.png}
  \end{center}
\end{frame}

\section{Workflow}
\begin{frame}{Project Workflow}
  \begin{center}
    \includegraphics[width=0.45\textwidth,height=0.75\textheight,keepaspectratio]{11_workflow.png}
  \end{center}
\end{frame}

\begin{frame}{Conclusion}
  \begin{itemize}
    \item Built a compact ANN for binary breast cancer classification
    \item Applied stratified split and feature standardization
    \item Evaluated with accuracy, confusion matrix, and classification report
    \item Demonstrated a confidence-aware predictive system
  \end{itemize}
  \vspace{0.6cm}
  {\small\textit{Educational demo only --- not a medical diagnostic tool.}}
\end{frame}

\begin{frame}
  \begin{center}
    {\LARGE Thank You}\\[0.8cm]
    {\large Questions?}
  \end{center}
\end{frame}

\end{document}
"""


# =============================================================================
# Summary
# =============================================================================
def print_summary(status: BuildStatus) -> None:
    log_stage("STAGE 6/6 - Build summary")
    if status.success:
        safe_print(f"{Colors.GREEN}{Colors.BOLD}")
        safe_print("=====================================")
        safe_print("PROJECT BUILD SUCCESSFUL")
        safe_print("=====================================")
        safe_print(Colors.RESET)
        safe_print(f"{Colors.GREEN}{MARK_OK} Notebook Executed{Colors.RESET}")
        safe_print(f"{Colors.GREEN}{MARK_OK} Screenshots Generated{Colors.RESET}")
        safe_print(f"{Colors.GREEN}{MARK_OK} Metrics Saved{Colors.RESET}")
        safe_print(f"{Colors.GREEN}{MARK_OK} Report Compiled{Colors.RESET}")
        safe_print(f"{Colors.GREEN}{MARK_OK} Presentation Compiled{Colors.RESET}")
        safe_print()
        safe_print(f"{Colors.BOLD}Output Files:{Colors.RESET}")
        safe_print("- report/main.pdf")
        safe_print("- presentation/presentation.pdf")
        safe_print("- screenshots/")
        safe_print("- metrics.txt")
        safe_print()
    else:
        safe_print(f"{Colors.RED}{Colors.BOLD}")
        safe_print("=====================================")
        safe_print("PROJECT BUILD FAILED")
        safe_print("=====================================")
        safe_print(Colors.RESET)

        def mark(ok: bool, label: str) -> None:
            if ok:
                icon = f"{Colors.GREEN}{MARK_OK}{Colors.RESET}"
            else:
                icon = f"{Colors.RED}{MARK_FAIL}{Colors.RESET}"
            safe_print(f"{icon} {label}")

        mark(status.notebook_ok, "Notebook Executed")
        mark(status.screenshots_ok, "Screenshots Generated")
        mark(status.metrics_ok, "Metrics Saved")
        mark(status.report_ok, "Report Compiled")
        mark(status.presentation_ok, "Presentation Compiled")
        safe_print()
        if status.errors:
            safe_print(f"{Colors.RED}{Colors.BOLD}Errors:{Colors.RESET}")
            for err in status.errors:
                safe_print(f"  - {err}")
            safe_print()


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    status = BuildStatus()
    safe_print(f"{Colors.BOLD}Breast Cancer ANN Mini Project - Automated Builder{Colors.RESET}")
    safe_print(f"{Colors.DIM}Root: {ROOT}{Colors.RESET}")
    safe_print(
        f"{Colors.DIM}Platform: {platform.system()} {platform.release()} | "
        f"Python {sys.version.split()[0]}{Colors.RESET}"
    )

    try:
        ensure_dirs()
        notebook = detect_notebook()
        log_ok(f"Detected notebook: {notebook.name}")
        install_missing_packages()

        try:
            execute_notebook(notebook)
            status.notebook_ok = True
        except Exception as exc:
            status.errors.append(f"Notebook: {exc}")
            log_err(str(exc))
            print_summary(status)
            return 1

        try:
            generate_screenshots_and_metrics()
            status.screenshots_ok = True
            status.metrics_ok = METRICS_FILE.exists()
            if not status.metrics_ok:
                raise RuntimeError("metrics.txt was not created")
        except Exception as exc:
            status.errors.append(f"Screenshots/Metrics: {exc}")
            log_err(str(exc))
            traceback.print_exc()
            print_summary(status)
            return 1

        try:
            compile_report()
            status.report_ok = (REPORT_DIR / "main.pdf").exists()
            if not status.report_ok:
                raise RuntimeError("report/main.pdf was not created")
        except Exception as exc:
            status.errors.append(f"Report: {exc}")
            log_err(str(exc))
            print_summary(status)
            return 1

        try:
            compile_presentation()
            status.presentation_ok = (PRESENTATION_DIR / "presentation.pdf").exists()
            if not status.presentation_ok:
                raise RuntimeError("presentation/presentation.pdf was not created")
        except Exception as exc:
            status.errors.append(f"Presentation: {exc}")
            log_err(str(exc))
            print_summary(status)
            return 1

        print_summary(status)
        return 0 if status.success else 1

    except KeyboardInterrupt:
        log_err("Build interrupted by user")
        status.errors.append("Interrupted by user")
        print_summary(status)
        return 130
    except Exception as exc:
        log_err(f"Fatal error: {exc}")
        traceback.print_exc()
        status.errors.append(str(exc))
        print_summary(status)
        return 1


if __name__ == "__main__":
    sys.exit(main())
