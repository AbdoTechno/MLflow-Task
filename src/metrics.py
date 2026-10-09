"""Evaluation metrics and plotting utilities."""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for servers and CI
import matplotlib.pyplot as plt
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X, y) -> dict[str, float]:
    """Calculate evaluation metrics on a given dataset split.

    ROC-AUC uses predicted probabilities for the positive class.
    """
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def plot_and_save_roc_curve(
    model,
    X_test,
    y_test,
    output_path: str = "baseline_roc_curve.png",
    display_name: str = "LogReg baseline",
    title: str = "Baseline ROC curve - credit default",
) -> str:
    """Generate and save an ROC curve plot to disk."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=display_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path
