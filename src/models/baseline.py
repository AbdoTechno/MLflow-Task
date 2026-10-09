"""Baseline Logistic Regression training with MLflow tracking."""

import mlflow
from sklearn.linear_model import LogisticRegression

from src.config import BASELINE_C, BASELINE_MAX_ITER, BASELINE_SOLVER
from src.metrics import evaluate_model, plot_and_save_roc_curve


def run_baseline(X_train, X_test, y_train, y_test) -> str:
    """Train a baseline Logistic Regression model and log metrics and plots to MLflow."""
    print("\n" + "=" * 70)
    print("TASK 1.3 - MANUALLY TRACKED BASELINE")
    print("=" * 70)

    with mlflow.start_run(run_name="logisticregression-baseline") as run:
        # Log hyperparameters and metadata tags
        mlflow.log_param("C", BASELINE_C)
        mlflow.log_param("max_iter", BASELINE_MAX_ITER)
        mlflow.log_param("solver", BASELINE_SOLVER)
        mlflow.set_tag("model_family", "LogisticRegression")
        mlflow.set_tag("pipeline_phase", "Baseline")

        # Train model
        model = LogisticRegression(
            C=BASELINE_C,
            max_iter=BASELINE_MAX_ITER,
            solver=BASELINE_SOLVER,
        ).fit(X_train, y_train)

        # Evaluate performance
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Generate and log ROC curve artifact
        plot_path = "baseline_roc_curve.png"
        plot_and_save_roc_curve(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_path=plot_path,
            display_name=f"LogReg baseline (C={BASELINE_C})",
        )
        mlflow.log_artifact(plot_path)

        print("Run name :", "logisticregression-baseline")
        print("Run ID   :", run.info.run_id)
        print("Metrics  :", " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return run.info.run_id
