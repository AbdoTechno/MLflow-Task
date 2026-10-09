"""Hyperparameter optimization (HPO) sweep with MLflow nested runs."""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression

from src.config import C_GRID, HPO_MAX_ITER
from src.metrics import evaluate_model


def run_grid_search(X_train, X_test, y_train, y_test, c_grid: list = None) -> str:
    """Run a grid search sweep across C values using MLflow nested runs.

    Parent run aggregates sweep metadata and overall best metric.
    Child runs log trial-specific parameters, metrics, model artifacts, and signatures.
    """
    if c_grid is None:
        c_grid = C_GRID

    print("\n" + "=" * 70)
    print("TASK 2.1 - NESTED-RUN HYPERPARAMETER SWEEP")
    print("=" * 70)

    best_f1 = -1.0
    trials = []

    with mlflow.start_run(run_name="logistic-regression-grid-search") as parent:
        mlflow.log_param("search_algorithm", "grid_search")
        mlflow.log_param("C_grid", str(c_grid))
        mlflow.set_tag("pipeline_phase", "HPO")

        for c_val in c_grid:
            with mlflow.start_run(run_name=f"trial_C_{c_val}", nested=True) as child:
                mlflow.log_param("C", c_val)
                mlflow.log_param("max_iter", HPO_MAX_ITER)

                model = LogisticRegression(
                    C=c_val,
                    max_iter=HPO_MAX_ITER,
                    solver="lbfgs",
                ).fit(X_train, y_train)

                m = evaluate_model(model, X_test, y_test)
                for metric_name in ("f1_score", "roc_auc", "accuracy"):
                    mlflow.log_metric(metric_name, m[metric_name])

                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    signature=signature,
                    input_example=X_train.iloc[:2],
                    serialization_format="cloudpickle",
                )

                trials.append({"C": c_val, "run_id": child.info.run_id, **m})
                best_f1 = max(best_f1, m["f1_score"])

        # Log aggregate best metric to parent run
        mlflow.log_metric("best_f1_score", best_f1)

        print(f"Parent run : logistic-regression-grid-search | ID {parent.info.run_id}")
        print(f"{'trial':<16}{'C':>8}{'F1':>9}{'ROC-AUC':>10}{'accuracy':>10} child run id")
        for t in trials:
            print(
                f"trial_C_{t['C']:<8}{t['C']:>8}{t['f1_score']:>9.4f}{t['roc_auc']:>10.4f}"
                f"{t['accuracy']:>10.4f} {t['run_id'][:12]}..."
            )
        print(f"best_f1_score recorded in parent: {best_f1:.4f}")
        return parent.info.run_id
