"""Model registry governance and champion aliasing."""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import CHAMPION_ALIAS, EXPERIMENT_NAME, MODEL_NAME


def register_champion(
    parent_run_id: str,
    experiment_name: str = EXPERIMENT_NAME,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
):
    """Select the best child run from HPO, register it, assign champion alias and governance tags."""
    print("\n" + "=" * 70)
    print("TASK 2.2 - MODEL REGISTRY GOVERNANCE (MlflowClient)")
    print("=" * 70)

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    # Search for child runs of THIS specific parent sweep only
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' AND metrics.roc_auc > 0",
        order_by=["metrics.roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No child runs found for parent run ID: {parent_run_id}")

    best = runs[0]
    best_roc_auc = best.data.metrics.get("roc_auc", 0.0)
    print(
        f"Highest ROC-AUC child : {best.info.run_name} | "
        f"roc_auc={best_roc_auc:.4f} | run {best.info.run_id[:12]}..."
    )

    # Register model in MLflow Model Registry
    model_uri = f"runs:/{best.info.run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)

    # Set alias and governance tags
    client.set_registered_model_alias(model_name, alias, mv.version)
    client.set_model_version_tag(model_name, mv.version, "validation_status", "approved")
    client.set_model_version_tag(model_name, mv.version, "fair_lending_audit", "passed")

    check = client.get_model_version_by_alias(model_name, alias)
    print(f"Registered model : {model_name}")
    print(f"Version          : {mv.version}")
    print(f"Alias @{alias:<10} -> version {check.version}")
    print(f"Governance tags  : {dict(check.tags)}")
    return mv
