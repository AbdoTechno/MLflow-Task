"""MLflow tracking setup and experiment configuration."""

import os
import mlflow
from src.config import ARTIFACTS_DIR, EXPERIMENT_NAME, TRACKING_URI


def setup_tracking(
    tracking_uri: str = TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    artifacts_dir: str = ARTIFACTS_DIR,
):
    """Initialize MLflow tracking URI and ensure the experiment is created."""
    print("\n" + "=" * 70)
    print("TASK 1.2 - MLFLOW TRACKING SETUP")
    print("=" * 70)

    mlflow.set_tracking_uri(tracking_uri)
    os.makedirs(artifacts_dir, exist_ok=True)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifacts_dir)

    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)

    print("Backend store  :", tracking_uri)
    print("Artifact store :", exp.artifact_location)
    print("Experiment     :", exp.name, f"(id={exp.experiment_id})")
    return exp
