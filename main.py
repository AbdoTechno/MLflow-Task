"""Main pipeline orchestration entry point for FinGuard Credit Default Risk."""

import argparse
import sys
from src.config import DEFAULT_CSV_PATH
from src.data import load_and_preprocess_data
from src.inference import verify_inference
from src.models.baseline import run_baseline
from src.models.hpo import run_grid_search
from src.registry import register_champion
from src.tracking import setup_tracking


def run_pipeline(csv_path: str = DEFAULT_CSV_PATH):
    """Execute the end-to-end FinGuard MLOps pipeline."""
    # Step 1: Ingestion & Preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_data(csv_path=csv_path)

    # Step 2: MLflow Tracking & Experiment Setup
    setup_tracking()

    # Step 3: Baseline Model Training & Tracking
    run_baseline(X_train, X_test, y_train, y_test)

    # Step 4: Hyperparameter Optimization Sweep (Nested Runs)
    parent_run_id = run_grid_search(X_train, X_test, y_train, y_test)

    # Step 5: Model Registry, Governance Tags, and Champion Alias
    register_champion(parent_run_id)

    # Step 6: Production Inference via Champion Alias
    verify_inference(X_test, y_test)


def main():
    parser = argparse.ArgumentParser(
        description="FinGuard Analytics: Tracked Training, Nested HPO, Model Registry, and Alias Inference."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV_PATH,
        help=f"Path to the credit card CSV dataset (default: {DEFAULT_CSV_PATH})",
    )
    args = parser.parse_args()
    run_pipeline(csv_path=args.csv_path)


if __name__ == "__main__":
    main()
