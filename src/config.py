"""Project configuration and default settings."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = str(BASE_DIR / "UCI_Credit_Card.csv")
ARTIFACTS_DIR = str(BASE_DIR / "mlflow_artifacts")

# Target & Features
TARGET_COLUMN = "default.payment.next.month"
ID_COLUMN = "ID"

# MLflow Tracking & Registry
TRACKING_URI = "sqlite:///mlflow_finguard.db"
EXPERIMENT_NAME = "credit-default-risk"
MODEL_NAME = "CreditDefaultRiskClassifier"
CHAMPION_ALIAS = "champion"

# Data Splitting & Preprocessing
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Hyperparameters
BASELINE_C = 0.1
BASELINE_MAX_ITER = 500
BASELINE_SOLVER = "lbfgs"
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
HPO_MAX_ITER = 500
