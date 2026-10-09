# FinGuard Analytics: End-to-End MLflow Pipeline

## Overview

FinGuard Analytics is a credit default risk classification system engineered to track, manage, and govern machine learning experiments using **MLflow**. The pipeline processes financial customer records, tracks baseline training and hyperparameter optimization (HPO), registers the best candidate model into the MLflow Model Registry, applies compliance governance tags, and executes alias-based production inference.

---

## What is MLflow?

**MLflow** is an open-source platform designed to manage the entire machine learning lifecycle, from experimentation and reproducibility to deployment and governance. It provides a standardized framework that eliminates "ad-hoc" experiment tracking (such as logging to spreadsheets or text files) and unifies model versioning.

### Core Components of MLflow

```
+-----------------------------------------------------------------------------------+
|                                  MLflow Platform                                  |
+---------------------+---------------------+------------------+--------------------+
|   MLflow Tracking   | MLflow Model Registry|  MLflow Models   |   MLflow Projects  |
|  - Parameters       |  - Centralized repo |  - Flavors       |  - Reproducible    |
|  - Metrics          |  - Versioning       |  - Signatures    |    packaging       |
|  - Artifacts/Plots  |  - Aliases/Tags     |  - Input example |  - Standard format |
|  - Parent/Child runs|  - Governance audit |  - Unified API   |    for execution   |
+---------------------+---------------------+------------------+--------------------+
```

1. **MLflow Tracking**:
   - An API and UI for logging parameters, code versions, metrics, and output files when running machine learning code.
   - Organizes work into **Experiments** (logical groups of runs) and individual **Runs** (executions of machine learning code).
   - Supports **Nested Runs** to structure hyperparameter sweeps where a parent run aggregates child trial iterations.

2. **MLflow Model Registry**:
   - A centralized model store, set of APIs, and UI to collaboratively manage the full lifecycle of an MLflow Model.
   - Provides model lineage, model versioning, transition states, model aliases (e.g., `@champion`), and governance metadata tagging.

3. **MLflow Models**:
   - A standard format for packaging machine learning models that can be consumed in diverse downstream tools (e.g., real-time REST serving, batch scoring via Apache Spark).
   - Stores **Model Signatures** (expected input/output schemas) and **Input Examples** to ensure data integrity during inference.

4. **Storage Architecture: Backend Store vs. Artifact Store**:
   - **Backend Store**: Stores experiment and run metadata, parameters, metrics, tags, and notes. Configured here using a persistent relational SQLite database (`sqlite:///mlflow_finguard.db`).
   - **Artifact Store**: A decoupled storage directory (`mlflow_artifacts/`) that holds heavy files such as model weights, conda/pip environment manifests, and generated evaluation plots.

---

## Project Structure

The project is structured modularly following MLOps software engineering standards:

```
MLflow-Task/
├── .gitignore                      # Git ignore rules for DB, artifacts, and cache
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Environment dependencies
├── UCI_Credit_Card.csv             # Default dataset
├── main.py                         # Primary pipeline orchestrator entry point
│
└── src/                            # Source package
    ├── __init__.py                 # Package initializer
    ├── config.py                   # Central configuration, paths, and hyperparameters
    ├── data.py                     # Ingestion, validation, splitting, and scaling
    ├── metrics.py                  # Evaluation metrics and visualization utilities
    ├── tracking.py                 # MLflow tracking URI and experiment initialization
    ├── registry.py                 # Champion model selection, registration, and governance
    ├── inference.py                # Model loading via alias and validation inference
    │
    └── models/                     # Model training sub-package
        ├── __init__.py
        ├── baseline.py             # Baseline model training and artifact logging
        └── hpo.py                  # Hyperparameter optimization with nested runs
```

---

## Pipeline Workflow

The pipeline executes through sequential tasks designed for production traceability:

### Task 1.1: Ingestion & Preprocessing (`src/data.py`)
- Reads the credit card default dataset (30,000 observations, 24 features).
- Drops arbitrary identifiers (`ID`) to prevent spurious correlations.
- Validates data completeness (verifies zero null values).
- Splits the dataset into 80% train and 20% test sets using **stratified sampling** to maintain the ~22.12% default rate across splits.
- Fits a `StandardScaler` strictly on the training partition and transforms test data to prevent data leakage.

### Task 1.2: MLflow Tracking Setup (`src/tracking.py`)
- Configures SQLite as the persistent backend store: `sqlite:///mlflow_finguard.db`.
- Sets a decoupled artifact repository: `mlflow_artifacts/`.
- Initializes the `credit-default-risk` experiment.

### Task 1.3: Baseline Model Training (`src/models/baseline.py`)
- Trains a baseline `LogisticRegression` model ($C=0.1$, `max_iter=500`, solver=`lbfgs`).
- Logs hyperparameters (`C`, `max_iter`, `solver`) and phase tags (`pipeline_phase: Baseline`).
- Evaluates metrics on the test partition: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
- Generates and logs an ROC Curve chart (`baseline_roc_curve.png`) as a run artifact.

### Task 2.1: Nested-Run Hyperparameter Optimization (`src/models/hpo.py`)
- Starts a **Parent Run** (`logistic-regression-grid-search`) documenting the sweep space.
- Executes **Child Runs** (`trial_C_{value}`) across $C \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$.
- Logs individual child metrics, parameters, model signatures, and model artifacts.
- Records the overall best F1-score onto the parent run for easy filtering.

### Task 2.2: Model Registry Governance (`src/registry.py`)
- Queries the MLflow Tracking API (`MlflowClient`) to isolate child runs belonging to the specific HPO sweep.
- Identifies the top-performing candidate based on highest test ROC-AUC.
- Registers the candidate into the MLflow Model Registry under `CreditDefaultRiskClassifier`.
- Assigns the `@champion` alias to the registered version.
- Attaches corporate governance tags:
  - `validation_status`: `approved`
  - `fair_lending_audit`: `passed`

### Task 2.3: Production Inference via Alias (`src/inference.py`)
- Loads the deployed champion model dynamically using the URI: `models:/CreditDefaultRiskClassifier@champion`.
- Executes inference on a sample batch from the holdout test set without manual version hardcoding.
- Validates predictions against ground truth labels.

---

## Installation & Requirements

### 1. Prerequisites
- Python 3.10 or higher.
- A virtual or Conda environment is strongly recommended.

### 2. Install Dependencies
Install all required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `mlflow >= 2.15.0`
- `scikit-learn >= 1.2.0`
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `matplotlib >= 3.7.0`
- `cloudpickle >= 2.2.0`

---

## Running the Pipeline

### Option A: Standard Execution (Default Dataset)
Run the pipeline orchestrator:

```bash
python main.py
```

### Option B: Custom Dataset Path
Provide an explicit path to a dataset file:

```bash
python main.py path/to/UCI_Credit_Card.csv
```

---

## Viewing the MLflow UI

To inspect runs, compare metric curves, view ROC plots, and audit model versions in the MLflow web interface, launch the server using the SQLite backend store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_finguard.db
```

Once running, open your browser and navigate to:
```
http://127.0.0.1:5000
```

### What to Explore in the UI:
1. **Experiments Tab (`credit-default-risk`)**:
   - Inspect the baseline run alongside the parent grid search run.
   - Expand the parent run to see the nested child trials.
   - Select multiple runs and click **Compare** to plot parameter-metric correlations.
2. **Artifacts View**:
   - View `baseline_roc_curve.png`.
   - Inspect the `model/` artifact directory containing `MLmodel`, conda environments, and serialized model weights.
3. **Models Tab (`CreditDefaultRiskClassifier`)**:
   - Review registered model versions and history.
   - Check the `@champion` alias pointer.
   - Verify governance tags (`validation_status: approved`, `fair_lending_audit: passed`).

---

## Technical Considerations

- **Serialization Format**: In MLflow 3.x, scikit-learn models default to the `skops` format. On certain Windows environments, `skops` can encounter system DLL conflicts when querying PyTorch optional dependencies. The pipeline specifies `serialization_format="cloudpickle"` to ensure robust, crash-free model logging.
- **Model Signatures & Input Examples**: Every logged model includes an explicit schema signature inferred from `X_train` and an input example slice, preventing schema drift during downstream serving.
- **SQLite Concurrency**: SQLite is utilized for simple, zero-configuration local metadata persistence. For multi-node distributed team setups, replace the tracking URI with PostgreSQL or MySQL.
