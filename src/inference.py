"""Model inference module using MLflow model alias."""

import mlflow.pyfunc
import pandas as pd

from src.config import CHAMPION_ALIAS, MODEL_NAME


def verify_inference(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    n_samples: int = 5,
) -> pd.DataFrame:
    """Load the champion model by alias from the Model Registry and test inference on a sample."""
    print("\n" + "=" * 70)
    print("TASK 2.3 - PRODUCTION INFERENCE VIA ALIAS")
    print("=" * 70)

    model_uri = f"models:/{model_name}@{alias}"
    production_model = mlflow.pyfunc.load_model(model_uri)

    sample = X_test.iloc[:n_samples]
    truth = y_test.iloc[:n_samples].values
    pred = production_model.predict(sample)

    table = pd.DataFrame({
        "Sample": range(1, n_samples + 1),
        "Ground Truth": truth,
        "Predicted": pred,
    })
    table["Match"] = (table["Ground Truth"] == table["Predicted"]).map({True: "yes", False: "NO"})

    print("Loaded:", model_uri)
    print(table.to_string(index=False))
    correct_count = int((table["Match"] == "yes").sum())
    print(f"Correct on this slice: {correct_count} / {n_samples}")

    return table
