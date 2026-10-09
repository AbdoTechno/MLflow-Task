"""Data ingestion and preprocessing module."""

import os
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_CSV_PATH, ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_and_preprocess_data(
    csv_path: str = DEFAULT_CSV_PATH,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Ingest dataset from CSV, validate integrity, and scale numerical features.

    Scaling is fitted strictly on the training partition to prevent data leakage.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: '{csv_path}'")

    print("\n" + "=" * 70)
    print("TASK 1.1 - INGESTION AND PREPROCESSING")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Ingested dataset dimensions : {df.shape[0]} rows x {df.shape[1]} columns")

    counts = df[target_column].value_counts().sort_index()
    print(f"Class balance on {target_column}:")
    for label, count in counts.items():
        print(f"  class {int(label)} : {count:>6} ({count / len(df):.2%})")

    # Drop identifier column if present
    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    # Validate missing values
    null_count = int(X.isnull().sum().sum())
    assert null_count == 0, f"Found {null_count} null values in feature columns."

    X = X.astype(float)
    print(
        f"Feature columns: {X.shape[1]} | null values: {null_count} | "
        f"dtypes: {sorted(X.dtypes.astype(str).unique())}"
    )

    # Stratified split to preserve class imbalance ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standard scaling fitted only on train split
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

    print(
        f"Train: {X_train_s.shape} (default rate {y_train.mean():.2%}) | "
        f"Test: {X_test_s.shape} (default rate {y_test.mean():.2%})"
    )

    return X_train_s, X_test_s, y_train, y_test
