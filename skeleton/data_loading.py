"""Load OpenML data and supply an example split and preprocessing.

Adapt and justify these choices for your study. Update DATASETS when replacing
a dataset; each ID identifies a specific OpenML dataset version.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# OpenML dataset IDs.
DATASETS = {
    "breast-w": 15,
    "credit-g": 31,
    "phoneme": 1489,
    "electricity": 151,
    "covertype": 1596,
}


@dataclass
class DataSplits:
    """Raw features and encoded labels for the three data partitions."""

    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


def load_dataset(
    name: str, cache_dir: str | Path = "data_cache"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load all rows of a pinned dataset; reject missing labels rather than drop rows."""

    openml.config.set_root_cache_directory(str(Path(cache_dir).expanduser().resolve()))
    dataset = openml.datasets.get_dataset(DATASETS[name], download_data=True)
    if not dataset.default_target_attribute:
        raise ValueError("The dataset has no default target; specify one for your study")
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    if X.empty or not isinstance(y, pd.Series):
        raise ValueError("Expected non-empty features and one classification target")
    if y.isna().any():
        raise ValueError("The target contains missing labels; choose and justify how to handle them")
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes")
    return X.reset_index(drop=True), LabelEncoder().fit_transform(y)


def load_and_split(
    name: str,
    cache_dir: str | Path = "data_cache",
    max_samples: int | None = None,
    seed: int = 17,
) -> DataSplits:
    """Create an example stratified 60/20/20 split; cap rows only for small checks."""

    X, y = load_dataset(name, cache_dir)
    if max_samples is not None and max_samples < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples, stratify=y, random_state=seed
        )
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_valid,
    )
    return DataSplits(
        X_train=X_train.reset_index(drop=True),
        X_valid=X_valid.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=np.asarray(y_train),
        y_valid=np.asarray(y_valid),
        y_test=np.asarray(y_test),
    )


def make_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing using training dtypes only."""

    numeric_columns = list(X_train.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [c for c in X_train.columns if c not in numeric_columns]
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_columns)
        )
    if categorical_columns:
        categorical = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        )
        transformers.append(("categorical", categorical, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(splits: DataSplits) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on training data and transform training/validation features."""

    preprocessor = make_preprocessor(splits.X_train)
    return (
        np.asarray(preprocessor.fit_transform(splits.X_train), dtype=np.float32),
        np.asarray(preprocessor.transform(splits.X_valid), dtype=np.float32),
    )


def prepare_final_data(
    splits: DataSplits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit preprocessing and prepare the complete held-out test partition."""

    X_train_valid = pd.concat([splits.X_train, splits.X_valid], ignore_index=True)
    y_train_valid = np.concatenate([splits.y_train, splits.y_valid])
    preprocessor = make_preprocessor(X_train_valid)
    X_fit = np.asarray(preprocessor.fit_transform(X_train_valid), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(splits.X_test), dtype=np.float32)
    return X_fit, y_train_valid, X_test, np.asarray(splits.y_test)

