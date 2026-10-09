"""Optional shared forest helpers and student-completed predictive evaluation.

Reuse or replace these interfaces, including with package-native scoring and
search spaces. Unused helpers may be removed. Preserve the common
comparison and record the metric definitions and objective direction you use.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, r2_score, recall_score, precision_score,
                             f1_score, roc_auc_score, log_loss)

Config = dict[str, Any]
Evaluator = Callable[[Config, int, int], dict[str, Any]]

# Example parallelism per forest; -1 uses all available CPU cores.
# Choose this for your hardware and report it when comparing runtimes.
N_JOBS = 4

# Optional starting point. Choose and justify a shared space and sampling rules,
# or use your optimiser package's search-space tools. Keep tree count separate.
SEARCH_SPACE = {
    "max_depth": (None, 4, 8, 16, 32),
    "max_features": ("sqrt", 0.5, 1.0, "log2"),
    "min_samples_leaf": (1, 2, 4, 8, 16),
    "max_samples": (None, 2, 4, 16, 64)
}


def sample_configuration(rng: np.random.Generator) -> Config: # Done
    """TODO if using this helper: sample a legal configuration using rng.

    Return the hyperparameters to pass to RandomForestClassifier. Account for
    dependencies between parameters if you extend the example search space.
    """
    config = {}
    for param, values in SEARCH_SPACE.items():
        config[param] = values[rng.integers(len(values))]

    return config


def make_classifier(config: Config, n_estimators: int, seed: int) -> RandomForestClassifier:
    """Use {} for the untuned baseline; omitted parameters keep library defaults.

    Tree count, seed, and parallelism are set here, outside the search space.
    Invalid configurations are left for scikit-learn to reject during fitting.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=N_JOBS,
        **config,
    )


def predictive_metrics(
    model: Any, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """TODO: evaluate the fitted model using your chosen predictive metrics.

    Return metric names mapped to scalar values. Choose the prediction outputs
    your metrics require. Use the same definitions for validation, final testing,
    and the foundation comparison. This function must not fit the model.
    """
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    y = np.argmax(y, axis=1) if y.ndim > 1 else y  # Convert one-hot to class labels if needed

    # Metrics

    if y_proba.shape[1] == 2:
        auroc = roc_auc_score(y, y_proba[:, 1])
    else:
        auroc = roc_auc_score(y, y_proba, multi_class="ovr",
                              average="macro", labels=model.classes_)
        
    accuracy = accuracy_score(y, y_pred) # accuracy
    r2 = r2_score(y, y_pred)  # R^2 score
    precision = precision_score(y, y_pred, average="macro", zero_division=0)  # precision
    recall = recall_score(y, y_pred, average="macro", zero_division=0)  # recall
    f1 = f1_score(y, y_pred, average="macro")  # F1 score

    return {"accuracy": accuracy,
             "r2": r2, 
             "precision": precision,
             "recall": recall,
             "auroc": auroc,
             "f1": f1}


def validation_objective(metrics: dict[str, float]) -> float:
    """TODO: return the scalar objective used to compare configurations.

    Explain its relationship to the primary metric and whether higher or lower
    is better. Apply that direction consistently in all optimisers.
    """
    return metrics["auroc"] # Simple test, it doesn't mean it will be our final evaluation metric
                            # Higher is better here

    #raise NotImplementedError("Implement validation_objective in random_forest.py")


def make_evaluator(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Evaluator:
    """Evaluate on one split; timing includes model construction, fit, and scoring.

    The returned dictionary is an optional in-memory interface.
    """

    def evaluate(config: Config, n_trees: int, seed: int) -> dict[str, Any]:
        start = perf_counter()
        model = make_classifier(config, n_trees, seed)
        model.fit(X_train, y_train)
        metrics = predictive_metrics(model, X_valid, y_valid)
        objective = validation_objective(metrics)

        return {
            "configuration": dict(config),
            "metrics": metrics,
            "objective": float(objective),
            "n_trees": int(n_trees),
            "elapsed_sec": float(perf_counter() - start),
        }

    return evaluate


def final_test_evaluation(
    config: Config,
    n_estimators: int,
    seed: int,
    X_train_valid: np.ndarray,
    y_train_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Fit on all non-test data and score the test set, with the same timing scope."""

    start = perf_counter()
    model = make_classifier(config, n_estimators, seed)
    model.fit(X_train_valid, y_train_valid)
    metrics = predictive_metrics(model, X_test, y_test)
    return {
        "metrics": metrics,
        "elapsed_sec": float(perf_counter() - start),
    }
