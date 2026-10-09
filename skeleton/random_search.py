"""Optional interface for Random Search in the common forest space.

Implement this loop, connect a package, or use another organisation. Choose how
to retain the results needed to analyse search progress and computational effort.
"""

from __future__ import annotations

from typing import Any

import random_forest
from random_forest import Config, Evaluator
import numpy as np


def optimise_random_search(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]: # DONE
    """TODO: randomly sample and evaluate up to n_trials configurations.

    Use the shared search space and train each forest with n_trees trees.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    
    history = []
    best_config = None # this cool guy will be our best handler
    best_objective = float("-inf")  # Assuming higher is better; adjust if lower is better
    cumulative_time = 0.0

    rng = np.random.default_rng([seed, 0])
    
    for trial in range(n_trials):
        rf_sample_config = random_forest.sample_configuration(rng)
        results = evaluator(rf_sample_config, n_trees, int(rng.integers(2**31 - 1)))
        elapsed_sec = results["elapsed_sec"]
        cumulative_time += elapsed_sec
        history.append({
            "trial": trial,
            "configuration": rf_sample_config,
            "objective": results["objective"],
            "elapsed_sec": elapsed_sec,
            "cumulative_time": cumulative_time,
            "metrics": results["metrics"],
        })

        # Update the best configuration if this one is better
        if results["objective"] > best_objective:
            best_objective = results["objective"]
            best_config = rf_sample_config
        
    return best_config, history
