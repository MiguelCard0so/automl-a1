"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations

from typing import Any

import random_forest as rf
from random_forest import Config, Evaluator
import numpy as np
from sambo import Optimizer


def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    if n_trials <= 0:
            raise ValueError("n_trials must be positive")

    rng = np.random.default_rng(seed)

    keys = list(rf.SEARCH_SPACE)
    choices = {k: list(rf.SEARCH_SPACE[k]) for k in keys}
    bounds = [(0, len(choices[k]) - 1) for k in keys]

    # Define a function to convert a sampled point in the search space to a configuration
    def to_config(x: np.ndarray) -> Config:
        return {k: choices[k][int(round(i))] for k, i in zip(keys, x)}


    optimizer = Optimizer(fun=None, bounds=bounds, estimator="et", rng=seed) # smbo optimizer with Gaussian Process surrogate model
    
    tracked_configs = [] # List to track evaluated configurations
    best_config = None # this cool guy will be our best handler
    best_objective = float("-inf")  # Assuming higher is better; adjust if lower

    cumulative_time = 0.0  # To track the cumulative time taken for evaluations

    for trial in range(n_trials):
        # Sample a new configuration, potentially using information from tracked_configs
        x = optimizer.ask(1)[0]  # Ask for one new point
        config = to_config(x)
        
        # Evaluate the configuration
        results = evaluator(config, n_trees, int(rng.integers(2**31 - 1)))

        cumulative_time += results["elapsed_sec"]
        # Track the evaluated configuration and its objective
        tracked_configs.append({
             "trial": trial,
             "configuration": config,
             "objective": results["objective"],
             "elapsed_sec": results["elapsed_sec"],
             "metrics": results["metrics"],
        })

        optimizer.tell([-results["objective"]])  # Update the optimizer with the new result 
                                                 # (negative, since smbo minimizes by default)

        # Update the best configuration if this one is better
        if results["objective"] > best_objective:
            best_objective = results["objective"]
            best_config = config


    return best_config, tracked_configs