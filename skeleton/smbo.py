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
    """SMBO with an Extra Trees surrogate (sambo.Optimizer).

    Uses the shared search space (each hyperparameter is mapped to an index
    into its list of choices) and trains each forest with n_trees trees.
    Uses n_trials evaluations in total. The best configuration is the one with
    the highest validation objective (higher is better).
    Returns the best configuration and the full evaluation history.
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


    # SMBO optimizer with an Extra Trees ("et") surrogate model.
    # Note: the space is discrete, so the optimizer may re-propose a
    # configuration that was already evaluated.
    optimizer = Optimizer(fun=None, bounds=bounds, estimator="et", rng=seed)
    
    tracked_configs = [] # List to track evaluated configurations
    best_config = None # this cool guy will be our best handler
    best_objective = float("-inf")  # Higher is better

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
             "cumulative_time": cumulative_time,
             "metrics": results["metrics"],
        })

        optimizer.tell([-results["objective"]])  # Update the optimizer with the new result 
                                                 # (negative, since sambo minimizes by default)

        # Update the best configuration if this one is better
        if results["objective"] > best_objective:
            best_objective = results["objective"]
            best_config = config


    return best_config, tracked_configs