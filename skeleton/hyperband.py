"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

import random_forest
from random_forest import Config, Evaluator
import numpy as np

def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any, list[float]]: 
    """Hyperband: several successive-halving brackets.

    Each bracket s starts with many configurations trained with few trees and
    repeatedly keeps the best 1/reduction_factor, giving the survivors
    reduction_factor times more trees (refit from scratch at each rung).
    Every bracket ends at max_trees. The final configuration is the one with
    the highest validation objective among all evaluations made at full
    fidelity (max_trees), since scores at different tree counts are not
    directly comparable. Higher objective is better.

    Returns (best configuration, history of all evaluations,
    list with the total evaluation time of each bracket).
    """
    rf = reduction_factor # reduction factor to be used in the hyperband algorithm

    s_max = int(np.floor(np.log(max_trees / min_trees) / np.log(rf) + 1e-9)) # maximum number of stages in the hyperband algorithm

    best_config = None
    best_objective = float("-inf")

    history = []  # To store the results of all evaluations
    cumulative_time_list = []  # total evaluation time of each bracket

    next_id = 0  # To assign unique IDs to configurations
    run_time = 0.0  # cumulative evaluation time over the whole run

    # Offset so Hyperband's sampled configurations are independent from the
    # ones random search draws with default_rng(seed + ...).
    seed_offset = 10_000
    
    for s in range(s_max, -1, -1):
        n_configs = int(np.ceil((s_max + 1) / (s + 1) * rf ** s))
        n_trees = max_trees / rf ** s

        config_seed = []
        for i in range(n_configs):
            config_seed.append(seed + seed_offset + next_id)
            next_id += 1
            
        configs = [random_forest.sample_configuration(np.random.default_rng(cs)) for cs in config_seed]

        bracket_time = 0.0  # evaluation time spent in this bracket

        for r in range(s + 1):
            n_trees_i = min(max_trees, int(round(n_trees * rf ** r)))
            n_configs_i = max(1, len(configs) // rf)

            results = [evaluator(config, n_trees_i, cs) for cs, config in zip(config_seed, configs)]

            # to store the history of evaluations for analysis, including stage, round, number of trees, configuration seed, configuration, and objective value
            for config_seed_i, config, result in zip(config_seed, configs, results):
                bracket_time += result["elapsed_sec"]
                run_time += result["elapsed_sec"]
                history.append({
                                "stage": s,
                                "round": r,
                                "n_trees": n_trees_i,
                                "config_seed": config_seed_i,
                                "configuration": config,
                                "objective": result["objective"],
                                "elapsed_sec": result["elapsed_sec"],
                                "cumulative_time": run_time,
                        })

            # Only full-fidelity evaluations are eligible for the final selection
            if n_trees_i == max_trees:
                k = max(range(len(results)), key=lambda i: results[i]["objective"])
                if results[k]["objective"] > best_objective:
                    best_objective = results[k]["objective"]
                    best_config = configs[k]

            order = sorted(range(len(results)), key=lambda k: results[k]["objective"], reverse=True)  # For our case, higher is always better
            # Sort results and seeds based on their objective values
            keep = order[:n_configs_i]
            configs = [configs[k] for k in keep]
            config_seed = [config_seed[k] for k in keep]
   
        cumulative_time_list.append(bracket_time)

    return best_config, history, cumulative_time_list