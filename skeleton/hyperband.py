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
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.
    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """
    rf = reduction_factor # reduction factor to be used in the hyperband algorithm

    s_max = int(np.floor(np.log(max_trees / min_trees) / np.log(rf) + 1e-9)) # maximum number of stages in the hyperband algorithm

    best_config = None

    history = []  # To store the results of all evaluations
    cumulative_time_list = []

    next_id = 0  # To assign unique IDs to configurations
    
    for s in range(s_max, -1, -1):
        n_configs = int(np.ceil((s_max + 1) / (s + 1) * rf ** s))
        n_trees = max_trees / rf ** s

        config_seed = []
        for i in range(n_configs):
            config_seed.append(seed + next_id)
            next_id += 1
            
        configs = [random_forest.sample_configuration(np.random.default_rng(cs)) for cs in config_seed]

        for r in range(s + 1):
            n_trees_i = min(max_trees, int(round(n_trees * rf ** r)))
            n_configs_i = max(1, len(configs) // rf)

            results = [evaluator(config, n_trees_i, cs) for cs, config in zip(config_seed, configs)]

            total_time = 0.0
            
            cumulative_time = 0
            # to store the history of evaluations for analysis, including stage, round, number of trees, configuration seed, configuration, and objective value
            for config_seed_i, config, result in zip(config_seed, configs, results):
                cumulative_time += result["elapsed_sec"]
                history.append({
                                "stage": s,
                                "round": r,
                                "n_trees": n_trees_i,
                                "config_seed": config_seed_i,
                                "configuration": config,
                                "objective": result["objective"],
                                "elapsed_sec": result["elapsed_sec"],
                                "cumulative_time": cumulative_time
                        })

            total_time += cumulative_time

            order = sorted(range(len(results)), key=lambda k: results[k]["objective"], reverse=True)  # For our case, higher is always better
            # Sort results and seeds based on their objective values
            keep = order[:n_configs_i]
            configs = [configs[k] for k in keep]
            config_seed = [config_seed[k] for k in keep]
   
        # After all rounds in the current stage, check if the best configuration from this stage is better than the overall best
        top = max(results, key=lambda x: x["objective"])
        if best_config is None or top["objective"] > best_config["objective"]:
            best_config = top

        cumulative_time_list.append(total_time)

    return best_config["configuration"], history, cumulative_time_list
