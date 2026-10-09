"""Example runner for a shared split, forest search, and final evaluation.

Adapt this flow to your experimental design. Results stay in memory; choose how
to save them and record the settings needed to reproduce your study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from data_loading import DATASETS, load_and_split, prepare_data, prepare_final_data
from hyperband import optimise_hyperband
from random_forest import final_test_evaluation, make_evaluator
from random_search import optimise_random_search
from smbo import optimise_smbo
#from tabular_foundation import run_foundation_model
import matplotlib.pyplot as plt
import numpy as np

# Update this if the provided largest dataset is replaced.
FOUNDATION_DATASET = "covertype"

# n_trials is the example evaluation budget for each of Random Search and SMBO.
# Choose budgets and a Hyperband schedule that support your justified comparison.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_samples": 2_500,
        "min_trees": 3,
        "max_trees": 27,
        "n_trials": 9,
    },
    "course": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 81,
        "n_trials": 16,
    },
    "full": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 24,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="electricity", choices=[*DATASETS, "all"])
    parser.add_argument("--profile", default="course", choices=PROFILES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "random", "smbo", "hyperband"],
        choices=["default", "random", "smbo", "hyperband", "foundation"],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    return parser.parse_args()


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return example in-memory results; their structure is yours to adapt."""

    if args.methods == ["foundation"] and name != FOUNDATION_DATASET:
        if args.dataset == "all":
            return []
        raise ValueError(f"Foundation-only runs require --dataset {FOUNDATION_DATASET}")

    profile = PROFILES[args.profile]
    splits = load_and_split(name, args.cache_dir, profile["max_samples"], args.split_seed)
    results = []
    forest_methods = {"default", "random", "smbo", "hyperband"}.intersection(args.methods)
    if forest_methods:
        X_train, X_valid = prepare_data(splits)
        evaluator = make_evaluator(
            X_train, splits.y_train, X_valid, splits.y_valid
        )
        final_arrays = prepare_final_data(splits)
        min_trees = int(profile["min_trees"])
        max_trees = int(profile["max_trees"])

        for method in ("default", "random", "hyperband", "smbo"):
            if method not in forest_methods:
                continue
            start = perf_counter()
            cumulative_time_list = None
            if method == "default":
                config = {}  # Library defaults, with the common tree count.
                history = [evaluator(config, max_trees, args.seed)]
            elif method == "random":
                config, history = optimise_random_search(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            elif method == "smbo":
                config, history = optimise_smbo(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            else:
                config, history, cumulative_time_list = optimise_hyperband(
                    evaluator, min_trees, max_trees, args.seed
                )
            search_seconds = perf_counter() - start
            final_result = final_test_evaluation(
                config, max_trees, args.seed, *final_arrays
            )
            print(f"{name} / {method} / seed {args.seed}: {final_result}", flush=True)
            print(f"{name} / {method} / selected configuration: {config}", flush=True)
            result_entry = {
                "dataset": name,
                "method": method,
                "seed": args.seed,
                "configuration": config,
                "history": history,
                "search_seconds": search_seconds,
                "final_result": final_result,
            }
            if cumulative_time_list is not None:
                result_entry["bracket_times"] = cumulative_time_list
            results.append(result_entry)

    # Foundation model block (disabled; also uncomment the import at the top).
    # if "foundation" in args.methods and name == FOUNDATION_DATASET:
    #     result = run_foundation_model(splits, seed=args.seed)
    #     print(f"{name} / foundation / seed {args.seed}: {result}", flush=True)
    #     results.append({
    #         "dataset": name,
    #         "method": "foundation",
    #         "seed": args.seed,
    #         "result": result,
    #     })
    return results


def _json_default(obj: Any) -> Any:
    """Make numpy scalars/arrays and other odd types JSON serialisable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def main() -> None:
    """Run the selected examples; add result saving before the main study."""

    args = parse_args()
    Path("results").mkdir(exist_ok=True)
    max_trees = int(PROFILES[args.profile]["max_trees"])
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Running {name} with seed {args.seed}", flush=True)
        results = run_dataset(name, args)
        
        # save results in a format of your choice, along with the settings
        # needed to reproduce the run. Retain enough information for your plots
        # and tables.

        ### SAVE RESULTS into a json file ###
        with open(f"results/results_{name}_seed{args.seed}.json", "w") as f:
            structured_results = {
                "dataset": name,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "profile": args.profile,
                "settings": PROFILES[args.profile],
                "methods": [result["method"] for result in results],
                "results": results,
            }
            json.dump(structured_results, f, indent=4, default=_json_default)


        ### PLOT RESULTS(results, name, args.seed) ###
        
        # Plot objective values over evaluations/iterations for each method
        fig_objective, ax_objective = plt.subplots()
        for result in results:
            method = result["method"]
            if "history" in result:
                objective_values = [entry["objective"] for entry in result["history"]]
                ax_objective.plot(objective_values, label=method)
            elif method == "foundation":
                ax_objective.axhline(
                    result["result"]["auroc"],
                    color="red",
                    linestyle="--",
                    label="Foundation",
                )

        ax_objective.set_xlabel("Evaluation")
        ax_objective.set_ylabel("Objective Value")
        ax_objective.set_title(f"Objective Value over Evaluations for {name} (seed={args.seed})")
        
        if results:
            ax_objective.legend()
        fig_objective.tight_layout()
        fig_objective.savefig(f"results/objective_plot_{name}_seed{args.seed}.png")
        plt.close(fig_objective)

        timed_results = [result for result in results if "search_seconds" in result]

        # Plot search time for each method
        if timed_results:
            fig_time, ax_time = plt.subplots()
            ax_time.bar(
                [result["method"] for result in timed_results],
                [result["search_seconds"] for result in timed_results],
            )
            ax_time.set_xlabel("Method")
            ax_time.set_ylabel("Search Time (seconds)")
            ax_time.set_title(f"Search Time by Method for {name} (seed={args.seed})")
            fig_time.tight_layout()
            fig_time.savefig(f"results/time_plot_{name}_seed{args.seed}.png")
            plt.close(fig_time)

        # Plot final test AUROC versus total search wall-clock time
        fig_final_auroc, ax_final_auroc = plt.subplots()
        for result in results:
            if "final_result" not in result or "search_seconds" not in result:
                continue

            score = result["final_result"]["metrics"]["auroc"]
            seconds = result["search_seconds"]
            ax_final_auroc.scatter(seconds, score)

            ax_final_auroc.annotate(result["method"], (seconds, score), xytext=(5,4), textcoords="offset points")
        
        ax_final_auroc.set_xlabel("Search wall-clock time (in seconds)")
        ax_final_auroc.set_ylabel("Final test AUROC")
        ax_final_auroc.set_title("Final quality versus search cost")
        fig_final_auroc.tight_layout()
        fig_final_auroc.savefig(f"results/testAUROC_plot_{name}_seed{args.seed}.png")
        plt.close(fig_final_auroc)

        # Plot best validation AUROC versus cumulative evaluation time.
        # The incumbent is only updated by full-fidelity evaluations
        # (n_trees == max_trees), so low-fidelity Hyperband scores are not
        # mixed with full-fidelity ones.
        fig_best, ax_best = plt.subplots()
        for result in results:
            history = result.get("history", [])
            timed = [
                entry for entry in history
                if "objective" in entry and "elapsed_sec" in entry
            ]
            
            if not timed:
                continue
            
            elapsed = np.cumsum([entry["elapsed_sec"] for entry in timed])
            best = np.nan
            best_so_far = []
            for entry in timed:
                if entry.get("n_trees", max_trees) == max_trees:
                    best = np.fmax(best, entry["objective"])
                best_so_far.append(best)
            ax_best.step(elapsed, best_so_far, where="post", label=result["method"])

        ax_best.set_xlabel("Cumulative evaluation time (seconds)")
        ax_best.set_ylabel("Best validation AUROC so far")
        ax_best.set_title("Search progress by evaluation time")
        if ax_best.has_data():
            ax_best.legend()
        fig_best.tight_layout()
        fig_best.savefig(f"results/validationAUROC_plot_{name}_seed{args.seed}.png")
        plt.close(fig_best)

 
if __name__ == "__main__":
    main()