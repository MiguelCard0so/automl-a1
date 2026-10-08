# Optional student skeleton

Use or adapt this code, connect suitable packages, or build your own pipeline.
The handout defines the requirements; supplied settings are examples.

## Set up

Use Python 3.11 or newer. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Add dependencies for your chosen optimisers and foundation model, recording
versions and any model-access steps.

## Get started

Complete or replace the relevant TODOs before running. For an initial baseline
check, implement the scoring functions in `random_forest.py`, then run:

```bash
python experiment.py --dataset breast-w --profile smoke --methods default
```

Use `python experiment.py --help` for available options.

Example profiles and budgets are in `experiment.py`: `smoke` caps rows for small
checks, `course` uses all rows, and `full` uses all rows with more trees. These
names do not indicate grading levels. `N_JOBS` in `random_forest.py` controls
parallelism for forest fits.

**Results are not saved automatically.** Add result saving in `experiment.py`
or your own pipeline before the main study.

Replace this README with your installation, pipeline-check, experiment, and
figure/table-generation instructions.
