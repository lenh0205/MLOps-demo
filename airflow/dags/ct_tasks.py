"""Continuous Training task logic -- Phase 8.

Plain, Airflow-free functions so they are importable both by continuous_training_dag.py's
PythonOperators and directly by pytest -- no `apache-airflow` install needed to test this
module, the same "pure-function coverage here, real stack for the rest" split every other
analysis/ script in this repo already uses.

continuous_training_dag.py is the thin Airflow wiring around these functions plus the two
steps that genuinely need a live MLflow server (train, evaluate-against-champion) and so
can't be exercised by pytest the same way -- verified manually against the real stack
instead, same as promote_model.py / drift_monitor.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

REGISTERED_MODEL = "product-recommender"
K = 5
DEFAULT_CANDIDATE_PURCHASE_WEIGHT = 5.0

REQUIRED_COLUMNS = {"user_id", "product_id", "event", "timestamp"}
MIN_ROWS = 100


def validate_dataset(path: Path, min_rows: int = MIN_ROWS) -> None:
    """The `check_data` task: fail the DAG early rather than training
    on garbage. Raises instead of returning a bool -- Airflow marks a task failed on any
    exception, which is exactly "stop before train runs"."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) < min_rows:
        raise ValueError(f"{path} has only {len(rows)} rows, expected at least {min_rows}")

    for row in rows:
        if any(value in (None, "") for value in row.values()):
            raise ValueError(f"{path} has a row with a null/empty value: {row}")


def passes_quality_gate(
    candidate_hit_rate: float, champion_hit_rate: float, tolerance: float = 0.0
) -> bool:
    """The `evaluate` task's quality gate: the candidate must be >=
    the current champion's hit_rate_at_5, or within `tolerance` of it. `tolerance=0.0` is
    the default -- strict >=. A candidate that regresses offline accuracy,
    even slightly, should not become `@challenger`."""
    return candidate_hit_rate >= champion_hit_rate - tolerance
