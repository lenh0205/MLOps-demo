"""Continuous Training DAG -- Phase 8 of docs/PLAN.md (5.11, 3.3).

    check_data -> train -> evaluate -> register

Triggered exclusively by `analysis/drift_monitor.py`'s `POST /api/v1/dags/
continuous_training/dagRuns` call once drift is persistent, past the sample floor, and
data quality is OK (docs/PLAN.md 5.11's gate lives in drift_monitor.py, not here) -- hence
`schedule=None`. This DAG never polls for drift itself, and it never calls
`promote_model.py`: it stops at registering a candidate and pointing `@challenger` at it.
Whether the candidate becomes `@champion` is decided by the existing A/B + promotion
machinery, run by a human, exactly as docs/PLAN.md section 2's "CT != CD" split describes.

`evaluate` is a ShortCircuitOperator rather than a plain PythonOperator: when the
candidate fails the quality gate, it returns False, Airflow skips `register` (nothing is
registered, no alias moves), and the DAG still finishes as a normal, non-failed run --
"stop, nothing registered" is a completed pipeline outcome here, not an error.

Each task is a plain PythonOperator/ShortCircuitOperator -- no custom operators, no
provider packages beyond the base image, same "smallest thing that works" spirit as
evaluate_ab.py's hand-rolled z-test (docs/PLAN.md 5.7).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# The repo's trainer/ is bind-mounted into this container (docker-compose.yml's `airflow`
# service) at CT_REPO_ROOT/trainer so `train.py`'s own module (recommender.py,
# evaluate_offline.py) resolve the same way they do everywhere else in this repo -- see
# docs/PLAN.md's "Flat modules, no package" convention.
REPO_ROOT = Path(os.environ.get("CT_REPO_ROOT", "/opt/airflow/repo"))
TRAINER_DIR = str(REPO_ROOT / "trainer")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

# ct_tasks.py lives next to this file in the dags folder, which Airflow already adds to
# sys.path when it imports a DAG file -- no extra path wiring needed for this one.
from ct_tasks import (  # noqa: E402
    DEFAULT_CANDIDATE_PURCHASE_WEIGHT,
    K,
    MIN_ROWS,
    REGISTERED_MODEL,
    passes_quality_gate,
    validate_dataset,
)

DATA_PATH = REPO_ROOT / "data" / "interactions.csv"
CANDIDATE_PURCHASE_WEIGHT = float(
    os.environ.get("CT_PURCHASE_WEIGHT", DEFAULT_CANDIDATE_PURCHASE_WEIGHT)
)


def check_data(**_context) -> None:
    validate_dataset(DATA_PATH, min_rows=MIN_ROWS)


def train(ti, dag_run, **_context) -> None:
    # Imported here, not at module load time: this needs `mlflow` (and train.py's own
    # pandas/sklearn imports) installed in the Airflow container's environment
    # (docker-compose.yml's `_PIP_ADDITIONAL_REQUIREMENTS`), which ct_tasks.py's pure
    # functions above deliberately do not require, so pytest can import this module's
    # sibling without any of that installed.
    from train import retrain_candidate

    # `{"conf": {"purchase_weight": ...}}` on a manual trigger overrides the env default --
    # mainly so this DAG can be exercised end to end with a deliberately worse candidate
    # (verifying the evaluate task's FAIL branch) without recreating the container just to
    # change CT_PURCHASE_WEIGHT.
    purchase_weight = (dag_run.conf or {}).get("purchase_weight", CANDIDATE_PURCHASE_WEIGHT)

    version, metrics = retrain_candidate(
        data_path=DATA_PATH, purchase_weight=purchase_weight, k=K
    )
    ti.xcom_push(key="candidate_version", value=version)
    ti.xcom_push(key="candidate_metrics", value=metrics)


def evaluate(ti, **_context) -> bool:
    """Returns True/False rather than raising -- ShortCircuitOperator reads the return
    value to decide whether to skip `register` (docs/PLAN.md 5.11's FAIL branch)."""
    from mlflow.tracking import MlflowClient

    candidate_metrics = ti.xcom_pull(key="candidate_metrics", task_ids="train")
    candidate_hit_rate = candidate_metrics[f"hit_rate_at_{K}"]

    client = MlflowClient()
    champion_mv = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    champion_run = client.get_run(champion_mv.run_id)
    champion_hit_rate = champion_run.data.metrics.get(f"hit_rate_at_{K}", 0.0)

    passed = passes_quality_gate(candidate_hit_rate, champion_hit_rate)
    print(
        f"candidate hit_rate_at_{K}={candidate_hit_rate:.3f} vs "
        f"champion (v{champion_mv.version})={champion_hit_rate:.3f} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def register(ti, **_context) -> None:
    from mlflow.tracking import MlflowClient

    version = ti.xcom_pull(key="candidate_version", task_ids="train")
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
    print(f"@challenger -> v{version}")


with DAG(
    dag_id="continuous_training",
    description="Continuous Training: check_data -> train -> evaluate -> register (docs/PLAN.md 5.11)",
    schedule=None,  # triggered only by drift_monitor.py's REST API call, never on a timer
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops-demo", "phase-8", "continuous-training"],
) as dag:
    check_data_task = PythonOperator(task_id="check_data", python_callable=check_data)
    train_task = PythonOperator(task_id="train", python_callable=train)
    evaluate_task = ShortCircuitOperator(task_id="evaluate", python_callable=evaluate)
    register_task = PythonOperator(task_id="register", python_callable=register)

    check_data_task >> train_task >> evaluate_task >> register_task
