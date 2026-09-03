"""Train, evaluate, log and register V1 and V2 — the Phase 1 deliverable.

Runs the whole offline half of the lifecycle in one shot:

    interactions.csv → temporal holdout → fit(pw=1) and fit(pw=5)
                     → log params + real metrics → register product-recommender v1/v2
                     → point @champion at V1 and @challenger at V2

There is deliberately **no** `mlflow.set_tracking_uri(...)` call. The SDK reads
`MLFLOW_TRACKING_URI` from the environment, which is `http://localhost:5000` from a local
venv and `http://mlflow:5000` inside Compose. That indirection is the only reason this
same file runs unchanged in Phase 1 and Phase 3 (see docs/PLAN.md section 3.2).

Usage:
    python trainer/train.py
    python trainer/train.py --data data/interactions.csv --k 5
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles still default to cp1252, which cannot encode arrows (nor, later,
# monitor.py's alert emoji). Fail soft on the console rather than crashing a training
# run over a decorative character. See docs/PLAN.md section 7 item 11.
# Guarded with hasattr -- found running this module inside an Airflow task (Phase 8):
# Airflow replaces sys.stdout with its own StreamLogWriter, which has no .reconfigure(),
# so the unguarded call raised AttributeError before a single line of train.py ran.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from evaluate_offline import evaluate, temporal_holdout
from recommender import ProductRecommender

EXPERIMENT = "product-recommendation"
REGISTERED_MODEL = "product-recommender"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "interactions.csv"

# cloudpickle stores the model class *by reference*, so without shipping this file the
# artifact only loads on a machine that already happens to have `recommender` importable.
# Packaging it is what makes the pyfunc artifact genuinely self-contained -- which is the
# entire reason we chose pyfunc over pickle (docs/PLAN.md section 2).
CODE_PATHS = [str(Path(__file__).resolve().parent / "recommender.py")]

# V1 vs V2 is exactly one parameter. Keep it that way: the demo's whole point is that the
# two registered versions are genuinely different artifacts, not that one is cleverer.
VARIANTS = (
    ("v1", 1.0, "champion"),    # the incumbent — what production serves on a cold start
    ("v2", 5.0, "challenger"),  # the contender — the A/B test decides whether it wins
)


def load_interactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python data/generate_data.py` first."
        )
    return pd.read_csv(path)


def git_commit() -> str:
    """The commit that produced this run, so a registered version's lineage is an
    answerable query instead of an assumption (docs/PLAN.md 5.4, Phase 7)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dataset_hash(path: Path) -> str:
    """SHA-256 of the exact dataset file trained on — pairs with git_commit to answer
    "which code and which data produced this version" (docs/PLAN.md 5.4, Phase 7)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_one(
    purchase_weight: float,
    train_df: pd.DataFrame,
    holdout: dict[str, str],
    k: int,
    lineage_tags: dict[str, str],
) -> tuple[str, dict[str, float]]:
    """One MLflow run: fit, evaluate, log, register. Returns (version, metrics)."""
    with mlflow.start_run(run_name=f"pw={purchase_weight:g}") as run:
        mlflow.set_tags({**lineage_tags, "training_timestamp": datetime.now(timezone.utc).isoformat()})

        model = ProductRecommender(purchase_weight=purchase_weight).fit(train_df)
        metrics = evaluate(model, holdout, k=k)

        mlflow.log_param("purchase_weight", purchase_weight)
        mlflow.log_param("algorithm", "item-item cosine")
        mlflow.log_param("k", k)
        mlflow.log_param("n_train_events", len(train_df))
        mlflow.log_param("n_holdout_users", len(holdout))
        mlflow.log_metrics(metrics)

        info = mlflow.pyfunc.log_model(
            name="model",  # MLflow 3 renamed artifact_path -> name
            python_model=model,
            # k is in the example on purpose: MLflow enforces required signature columns,
            # so callers always send both rather than relying on an "optional" column.
            input_example=pd.DataFrame({"user_id": ["U001"], "k": [k]}),
            registered_model_name=REGISTERED_MODEL,
            code_paths=CODE_PATHS,
        )

        version = info.registered_model_version
        print(
            f"  run {run.info.run_id[:8]}  pw={purchase_weight:g}  "
            f"-> {REGISTERED_MODEL} v{version}  "
            + "  ".join(f"{name}={value:.3f}" for name, value in sorted(metrics.items()))
        )
        return str(version), metrics


def retrain_candidate(
    data_path: Path = DEFAULT_DATA,
    purchase_weight: float = 5.0,
    k: int = 5,
) -> tuple[str, dict[str, float]]:
    """Continuous Training's `train` task (docs/PLAN.md 5.11) — one new candidate
    version, not a fresh v1/v2 pair.

    Deliberately distinct from main()'s bootstrap path: main() skips training entirely
    once both variants exist (the idempotency guard above, section 5.4), which is correct
    for the one-shot Compose bootstrap but wrong for a deliberate retrain. This function
    has no idempotency check — every call registers a new version — and it never touches
    an alias itself. Pointing `@challenger` at the result is the CT DAG's `register` task
    alone (airflow/dags/ct_tasks.py); `@champion` still only ever moves through the
    human-run promote_model.py (5.9).

    `purchase_weight` defaults to V2's value (5.0): the point of this phase is proving the
    retrain pipeline runs end to end, not searching hyperparameters, so it reuses the
    weight that already won the A/B test rather than guessing a new one.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set. Point it at the tracking server, e.g.\n"
            "    export MLFLOW_TRACKING_URI=http://localhost:5000"
        )

    interactions = load_interactions(data_path)
    train_df, holdout = temporal_holdout(interactions)
    mlflow.set_experiment(EXPERIMENT)

    lineage_tags = {
        "git_commit": git_commit(),
        "dataset_hash": dataset_hash(data_path),
        "trigger": "continuous_training",
    }
    return train_one(purchase_weight, train_df, holdout, k, lineage_tags)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--retrain", action="store_true",
        help="skip the v1/v2 bootstrap and register exactly one new candidate version "
        "instead (docs/PLAN.md 5.11) — manual equivalent of the CT DAG's train task, "
        "for testing outside Airflow. Does not touch any alias.",
    )
    parser.add_argument(
        "--purchase-weight", type=float, default=5.0,
        help="candidate's purchase_weight when --retrain is set (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.retrain:
        version, metrics = retrain_candidate(args.data, args.purchase_weight, args.k)
        print(f"\ncandidate -> {REGISTERED_MODEL} v{version} (no alias moved)")
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set. Point it at the tracking server, e.g.\n"
            "    export MLFLOW_TRACKING_URI=http://localhost:5000"
        )
    print(f"tracking -> {tracking_uri}")

    interactions = load_interactions(args.data)
    train_df, holdout = temporal_holdout(interactions)
    print(
        f"data     -> {len(interactions)} events, "
        f"{interactions['user_id'].nunique()} users, "
        f"{interactions['product_id'].nunique()} products\n"
        f"holdout  -> {len(holdout)} users evaluated on their most recent event"
    )

    mlflow.set_experiment(EXPERIMENT)

    lineage_tags = {"git_commit": git_commit(), "dataset_hash": dataset_hash(args.data)}
    print(f"lineage  -> git_commit={lineage_tags['git_commit'][:8]}  dataset_hash={lineage_tags['dataset_hash'][:12]}")

    client = MlflowClient()

    # Idempotency (docs/PLAN.md 5.4): re-running this against a registry that already has
    # both variants would otherwise mint v3/v4 on every `docker compose up`. Skip instead
    # of training again, and drive serving by alias/version numbers already in place.
    existing_versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if len(existing_versions) >= len(VARIANTS):
        print(
            f"{REGISTERED_MODEL} already has {len(existing_versions)} version(s) — "
            "skipping training (idempotent)."
        )
        for _, _, alias in VARIANTS:
            mv = client.get_model_version_by_alias(REGISTERED_MODEL, alias)
            print(f"  @{alias} -> v{mv.version}")
        return

    results: dict[str, dict[str, float]] = {}
    for label, purchase_weight, alias in VARIANTS:
        version, metrics = train_one(purchase_weight, train_df, holdout, args.k, lineage_tags)
        results[label] = metrics

        # Serving resolves models by alias, never by number (docs/PLAN.md 5.5), so the
        # aliases have to exist before any model API can start. Re-running the trainer
        # creates new versions and re-points the aliases at them, which keeps the script
        # idempotent in effect even though version numbers keep climbing.
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
        print(f"  alias @{alias} -> v{version}")

    hit = f"hit_rate_at_{args.k}"
    v1, v2 = results["v1"][hit], results["v2"][hit]
    print(
        f"\noffline {hit}: V1={v1:.3f}  V2={v2:.3f}  (delta={v2 - v1:+.3f})\n"
        "Near-ties here are the intended result, not a bug — this is exactly the\n"
        "situation the A/B test exists to settle (docs/PLAN.md section 2)."
    )


if __name__ == "__main__":
    main()
