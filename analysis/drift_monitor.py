"""Drift detection -- Phase 7.

Answers two questions upstream of the performance question monitor.py asks
("is the model producing worse outcomes?"):

    data drift            has PRODUCTION DATA changed?     event-type mix (click vs
                           purchase) in experiment_events, recent window vs the
                           training data's mix (data/interactions.csv)
    recommendation drift  has MODEL BEHAVIOUR changed?     product distribution inside
                           `recommendations` (jsonb), recent window vs a reference taken
                           from that model version's own older events

Both reduce to the same shape -- compare a reference categorical distribution to a recent
one -- so one function, `psi()` (population stability index), serves both. Hand-rolled,
no new dependency, same spirit as evaluate_ab.py's two-proportion z-test.

Drift does not auto-rollback. This script prints a table and a verdict, nothing more --
the same human-in-the-loop principle as promote_model.py / rollback_model.py:

    DRIFT DETECTED
          |
          +-- performance OK       -> keep monitoring
          +-- performance degraded -> rollback / retrain

Usage:
    python analysis/drift_monitor.py                # inspects whichever version @champion resolves to
    python analysis/drift_monitor.py --version v2
    python analysis/drift_monitor.py --window 5m --min-sample-size 500

Phase 8 addition: each run appends its verdict to a small local
history file and, if drift has been persistent across the last two checks *and* the
sample floor is met *and* the training data still passes a basic quality check, calls
Airflow's REST API to kick off the `continuous_training` DAG (check_data -> train ->
evaluate -> register) and stops -- it does not poll the DAG's outcome. That gate is why a
single noisy DRIFT reading never triggers a retrain by itself:

    python analysis/drift_monitor.py --airflow-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Windows consoles default to cp1252 and choke on non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncpg
import httpx
from mlflow.tracking import MlflowClient

from evaluate_ab import DEFAULT_DATABASE_URL
from monitor import check_alerts, fetch_windowed_stats, parse_duration

REGISTERED_MODEL = "product-recommender"
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "interactions.csv"

# Phase 8: the Continuous Training trigger gate and its Airflow call.
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent / ".drift_history.json"
HISTORY_LIMIT = 5  # only the last 2 checks matter to should_trigger_ct; a few more is plenty for debugging
CT_DAG_ID = "continuous_training"
DEFAULT_AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
DEFAULT_AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
DEFAULT_AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "admin")
REQUIRED_DATA_COLUMNS = {"user_id", "product_id", "event", "timestamp"}


@dataclass
class DriftResult:
    verdict: str  # "OK" | "WARNING" | "DRIFT"
    timestamp: str  # UTC ISO-8601, when this check ran


_VERDICT_ORDER = {"n/a": -1, "OK": 0, "WARNING": 1, "DRIFT": 2}


def combined_verdict(data_status: str, rec_status: str) -> str:
    """The worse of the data-drift and recommendation-drift verdicts for one check --
    what gets appended to the history should_trigger_ct reads."""
    worse = max(data_status, rec_status, key=lambda status: _VERDICT_ORDER.get(status, -1))
    return worse if _VERDICT_ORDER.get(worse, -1) >= 0 else "OK"


def load_drift_history(path: Path) -> list[DriftResult]:
    if not path.exists():
        return []
    return [DriftResult(**item) for item in json.loads(path.read_text())]


def save_drift_history(path: Path, history: list[DriftResult]) -> None:
    path.write_text(json.dumps([asdict(r) for r in history[-HISTORY_LIMIT:]]))


def data_quality_ok(data_path: Path) -> bool:
    """A cheap schema/null check on the training data -- the same
    thing the CT DAG's own check_data task validates, so the trigger never fires on data
    the DAG would immediately reject anyway."""
    try:
        with data_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not REQUIRED_DATA_COLUMNS.issubset(reader.fieldnames or []):
                return False
            return all(
                value not in (None, "") for row in reader for value in row.values()
            )
    except OSError:
        return False


def should_trigger_ct(
    drift_history: list[DriftResult], sample_size: int, min_sample_size: int, data_ok: bool
) -> bool:
    """The combined gate. Drift alone is not enough: it has to be
    persistent across the last two checks (not one noisy reading), the window has to have
    enough traffic to trust, and the data has to still look sane."""
    return (
        len(drift_history) >= 2
        and all(result.verdict == "DRIFT" for result in drift_history[-2:])
        and sample_size >= min_sample_size
        and data_ok
    )


def trigger_continuous_training(airflow_url: str, username: str, password: str) -> str:
    """POSTs to Airflow's REST API and returns the new run's id. Raises on any HTTP
    failure -- the caller decides how to report that, this function does not swallow it."""
    response = httpx.post(
        f"{airflow_url}/api/v1/dags/{CT_DAG_ID}/dagRuns",
        json={},  # let Airflow assign dag_run_id/logical_date
        auth=(username, password),
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json().get("dag_run_id", "?")

# Same convention as promote_model.py's LABEL_TO_VERSION, inverted -- v1/v2 register in
# that fixed order in trainer/train.py's VARIANTS tuple.
VERSION_TO_LABEL = {"1": "v1", "2": "v2"}

# Fixed thresholds -- not a tuned model.
PSI_WARNING = 0.10
PSI_DRIFT = 0.25

# Floors ref/recent percentages away from zero so a category present on only one side
# doesn't produce log(0) or a division by zero.
EPSILON = 1e-4


def psi(reference: dict[str, float], recent: dict[str, float]) -> float:
    """Population stability index between two categorical distributions.
    Both dicts are category -> proportion (need not share the same categories)."""
    total = 0.0
    for category in set(reference) | set(recent):
        ref_pct = max(reference.get(category, 0.0), EPSILON)
        rec_pct = max(recent.get(category, 0.0), EPSILON)
        total += (rec_pct - ref_pct) * math.log(rec_pct / ref_pct)
    return total


def classify(value: float) -> str:
    if value >= PSI_DRIFT:
        return "DRIFT"
    if value >= PSI_WARNING:
        return "WARNING"
    return "OK"


def normalize(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def resolve_champion_label() -> str:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    return VERSION_TO_LABEL.get(mv.version, f"v{mv.version}")


# --------------------------------------------------------------------------- data drift


def training_event_mix(data_path: Path) -> dict[str, float]:
    """Reference distribution: the click/purchase split data/interactions.csv had at
    training time."""
    counts: dict[str, int] = {}
    with data_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["event"]] = counts.get(row["event"], 0) + 1
    return normalize(counts)


async def recent_event_mix(database_url: str, version: str, window_seconds: float) -> tuple[dict[str, float], int]:
    """Among `version`'s requests in the recent window that were actually engaged with
    (clicked or purchased), the click-vs-purchase split -- the same two categories
    data/interactions.csv encodes per row. Requests with neither are not "an event" any
    more than an unclicked impression would be a row in that CSV."""
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            """
            SELECT
                count(*) FILTER (WHERE purchased) AS purchase,
                count(*) FILTER (WHERE clicked AND NOT purchased) AS click
            FROM experiment_events
            WHERE model_version = $1 AND created_at >= now() - make_interval(secs => $2)
            """,
            version,
            window_seconds,
        )
    finally:
        await conn.close()
    counts = {"click": row["click"], "purchase": row["purchase"]}
    return normalize(counts), sum(counts.values())


# ------------------------------------------------------------------ recommendation drift


async def product_distribution(
    database_url: str, version: str, *, since: float | None = None, before: float | None = None
) -> tuple[dict[str, float], int]:
    """Flattened product_id counts across the `recommendations` jsonb array, for
    `version`, optionally windowed on created_at. Exactly one of since/before (or
    neither, for "all time") is expected."""
    conditions = ["model_version = $1"]
    params: list = [version]
    if since is not None:
        params.append(since)
        conditions.append(f"created_at >= now() - make_interval(secs => ${len(params)})")
    if before is not None:
        params.append(before)
        conditions.append(f"created_at < now() - make_interval(secs => ${len(params)})")
    where = " AND ".join(conditions)

    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(
            f"""
            SELECT jsonb_array_elements_text(recommendations) AS product_id, count(*) AS n
            FROM experiment_events
            WHERE {where}
            GROUP BY product_id
            """,
            *params,
        )
    finally:
        await conn.close()
    counts = {row["product_id"]: row["n"] for row in rows}
    return normalize(counts), sum(counts.values())


async def request_count(
    database_url: str, version: str, *, since: float | None = None, before: float | None = None
) -> int:
    conditions = ["model_version = $1"]
    params: list = [version]
    if since is not None:
        params.append(since)
        conditions.append(f"created_at >= now() - make_interval(secs => ${len(params)})")
    if before is not None:
        params.append(before)
        conditions.append(f"created_at < now() - make_interval(secs => ${len(params)})")
    where = " AND ".join(conditions)

    conn = await asyncpg.connect(database_url)
    try:
        return await conn.fetchval(f"SELECT count(*) FROM experiment_events WHERE {where}", *params)
    finally:
        await conn.close()


# ----------------------------------------------------------------------------- reporting


def print_mix_table(reference: dict[str, float], recent: dict[str, float]) -> None:
    categories = sorted(set(reference) | set(recent))
    print(f"{'':12}{'reference':>12}{'recent':>12}")
    for category in categories:
        print(f"{category:12}{reference.get(category, 0.0):11.1%} {recent.get(category, 0.0):11.1%}")


async def run(args: argparse.Namespace) -> None:
    version = args.version or resolve_champion_label()
    window_seconds = args.window_seconds

    print(f"drift check -- model {version}, recent window {args.window}\n")

    # -- data drift: training mix vs recent engaged-request mix
    reference_mix = training_event_mix(args.data)
    recent_mix, recent_engaged_n = await recent_event_mix(args.database_url, version, window_seconds)
    data_psi = psi(reference_mix, recent_mix) if recent_engaged_n else 0.0
    data_status = classify(data_psi) if recent_engaged_n else "n/a"

    print(f"DATA DRIFT -- event-type mix, training data vs recent {version} traffic (n={recent_engaged_n})")
    print_mix_table(reference_mix, recent_mix)
    caveat = "" if recent_engaged_n >= args.min_sample_size else f"  (low sample, n={recent_engaged_n})"
    print(f"PSI = {data_psi:.3f} -> {data_status}{caveat}\n")

    # -- recommendation drift: this version's older recommendations vs its recent ones
    reference_products, reference_n = await product_distribution(args.database_url, version, before=window_seconds)
    recent_products, recent_products_n = await product_distribution(args.database_url, version, since=window_seconds)
    reference_requests = await request_count(args.database_url, version, before=window_seconds)
    recent_requests = await request_count(args.database_url, version, since=window_seconds)
    has_both_sides = reference_products and recent_products
    rec_psi = psi(reference_products, recent_products) if has_both_sides else 0.0
    rec_status = classify(rec_psi) if has_both_sides else "n/a"

    print(
        f"RECOMMENDATION DRIFT -- product distribution, {version}'s older events "
        f"(n={reference_requests} requests) vs recent (n={recent_requests} requests)"
    )
    low_sample = min(reference_requests, recent_requests) < args.min_sample_size
    caveat = f"  (low sample, n={min(reference_requests, recent_requests)})" if low_sample else ""
    print(f"PSI = {rec_psi:.3f} -> {rec_status}{caveat}\n")

    # -- tie to performance, same windowed/gated check monitor.py uses
    stats = await fetch_windowed_stats(args.database_url, window_seconds)
    for v in ("v1", "v2"):
        stats.setdefault(v, {"requests": 0, "clicks": 0, "purchases": 0})
    baseline_ctr = {"v1": args.baseline_ctr_v1, "v2": args.baseline_ctr_v2}
    alerts = check_alerts(stats, baseline_ctr, args.alert_ratio, args.min_sample_size)

    any_drift = data_status == "DRIFT" or rec_status == "DRIFT" or data_status == "WARNING" or rec_status == "WARNING"
    if any_drift:
        print("DRIFT DETECTED")
        print("      |")
        if alerts:
            for alert in alerts:
                print(f"      +-- performance degraded: {alert}")
            print("      +-- two independent signals agree -> rollback / retrain justified (analysis/rollback_model.py)")
        else:
            print("      +-- performance OK -> keep monitoring (drift alone is not a rollback trigger)")
    else:
        print("no drift detected.")

    if args.no_ct_trigger:
        return

    # -- Phase 8: persist this check's verdict and, only if drift has
    # been persistent across the last two checks *and* there's enough traffic to trust it
    # *and* the training data still looks sane, hand off to Airflow. One HTTP call, then
    # stop -- the DAG's own progress is Airflow's UI/logs to inspect (human-in-the-loop),
    # not something this script polls for.
    verdict = combined_verdict(data_status, rec_status)
    history = load_drift_history(args.history_path)
    history.append(DriftResult(verdict=verdict, timestamp=datetime.now(timezone.utc).isoformat()))
    save_drift_history(args.history_path, history)

    combined_sample_size = min(recent_engaged_n, recent_requests)
    data_ok = data_quality_ok(args.data)
    if should_trigger_ct(history, combined_sample_size, args.min_sample_size, data_ok):
        print(
            f"\nCT TRIGGER -- drift persistent across the last 2 checks, "
            f"n={combined_sample_size} >= {args.min_sample_size}, data OK -> "
            f"POST {args.airflow_url}/api/v1/dags/{CT_DAG_ID}/dagRuns"
        )
        try:
            dag_run_id = trigger_continuous_training(
                args.airflow_url, args.airflow_username, args.airflow_password
            )
            print(f"triggered {CT_DAG_ID} -> dag_run_id={dag_run_id}")
        except httpx.HTTPError as exc:
            print(f"could not trigger {CT_DAG_ID}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--version", choices=["v1", "v2"], default=None,
        help="which model version's traffic to inspect (default: resolve @champion)",
    )
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="training data, for the data-drift reference")
    parser.add_argument(
        "--window", default=os.environ.get("MONITOR_WINDOW", "5m"),
        help="how far back 'recent' looks, e.g. 5m/300s/1h (default: %(default)s)",
    )
    parser.add_argument(
        "--min-sample-size", type=int, default=int(os.environ.get("MIN_SAMPLE_SIZE", 500)),
        help="flag PSI values computed below this many requests as low-sample (default: %(default)s)",
    )
    parser.add_argument("--alert-ratio", type=float, default=float(os.environ.get("ALERT_RATIO", 0.8)))
    parser.add_argument("--baseline-ctr-v1", type=float, default=float(os.environ.get("BASELINE_CTR_V1", 0.072)))
    parser.add_argument("--baseline-ctr-v2", type=float, default=float(os.environ.get("BASELINE_CTR_V2", 0.110)))
    parser.add_argument(
        "--history-path", type=Path, default=DEFAULT_HISTORY_PATH,
        help="where consecutive-check drift history for the CT trigger gate is kept (default: %(default)s)",
    )
    parser.add_argument("--airflow-url", default=DEFAULT_AIRFLOW_URL, help="default: %(default)s")
    parser.add_argument("--airflow-username", default=DEFAULT_AIRFLOW_USERNAME)
    parser.add_argument("--airflow-password", default=DEFAULT_AIRFLOW_PASSWORD)
    parser.add_argument(
        "--no-ct-trigger", action="store_true",
        help="skip the Phase 8 history update and Airflow trigger check entirely",
    )
    args = parser.parse_args()
    args.window_seconds = parse_duration(args.window)

    if not os.environ.get("MLFLOW_TRACKING_URI") and args.version is None:
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set and --version was not given. Either export it, "
            "e.g.\n    export MLFLOW_TRACKING_URI=http://localhost:5000\nor pass --version v1/v2 directly."
        )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
