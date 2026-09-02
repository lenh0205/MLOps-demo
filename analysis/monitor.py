"""Monitoring -- Phase 6 of docs/PLAN.md.

Polls events-db's experiment_events table (the same source evaluate_ab.py reads) and
prints a Requests/CTR/CVR/error-rate table per model version, on the interval docs/PLAN.md
5.8 describes. Two layers, kept visibly separate:

    ML / business layer (CTR, CVR)   -- the focus, computed from experiment_events, which
                                         already carries model_version/clicked/purchased
                                         per request. No new storage needed.
    System / API layer (error rate)  -- minimal: pulled from model-v1/model-v2's own
                                         /metrics (docs/PLAN.md 5.5), since experiment_events
                                         has no error rows.

Alerting is one deliberate line, not a rules engine (docs/PLAN.md 5.8):

    if current_ctr < baseline_ctr * ALERT_RATIO: alert(...)

`baseline_ctr` is deliberately a parameter here, not something this script derives itself:
"the promotion decision itself supplies the number monitoring later holds the model to"
-- i.e. it is evaluate_ab.py's own printed CTR for each version from the healthy A/B test
(stage a). Defaults below are this repo's own Phase 5 run (docs/PLAN.md 6): V1 7.2%, V2
11.0%. Pass --baseline-ctr-v1/v2 if you re-run the simulation with different numbers.

Usage:
    python analysis/monitor.py                 # poll every 10s, forever, until Ctrl+C
    python analysis/monitor.py --once           # single snapshot, then exit
    python analysis/monitor.py --alert-ratio 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Windows consoles default to cp1252 and choke on non-ASCII, incl. the alert's emoji --
# see docs/PLAN.md 7.11.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

from evaluate_ab import DEFAULT_DATABASE_URL, fetch_stats

VARIANTS = ("v1", "v2")

DEFAULT_MODEL_URLS = {
    "v1": os.environ.get("MODEL_V1_METRICS_URL", "http://localhost:8001"),
    "v2": os.environ.get("MODEL_V2_METRICS_URL", "http://localhost:8002"),
}


async def fetch_error_rates(model_urls: dict[str, str]) -> dict[str, float]:
    """errors_total / requests_total from each model API's own /metrics (docs/PLAN.md
    5.5) -- experiment_events has no error rows, so this is the one thing the ML-layer
    query above can't answer."""
    rates: dict[str, float] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for variant, url in model_urls.items():
            try:
                response = await client.get(f"{url}/metrics")
                response.raise_for_status()
                body = response.json()
                total = body["requests_total"]
                rates[variant] = body["errors_total"] / total if total else 0.0
            except httpx.HTTPError as exc:
                print(f"[monitor] could not reach {url}/metrics: {exc}")
                rates[variant] = 0.0
    return rates


def print_table(stats: dict[str, dict], error_rates: dict[str, float]) -> None:
    def row(label: str, cells: dict[str, str]) -> str:
        return f"{label:<16}" + "".join(f"{cells[v]:>12}" for v in VARIANTS)

    print(row("", {v: v.upper() for v in VARIANTS}))
    print(row("Requests", {v: str(stats[v]["requests"]) for v in VARIANTS}))
    print(
        row(
            "CTR",
            {
                v: f"{stats[v]['clicks'] / stats[v]['requests']:.1%}" if stats[v]["requests"] else "n/a"
                for v in VARIANTS
            },
        )
    )
    print(
        row(
            "CVR",
            {
                v: f"{stats[v]['purchases'] / stats[v]['requests']:.1%}" if stats[v]["requests"] else "n/a"
                for v in VARIANTS
            },
        )
    )
    print(row("Error rate", {v: f"{error_rates[v]:.1%}" for v in VARIANTS}))


def check_alerts(stats: dict[str, dict], baseline_ctr: dict[str, float], alert_ratio: float) -> list[str]:
    alerts = []
    for variant in VARIANTS:
        s = stats[variant]
        if s["requests"] == 0:
            continue
        current_ctr = s["clicks"] / s["requests"]
        baseline = baseline_ctr[variant]
        threshold = baseline * alert_ratio
        if current_ctr < threshold:
            alerts.append(
                f"\U0001f6a8 model {variant} CTR {current_ctr:.1%} < threshold "
                f"{threshold:.1%} (baseline {baseline:.1%})"
            )
    return alerts


async def poll_once(args: argparse.Namespace) -> None:
    stats = await fetch_stats(args.database_url)
    for variant in VARIANTS:
        stats.setdefault(variant, {"requests": 0, "clicks": 0, "purchases": 0})

    error_rates = await fetch_error_rates({"v1": args.model_v1_url, "v2": args.model_v2_url})

    print_table(stats, error_rates)
    baseline_ctr = {"v1": args.baseline_ctr_v1, "v2": args.baseline_ctr_v2}
    for alert in check_alerts(stats, baseline_ctr, args.alert_ratio):
        print(alert)
    print()


async def run(args: argparse.Namespace) -> None:
    while True:
        await poll_once(args)
        if args.once:
            return
        await asyncio.sleep(args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    parser.add_argument("--model-v1-url", default=DEFAULT_MODEL_URLS["v1"])
    parser.add_argument("--model-v2-url", default=DEFAULT_MODEL_URLS["v2"])
    parser.add_argument(
        "--interval", type=float, default=float(os.environ.get("POLL_INTERVAL_SECONDS", 10)),
        help="seconds between polls (default: %(default)s)",
    )
    parser.add_argument("--once", action="store_true", help="print a single snapshot and exit")
    parser.add_argument(
        "--alert-ratio", type=float, default=float(os.environ.get("ALERT_RATIO", 0.8)),
        help="alert when CTR falls below baseline * this ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-ctr-v1", type=float, default=float(os.environ.get("BASELINE_CTR_V1", 0.072)),
        help="V1's CTR from the healthy A/B test, i.e. evaluate_ab.py's own number (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline-ctr-v2", type=float, default=float(os.environ.get("BASELINE_CTR_V2", 0.110)),
        help="V2's CTR from the healthy A/B test (default: %(default)s)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
