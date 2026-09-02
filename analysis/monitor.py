"""Monitoring -- Phase 6 of docs/PLAN.md, windowing corrected in Phase 7 (5.8).

Polls events-db's experiment_events table (the same source evaluate_ab.py reads) and
prints a Requests/CTR/CVR/error-rate table per model version, on the interval docs/PLAN.md
5.8 describes. Two layers, kept visibly separate:

    ML / business layer (CTR, CVR)   -- the focus, computed from experiment_events, which
                                         already carries model_version/clicked/purchased
                                         per request. No new storage needed.
    System / API layer (error rate)  -- minimal: pulled from model-v1/model-v2's own
                                         /metrics (docs/PLAN.md 5.5), since experiment_events
                                         has no error rows.

**Recent window, not cumulative.** Phase 6 queried *all* experiment_events since
promotion, which is misleading for anything longer-lived than a demo -- a real regression
dilutes into an ever-growing average for a long time before crossing a threshold. Phase 7
windows the CTR/CVR query on `created_at` instead (`--window`, default `MONITOR_WINDOW` env
or "5m"), so this always compares *recent* behaviour against the frozen A/B baseline, not
everything-ever-recorded behaviour.

**Minimum-sample guard.** A recent window is small right after a deploy, and a 3-request /
0-click window should not fire the same alert as a genuine degradation. The alert check is
gated on a sample-size floor (`--min-sample-size`, default `MIN_SAMPLE_SIZE` env or 500) --
not a statistical framework, just "don't alert on noise":

    if requests >= MIN_SAMPLE_SIZE and current_ctr < baseline_ctr * ALERT_RATIO: alert(...)

`baseline_ctr` is deliberately a parameter here, not something this script derives itself:
"the promotion decision itself supplies the number monitoring later holds the model to"
-- i.e. it is evaluate_ab.py's own printed CTR for each version from the healthy A/B test
(stage a). Defaults below are this repo's own Phase 5 run (docs/PLAN.md 6): V1 7.2%, V2
11.0%. Pass --baseline-ctr-v1/v2 if you re-run the simulation with different numbers.

Usage:
    python analysis/monitor.py                 # poll every 10s, forever, until Ctrl+C
    python analysis/monitor.py --once           # single snapshot, then exit
    python analysis/monitor.py --window 2m --min-sample-size 200
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Windows consoles default to cp1252 and choke on non-ASCII, incl. the alert's emoji --
# see docs/PLAN.md 7.11.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncpg
import httpx

from evaluate_ab import DEFAULT_DATABASE_URL

VARIANTS = ("v1", "v2")

_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600}


def parse_duration(value: str) -> float:
    """"5m" -> 300.0, "45s" -> 45.0, "2h" -> 7200.0, a bare number -> itself, in seconds."""
    value = value.strip().lower()
    if value and value[-1] in _DURATION_UNITS:
        return float(value[:-1]) * _DURATION_UNITS[value[-1]]
    return float(value)


async def fetch_windowed_stats(database_url: str, window_seconds: float) -> dict[str, dict[str, int]]:
    """Same shape as evaluate_ab.fetch_stats, but scoped to the last `window_seconds` of
    `created_at` instead of the whole table -- see the module docstring."""
    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT model_version,
                   count(*) AS requests,
                   count(*) FILTER (WHERE clicked) AS clicks,
                   count(*) FILTER (WHERE purchased) AS purchases
            FROM experiment_events
            WHERE created_at >= now() - make_interval(secs => $1)
            GROUP BY model_version
            """,
            window_seconds,
        )
    finally:
        await conn.close()
    return {row["model_version"]: dict(row) for row in rows}

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


def check_alerts(
    stats: dict[str, dict], baseline_ctr: dict[str, float], alert_ratio: float, min_sample_size: int
) -> list[str]:
    alerts = []
    for variant in VARIANTS:
        s = stats[variant]
        if s["requests"] < min_sample_size:
            continue
        current_ctr = s["clicks"] / s["requests"]
        baseline = baseline_ctr[variant]
        threshold = baseline * alert_ratio
        if current_ctr < threshold:
            alerts.append(
                f"\U0001f6a8 model {variant} CTR {current_ctr:.1%} < threshold "
                f"{threshold:.1%} (baseline {baseline:.1%}, n={s['requests']})"
            )
    return alerts


async def poll_once(args: argparse.Namespace) -> None:
    stats = await fetch_windowed_stats(args.database_url, args.window_seconds)
    for variant in VARIANTS:
        stats.setdefault(variant, {"requests": 0, "clicks": 0, "purchases": 0})

    error_rates = await fetch_error_rates({"v1": args.model_v1_url, "v2": args.model_v2_url})

    print(f"[recent window: {args.window}]")
    print_table(stats, error_rates)
    baseline_ctr = {"v1": args.baseline_ctr_v1, "v2": args.baseline_ctr_v2}
    for alert in check_alerts(stats, baseline_ctr, args.alert_ratio, args.min_sample_size):
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
        "--window", default=os.environ.get("MONITOR_WINDOW", "5m"),
        help="how far back created_at is queried, e.g. 5m/300s/1h (default: %(default)s)",
    )
    parser.add_argument(
        "--min-sample-size", type=int, default=int(os.environ.get("MIN_SAMPLE_SIZE", 500)),
        help="suppress the alert check below this many requests in the window (default: %(default)s)",
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
    args.window_seconds = parse_duration(args.window)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
