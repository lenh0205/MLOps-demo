"""Online evaluation -- Phase 5.

Queries events-db's experiment_events table (populated by ab-router + simulate_traffic.py),
prints a CTR/CVR table per model version, and names a winner backed by a two-proportion
z-test on CTR. That is as far as the statistics go -- "~5 lines", not a
full experimentation platform.

This answers a different question than trainer/evaluate_offline.py: offline hit_rate_at_5
says which model looks better on held-out history; this says which one real (simulated)
users actually clicked on. The two are allowed to disagree -- that is why
both exist.

Usage:
    python analysis/evaluate_ab.py
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from math import erf, sqrt

# Windows consoles default to cp1252 and choke on non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncpg

# Host-side default: events-db's port published to localhost (docker-compose.yml), not the
# Compose-network hostname ab-router itself uses.
DEFAULT_DATABASE_URL = os.environ.get(
    "EVENTS_DATABASE_URL", "postgresql://ab_events:ab_events@localhost:5433/ab_events"
)

SIGNIFICANCE_LEVEL = 0.01


def _normal_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def two_proportion_z_test(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sided z-test for a difference in CTR between two variants."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 2 * (1 - _normal_cdf(abs(z)))


async def fetch_stats(database_url: str) -> dict[str, dict[str, int]]:
    conn = await asyncpg.connect(database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT model_version,
                   count(*) AS requests,
                   count(*) FILTER (WHERE clicked) AS clicks,
                   count(*) FILTER (WHERE purchased) AS purchases
            FROM experiment_events
            GROUP BY model_version
            """
        )
    finally:
        await conn.close()
    return {row["model_version"]: dict(row) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args()

    stats = asyncio.run(fetch_stats(args.database_url))
    missing = [v for v in ("v1", "v2") if stats.get(v, {}).get("requests", 0) == 0]
    if missing:
        raise SystemExit(
            f"no events recorded for {missing} -- run analysis/simulate_traffic.py against "
            "the router first."
        )

    print(f"{'':10}{'requests':>10}{'clicks':>9}{'CTR':>8}{'purchases':>11}{'CVR':>8}")
    for variant in ("v1", "v2"):
        s = stats[variant]
        ctr = s["clicks"] / s["requests"]
        cvr = s["purchases"] / s["requests"]
        print(f"{variant:10}{s['requests']:10d}{s['clicks']:9d}{ctr:7.1%}{s['purchases']:11d}{cvr:7.1%}")

    v1, v2 = stats["v1"], stats["v2"]
    ctr1, ctr2 = v1["clicks"] / v1["requests"], v2["clicks"] / v2["requests"]
    z, p_value = two_proportion_z_test(v1["clicks"], v1["requests"], v2["clicks"], v2["requests"])
    winner = "v1" if ctr1 > ctr2 else "v2"
    significance = (
        f"p < {SIGNIFICANCE_LEVEL}" if p_value < SIGNIFICANCE_LEVEL else f"p = {p_value:.3f}, not significant"
    )
    print(f"\n-> {winner} wins on CTR (two-proportion z-test, z={z:.2f}, {significance})")


if __name__ == "__main__":
    main()
