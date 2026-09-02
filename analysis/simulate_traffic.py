"""Synthetic traffic simulator -- Phase 5 (stage a) / Phase 6 (stage b) of docs/PLAN.md.

Sends synthetic users through the A/B router, decides click/purchase per the per-variant
probabilities for the requested stage, and reports the outcome back via /feedback. Two
stages, one script (--stage a|b), not two scripts -- see docs/PLAN.md 5.7:

    stage a (healthy, during the A/B test)   V1 CTR  8%   V2 CTR 11%  -> V2 wins online
    stage b (degraded, after promotion)      V1 CTR  8%   V2 CTR  4%  -> champion decays

The per-variant probabilities are simulation inputs, deliberately fixed -- nothing here
infers degradation from the model; stage b stages an outage so Phase 6's monitor/rollback
have something real to react to.

Usage:
    python analysis/simulate_traffic.py                  # stage a, 10000 requests
    python analysis/simulate_traffic.py --stage b --n 5000
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

# Windows consoles default to cp1252 and choke on non-ASCII -- see docs/PLAN.md 7.11.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

DEFAULT_ROUTER_URL = "http://localhost:8000"
DEFAULT_N = 10000
DEFAULT_CONCURRENCY = 50

# variant -> CTR, by stage. docs/PLAN.md 5.7's table.
STAGE_CTR = {
    "a": {"v1": 0.08, "v2": 0.11},
    "b": {"v1": 0.08, "v2": 0.04},
}

# variant -> P(purchase | clicked), held fixed across stages so CVR simply follows CTR
# down when a variant degrades. Chosen so stage a reproduces docs/PLAN.md 5.7's example
# table exactly: V1 8.0%/2.4%, V2 11.0%/3.6%.
PURCHASE_GIVEN_CLICK = {"v1": 0.30, "v2": 0.3273}


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    router_url: str,
    user_id: str,
    ctr_by_variant: dict[str, float],
    rng: random.Random,
) -> tuple[str, bool, bool]:
    async with semaphore:
        response = await client.post(f"{router_url}/recommend", json={"user_id": user_id})
        response.raise_for_status()
        body = response.json()
        variant = body["model_version"]

        clicked = rng.random() < ctr_by_variant[variant]
        purchased = clicked and rng.random() < PURCHASE_GIVEN_CLICK[variant]

        feedback = await client.post(
            f"{router_url}/feedback",
            json={"request_id": body["request_id"], "clicked": clicked, "purchased": purchased},
        )
        feedback.raise_for_status()
        return variant, clicked, purchased


async def run(args: argparse.Namespace) -> None:
    ctr_by_variant = STAGE_CTR[args.stage]
    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(args.concurrency)

    counts = {"v1": {"requests": 0, "clicks": 0, "purchases": 0}, "v2": {"requests": 0, "clicks": 0, "purchases": 0}}

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [
            run_one(client, semaphore, args.router_url, f"sim-{args.stage}-{i:05d}", ctr_by_variant, rng)
            for i in range(args.n)
        ]
        for coro in asyncio.as_completed(tasks):
            variant, clicked, purchased = await coro
            counts[variant]["requests"] += 1
            counts[variant]["clicks"] += int(clicked)
            counts[variant]["purchases"] += int(purchased)

    print(f"stage {args.stage} -> {args.n} requests via {args.router_url}")
    for variant in ("v1", "v2"):
        c = counts[variant]
        n = c["requests"]
        ctr = c["clicks"] / n if n else 0.0
        cvr = c["purchases"] / n if n else 0.0
        print(
            f"  {variant}: requests={n:5d}  clicks={c['clicks']:4d} ({ctr:5.1%})  "
            f"purchases={c['purchases']:4d} ({cvr:5.1%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["a", "b"], default="a", help="traffic profile (default: a)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="total requests to send (default: %(default)s)")
    parser.add_argument("--router-url", default=DEFAULT_ROUTER_URL)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
