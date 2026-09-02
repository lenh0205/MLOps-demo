"""Generate the synthetic click/purchase interaction log for the demo.

Standard library only, on purpose: this has to run before anything is installed, and the
unit tests import it to build fixtures without needing the CSV on disk.

Modelling assumptions — these are deliberate, and they are what make V1 vs V2 a real
experiment rather than theatre:

* Each user has a primary product category they mostly interact with, and a minority have
  a secondary one. That affinity is the signal the recommender has to recover.
* A purchase is a stronger intent signal than a click: purchases land inside a user's
  affinity far more often than clicks do. If purchases carried no more information than
  clicks, weighting them more heavily (V2) could not possibly help.
* One event per (user, product) pair, with the event type drawn once when the product is
  first selected. This keeps leave-one-out evaluation honest: the held-out product never
  appears earlier in the same user's history, so the recommender cannot be denied credit
  for it by the "already seen" filter.

Note what is NOT encoded here: nothing forces the offline winner to also be the online
winner. Phase 5's traffic simulator decides online behaviour independently, which is
exactly how offline and online conclusions come to disagree in real life.

Usage:
    python data/generate_data.py                     # writes data/interactions.csv
    python data/generate_data.py --seed 7 --users 500
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

CLICK = "click"
PURCHASE = "purchase"

DEFAULT_SEED = 42
DEFAULT_USERS = 200
DEFAULT_PRODUCTS = 40
DEFAULT_CATEGORIES = 5

MIN_INTERACTIONS = 6
MAX_INTERACTIONS = 18

P_SECOND_CATEGORY = 0.25   # share of users with a secondary interest
P_FROM_PRIMARY = 0.70      # draw from primary category
P_FROM_SECONDARY = 0.15    # draw from secondary category (else: uniform noise)

P_PURCHASE_IN_AFFINITY = 0.22
P_PURCHASE_OUTSIDE = 0.04

WINDOW_DAYS = 60
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)

FIELDNAMES = ["user_id", "product_id", "event", "timestamp"]


def product_ids(n_products: int) -> list[str]:
    return [f"P{i:02d}" for i in range(1, n_products + 1)]


def category_pools(products: list[str], n_categories: int) -> list[list[str]]:
    """Split the catalogue into contiguous, roughly equal category pools."""
    size = len(products) // n_categories
    pools = [products[i * size:(i + 1) * size] for i in range(n_categories)]
    pools[-1].extend(products[n_categories * size:])  # any remainder joins the last pool
    return pools


def generate_interactions(
    seed: int = DEFAULT_SEED,
    n_users: int = DEFAULT_USERS,
    n_products: int = DEFAULT_PRODUCTS,
    n_categories: int = DEFAULT_CATEGORIES,
) -> list[dict[str, str]]:
    """Return interaction rows as dicts. Deterministic for a given seed."""
    rng = random.Random(seed)
    products = product_ids(n_products)
    pools = category_pools(products, n_categories)

    rows: list[dict[str, str]] = []

    for user_index in range(1, n_users + 1):
        user_id = f"U{user_index:03d}"

        primary = rng.randrange(n_categories)
        secondary = None
        if n_categories > 1 and rng.random() < P_SECOND_CATEGORY:
            secondary = rng.choice([c for c in range(n_categories) if c != primary])

        affinity = set(pools[primary])
        if secondary is not None:
            affinity |= set(pools[secondary])

        target = rng.randint(MIN_INTERACTIONS, MAX_INTERACTIONS)
        chosen: dict[str, str] = {}

        # Bounded attempts: the primary pool is small, so a user wanting many distinct
        # products necessarily picks up noise. That is realistic and keeps this terminating.
        for _ in range(target * 10):
            if len(chosen) >= target:
                break

            draw = rng.random()
            if draw < P_FROM_PRIMARY:
                pool = pools[primary]
            elif secondary is not None and draw < P_FROM_PRIMARY + P_FROM_SECONDARY:
                pool = pools[secondary]
            else:
                pool = products

            product = rng.choice(pool)
            if product in chosen:
                continue  # one row per (user, product): the event type is drawn once

            purchase_p = P_PURCHASE_IN_AFFINITY if product in affinity else P_PURCHASE_OUTSIDE
            chosen[product] = PURCHASE if rng.random() < purchase_p else CLICK

        # Give each user a browsing session with strictly increasing timestamps, so
        # "the user's last interaction" is well defined for the leave-one-out split.
        order = list(chosen.items())
        rng.shuffle(order)
        moment = EPOCH + timedelta(minutes=rng.randrange(WINDOW_DAYS * 24 * 60 // 2))
        for product, event in order:
            moment += timedelta(minutes=rng.randint(5, 24 * 60))
            rows.append(
                {
                    "user_id": user_id,
                    "product_id": product,
                    "event": event,
                    "timestamp": moment.isoformat(),
                }
            )

    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> str:
    users = {row["user_id"] for row in rows}
    products = {row["product_id"] for row in rows}
    purchases = sum(1 for row in rows if row["event"] == PURCHASE)
    return (
        f"{len(rows)} events · {len(users)} users · {len(products)} products · "
        f"{purchases} purchases ({purchases / len(rows):.1%}) · "
        f"{len(rows) / len(users):.1f} events/user"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCTS)
    parser.add_argument("--categories", type=int, default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "interactions.csv",
    )
    args = parser.parse_args()

    rows = generate_interactions(
        seed=args.seed,
        n_users=args.users,
        n_products=args.products,
        n_categories=args.categories,
    )
    write_csv(rows, args.output)
    print(f"wrote {args.output}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
