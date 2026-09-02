"""Offline evaluation — the "does this model look promising?" half of the story.

Leave-one-out over time: hold out each user's most recent interaction, fit on everything
else, and check whether the held-out product appears in that user's top-k.

A note on the metric name. With exactly one held-out item per user, the standard
"precision@5" is a misleading label: the best achievable value is 1/5. What we actually
compute is the **hit rate** (equivalently recall@5 here) — the share of users whose held-out
product was recommended. It is the number worth putting in MLflow; precision@5 is just
hit_rate / k if anyone asks.

Random baseline for context: k / |catalogue| (5/40 = 12.5% in the default dataset).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_K = 5
MIN_INTERACTIONS = 4


def temporal_holdout(
    interactions: pd.DataFrame,
    min_interactions: int = MIN_INTERACTIONS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Split into (train, holdout).

    `holdout` maps user_id -> the product held out for them. Users with fewer than
    `min_interactions` events keep their full history and are not evaluated: holding an
    item out of a 2-event history mostly measures noise.
    """
    if "timestamp" not in interactions.columns:
        raise ValueError("a 'timestamp' column is required to hold out the latest event")

    df = interactions.sort_values(["user_id", "timestamp"], kind="stable")
    counts = df.groupby("user_id")["product_id"].transform("size")
    eligible = df[counts >= min_interactions]

    last_rows = eligible.groupby("user_id", sort=False).tail(1)
    holdout = dict(zip(last_rows["user_id"].astype(str), last_rows["product_id"].astype(str)))

    train = df.drop(index=last_rows.index)
    return train, holdout


def hit_rate_at_k(model, holdout: dict[str, str], k: int = DEFAULT_K) -> float:
    """Share of held-out products that appear in the user's top-k."""
    if not holdout:
        raise ValueError("holdout is empty — nothing to evaluate")

    hits = sum(
        1 for user_id, product in holdout.items() if product in model.recommend(user_id, k)
    )
    return hits / len(holdout)


def catalog_coverage_at_k(model, holdout: dict[str, str], k: int = DEFAULT_K) -> float:
    """Share of the catalogue that ever gets recommended.

    Worth logging next to the hit rate: a model can win on accuracy while collapsing onto
    a handful of popular products, and that shows up online as fatigue long before it
    shows up in an offline accuracy number.
    """
    if not model.products_:
        raise RuntimeError("model is not fitted — call fit() first")

    recommended = {
        product for user_id in holdout for product in model.recommend(user_id, k)
    }
    return len(recommended) / len(model.products_)


def evaluate(model, holdout: dict[str, str], k: int = DEFAULT_K) -> dict[str, float]:
    """The metric bundle logged to MLflow for each run."""
    return {
        f"hit_rate_at_{k}": hit_rate_at_k(model, holdout, k),
        f"catalog_coverage_at_{k}": catalog_coverage_at_k(model, holdout, k),
        f"random_baseline_at_{k}": k / len(model.products_),
    }
