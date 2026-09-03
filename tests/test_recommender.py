"""Phase 0 unit tests — pure Python, no MLflow server and no containers involved."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate_offline import catalog_coverage_at_k, hit_rate_at_k, temporal_holdout
from generate_data import generate_interactions
from recommender import ProductRecommender


@pytest.fixture(scope="module")
def interactions() -> pd.DataFrame:
    return pd.DataFrame(generate_interactions(seed=42))


def rows(*triples: tuple[str, str, str]) -> pd.DataFrame:
    """Build a hand-crafted interaction frame: (user_id, product_id, event)."""
    return pd.DataFrame(triples, columns=["user_id", "product_id", "event"])


# --------------------------------------------------------------------------- fitting


def test_fit_builds_a_square_symmetric_similarity_matrix(interactions):
    model = ProductRecommender().fit(interactions)

    n = len(model.products_)
    assert model.similarity_.shape == (n, n)
    assert np.allclose(model.similarity_, model.similarity_.T)
    assert np.allclose(np.diag(model.similarity_), 1.0)


def test_fit_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing column"):
        ProductRecommender().fit(pd.DataFrame({"user_id": ["U1"]}))


def test_recommend_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        ProductRecommender().recommend("U001")


def test_event_weights_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        ProductRecommender(purchase_weight=0)


# -------------------------------------------------------------------- recommending


def test_recommend_returns_k_unseen_products(interactions):
    model = ProductRecommender(purchase_weight=5).fit(interactions)

    for user_id in ["U001", "U050", "U200"]:
        recommendations = model.recommend(user_id, k=5)

        assert len(recommendations) == 5
        assert len(set(recommendations)) == 5, "no duplicates"
        assert set(recommendations) <= set(model.products_), "only real products"
        assert not set(recommendations) & set(model.user_history_[user_id]), "nothing already seen"


def test_cold_start_user_falls_back_to_popularity(interactions):
    model = ProductRecommender().fit(interactions)

    assert model.recommend("NOT-A-USER", k=5) == model.popularity_[:5]


def test_purchase_weight_changes_the_ranking():
    """V1 and V2 must be genuinely different models, not the same model twice.

    U1 has only PX. PA co-occurs with PX through two *clicking* users; PB co-occurs
    through one *purchasing* user. Weighting purchases 5x flips which one wins.
    """
    data = rows(
        ("U1", "PX", "click"),
        ("U2", "PX", "click"),
        ("U2", "PA", "click"),
        ("U3", "PX", "click"),
        ("U3", "PA", "click"),
        ("U4", "PX", "purchase"),
        ("U4", "PB", "purchase"),
    )

    v1 = ProductRecommender(purchase_weight=1).fit(data)
    v2 = ProductRecommender(purchase_weight=5).fit(data)

    assert v1.recommend("U1", k=1) == ["PA"]
    assert v2.recommend("U1", k=1) == ["PB"]


def test_recommend_never_exceeds_the_unseen_ceiling(interactions):
    """`k` larger than the catalogue is capped by the already-seen filter, not padded past it.

    The filter is load-bearing, so the most a user can ever be
    shown is |catalogue| - |their history|. Padding fills up to that ceiling and stops;
    it must never re-suggest something the user has already interacted with.
    """
    model = ProductRecommender().fit(interactions)

    seen = set(model.user_history_["U001"])
    ceiling = len(model.products_) - len(seen)

    recommendations = model.recommend("U001", k=len(model.products_) - 1)

    assert len(recommendations) == ceiling
    assert len(set(recommendations)) == ceiling      # padding introduced no duplicates
    assert not set(recommendations) & seen           # and respected the filter


def test_recommend_pads_from_popularity_when_scores_run_out():
    """When nothing co-occurs with the user's history, popularity fills the list.

    Each user here touches a different product, so no two products share a user and every
    similarity off the diagonal is 0. U1's scored candidate list is therefore empty and
    the popularity fallback is the only thing that can produce recommendations.
    """
    data = rows(
        ("U1", "PX", "click"),
        ("U2", "PA", "click"),
        ("U3", "PB", "click"),
        ("U4", "PC", "click"),
    )
    model = ProductRecommender().fit(data)

    recommendations = model.recommend("U1", k=3)

    assert set(recommendations) == {"PA", "PB", "PC"}   # everything unseen, via padding
    assert "PX" not in recommendations


def test_recommend_k_zero_is_empty(interactions):
    assert ProductRecommender().fit(interactions).recommend("U001", k=0) == []


# ------------------------------------------------------------- pyfunc predict contract


def test_predict_returns_one_list_per_row(interactions):
    model = ProductRecommender().fit(interactions)

    model_input = pd.DataFrame({"user_id": ["U001", "U002", "U003"], "k": [5, 5, 3]})
    predictions = model.predict(None, model_input)

    assert [len(p) for p in predictions] == [5, 5, 3]
    assert predictions[0] == model.recommend("U001", 5)


def test_predict_defaults_k_when_column_absent(interactions):
    model = ProductRecommender().fit(interactions)

    predictions = model.predict(None, pd.DataFrame({"user_id": ["U001"]}))

    assert len(predictions[0]) == 5


def test_predict_requires_user_id(interactions):
    model = ProductRecommender().fit(interactions)

    with pytest.raises(ValueError, match="user_id"):
        model.predict(None, pd.DataFrame({"customer": ["U001"]}))


# ------------------------------------------------------------------ offline evaluation


def test_holdout_removes_exactly_one_event_per_eligible_user(interactions):
    train, holdout = temporal_holdout(interactions)

    assert len(train) == len(interactions) - len(holdout)
    for user_id, product in holdout.items():
        user_rows = train[train["user_id"] == user_id]
        assert product not in set(user_rows["product_id"]), "held-out product must not leak"


def test_model_beats_the_random_baseline(interactions):
    """The planted category affinities have to be recoverable, or nothing downstream means anything."""
    train, holdout = temporal_holdout(interactions)
    model = ProductRecommender(purchase_weight=5).fit(train)

    hit_rate = hit_rate_at_k(model, holdout, k=5)
    random_baseline = 5 / len(model.products_)

    assert hit_rate > random_baseline * 1.5, f"hit rate {hit_rate:.3f} vs baseline {random_baseline:.3f}"


def test_coverage_is_a_fraction(interactions):
    train, holdout = temporal_holdout(interactions)
    model = ProductRecommender().fit(train)

    assert 0.0 < catalog_coverage_at_k(model, holdout, k=5) <= 1.0
