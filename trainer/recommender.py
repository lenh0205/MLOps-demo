"""Item-item collaborative filtering recommender, packaged as an MLflow pyfunc model.

The algorithm is deliberately small — the exercise is about the MLOps lifecycle, not the
recommender. V1 and V2 differ in exactly one parameter, `purchase_weight`:

    V1: click = 1, purchase = 1   (all interactions equal)
    V2: click = 1, purchase = 5   (purchases are the stronger intent signal)

Being a `mlflow.pyfunc.PythonModel` is what lets MLflow package the fitted object with a
signature and load it back in the serving container with `mlflow.pyfunc.load_model`, with
no bespoke serialization on our side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel
from sklearn.metrics.pairwise import cosine_similarity

CLICK = "click"
PURCHASE = "purchase"
DEFAULT_K = 5

REQUIRED_COLUMNS = {"user_id", "product_id", "event"}


class ProductRecommender(PythonModel):
    """Score products by cosine similarity to the ones a user already interacted with."""

    def __init__(self, purchase_weight: float = 1.0, click_weight: float = 1.0) -> None:
        if purchase_weight <= 0 or click_weight <= 0:
            raise ValueError("event weights must be positive")

        self.purchase_weight = float(purchase_weight)
        self.click_weight = float(click_weight)

        # Fitted state (cloudpickled with the instance when MLflow logs the model).
        self.products_: list[str] = []
        self.similarity_: np.ndarray | None = None
        self.user_history_: dict[str, dict[str, float]] = {}
        self.popularity_: list[str] = []
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------------ training

    def event_weight(self, event: str) -> float:
        """Unknown event types are treated as the weakest signal we know about."""
        return self.purchase_weight if event == PURCHASE else self.click_weight

    def fit(self, interactions: pd.DataFrame) -> "ProductRecommender":
        missing = REQUIRED_COLUMNS - set(interactions.columns)
        if missing:
            raise ValueError(f"interactions is missing column(s): {sorted(missing)}")
        if interactions.empty:
            raise ValueError("cannot fit on an empty interaction log")

        df = interactions.loc[:, ["user_id", "product_id", "event"]].astype(str).copy()
        df["weight"] = df["event"].map(self.event_weight)

        # products x users, weighted. Cosine over the rows gives product-to-product
        # similarity: two products are similar when the same people engage with both.
        matrix = pd.pivot_table(
            df,
            index="product_id",
            columns="user_id",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )

        self.products_ = list(matrix.index)
        self._index = {product: i for i, product in enumerate(self.products_)}
        self.similarity_ = cosine_similarity(matrix.to_numpy(dtype=float))

        self.user_history_ = {
            user: dict(zip(group["product_id"], group["weight"]))
            for user, group in df.groupby("user_id", sort=False)
        }

        totals = df.groupby("product_id")["weight"].sum().sort_values(ascending=False)
        self.popularity_ = list(totals.index)

        return self

    # ----------------------------------------------------------------- inference

    def recommend(self, user_id: str, k: int = DEFAULT_K) -> list[str]:
        """Top-k product ids for one user, excluding what they already interacted with."""
        if self.similarity_ is None:
            raise RuntimeError("model is not fitted — call fit() first")

        k = max(int(k), 0)
        if k == 0:
            return []

        history = self.user_history_.get(str(user_id))
        if not history:
            # Cold start: no history to compare against, so fall back to popularity.
            return self.popularity_[:k]

        scores = np.zeros(len(self.products_), dtype=float)
        for product, weight in history.items():
            index = self._index.get(product)
            if index is not None:
                scores += weight * self.similarity_[index]

        seen = set(history)
        ranked = [
            self.products_[i]
            for i in np.argsort(-scores)
            if scores[i] > 0 and self.products_[i] not in seen
        ]
        recommendations = ranked[:k]

        # Thin catalogue corners can leave fewer than k scored candidates.
        for product in self.popularity_:
            if len(recommendations) >= k:
                break
            if product not in seen and product not in recommendations:
                recommendations.append(product)

        return recommendations

    def predict(self, context, model_input, params=None) -> list[list[str]]:
        """MLflow pyfunc entry point.

        `model_input` is a DataFrame with a `user_id` column and an optional `k` column.
        Both are always sent by the model API so the logged signature stays satisfied —
        MLflow enforces required columns, so an "optional" column is not worth the risk.
        """
        if isinstance(model_input, dict):
            model_input = pd.DataFrame(model_input)
        if "user_id" not in model_input.columns:
            raise ValueError("model_input must have a 'user_id' column")

        if "k" in model_input.columns:
            ks = model_input["k"].tolist()
        else:
            ks = [DEFAULT_K] * len(model_input)

        return [
            self.recommend(user_id, k)
            for user_id, k in zip(model_input["user_id"].tolist(), ks)
        ]
