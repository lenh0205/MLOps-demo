"""A/B router — Phase 4.

Deterministically assigns each user_id to v1 or v2 by hashing the user_id (never
random — a user must stay in the same bucket across requests or the experiment is
meaningless), proxies /recommend to that model API, logs the assignment + response to
`events-db`'s `experiment_events` table, and exposes /feedback so a later request can
attach click/purchase outcomes to the same row via `request_id`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

# Windows consoles default to cp1252 and choke on non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DEFAULT_K = 5
SPLIT_PCT = int(os.environ.get("SPLIT_PCT", "50"))

MODEL_URLS = {
    "v1": os.environ.get("MODEL_V1_URL", "http://model-v1:8000"),
    "v2": os.environ.get("MODEL_V2_URL", "http://model-v2:8000"),
}

# A default lets `app.py` be imported (e.g. by pytest, for select_variant) without a
# live events-db — the pool is only actually opened in the startup hook below.
DATABASE_URL = os.environ.get(
    "EVENTS_DATABASE_URL", "postgresql://ab_events:ab_events@events-db:5432/ab_events"
)

DB_CONNECT_RETRIES = 5
DB_CONNECT_RETRY_DELAY_SECONDS = 2


def select_variant(user_id: str) -> str:
    """Deterministic hash of user_id -> "v1" or "v2".

    Never random: the same user must land in the same bucket on every request, or the
    experiment being run is meaningless. SPLIT_PCT (env, default 50)
    is the percentage of the md5 hash space routed to v1.
    """
    h = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "v1" if h % 100 < SPLIT_PCT else "v2"


class RecommendRequest(BaseModel):
    user_id: str
    k: int = DEFAULT_K


class RecommendResponse(BaseModel):
    request_id: str
    user_id: str
    model_version: str
    recommendations: list[str]


class FeedbackRequest(BaseModel):
    request_id: str
    clicked: bool = False
    purchased: bool = False


class Metrics:
    """Hand-rolled, like model-api's — the split between arms is the cheapest sanity
    check that deterministic hashing is actually ~50/50."""

    def __init__(self) -> None:
        self.requests_by_variant = {"v1": 0, "v2": 0}
        self.errors_total = 0

    def record(self, variant: str, error: bool) -> None:
        if error:
            self.errors_total += 1
        else:
            self.requests_by_variant[variant] += 1


metrics = Metrics()


@asynccontextmanager
async def lifespan(app: FastAPI):
    last_error: Exception | None = None
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            app.state.pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
            break
        except Exception as exc:  # noqa: BLE001 - retry any connect failure, then raise
            last_error = exc
            if attempt < DB_CONNECT_RETRIES:
                print(
                    f"[ab-router] events-db connect attempt {attempt}/{DB_CONNECT_RETRIES} "
                    f"failed: {exc} — retrying in {DB_CONNECT_RETRY_DELAY_SECONDS}s"
                )
                time.sleep(DB_CONNECT_RETRY_DELAY_SECONDS)
    else:
        raise RuntimeError(f"could not connect to events-db after {DB_CONNECT_RETRIES} attempts") from last_error

    app.state.http = httpx.AsyncClient(timeout=10.0)

    yield

    await app.state.pool.close()
    await app.state.http.aclose()


app = FastAPI(title="A/B router", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics() -> dict:
    return {
        "model_v1_requests": metrics.requests_by_variant["v1"],
        "model_v2_requests": metrics.requests_by_variant["v2"],
        "errors_total": metrics.errors_total,
    }


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest) -> RecommendResponse:
    variant = select_variant(request.user_id)
    upstream_url = f"{MODEL_URLS[variant]}/recommend"

    try:
        response = await app.state.http.post(
            upstream_url, json={"user_id": request.user_id, "k": request.k}
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        metrics.record(variant, error=True)
        raise HTTPException(status_code=502, detail=f"upstream {variant} error: {exc}") from exc

    body = response.json()
    request_id = uuid.uuid4()

    async with app.state.pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO experiment_events (request_id, user_id, model_version, recommendations)
            VALUES ($1, $2, $3, $4)
            """,
            request_id,
            request.user_id,
            variant,
            json.dumps(body["recommendations"]),
        )

    metrics.record(variant, error=False)

    return RecommendResponse(
        request_id=str(request_id),
        user_id=request.user_id,
        model_version=variant,
        recommendations=body["recommendations"],
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict:
    """Turns "randomly choosing between two models" into an actual A/B test:
    the caller reports what the user did with the recommendations
    a prior /recommend call returned, keyed by that call's request_id."""
    try:
        request_uuid = uuid.UUID(request.request_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="request_id must be a UUID") from exc

    async with app.state.pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE experiment_events
            SET clicked = clicked OR $2, purchased = purchased OR $3
            WHERE request_id = $1
            """,
            request_uuid,
            request.clicked,
            request.purchased,
        )

    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="request_id not found")
    return {"status": "recorded"}
