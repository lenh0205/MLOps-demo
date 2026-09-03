"""FastAPI service that loads one version of `product-recommender` from the MLflow
registry and serves it over REST — Phase 2.

One image, one env var: the same app serves any version, chosen entirely by `MODEL_URI`.

    MODEL_URI=models:/product-recommender@champion   # production — resolved by alias
    MODEL_URI=models:/product-recommender/2          # A/B experiment arm — pinned by number

There is deliberately no `mlflow.set_tracking_uri(...)` call here either — the SDK reads
`MLFLOW_TRACKING_URI` from the environment, same as trainer/train.py.

Run locally (Phase 2):
    MLFLOW_TRACKING_URI=http://localhost:5000 \
    MODEL_URI=models:/product-recommender/1 \
    uvicorn app:app --host 0.0.0.0 --port 8001
"""

from __future__ import annotations

import os
import re
import sys
import time

# Windows consoles default to cp1252 and choke on non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

DEFAULT_K = 5
LOAD_RETRIES = 5
LOAD_RETRY_DELAY_SECONDS = 2

# "models:/<name>/<version>" (A/B arms) vs "models:/<name>@<alias>" (production).
_VERSION_URI = re.compile(r"^models:/(?P<name>[^/@]+)/(?P<version>\d+)$")
_ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/]+)$")


class RecommendRequest(BaseModel):
    user_id: str
    k: int = DEFAULT_K


class RecommendResponse(BaseModel):
    user_id: str
    model_version: str
    recommendations: list[str]


def resolve_version(model_uri: str) -> str:
    """Resolve MODEL_URI to a concrete registry version number, for /health to report.

    An alias is resolved once, at *load* time — moving `@champion`
    later does not change what a running container reports until it reloads. Doing this
    resolution ourselves, rather than trusting `load_model`'s return value, is what lets
    /health show the resolved version rather than just echoing the alias name back.
    """
    version_match = _VERSION_URI.match(model_uri)
    if version_match:
        return version_match.group("version")

    alias_match = _ALIAS_URI.match(model_uri)
    if alias_match:
        name, alias = alias_match.group("name"), alias_match.group("alias")
        return MlflowClient().get_model_version_by_alias(name, alias).version

    raise ValueError(
        "MODEL_URI must look like 'models:/<name>/<version>' or 'models:/<name>@<alias>', "
        f"got {model_uri!r}"
    )


class ModelHandle:
    """The one loaded model, swappable in place by /reload without restarting the app."""

    def __init__(self, model_uri: str) -> None:
        self.model_uri = model_uri
        self.model = None
        self.model_version: str | None = None

    def load(self) -> None:
        """Resolve + load, retrying — the registry may not be populated yet on a cold
        stack start (gotcha 3/4)."""
        last_error: Exception | None = None
        for attempt in range(1, LOAD_RETRIES + 1):
            try:
                version = resolve_version(self.model_uri)
                model = mlflow.pyfunc.load_model(self.model_uri)
            except Exception as exc:  # noqa: BLE001 - retry any load failure, then raise
                last_error = exc
                if attempt < LOAD_RETRIES:
                    print(
                        f"[model-api] load attempt {attempt}/{LOAD_RETRIES} failed: {exc} "
                        f"— retrying in {LOAD_RETRY_DELAY_SECONDS}s"
                    )
                    time.sleep(LOAD_RETRY_DELAY_SECONDS)
                continue
            self.model_version = version
            self.model = model
            print(f"[model-api] loaded {self.model_uri} -> version {version}")
            return
        raise RuntimeError(f"could not load {self.model_uri!r} after {LOAD_RETRIES} attempts") from last_error

    def predict(self, user_id: str, k: int) -> list[str]:
        result = self.model.predict(pd.DataFrame({"user_id": [user_id], "k": [k]}))
        return list(result[0])


class Metrics:
    """Hand-rolled — the `/metrics` seam a real Prometheus would later scrape,
    not a metrics platform."""

    def __init__(self) -> None:
        self.requests_total = 0
        self.errors_total = 0
        self._latency_total_ms = 0.0

    def record(self, latency_ms: float, error: bool) -> None:
        self.requests_total += 1
        self.errors_total += int(error)
        self._latency_total_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self.requests_total == 0:
            return 0.0
        return self._latency_total_ms / self.requests_total


def get_model_uri() -> str:
    model_uri = os.environ.get("MODEL_URI")
    if not model_uri:
        raise SystemExit(
            "MODEL_URI is not set. Point it at a registered version or alias, e.g.\n"
            "    export MODEL_URI=models:/product-recommender/1"
        )
    return model_uri


metrics = Metrics()
handle = ModelHandle(get_model_uri())
handle.load()  # load once at startup, not per request

app = FastAPI(title="product-recommender model API")


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_version": handle.model_version,
        "model_uri": handle.model_uri,
    }


@app.get("/metrics")
def get_metrics() -> dict:
    return {
        "requests_total": metrics.requests_total,
        "errors_total": metrics.errors_total,
        "avg_latency_ms": round(metrics.avg_latency_ms, 2),
    }


@app.post("/reload")
def reload_model() -> dict:
    """Re-resolve MODEL_URI and swap the in-memory model — the deployment mechanism
    behind promotion and rollback. No rebuild, no restart."""
    handle.load()
    return {"status": "reloaded", "model_version": handle.model_version, "model_uri": handle.model_uri}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    start = time.perf_counter()
    error = False
    try:
        recommendations = handle.predict(request.user_id, request.k)
    except Exception as exc:
        error = True
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        metrics.record((time.perf_counter() - start) * 1000, error)

    return RecommendResponse(
        user_id=request.user_id,
        model_version=handle.model_version,
        recommendations=recommendations,
    )
