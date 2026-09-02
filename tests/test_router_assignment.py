"""Phase 4 unit tests — deterministic A/B assignment, no server or DB involved.

ab-router/app.py's DATABASE_URL falls back to a default when EVENTS_DATABASE_URL is
unset, and the asyncpg pool is only opened in the FastAPI startup hook, so importing
the module here never touches a real Postgres connection.

Loaded by explicit file path, not `import app`: pyproject.toml's pythonpath lists both
model-api/ and ab-router/, and both directories have an app.py (docs/PLAN.md layout,
section 4) — a bare `import app` would silently resolve to whichever comes first on
pythonpath (model-api's, which requires MODEL_URI at import time) rather than this one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_APP_PATH = Path(__file__).resolve().parent.parent / "ab-router" / "app.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("ab_router_app", _APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load_app_module()
select_variant = app.select_variant


def test_same_user_always_gets_the_same_variant():
    user_id = "U12345"
    first = select_variant(user_id)
    for _ in range(50):
        assert select_variant(user_id) == first


def test_assignment_is_roughly_balanced_across_many_users():
    users = [f"U{i:05d}" for i in range(2000)]
    counts = {"v1": 0, "v2": 0}
    for user_id in users:
        counts[select_variant(user_id)] += 1

    # md5-hash bucketing over enough users lands close to 50/50, not exactly.
    assert 800 < counts["v1"] < 1200
    assert 800 < counts["v2"] < 1200


def test_split_pct_zero_sends_everyone_to_v2(monkeypatch):
    monkeypatch.setenv("SPLIT_PCT", "0")
    reloaded = _load_app_module()
    assert all(reloaded.select_variant(f"U{i}") == "v2" for i in range(20))


def test_split_pct_hundred_sends_everyone_to_v1(monkeypatch):
    monkeypatch.setenv("SPLIT_PCT", "100")
    reloaded = _load_app_module()
    assert all(reloaded.select_variant(f"U{i}") == "v1" for i in range(20))
