# MLOps Demo — Product Recommender Lifecycle

A teaching demo of the **MLOps lifecycle**, end to end:

```
dataset → training → experiment tracking → model registry
        → REST serving (2 versions) → A/B traffic routing
        → outcome collection → offline vs online comparison → promotion
        → monitoring → detected degradation → rollback
```

The recommender itself is deliberately trivial (item-item cosine similarity, two
versions that differ only in one weight). **The point of this repo is the plumbing
around the model, not the model.** If you came here expecting a clever algorithm,
you're in the wrong place — if you came here to see how a model goes from a training
script to a registry to a REST API to an A/B test to a promotion/rollback, keep reading.

---

## 1. Prerequisites

- **WSL2 Ubuntu** if you're on Windows — run everything from a WSL shell, not
  PowerShell; paths under `/mnt/...` are slow.
- **Docker** and **Docker Compose**.
- **[uv](https://docs.astral.sh/uv/)** for running things outside containers (venv, tests,
  `analysis/` scripts) — it also manages the Python 3.12 interpreter itself, so a system
  Python install isn't required.

---

## 2. Quickstart — bring the whole thing up

```bash
# 1. One-time: get the venv for local tooling (tests, analysis/ scripts, data generation)
uv sync    # creates .venv from pyproject.toml + uv.lock, Python 3.12 pinned via .python-version

uv run pytest    # 70 tests, no live stack needed

# 2. Bring up MLflow + our services in one command (the -p / --project-directory /
#    --env-file flags matter: a bare `docker compose -f a -f b up` silently
#    resolves relative paths and loads .env from the wrong compose file)
docker compose -f mlflow-stack/docker-compose.yml -f docker-compose.yml \
  -p mlflow-stack --project-directory . --env-file mlflow-stack/.env up -d --build
```

That single command starts, in order: MLflow + Postgres + RustFS (official stack), the
`trainer` (registers `product-recommender` v1/v2 and sets `@champion`/`@challenger`
aliases, then exits — idempotent, so re-running the stack won't mint v3/v4), the three
model APIs, `events-db`, and `ab-router`.

Check it's up:

```bash
docker compose -f mlflow-stack/docker-compose.yml ps      # is the MLflow stack already running?
curl localhost:5000                  # MLflow UI (also reachable from Windows browser)
curl localhost:8001/health           # → {"model_version": "1", ...}
curl localhost:8002/health           # → {"model_version": "2", ...}
curl localhost:8003/health           # → resolves @champion
curl localhost:8000/health           # ab-router
```

Tear down:

```bash
docker compose -f mlflow-stack/docker-compose.yml -f docker-compose.yml \
  -p mlflow-stack --project-directory . --env-file mlflow-stack/.env down -v   # -v also drops volumes/data
```

---

## 3. Walking through the demo (golden path)

Once the stack is up, this is the story in order — each step is one command.

**① Train.** Already ran automatically as the `trainer` container. Confirm in the MLflow
UI (`http://localhost:5000`): experiment `product-recommendation` has two runs
(`purchase_weight` 1 and 5) with a real `hit_rate_at_5` metric each (~0.600 vs ~0.605 —
a near-tie on purpose, see §7 below).

**② Register.** Same UI, Models tab: `product-recommender` has v1 and v2, each traceable
to its run.

**③ Serve.** `curl -X POST localhost:8001/recommend -d '{"user_id":"U1","k":5}' -H 'content-type: application/json'`
vs the same against `:8002` — same request, different model, different ranking.

**④ A/B test traffic.**

```bash
uv run python analysis/simulate_traffic.py --stage a --n 10000
```

Sends synthetic users through `ab-router` (deterministic `md5(user_id)` hashing splits
them ~50/50 into v1/v2), each one calling `/recommend` then `/feedback`. Every assignment
and outcome lands in `events-db.experiment_events`.

**⑤ Evaluate online.**

```bash
uv run python analysis/evaluate_ab.py
```

Prints a CTR/CVR table per version and a two-proportion z-test verdict — e.g. "V2 wins
(z=-6.54, p<0.01)" even though the two models were an offline near-tie.

**⑥ Promote.**

```bash
uv run python analysis/promote_model.py v2
```

Moves the `@champion` alias to v2 via `MlflowClient`, then `POST`s `/reload` on
`model-champion`. `curl localhost:8003/health` now reports version 2 — no rebuild, no
redeploy, no config change.

**⑦ Degrade + monitor.**

```bash
uv run python analysis/simulate_traffic.py --stage b --product-shift --n 5000
uv run python analysis/monitor.py --once
uv run python analysis/drift_monitor.py
```

Stage B simulates the now-promoted model decaying online. `monitor.py` compares a recent
window of CTR against the A/B baseline and prints a `🚨` alert once it crosses the
threshold (gated on a minimum sample size so it doesn't fire on noise). `drift_monitor.py`
independently checks whether the input/recommendation distribution has drifted (PSI) —
`--product-shift` narrows the simulated user pool so the champion's recommended products
visibly concentrate, giving drift something real to detect alongside the CTR drop.

**⑧ Roll back.**

```bash
uv run python analysis/rollback_model.py
```

Same alias-move-then-`/reload` operation as promotion, just defaulted back to v1.
`curl localhost:8003/health` flips back to version 1.

The punchline: ⑥–⑧ are the *same* one-line alias operation. The running container never
changes — only which registered model version it resolves to.

---

## 4. Architecture at a glance

```
        ┌───────────────────────────────────────────┐
        │  Official MLflow Compose (untouched)       │
        │  mlflow :5000  ·  postgres  ·  rustfs      │
        └───────────────┬───────────────────────────┘
                         │ tracking + registry API
                  ┌──────┴──────┐
                  │   trainer   │  one-shot: trains V1 + V2, logs runs,
                  └──────┬──────┘  registers product-recommender v1/v2
                         │
              ┌──────────┴──────────┬─────────────────┐
              ▼                     ▼                  ▼
       ┌─────────────┐       ┌─────────────┐    ┌───────────────┐
       │ model-v1    │       │ model-v2    │    │ model-champion│  resolves
       │ :8001       │       │ :8002       │    │ :8003         │  @champion
       └──────┬──────┘       └──────┬──────┘    └───────────────┘
              └──────────┬──────────┘
                         ▼
                  ┌─────────────┐        ┌───────────────┐
                  │  ab-router  │───────▶│  events-db    │
                  │  :8000      │        │  (postgres)   │
                  └──────┬──────┘        └───────┬───────┘
                         ▼                  ┌────┴─────┐
                      Client                ▼          ▼
                                   evaluate_ab.py   monitor.py / drift_monitor.py
                                   CTR V1 vs V2     🚨 alerts
                                         │               │
                                         ▼               ▼
                                  promote_model.py  rollback_model.py
                                  @champion → v2    @champion → v1
```

**There is no model-registry container.** The registry is a capability of the MLflow
server itself, backed by the same Postgres the tracking data lives in — don't go
looking for a separate registry service to add.

| Port | Service | What it is |
|---|---|---|
| `5000` | `mlflow` | MLflow UI + tracking/registry API (official stack) |
| `8000` | `ab-router` | Public entry point — A/B traffic routing |
| `8001` | `model-v1` | Model API pinned to `product-recommender` v1 (A/B arm) |
| `8002` | `model-v2` | Model API pinned to v2 (A/B arm) |
| `8003` | `model-champion` | Model API resolving `models:/product-recommender@champion` — the "production" endpoint promotion/rollback act on |
| `5433` | `events-db` | Postgres holding A/B assignment + outcome events |
| `5432` | `mlflow-postgres` | MLflow's own backend store (tracking + registry metadata) — not ours to touch |
| `9000`/`9001` | `storage` (RustFS) | Artifact store behind MLflow — never talked to directly by our code |

**Two versions, one image.** `model-v1`, `model-v2`, and `model-champion` are the exact
same Docker image (`mlops-demo/model-api`); only the `MODEL_URI` env var differs. That's
the whole "one application artifact, different model versions" story this demo is built
to tell.

---

## 5. Repository layout — where to look for what

```
MLOps-demo/
├── docker-compose.yml     # OUR services (trainer, model APIs, router, events-db)
├── pyproject.toml         # pytest config — pythonpath into each service dir
│
├── mlflow-stack/          # vendored official MLflow Compose clone — DO NOT EDIT
│   ├── docker-compose.yml #   (except mlflow-stack/.env, which is ours)
│   └── .env
│
├── data/
│   ├── generate_data.py   # stdlib-only synthetic dataset generator (seeded)
│   └── interactions.csv   # the dataset itself
│
├── trainer/               # trains + logs + registers v1/v2 — one-shot container
│   ├── recommender.py     #   the model class (mlflow.pyfunc.PythonModel)
│   ├── evaluate_offline.py#   leave-one-out hit_rate@5 + coverage
│   ├── train.py           #   entry point
│   └── Dockerfile
│
├── model-api/             # FastAPI serving one model version each
│   ├── app.py              #   /health /metrics /reload /recommend
│   └── Dockerfile
│
├── ab-router/              # deterministic hash-routing between v1/v2
│   ├── app.py               #   /recommend /feedback /metrics /health
│   ├── schema.sql            #   experiment_events table (auto-applied on first boot)
│   └── Dockerfile
│
├── analysis/               # everything you run *against* the live stack
│   ├── simulate_traffic.py #   generates synthetic users + click/purchase feedback
│   ├── evaluate_ab.py      #   CTR/CVR table + two-proportion z-test
│   ├── monitor.py          #   windowed CTR/CVR + error-rate alerting
│   ├── drift_monitor.py    #   PSI-based data & recommendation drift detection
│   ├── promote_model.py    #   moves @champion → a version, then POST /reload
│   └── rollback_model.py   #   promote_model.py with target defaulted to v1
│
└── tests/                  # pytest, 46 tests, all pure-function / no live stack needed
```

**Flat modules, no package.** `pyproject.toml` puts `trainer`, `data`, `model-api`,
`ab-router`, and `analysis` directly on pytest's `pythonpath`. Imports inside the repo
look like `from recommender import ProductRecommender`, never
`from trainer.recommender import ...`. If you add a new service directory, add it to
that `pythonpath` list too.

---

## 6. Where to look, by role

**Data scientist** — the model and the offline evaluation:
- `data/generate_data.py` / `data/interactions.csv` — the synthetic dataset. Regenerate
  with `python data/generate_data.py` (seeded, deterministic) or
  `python data/generate_data.py --seed 7 --users 500` for a different sample.
- `trainer/recommender.py` — the `mlflow.pyfunc.PythonModel`. V1/V2 differ only by
  `purchase_weight` (1 vs 5); that's the only "model change" this demo makes on purpose.
- `trainer/evaluate_offline.py` — leave-one-out `hit_rate_at_5` / `catalog_coverage_at_5`
  / `random_baseline_at_5`. Note it's `hit_rate`, not `precision` — see the design
  invariants (§7) for why.
- `trainer/train.py` — the training entry point; logs params/metrics/tags to MLflow and
  registers both versions. Run it directly against a live MLflow with
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python trainer/train.py`.
- Offline near-ties (0.600 vs 0.605 hit rate) are the intended result, not a bug — don't
  tune the data to manufacture a bigger gap; the whole demo is built on "offline can't
  decide, so run an A/B test."

**Backend engineer** — the serving and routing layer:
- `model-api/app.py` — FastAPI app, loads one `MODEL_URI` at startup
  (`models:/product-recommender/N` or `models:/product-recommender@champion`), serves
  `/health`, `/metrics`, `/reload`, `/recommend`. All three model containers run this same
  file; only the env var differs.
- `ab-router/app.py` — `select_variant()` does deterministic `md5(user_id) % 100` hash
  routing (tunable via `SPLIT_PCT`), proxies to `model-v1`/`model-v2` over `httpx`, and
  writes every assignment + later `/feedback` outcome to Postgres
  (`ab-router/schema.sql` is the schema, auto-applied on first container boot).
- Both APIs expose `/metrics` (hand-rolled counters — no Prometheus here, see §8) — that's
  the seam a real observability stack would plug into.
- `/reload` is the actual deployment mechanism: an MLflow alias resolves once at model
  load time, so moving `@champion` does nothing to a running container until `/reload` is
  called.

**DevOps / MLOps engineer** — infra, containers, lifecycle scripts:
- `docker-compose.yml` (root) — our services only; joins the official MLflow stack's
  network by name (`mlflow-network`, external). See the "bring the whole thing up"
  invocation in §2 above — the `-p`, `--project-directory`, and `--env-file` flags are
  load-bearing, not cosmetic.
- `mlflow-stack/` — the vendored official MLflow Compose clone. **Only `.env` inside it is
  ours to edit**; everything else stays untouched, with overrides layered in from our own
  `docker-compose.yml` instead (e.g. the `MLFLOW_SERVER_ALLOWED_HOSTS` override needed for
  container-to-container calls to succeed mlflow 3.8.1's Host-header check).
- `trainer/Dockerfile`, `model-api/Dockerfile`, `ab-router/Dockerfile` — one image each;
  `model-api`'s image is reused three times (`model-v1`/`model-v2`/`model-champion`) via
  env var, not three builds.
- `analysis/promote_model.py` / `rollback_model.py` — the actual promotion/rollback
  levers; both are `MlflowClient.set_registered_model_alias(...)` + `POST /reload`.
- `analysis/monitor.py` — polls `events-db` for windowed CTR/CVR
  (`MONITOR_WINDOW`, default 5m) gated on a minimum sample size (`MIN_SAMPLE_SIZE`,
  default 500) plus each API's own `/metrics` for error rate. Run continuously (default)
  or `--once` for a single snapshot.
- `analysis/drift_monitor.py` — hand-rolled PSI (population stability index) over two
  signals: training-data vs. recent event-type mix (data drift), and a model version's
  older vs. recent recommended-product distribution (recommendation drift). Thresholds:
  `< 0.10` OK, `0.10–0.25` WARNING, `>= 0.25` DRIFT.
- Rollback and auto-mitigation are **deliberately human-in-the-loop** — the monitor and
  drift detector only print alerts; no script calls rollback automatically. That's a
  considered choice, not a gap: automated rollback needs guard rails, hysteresis, and a
  kill switch that are all out of scope for this demo.

**Anyone extending or customizing this demo:**
- Read the design invariants (§7) before changing a component that touches them.
- Tests: `pytest` runs all 46 tests, none of which need the live Docker stack — they're
  pure-function tests against `recommender.py`, `train.py` helpers, the router's
  `select_variant`, `monitor.py`'s windowing/gating, and `drift_monitor.py`'s PSI math.
  Run a single test with `pytest tests/test_recommender.py::test_name` or filter with
  `pytest -k drift`.

---

## 7. Design invariants worth knowing before you touch anything

- **Pin the same `mlflow` version everywhere.** Version drift between the image that logs
  a model and the image that loads it is the classic way this demo breaks. Currently
  pinned to `3.8.1` — check a version exists as both a PyPI release *and* a
  `ghcr.io/mlflow/mlflow` image tag (`v`-prefixed there, unprefixed on PyPI) before
  bumping either.
- **Never call `mlflow.set_tracking_uri(...)` in code.** All services read
  `MLFLOW_TRACKING_URI` from the environment (`http://localhost:5000` locally,
  `http://mlflow:5000` inside Compose) — that's what lets the same file run in both
  contexts unmodified.
- **Dataset schema is load-bearing.** `timestamp` exists because the offline split holds
  out each user's *most recent* event; one row per `(user, product)` prevents a held-out
  item from also appearing earlier in that user's history.
- **The accuracy metric is `hit_rate_at_5`, not `precision_at_5`** — with one held-out item
  per user, precision@5 caps at 1/5, so that name would mislead.
- **`predict()` always receives both `user_id` and `k`.** MLflow enforces the logged input
  signature's required columns.
- **Alias for production, version numbers for the experiment.** `@champion`/`@challenger`
  are what production and promotion/rollback operate on; `model-v1`/`model-v2` stay
  pinned by version number because an A/B experiment has to name exactly which versions
  it's comparing.
- **There is no model-registry service to add.** It's a capability of the MLflow server,
  backed by its existing Postgres.

---

## 8. Explicitly out of scope

No frontend · no real e-commerce services · no Kafka/RabbitMQ · no Kubernetes · no cloud ·
no custom model registry · no feature store · no deep learning · no custom MLflow
Dockerfile · no full experimentation platform · no Prometheus/Grafana/OpenTelemetry (the
`/metrics` endpoints are the seam a real observability stack would consume) · no automated
rollback (rollback is a human reading a monitor alert, on purpose).
