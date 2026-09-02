-- experiment_events: one row per /recommend request through the A/B router
-- (docs/PLAN.md 5.6). Mounted into events-db via docker-entrypoint-initdb.d, so Postgres
-- runs this automatically the first time the events-db data volume is initialized.

CREATE TABLE IF NOT EXISTS experiment_events (
    request_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    recommendations JSONB NOT NULL,
    clicked BOOLEAN NOT NULL DEFAULT FALSE,
    purchased BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- evaluate_ab.py and monitor.py (Phase 5/6) both group by model_version.
CREATE INDEX IF NOT EXISTS idx_experiment_events_model_version
    ON experiment_events (model_version);
