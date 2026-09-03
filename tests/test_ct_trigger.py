"""Phase 8 unit tests -- drift_monitor.py's Continuous Training trigger gate
(docs/PLAN.md 5.11).

Pure-function coverage only: the actual `POST .../dagRuns` call is exercised manually
against the real Airflow container, same as this file's Phase 7 sibling did for the DB/
MLflow calls in drift_monitor.py (docs/PLAN.md 6).
"""

from __future__ import annotations

import pytest

from drift_monitor import (
    DriftResult,
    combined_verdict,
    data_quality_ok,
    load_drift_history,
    save_drift_history,
    should_trigger_ct,
)

CSV_HEADER = "user_id,product_id,event,timestamp\n"


@pytest.mark.parametrize(
    "data_status, rec_status, expected",
    [
        ("OK", "OK", "OK"),
        ("OK", "WARNING", "WARNING"),
        ("WARNING", "DRIFT", "DRIFT"),
        ("DRIFT", "OK", "DRIFT"),
        ("n/a", "OK", "OK"),
        ("n/a", "n/a", "OK"),
    ],
)
def test_combined_verdict_takes_the_worse_of_the_two(data_status, rec_status, expected):
    assert combined_verdict(data_status, rec_status) == expected


def test_drift_history_roundtrips_through_a_file(tmp_path):
    path = tmp_path / "history.json"
    history = [
        DriftResult(verdict="OK", timestamp="2026-01-01T00:00:00+00:00"),
        DriftResult(verdict="DRIFT", timestamp="2026-01-01T00:05:00+00:00"),
    ]

    save_drift_history(path, history)
    loaded = load_drift_history(path)

    assert loaded == history


def test_load_drift_history_missing_file_is_empty(tmp_path):
    assert load_drift_history(tmp_path / "does-not-exist.json") == []


def test_save_drift_history_caps_at_the_limit(tmp_path):
    path = tmp_path / "history.json"
    history = [DriftResult(verdict="OK", timestamp=str(i)) for i in range(10)]

    save_drift_history(path, history)

    assert len(load_drift_history(path)) == 5  # HISTORY_LIMIT


def test_data_quality_ok_accepts_a_well_formed_file(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text(CSV_HEADER + "U1,P1,click,2026-01-01T00:00:00+00:00\n")
    assert data_quality_ok(path) is True


def test_data_quality_ok_rejects_missing_columns(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text("user_id,product_id\nU1,P1\n")
    assert data_quality_ok(path) is False


def test_data_quality_ok_rejects_a_null_value(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text(CSV_HEADER + "U1,,click,2026-01-01T00:00:00+00:00\n")
    assert data_quality_ok(path) is False


def test_data_quality_ok_missing_file_is_false(tmp_path):
    assert data_quality_ok(tmp_path / "does-not-exist.csv") is False


DRIFT = DriftResult(verdict="DRIFT", timestamp="t")
OK = DriftResult(verdict="OK", timestamp="t")


def test_should_trigger_ct_requires_two_consecutive_drift_verdicts():
    assert should_trigger_ct([DRIFT], sample_size=1000, min_sample_size=500, data_ok=True) is False
    assert should_trigger_ct([OK, DRIFT], sample_size=1000, min_sample_size=500, data_ok=True) is False
    assert should_trigger_ct([DRIFT, DRIFT], sample_size=1000, min_sample_size=500, data_ok=True) is True


def test_should_trigger_ct_respects_the_sample_floor():
    history = [DRIFT, DRIFT]
    assert should_trigger_ct(history, sample_size=100, min_sample_size=500, data_ok=True) is False
    assert should_trigger_ct(history, sample_size=500, min_sample_size=500, data_ok=True) is True


def test_should_trigger_ct_respects_data_quality():
    history = [DRIFT, DRIFT]
    assert should_trigger_ct(history, sample_size=1000, min_sample_size=500, data_ok=False) is False
