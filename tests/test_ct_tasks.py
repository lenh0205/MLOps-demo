"""Phase 8 unit tests -- the Continuous Training DAG's Airflow-free task logic
(airflow/dags/ct_tasks.py).

Pure-function coverage only, same split as everywhere else in this repo: the two DAG
tasks that need a live MLflow server (train, evaluate-against-champion) are exercised
manually against the real stack, not here (docs/PLAN.md 6).
"""

from __future__ import annotations

import pytest

from ct_tasks import passes_quality_gate, validate_dataset

CSV_HEADER = "user_id,product_id,event,timestamp\n"


def _write_rows(path, n: int) -> None:
    rows = [CSV_HEADER]
    for i in range(n):
        rows.append(f"U{i:03d},P01,click,2026-01-01T00:00:{i % 60:02d}+00:00\n")
    path.write_text("".join(rows))


def test_validate_dataset_accepts_a_well_formed_file(tmp_path):
    path = tmp_path / "interactions.csv"
    _write_rows(path, 100)
    validate_dataset(path, min_rows=100)  # should not raise


def test_validate_dataset_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path / "does-not-exist.csv")


def test_validate_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text("user_id,product_id\nU1,P1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(path, min_rows=1)


def test_validate_dataset_rejects_too_few_rows(tmp_path):
    path = tmp_path / "interactions.csv"
    _write_rows(path, 5)
    with pytest.raises(ValueError, match="only 5 rows"):
        validate_dataset(path, min_rows=100)


def test_validate_dataset_rejects_a_null_value(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text(CSV_HEADER + "U1,,click,2026-01-01T00:00:00+00:00\n" * 100)
    with pytest.raises(ValueError, match="null/empty value"):
        validate_dataset(path, min_rows=100)


@pytest.mark.parametrize(
    "candidate, champion, tolerance, expected",
    [
        (0.605, 0.600, 0.0, True),   # strictly better
        (0.600, 0.600, 0.0, True),   # tie passes with the default strict >=
        (0.599, 0.600, 0.0, False),  # any regression fails with tolerance=0.0
        (0.590, 0.600, 0.02, True),  # within an explicit tolerance
        (0.570, 0.600, 0.02, False), # outside the tolerance
    ],
)
def test_passes_quality_gate(candidate, champion, tolerance, expected):
    assert passes_quality_gate(candidate, champion, tolerance) is expected
