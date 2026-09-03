"""Phase 7 unit tests -- PSI and its OK/WARNING/DRIFT classification.

Pure-function coverage only: drift_monitor.py's async DB/MLflow calls are exercised
manually against the real stack, same as evaluate_ab.py / promote_model.py were in
earlier phases.
"""

from __future__ import annotations

import pytest

from drift_monitor import PSI_DRIFT, PSI_WARNING, classify, normalize, psi, training_event_mix
from generate_data import generate_interactions, write_csv


def test_psi_is_zero_for_identical_distributions():
    dist = {"click": 0.7, "purchase": 0.3}
    assert psi(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_large_for_a_reversed_distribution():
    reference = {"click": 0.9, "purchase": 0.1}
    recent = {"click": 0.1, "purchase": 0.9}
    assert psi(reference, recent) >= PSI_DRIFT


def test_psi_handles_a_category_missing_on_one_side():
    reference = {"P01": 1.0}
    recent = {"P01": 0.5, "P02": 0.5}
    # Should not raise (no log(0)/ZeroDivisionError) and should register as drift --
    # a brand-new product taking half of recent traffic is a real behaviour shift.
    assert psi(reference, recent) > 0


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, "OK"), (PSI_WARNING - 0.01, "OK"), (PSI_WARNING, "WARNING"), (PSI_DRIFT - 0.01, "WARNING"), (PSI_DRIFT, "DRIFT"), (1.0, "DRIFT")],
)
def test_classify_thresholds(value, expected):
    assert classify(value) == expected


def test_normalize_empty_counts_is_empty():
    assert normalize({}) == {}


def test_normalize_sums_to_one():
    result = normalize({"a": 3, "b": 1})
    assert result["a"] == pytest.approx(0.75)
    assert sum(result.values()) == pytest.approx(1.0)


def test_training_event_mix_is_a_normalized_click_purchase_split(tmp_path):
    path = tmp_path / "interactions.csv"
    write_csv(generate_interactions(seed=42), path)

    mix = training_event_mix(path)

    assert set(mix) <= {"click", "purchase"}
    assert sum(mix.values()) == pytest.approx(1.0)
    # generate_data.py's own modelling assumption: purchases are the minority event.
    assert mix["purchase"] < mix["click"]
