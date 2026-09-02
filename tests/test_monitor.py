"""Phase 7 unit tests -- monitor.py's windowed-query duration parsing and the
minimum-sample-size alert gate. No DB involved: fetch_windowed_stats is exercised
manually against the real stack, same as evaluate_ab.fetch_stats was in Phase 5/6.
"""

from __future__ import annotations

import pytest

from monitor import check_alerts, parse_duration


@pytest.mark.parametrize(
    "value, expected",
    [("300", 300.0), ("300s", 300.0), ("5m", 300.0), ("2h", 7200.0), ("45s", 45.0), ("0.5h", 1800.0)],
)
def test_parse_duration(value, expected):
    assert parse_duration(value) == expected


def test_alert_suppressed_below_min_sample_size():
    # V1's CTR here (0%) is well below threshold, but n=10 shouldn't fire an alert --
    # a handful of requests right after a deploy is noise, not degradation.
    stats = {
        "v1": {"requests": 10, "clicks": 0, "purchases": 0},
        "v2": {"requests": 10, "clicks": 10, "purchases": 0},
    }
    baseline = {"v1": 0.08, "v2": 0.11}
    assert check_alerts(stats, baseline, alert_ratio=0.8, min_sample_size=500) == []


def test_alert_fires_once_min_sample_size_is_met():
    stats = {
        "v1": {"requests": 1000, "clicks": 40, "purchases": 0},   # 4.0% < 8.0%*0.8
        "v2": {"requests": 1000, "clicks": 110, "purchases": 0},  # 11.0%, not below 11.0%*0.8
    }
    baseline = {"v1": 0.08, "v2": 0.11}
    alerts = check_alerts(stats, baseline, alert_ratio=0.8, min_sample_size=500)
    assert len(alerts) == 1
    assert "v1" in alerts[0]
    assert "v2" not in alerts[0]


def test_no_alert_when_ctr_stays_above_threshold():
    stats = {
        "v1": {"requests": 1000, "clicks": 75, "purchases": 0},
        "v2": {"requests": 1000, "clicks": 100, "purchases": 0},
    }
    baseline = {"v1": 0.08, "v2": 0.11}
    assert check_alerts(stats, baseline, alert_ratio=0.8, min_sample_size=500) == []
