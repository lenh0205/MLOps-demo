"""Phase 7 unit test -- simulate_traffic.py's --product-shift user pool (docs/PLAN.md 5.10).

No router/events-db involved: run_one's HTTP calls are exercised manually against the
real stack, same as the rest of simulate_traffic.py.
"""

from __future__ import annotations

from simulate_traffic import load_user_ids


def test_load_user_ids_returns_sorted_unique_ids_from_the_csv(tmp_path):
    path = tmp_path / "interactions.csv"
    path.write_text(
        "user_id,product_id,event,timestamp\n"
        "U002,P01,click,2026-01-01T00:00:00+00:00\n"
        "U001,P02,purchase,2026-01-01T00:01:00+00:00\n"
        "U001,P03,click,2026-01-01T00:02:00+00:00\n"
    )

    assert load_user_ids(path) == ["U001", "U002"]
