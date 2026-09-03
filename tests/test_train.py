"""Phase 7 unit tests -- train.py's lineage-tag helpers.

Pure-function coverage only: the tags actually landing on a run is verified manually
against the real MLflow server, same as the rest of train.py's MLflow calls.
"""

from __future__ import annotations

from train import dataset_hash, git_commit


def test_git_commit_returns_a_40_char_hex_sha():
    commit = git_commit()
    assert len(commit) == 40
    assert all(c in "0123456789abcdef" for c in commit)


def test_dataset_hash_is_deterministic_and_content_sensitive(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("user_id,product_id,event,timestamp\nU001,P01,click,2026-01-01T00:00:00+00:00\n")
    b.write_text("user_id,product_id,event,timestamp\nU001,P01,purchase,2026-01-01T00:00:00+00:00\n")

    assert dataset_hash(a) == dataset_hash(a)
    assert dataset_hash(a) != dataset_hash(b)
    assert len(dataset_hash(a)) == 64  # sha256 hex digest
