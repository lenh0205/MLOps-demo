"""Promotion -- Phase 5.

Moves the `@champion` alias to a registered version, then reloads the production model API
so the change actually takes effect. The alias move alone changes nothing on a running
container -- an alias is resolved once, at load time -- so `/reload` is
not an afterthought here, it *is* the deployment step: no image rebuild, no
`docker compose up`, no config change, same container before and after.

rollback_model.py (Phase 6) is the same call with the opposite target -- that symmetry is
deliberate.

Usage:
    python analysis/promote_model.py v2        # after evaluate_ab.py names v2 the winner
    python analysis/promote_model.py 2         # equivalent -- a bare registry version works too
    python analysis/promote_model.py v1        # e.g. Phase 6's manual rollback
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows consoles default to cp1252 and choke on non-ASCII.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
from mlflow.tracking import MlflowClient

REGISTERED_MODEL = "product-recommender"
ALIAS = "champion"

# Convenience labels matching the A/B arms' names -- v1/v2 register in that order in
# trainer/train.py's VARIANTS tuple, so the mapping is fixed, not looked up.
LABEL_TO_VERSION = {"v1": "1", "v2": "2"}

DEFAULT_PROD_URL = "http://localhost:8003"  # model-champion (MODEL_URI=...@champion)


def resolve_target_version(target: str) -> str:
    return LABEL_TO_VERSION.get(target, target)


def promote(target: str, prod_url: str) -> None:
    version = resolve_target_version(target)

    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, ALIAS, version)
    print(f"@{ALIAS} -> v{version}")

    response = httpx.post(f"{prod_url}/reload", timeout=30.0)
    response.raise_for_status()
    body = response.json()
    print(f"{prod_url}/reload -> model_version={body['model_version']} (model_uri={body['model_uri']})")

    health = httpx.get(f"{prod_url}/health", timeout=10.0).json()
    print(f"{prod_url}/health -> model_version={health['model_version']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="version to promote: v1, v2, or a bare registry version number")
    parser.add_argument("--prod-url", default=DEFAULT_PROD_URL, help="model-champion base URL (default: %(default)s)")
    args = parser.parse_args()

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise SystemExit(
            "MLFLOW_TRACKING_URI is not set. Point it at the tracking server, e.g.\n"
            "    export MLFLOW_TRACKING_URI=http://localhost:5000"
        )

    promote(args.target, args.prod_url)


if __name__ == "__main__":
    main()
