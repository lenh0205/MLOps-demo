"""Rollback -- Phase 6 of docs/PLAN.md.

The exact inverse of promote_model.py, and deliberately built on it rather than
duplicating its logic: both scripts move the `@champion` alias and hit `/reload` on
model-champion (docs/PLAN.md 5.9), so this one imports `promote()` instead of re-copying
that alias-move + reload call. The only difference is intent and the default target (v1,
the pre-promotion version, vs. promote_model.py's no-default -- an operator promoting
always names a winner; an operator rolling back is usually going "back", so v1 is a safe
default here).

Rollback stays manual on purpose: monitor.py alerts, a human reads the alert and runs this
script. Wiring the monitor to call this automatically would need guard rails, hysteresis
and a kill switch this demo does not build:

    monitor.py -> \U0001f6a8 CTR below threshold -> operator decides -> rollback_model.py -> @champion -> v1

Usage:
    python analysis/rollback_model.py             # @champion -> v1 (the default target)
    python analysis/rollback_model.py v1           # equivalent, explicit
    python analysis/rollback_model.py 1            # a bare registry version also works
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows consoles default to cp1252 and choke on non-ASCII -- see docs/PLAN.md 7.11.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from promote_model import DEFAULT_PROD_URL, promote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "target", nargs="?", default="v1",
        help="version to roll back to: v1, v2, or a bare registry version number (default: v1)",
    )
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
