"""Backward-compatible legacy Track 4 wrapper.

Prefer `scripts/run_action_loss_diagnostic.py` in new phase-neutral reports.
"""

from scripts.run_action_loss_diagnostic import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
