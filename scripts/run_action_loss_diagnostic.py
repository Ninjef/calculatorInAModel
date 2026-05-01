"""Canonical entrypoint for operand action-loss diagnostics.

The legacy Track 4 script remains available for backward compatibility. This
phase-neutral name is preferred in phase-2 reports.
"""

from scripts.run_phase1_track4_action_loss_diagnostic import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
