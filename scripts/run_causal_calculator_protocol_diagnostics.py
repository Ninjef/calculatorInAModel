"""Canonical entrypoint for calculator protocol causal diagnostics.

The legacy phase-1 Track 3 runner remains available for its fixed manifest.
This canonical name forwards to the checkpoint-first diagnostic engine used by
phase-2 reports.
"""

from scripts.diagnose_calculator_protocol import *  # noqa: F403


if __name__ == "__main__":
    main()  # noqa: F405
