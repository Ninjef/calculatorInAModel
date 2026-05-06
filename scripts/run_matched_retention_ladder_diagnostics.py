import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_command(command: list[str], *, skip_if_exists: Path | None) -> None:
    if skip_if_exists is not None and skip_if_exists.exists():
        print(f"skip existing: {skip_if_exists}")
        return
    print(" ".join(command))
    subprocess.run(command, check=True)


def selected_label(row: dict[str, Any]) -> str:
    return f"step{int(row['step'])}"


def run_diagnostics(
    *,
    selected: list[dict[str, Any]],
    causal_samples: int,
    full_enum_samples: int,
    full_enum_batch_size: int,
    seed_base: int,
    skip_existing: bool,
) -> None:
    for index, row in enumerate(selected):
        checkpoint = Path(row["checkpoint"])
        run_dir = Path(row["run_dir"])
        label = selected_label(row)
        seed = seed_base + int(row["seed"] or 0) + index

        causal_dir = run_dir / f"{label}_canonical_causal_diagnostics"
        full_enum_root = run_dir / f"{label}_full_enum_action_loss"
        private_dir = run_dir / f"{label}_private_protocol_diagnostics"

        run_command(
            [
                sys.executable,
                "-m",
                "scripts.run_causal_calculator_protocol_diagnostics",
                "--checkpoint",
                str(checkpoint),
                "--samples",
                str(causal_samples),
                "--digits",
                "2",
                "--operand-max",
                "19",
                "--seed",
                str(seed + 1000),
                "--forced-result-sweep",
                "--forced-result-batch-size",
                "64",
                "--output-dir",
                str(causal_dir),
            ],
            skip_if_exists=causal_dir / "diagnostic_summary.json"
            if skip_existing
            else None,
        )
        run_command(
            [
                sys.executable,
                "scripts/run_full_enum_action_loss_diagnostic.py",
                "--checkpoint",
                str(checkpoint),
                "--samples",
                str(full_enum_samples),
                "--batch-size",
                str(full_enum_batch_size),
                "--digits",
                "2",
                "--operand-max",
                "19",
                "--temperature",
                "1.0",
                "--chunk-size",
                "64",
                "--seed",
                str(seed + 2000),
                "--output-root",
                str(full_enum_root),
            ],
            skip_if_exists=full_enum_root / "checkpoint_snapshots" / "full_enum_summary.json"
            if skip_existing
            else None,
        )
        run_command(
            [
                sys.executable,
                "scripts/diagnose_private_protocol.py",
                "--checkpoint",
                str(checkpoint),
                "--digits",
                "2",
                "--operand-max",
                "19",
                "--seed",
                str(seed + 3000),
                "--output-dir",
                str(private_dir),
            ],
            skip_if_exists=private_dir / "private_protocol_summary.json"
            if skip_existing
            else None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run canonical diagnostics for matched-retention selections."
    )
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--causal-samples", type=int, default=256)
    parser.add_argument("--full-enum-samples", type=int, default=128)
    parser.add_argument("--full-enum-batch-size", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=91000)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text())
    run_diagnostics(
        selected=summary["selected"],
        causal_samples=args.causal_samples,
        full_enum_samples=args.full_enum_samples,
        full_enum_batch_size=args.full_enum_batch_size,
        seed_base=args.seed_base,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
