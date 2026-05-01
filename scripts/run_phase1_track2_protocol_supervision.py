import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OVERFIT = ROOT / "scripts" / "overfit_one_batch.py"


BASE_2DIGIT = [
    "--digits",
    "2",
    "--steps",
    "1000",
    "--eval-samples",
    "512",
    "--operand-max",
    "19",
    "--calculator-operand-vocab-size",
    "20",
    "--n-layer",
    "2",
    "--n-head",
    "1",
    "--n-embd",
    "16",
    "--mlp-expansion",
    "1",
    "--calculator-hook-after-layer",
    "1",
    "--calculator-read-position",
    "operands",
]


TRANSITION_SNAPSHOTS = [
    "--snapshot-every",
    "100",
    "--snapshot-samples",
    "64",
]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_cmd(
    *,
    variant: str,
    injection_mode: str,
    seed: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(OVERFIT),
        "--variant",
        variant,
        *BASE_2DIGIT,
        "--calculator-injection-mode",
        injection_mode,
        "--seed",
        str(seed),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def non_bottleneck_commands(seed: int) -> list[list[str]]:
    return [
        train_cmd(variant="model-c", injection_mode="add", seed=seed),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.003"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.01"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.03"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=[
                "--aux-operand-loss-weight",
                "0.03",
                "--aux-operand-loss-decay-steps",
                "1000",
                *TRANSITION_SNAPSHOTS,
            ],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=[
                "--aux-operand-loss-weight",
                "0.03",
                "--aux-operand-loss-decay-steps",
                "1000",
                "--aux-operand-loss-floor",
                "0.003",
                *TRANSITION_SNAPSHOTS,
            ],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="add",
            seed=seed,
            extra=[
                "--aux-operand-loss-weight",
                "0.03",
                "--aux-operand-loss-decay-steps",
                "300",
                *TRANSITION_SNAPSHOTS,
            ],
        ),
    ]


def bottleneck_commands(seed: int) -> list[list[str]]:
    return [
        train_cmd(variant="model-b", injection_mode="replace", seed=seed),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=["--oracle-train"],
        ),
        train_cmd(variant="model-c", injection_mode="replace", seed=seed),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.003"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.01"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=["--aux-operand-loss-weight", "0.03"],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=[
                "--aux-operand-loss-weight",
                "0.03",
                "--aux-operand-loss-decay-steps",
                "1000",
                *TRANSITION_SNAPSHOTS,
            ],
        ),
        train_cmd(
            variant="model-c",
            injection_mode="replace",
            seed=seed,
            extra=[
                "--aux-operand-loss-weight",
                "0.03",
                "--aux-operand-loss-decay-steps",
                "1000",
                "--aux-operand-loss-floor",
                "0.003",
                *TRANSITION_SNAPSHOTS,
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track 2 protocol-supervision and retention experiments."
    )
    parser.add_argument(
        "--track",
        choices=["non-bottleneck", "bottleneck", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Skip the first N generated commands, for resuming an interrupted matrix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N commands after applying --skip-first.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed in args.seeds:
        commands: list[list[str]] = []
        if args.track in {"non-bottleneck", "all"}:
            commands.extend(non_bottleneck_commands(seed))
        if args.track in {"bottleneck", "all"}:
            commands.extend(bottleneck_commands(seed))
        if args.skip_first < 0:
            raise ValueError("--skip-first must be non-negative")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive when provided")
        commands = commands[args.skip_first :]
        if args.limit is not None:
            commands = commands[: args.limit]
        for cmd in commands:
            run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
