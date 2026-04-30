import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OVERFIT = ROOT / "scripts" / "overfit_one_batch.py"
DIAGNOSE = ROOT / "scripts" / "diagnose_calculator_protocol.py"


BASE_2DIGIT = [
    "--digits",
    "2",
    "--steps",
    "1000",
    "--eval-samples",
    "512",
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
]


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_cmd(
    *,
    variant: str,
    operand_max: int,
    operand_vocab_size: int,
    seed: int,
    extra: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(OVERFIT),
        "--variant",
        variant,
        *BASE_2DIGIT,
        "--operand-max",
        str(operand_max),
        "--calculator-operand-vocab-size",
        str(operand_vocab_size),
        "--seed",
        str(seed),
    ]
    if extra:
        cmd.extend(extra)
    return cmd


def run_ladder(args: argparse.Namespace) -> None:
    rungs = [(19, 20), (49, 50), (99, 100)]
    for operand_max, vocab_size in rungs:
        if operand_max not in args.operand_max:
            continue
        for seed in args.seeds:
            for variant in ("model-a", "model-b", "model-c"):
                run_command(
                    train_cmd(
                        variant=variant,
                        operand_max=operand_max,
                        operand_vocab_size=vocab_size,
                        seed=seed,
                    ),
                    dry_run=args.dry_run,
                )


def run_aux(args: argparse.Namespace) -> None:
    configs = [
        ["--aux-operand-loss-weight", "0.01"],
        ["--aux-operand-loss-weight", "0.03"],
        [
            "--aux-operand-loss-weight",
            "0.1",
            "--aux-operand-loss-decay-steps",
            "1000",
        ],
        [
            "--aux-operand-loss-weight",
            "0.1",
            "--aux-operand-loss-decay-steps",
            "300",
            "--aux-operand-loss-floor",
            "0.01",
        ],
    ]
    for seed in args.seeds:
        for extra in configs:
            run_command(
                train_cmd(
                    variant="model-c",
                    operand_max=19,
                    operand_vocab_size=20,
                    seed=seed,
                    extra=extra,
                ),
                dry_run=args.dry_run,
            )


def run_warmup(args: argparse.Namespace) -> None:
    configs = [
        ["--oracle-warmup-steps", "100", "--aux-operand-loss-weight", "0.01"],
        ["--oracle-warmup-steps", "300", "--aux-operand-loss-weight", "0.01"],
        ["--oracle-warmup-steps", "300", "--aux-operand-loss-weight", "0.03"],
    ]
    for seed in args.seeds:
        for extra in configs:
            run_command(
                train_cmd(
                    variant="model-c",
                    operand_max=19,
                    operand_vocab_size=20,
                    seed=seed,
                    extra=extra,
                ),
                dry_run=args.dry_run,
            )


def run_private_trajectory(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(OVERFIT),
        "--variant",
        "model-c",
        "--digits",
        "1",
        "--operand-max",
        "9",
        "--calculator-operand-vocab-size",
        "10",
        "--steps",
        "1000",
        "--eval-samples",
        "512",
        "--n-layer",
        "1",
        "--n-head",
        "1",
        "--n-embd",
        "4",
        "--mlp-expansion",
        "1",
        "--calculator-hook-after-layer",
        "1",
        "--snapshot-every",
        str(args.snapshot_every),
        "--snapshot-samples",
        str(args.snapshot_samples),
    ]
    for seed in args.seeds:
        run_command([*cmd, "--seed", str(seed)], dry_run=args.dry_run)


def run_probes(args: argparse.Namespace) -> None:
    if not args.checkpoints:
        raise ValueError("--track probes requires at least one --checkpoint")
    for checkpoint in args.checkpoints:
        cmd = [
            sys.executable,
            str(DIAGNOSE),
            "--checkpoint",
            str(checkpoint),
            "--digits",
            "2",
            "--operand-max",
            str(args.probe_operand_max),
            "--samples",
            "1024",
            "--probe",
            "--probe-steps",
            "400",
            "--probe-layers",
            "1",
            "2",
            "--probe-positions",
            "a",
            "b",
            "eq",
        ]
        run_command(cmd, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-bottleneck calculator protocol experiment tracks."
    )
    parser.add_argument(
        "--track",
        choices=["ladder", "aux", "warmup", "private-trajectory", "probes"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--operand-max",
        type=int,
        nargs="+",
        default=[19, 49, 99],
        help="Curriculum rungs for --track ladder.",
    )
    parser.add_argument("--checkpoint", dest="checkpoints", type=Path, nargs="*")
    parser.add_argument("--probe-operand-max", type=int, default=19)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--snapshot-samples", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.track == "ladder":
        run_ladder(args)
    elif args.track == "aux":
        run_aux(args)
    elif args.track == "warmup":
        run_warmup(args)
    elif args.track == "private-trajectory":
        run_private_trajectory(args)
    elif args.track == "probes":
        run_probes(args)
    else:
        raise ValueError(f"unknown track: {args.track}")


if __name__ == "__main__":
    main()
