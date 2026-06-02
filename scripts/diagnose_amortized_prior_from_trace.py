"""Fit the amortized hard-memory prior from train traces and test heldout traces."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path


def load_overfit_module():
    script_path = Path(__file__).resolve().parent / "overfit_one_batch.py"
    spec = importlib.util.spec_from_file_location("overfit_prior_diag", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_int_set(raw: str) -> set[int]:
    if not raw:
        return set()
    values = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def row_pair(row: dict[str, str]) -> tuple[int, int]:
    return int(row["true_a"]), int(row["true_b"])


def row_route(row: dict[str, str]) -> int:
    if "calculator_hook_route" not in row or row["calculator_hook_route"] == "":
        raise ValueError("trace rows do not include calculator_hook_route")
    return int(row["calculator_hook_route"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-digits", type=int, default=2)
    parser.add_argument("--operand-vocab-size", type=int, default=20)
    parser.add_argument("--result-vocab-size", type=int, default=39)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--feature-mode",
        choices=["embedding", "numeric"],
        default="embedding",
    )
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--split-mode",
        choices=["prompt_heldout", "route_heldout"],
        default="prompt_heldout",
        help=(
            "prompt_heldout fits train_prompt_trace_rows and evaluates "
            "heldout_prompt_trace_rows. route_heldout withholds rows whose "
            "calculator_hook_route is listed in --heldout-routes."
        ),
    )
    parser.add_argument(
        "--heldout-routes",
        type=parse_int_set,
        default=set(),
        help="Comma-separated calculator_hook_route ids for route_heldout mode.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overfit = load_overfit_module()
    train_rows = read_rows(args.run_dir / "train_prompt_trace_rows.csv")
    heldout_rows = read_rows(args.run_dir / "heldout_prompt_trace_rows.csv")

    if args.split_mode == "prompt_heldout":
        fit_rows = train_rows
        eval_train_rows = train_rows
        eval_heldout_rows = heldout_rows
    else:
        if not args.heldout_routes:
            raise ValueError("--heldout-routes is required for route_heldout mode")
        fit_rows = [
            row for row in train_rows if row_route(row) not in args.heldout_routes
        ]
        eval_train_rows = fit_rows
        eval_heldout_rows = [
            row for row in train_rows if row_route(row) in args.heldout_routes
        ]
        if not fit_rows or not eval_heldout_rows:
            raise ValueError("route_heldout split must leave fit and heldout rows")

    memory = overfit.ResultBoundaryPromptHardMemory(
        entries={},
        target_mode="zero_improvement",
        scoring_bottleneck_mode="answer_decoder",
        expected_prompt_count=len(fit_rows),
    )
    for row in fit_rows:
        memory.entries[tuple(overfit.tokenize(row["prompt"]))] = {
            "best_result": int(row["calculator_result"]),
            "best_loss": 0.0,
        }

    prior = overfit.init_result_boundary_amortized_prior(
        operand_vocab_size=args.operand_vocab_size,
        result_vocab_size=args.result_vocab_size,
        hidden_size=args.hidden_size,
        feature_mode=args.feature_mode,
        lr=args.lr,
        min_entries=1,
        replay_batch_size=0,
        device="cpu",
    )
    rng = random.Random(args.seed)
    train_metrics: dict[str, float] = {}
    for _ in range(args.steps):
        train_metrics = overfit.train_result_boundary_amortized_prior(
            prior,
            memory,
            num_digits=args.num_digits,
            device="cpu",
            rng=rng,
        )

    train_pairs = [row_pair(row) for row in eval_train_rows]
    heldout_pairs = [row_pair(row) for row in eval_heldout_rows]
    train_prior = overfit.evaluate_result_boundary_amortized_prior(
        prior,
        train_pairs,
        device="cpu",
    )
    heldout_prior = overfit.evaluate_result_boundary_amortized_prior(
        prior,
        heldout_pairs,
        device="cpu",
    )
    memory_matches_true = sum(
        int(row["calculator_result"]) == int(row["true_sum"]) for row in fit_rows
    ) / max(len(fit_rows), 1)
    metrics = {
        "split_mode": args.split_mode,
        "heldout_routes": sorted(args.heldout_routes),
        "fit_rows": len(fit_rows),
        "train_rows": len(eval_train_rows),
        "heldout_rows": len(eval_heldout_rows),
        "memory_target_matches_true": memory_matches_true,
        "prior_train_fit_accuracy_vs_memory": train_metrics[
            "result_boundary_target_amortized_prior_train_accuracy"
        ],
        "prior_train_accuracy_vs_true": train_prior["accuracy"],
        "prior_train_confidence": train_prior["confidence"],
        "prior_heldout_accuracy_vs_true": heldout_prior["accuracy"],
        "prior_heldout_confidence": heldout_prior["confidence"],
        "steps": args.steps,
        "hidden_size": args.hidden_size,
        "feature_mode": args.feature_mode,
        "lr": args.lr,
        "seed": args.seed,
    }
    text = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
