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
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overfit = load_overfit_module()
    train_rows = read_rows(args.run_dir / "train_prompt_trace_rows.csv")
    heldout_rows = read_rows(args.run_dir / "heldout_prompt_trace_rows.csv")

    memory = overfit.ResultBoundaryPromptHardMemory(
        entries={},
        target_mode="zero_improvement",
        scoring_bottleneck_mode="answer_decoder",
        expected_prompt_count=len(train_rows),
    )
    for row in train_rows:
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

    train_pairs = [
        (int(row["true_a"]), int(row["true_b"])) for row in train_rows
    ]
    heldout_pairs = [
        (int(row["true_a"]), int(row["true_b"])) for row in heldout_rows
    ]
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
        int(row["calculator_result"]) == int(row["true_sum"]) for row in train_rows
    ) / max(len(train_rows), 1)
    metrics = {
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
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
