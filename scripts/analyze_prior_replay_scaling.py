#!/usr/bin/env python3
"""Account for amortized-prior replay cost as calculator count grows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PriorReplayScalingRow:
    calculator_count: int
    prompt_memory_entries_per_calculator: int
    prompt_memory_entries_total: int
    forced_candidate_evals_per_calculator: int
    forced_candidate_evals_total: int
    prior_fit_examples_per_calculator: int
    prior_fit_examples_total: int
    full_fit_examples_per_calculator: int
    full_fit_examples_total: int
    prior_updates_per_calculator: int
    prior_params_per_calculator: int
    prior_params_total: int
    candidate_plus_prior_examples_total: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def result_vocab_size(operand_max: int) -> int:
    if operand_max < 0:
        raise ValueError("operand_max must be non-negative")
    return (2 * (operand_max + 1)) - 1


def prompt_count(operand_max: int) -> int:
    if operand_max < 0:
        raise ValueError("operand_max must be non-negative")
    operand_vocab_size = operand_max + 1
    return operand_vocab_size * operand_vocab_size


def train_prompt_count(operand_max: int, heldout_fraction: float) -> int:
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in [0, 1)")
    return round(prompt_count(operand_max) * (1.0 - heldout_fraction))


def prior_parameter_count(
    *,
    operand_vocab_size: int,
    result_vocab: int,
    hidden_size: int,
    feature_mode: str,
) -> int:
    if operand_vocab_size <= 0 or result_vocab <= 0 or hidden_size <= 0:
        raise ValueError("vocab sizes and hidden_size must be positive")
    if feature_mode not in {"numeric", "embedding"}:
        raise ValueError("feature_mode must be numeric or embedding")
    embedding_params = operand_vocab_size * hidden_size if feature_mode == "embedding" else 0
    input_size = hidden_size * 2 if feature_mode == "embedding" else 2
    first_layer = (input_size * hidden_size) + hidden_size
    second_layer = (hidden_size * result_vocab) + result_vocab
    return embedding_params + first_layer + second_layer


def compute_row(
    *,
    calculator_count: int,
    operand_max: int,
    heldout_fraction: float,
    memory_fill_steps: int,
    batch_size: int,
    sample_count: int,
    prior_fit_examples: int,
    full_fit_examples: int,
    prior_updates: int,
    prior_hidden_size: int,
    prior_feature_mode: str,
) -> PriorReplayScalingRow:
    if calculator_count <= 0:
        raise ValueError("calculator_count must be positive")
    if memory_fill_steps < 0:
        raise ValueError("memory_fill_steps must be non-negative")
    if batch_size <= 0 or sample_count <= 0:
        raise ValueError("batch_size and sample_count must be positive")
    if prior_fit_examples < 0 or full_fit_examples < 0 or prior_updates < 0:
        raise ValueError("prior cost counts must be non-negative")

    prompts = train_prompt_count(operand_max, heldout_fraction)
    forced_candidate_evals = memory_fill_steps * batch_size * sample_count
    prior_params = prior_parameter_count(
        operand_vocab_size=operand_max + 1,
        result_vocab=result_vocab_size(operand_max),
        hidden_size=prior_hidden_size,
        feature_mode=prior_feature_mode,
    )
    forced_total = forced_candidate_evals * calculator_count
    prior_fit_total = prior_fit_examples * calculator_count

    return PriorReplayScalingRow(
        calculator_count=calculator_count,
        prompt_memory_entries_per_calculator=prompts,
        prompt_memory_entries_total=prompts * calculator_count,
        forced_candidate_evals_per_calculator=forced_candidate_evals,
        forced_candidate_evals_total=forced_total,
        prior_fit_examples_per_calculator=prior_fit_examples,
        prior_fit_examples_total=prior_fit_total,
        full_fit_examples_per_calculator=full_fit_examples,
        full_fit_examples_total=full_fit_examples * calculator_count,
        prior_updates_per_calculator=prior_updates,
        prior_params_per_calculator=prior_params,
        prior_params_total=prior_params * calculator_count,
        candidate_plus_prior_examples_total=forced_total + prior_fit_total,
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def markdown_table(rows: list[PriorReplayScalingRow]) -> str:
    header = [
        "| calculators | prompt-memory entries | candidate evals | prior fit examples | full-fit examples | prior params | candidate + prior examples |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = [
        (
            f"| {format_int(row.calculator_count)} | "
            f"{format_int(row.prompt_memory_entries_total)} "
            f"({format_int(row.prompt_memory_entries_per_calculator)} each) | "
            f"{format_int(row.forced_candidate_evals_total)} "
            f"({format_int(row.forced_candidate_evals_per_calculator)} each) | "
            f"{format_int(row.prior_fit_examples_total)} "
            f"({format_int(row.prior_fit_examples_per_calculator)} each) | "
            f"{format_int(row.full_fit_examples_total)} "
            f"({format_int(row.full_fit_examples_per_calculator)} each) | "
            f"{format_int(row.prior_params_total)} "
            f"({format_int(row.prior_params_per_calculator)} each) | "
            f"{format_int(row.candidate_plus_prior_examples_total)} |"
        )
        for row in rows
    ]
    return "\n".join(header + body)


def build_rows(args: argparse.Namespace) -> list[PriorReplayScalingRow]:
    return [
        compute_row(
            calculator_count=calculator_count,
            operand_max=args.operand_max,
            heldout_fraction=args.heldout_fraction,
            memory_fill_steps=args.memory_fill_steps,
            batch_size=args.batch_size,
            sample_count=args.sample_count,
            prior_fit_examples=args.prior_fit_examples,
            full_fit_examples=args.full_fit_examples,
            prior_updates=args.prior_updates,
            prior_hidden_size=args.prior_hidden_size,
            prior_feature_mode=args.prior_feature_mode,
        )
        for calculator_count in args.calculator_counts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate how the measured amortized-prior replay recipe scales "
            "when independent active calculators each need their own prompt "
            "memory and prior."
        )
    )
    parser.add_argument("--calculator-counts", type=parse_int_list, default=[1, 4, 16, 64])
    parser.add_argument("--operand-max", type=int, default=29)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--memory-fill-steps",
        type=int,
        default=192,
        help=(
            "Streaming source steps with sparse candidate scoring before prompt "
            "memory is full. The op29 capped-prior lineage used about 192."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--prior-fit-examples", type=int, default=1_254_817)
    parser.add_argument("--full-fit-examples", type=int, default=1_080_000)
    parser.add_argument("--prior-updates", type=int, default=2_000)
    parser.add_argument("--prior-hidden-size", type=int, default=128)
    parser.add_argument("--prior-feature-mode", choices=("numeric", "embedding"), default="numeric")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    if args.format == "json":
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    else:
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
