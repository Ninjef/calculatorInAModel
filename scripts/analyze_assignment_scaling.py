#!/usr/bin/env python3
"""Account for calculator assignment cost as calculator count grows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AssignmentScalingRow:
    operand_max: int
    result_vocab_size: int
    sample_count: int
    scored_fraction: float
    scoring_reduction_fraction: float
    prompts_per_calculator: int
    assignment_steps: int
    calculator_count: int
    exact_forced_evals: int
    sampled_forced_evals: int
    forced_eval_savings: int
    result_head_params_per_calculator: int
    result_head_params_total: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def result_vocab_size(operand_max: int) -> int:
    if operand_max < 0:
        raise ValueError("operand_max must be non-negative")
    operand_vocab_size = operand_max + 1
    return (2 * operand_vocab_size) - 1


def default_prompt_count(operand_max: int) -> int:
    operand_vocab_size = operand_max + 1
    return operand_vocab_size * operand_vocab_size


def result_head_parameter_count(
    result_vocab: int,
    *,
    n_embd: int,
    span_width: int,
    hidden_size: int,
) -> int:
    if result_vocab <= 0:
        raise ValueError("result_vocab must be positive")
    if n_embd <= 0 or span_width <= 0:
        raise ValueError("n_embd and span_width must be positive")
    paired_width = 2 * n_embd * span_width
    if hidden_size <= 0:
        return (paired_width * result_vocab) + result_vocab
    return (
        (paired_width * hidden_size)
        + hidden_size
        + (hidden_size * result_vocab)
        + result_vocab
    )


def compute_row(
    *,
    operand_max: int,
    sample_count: int,
    assignment_steps: int,
    calculator_count: int,
    n_embd: int,
    span_width: int,
    hidden_size: int,
    prompts_per_calculator: int | None = None,
) -> AssignmentScalingRow:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if assignment_steps < 0:
        raise ValueError("assignment_steps must be non-negative")
    if calculator_count <= 0:
        raise ValueError("calculator_count must be positive")

    result_vocab = result_vocab_size(operand_max)
    prompts = (
        default_prompt_count(operand_max)
        if prompts_per_calculator is None
        else prompts_per_calculator
    )
    if prompts <= 0:
        raise ValueError("prompts_per_calculator must be positive")

    sampled_candidates = min(sample_count, result_vocab)
    exact_forced_evals = prompts * assignment_steps * result_vocab * calculator_count
    sampled_forced_evals = prompts * assignment_steps * sampled_candidates * calculator_count
    params_per_calculator = result_head_parameter_count(
        result_vocab,
        n_embd=n_embd,
        span_width=span_width,
        hidden_size=hidden_size,
    )
    scored_fraction = sampled_candidates / result_vocab

    return AssignmentScalingRow(
        operand_max=operand_max,
        result_vocab_size=result_vocab,
        sample_count=sampled_candidates,
        scored_fraction=scored_fraction,
        scoring_reduction_fraction=1.0 - scored_fraction,
        prompts_per_calculator=prompts,
        assignment_steps=assignment_steps,
        calculator_count=calculator_count,
        exact_forced_evals=exact_forced_evals,
        sampled_forced_evals=sampled_forced_evals,
        forced_eval_savings=exact_forced_evals - sampled_forced_evals,
        result_head_params_per_calculator=params_per_calculator,
        result_head_params_total=params_per_calculator * calculator_count,
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_table(rows: list[AssignmentScalingRow]) -> str:
    header = [
        "| operands | calculators | result classes | scored classes | exact evals | sampled evals | eval savings | result-head params |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = [
        (
            f"| 0..{row.operand_max} | {format_int(row.calculator_count)} | "
            f"{format_int(row.result_vocab_size)} | "
            f"{format_int(row.sample_count)} ({format_pct(row.scored_fraction)}) | "
            f"{format_int(row.exact_forced_evals)} | "
            f"{format_int(row.sampled_forced_evals)} | "
            f"{format_int(row.forced_eval_savings)} "
            f"({format_pct(row.scoring_reduction_fraction)}) | "
            f"{format_int(row.result_head_params_total)} "
            f"({format_int(row.result_head_params_per_calculator)} each) |"
        )
        for row in rows
    ]
    return "\n".join(header + body)


def build_rows(args: argparse.Namespace) -> list[AssignmentScalingRow]:
    return [
        compute_row(
            operand_max=operand_max,
            sample_count=args.sample_count,
            assignment_steps=args.assignment_steps,
            calculator_count=calculator_count,
            n_embd=args.n_embd,
            span_width=args.span_width,
            hidden_size=args.result_head_hidden_size,
            prompts_per_calculator=args.prompts_per_calculator,
        )
        for operand_max in args.operand_maxes
        for calculator_count in args.calculator_counts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate forced-candidate scorer cost and result-head parameter "
            "growth for exact versus sampled result assignment."
        )
    )
    parser.add_argument("--operand-maxes", type=parse_int_list, default=[19, 29, 39, 99])
    parser.add_argument("--calculator-counts", type=parse_int_list, default=[1, 4, 16, 64])
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--assignment-steps", type=int, default=630)
    parser.add_argument("--prompts-per-calculator", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--span-width", type=int, default=2)
    parser.add_argument("--result-head-hidden-size", type=int, default=64)
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
