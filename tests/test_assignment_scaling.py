import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "analyze_assignment_scaling.py"
SPEC = importlib.util.spec_from_file_location("analyze_assignment_scaling", SCRIPT_PATH)
assignment_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = assignment_scaling
SPEC.loader.exec_module(assignment_scaling)


def test_op29_policy_topk_scaling_counts_match_known_range():
    row = assignment_scaling.compute_row(
        operand_max=29,
        sample_count=24,
        assignment_steps=630,
        calculator_count=16,
        n_embd=32,
        span_width=2,
        hidden_size=64,
    )

    assert row.result_vocab_size == 59
    assert row.prompts_per_calculator == 900
    assert row.exact_forced_evals == 535_248_000
    assert row.sampled_forced_evals == 217_728_000
    assert row.forced_eval_savings == 317_520_000
    assert row.result_head_params_per_calculator == 12_091
    assert row.result_head_params_total == 193_456
    assert row.scored_fraction == pytest.approx(24 / 59)


def test_result_head_parameter_count_supports_linear_and_hidden_heads():
    assert (
        assignment_scaling.result_head_parameter_count(
            79,
            n_embd=32,
            span_width=2,
            hidden_size=64,
        )
        == 13_391
    )
    assert (
        assignment_scaling.result_head_parameter_count(
            59,
            n_embd=32,
            span_width=2,
            hidden_size=0,
        )
        == 7_611
    )


def test_sampling_never_scores_more_classes_than_exist():
    row = assignment_scaling.compute_row(
        operand_max=9,
        sample_count=24,
        assignment_steps=10,
        calculator_count=3,
        n_embd=32,
        span_width=2,
        hidden_size=64,
    )

    assert row.result_vocab_size == 19
    assert row.sample_count == 19
    assert row.exact_forced_evals == row.sampled_forced_evals
    assert row.scoring_reduction_fraction == 0.0
