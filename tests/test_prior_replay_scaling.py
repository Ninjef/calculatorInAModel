import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "analyze_prior_replay_scaling.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_prior_replay_scaling", SCRIPT_PATH)
prior_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prior_scaling
SPEC.loader.exec_module(prior_scaling)


def test_numeric_prior_parameter_count_matches_op29_h128():
    assert (
        prior_scaling.prior_parameter_count(
            operand_vocab_size=30,
            result_vocab=59,
            hidden_size=128,
            feature_mode="numeric",
        )
        == 7_995
    )


def test_embedding_prior_parameter_count_includes_embedding_table():
    assert (
        prior_scaling.prior_parameter_count(
            operand_vocab_size=30,
            result_vocab=59,
            hidden_size=128,
            feature_mode="embedding",
        )
        == 44_347
    )


def test_op29_capped_prior_scaling_row_uses_measured_costs():
    row = prior_scaling.compute_row(
        calculator_count=16,
        operand_max=29,
        heldout_fraction=0.2,
        memory_fill_steps=192,
        batch_size=64,
        sample_count=24,
        prior_fit_examples=1_254_817,
        full_fit_examples=1_080_000,
        prior_updates=2_000,
        prior_hidden_size=128,
        prior_feature_mode="numeric",
    )

    assert row.prompt_memory_entries_per_calculator == 720
    assert row.prompt_memory_entries_total == 11_520
    assert row.forced_candidate_evals_per_calculator == 294_912
    assert row.forced_candidate_evals_total == 4_718_592
    assert row.prior_fit_examples_total == 20_077_072
    assert row.full_fit_examples_total == 17_280_000
    assert row.prior_params_per_calculator == 7_995
    assert row.prior_params_total == 127_920
    assert row.candidate_plus_prior_examples_total == 24_795_664


def test_invalid_heldout_fraction_rejected():
    with pytest.raises(ValueError):
        prior_scaling.train_prompt_count(operand_max=29, heldout_fraction=1.0)
