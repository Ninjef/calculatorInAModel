import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from src.data import EQ_ID, PLUS_ID, VOCAB_SIZE, ArithmeticBatch, tokenize
from src.model import CalculatorHook, GPTConfig, HardAddSTE, TinyGPT, masked_cross_entropy
from scripts.summarize_matched_retention_ladder import (
    select_checkpoint,
    summarize_runs,
)


def _small_cfg() -> GPTConfig:
    return GPTConfig(n_embd=32, n_layer=2, n_head=2, block_size=8)


def _small_calculator_cfg(
    mode: str = "add",
    estimator: str = "ste",
    injection_mode: str = "add",
    bottleneck_mode: str = "none",
    output_format: str = "sum",
    answer_decoder_interaction: str = "none",
) -> GPTConfig:
    return GPTConfig(
        n_embd=32,
        n_layer=2,
        n_head=2,
        block_size=8,
        calculator_enabled=True,
        calculator_mode=mode,
        calculator_estimator=estimator,
        calculator_injection_mode=injection_mode,
        calculator_bottleneck_mode=bottleneck_mode,
        calculator_output_format=output_format,
        answer_decoder_interaction=answer_decoder_interaction,
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=10,
        calculator_result_vocab_size=19,
    )


def _small_scaled_calculator_cfg(scale: float, mode: str = "add") -> GPTConfig:
    cfg = _small_calculator_cfg(mode=mode)
    cfg.calculator_injection_scale = scale
    return cfg


def _write_matched_ladder_run(
    run_dir: Path,
    *,
    seed: int,
    interface_decay_steps: int,
    rows: list[dict[str, float]],
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint_snapshots").mkdir()
    metrics = {
        "seed": seed,
        "exact_match": 0.125,
        "adaptive_interface_loss_decay_steps": interface_decay_steps,
        "final_adaptive_interface_loss_weight": 0.0
        if interface_decay_steps > 0
        else 1.0,
        "final_aux_operand_loss_weight": 0.0,
        "final_input_proj_anchor_weight": 0.0,
        "trainable_parameter_groups": [
            {"name": "calculator_hook.pair_proj"},
            {"name": "upstream"},
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics) + "\n")
    with (run_dir / "diagnostic_snapshots.csv").open("w", newline="") as handle:
        snapshot_fieldnames = [
            "step",
            "normal_exact_match",
            "injection_zero_exact_match",
            "oracle_exact_match",
            "forced_zero_exact_match",
            "forced_random_exact_match",
            "pair_exact_match",
            "calculator_result_accuracy",
            "mean_pair_entropy",
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=snapshot_fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in snapshot_fieldnames}
            for row in rows
        )
    with (run_dir / "training_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "aux_operand_loss_weight",
                "adaptive_interface_loss_weight",
                "action_loss_full_enum_pair_logit_effective_pairs",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "aux_operand_loss_weight": row["aux_operand_loss_weight"],
                    "adaptive_interface_loss_weight": row[
                        "adaptive_interface_loss_weight"
                    ],
                    "action_loss_full_enum_pair_logit_effective_pairs": 399.0,
                }
            )


def test_forward_shape() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_cfg())
    x = torch.randint(0, VOCAB_SIZE, (2, 8))

    logits = model(x)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert logits.dtype == torch.float32


def test_matched_retention_ladder_selects_aux_zero_with_tie_breakers() -> None:
    rows = [
        {
            "step": 125,
            "aux_operand_loss_weight": 0.5,
            "condition": "constant",
            "pair_exact_match": 0.9,
            "injection_zero_exact_match": 0.0,
            "forced_random_exact_match": 0.0,
            "oracle_exact_match": 1.0,
            "calculator_result_accuracy": 0.9,
        },
        {
            "step": 150,
            "aux_operand_loss_weight": 0.0,
            "condition": "constant",
            "pair_exact_match": 0.25,
            "injection_zero_exact_match": 0.02,
            "forced_random_exact_match": 0.0,
            "oracle_exact_match": 0.9,
            "calculator_result_accuracy": 0.4,
        },
        {
            "step": 175,
            "aux_operand_loss_weight": 0.0,
            "condition": "constant",
            "pair_exact_match": 0.25,
            "injection_zero_exact_match": 0.0,
            "forced_random_exact_match": 0.0,
            "oracle_exact_match": 0.9,
            "calculator_result_accuracy": 0.3,
        },
    ]

    selected = select_checkpoint(rows)

    assert selected["step"] == 175


def test_matched_retention_ladder_requires_decayed_interface_zero(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "step": 150,
            "aux_operand_loss_weight": 0.0,
            "adaptive_interface_loss_weight": 0.2,
            "normal_exact_match": 0.2,
            "injection_zero_exact_match": 0.0,
            "oracle_exact_match": 0.9,
            "forced_zero_exact_match": 0.0,
            "forced_random_exact_match": 0.0,
            "pair_exact_match": 0.9,
            "calculator_result_accuracy": 0.9,
            "mean_pair_entropy": 5.9,
        },
        {
            "step": 175,
            "aux_operand_loss_weight": 0.0,
            "adaptive_interface_loss_weight": 0.0,
            "normal_exact_match": 0.1,
            "injection_zero_exact_match": 0.0,
            "oracle_exact_match": 0.9,
            "forced_zero_exact_match": 0.0,
            "forced_random_exact_match": 0.0,
            "pair_exact_match": 0.2,
            "calculator_result_accuracy": 0.3,
            "mean_pair_entropy": 5.9,
        },
    ]
    run_dir = tmp_path / "model-c-2digit-seed1"
    _write_matched_ladder_run(
        run_dir,
        seed=1,
        interface_decay_steps=150,
        rows=rows,
    )

    summary = summarize_runs([run_dir])

    assert summary["selected"][0]["step"] == 175
    assert summary["selected"][0]["condition"] == "decayed"
    assert summary["aggregate"][0]["mean_selected_pair_exact"] == 0.2


def test_mlp_expansion_changes_parameter_count() -> None:
    narrow_cfg = _small_cfg()
    wide_cfg = _small_cfg()
    narrow_cfg.mlp_expansion = 1
    wide_cfg.mlp_expansion = 4

    assert TinyGPT(narrow_cfg).num_params() < TinyGPT(wide_cfg).num_params()


def test_causal_mask_does_not_leak_future_tokens() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_cfg())
    model.eval()

    a = torch.randint(0, VOCAB_SIZE, (1, 8))
    b = a.clone()
    # change only the last position; logits at earlier positions must be unchanged
    b[0, -1] = (a[0, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        la = model(a)
        lb = model(b)

    assert torch.allclose(la[:, :-1, :], lb[:, :-1, :], atol=1e-6)


def test_calculator_off_preserves_forward_and_generate_contracts() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="off"))
    x = torch.randint(0, VOCAB_SIZE, (2, 8))

    logits = model(x)
    generated = model.generate(x[:, :3], max_new_tokens=2)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert generated.shape == (2, 5)


def test_calculator_hook_does_not_change_core_initialization() -> None:
    seed = 123
    torch.manual_seed(seed)
    model_a = TinyGPT(_small_cfg())
    torch.manual_seed(seed)
    model_b = TinyGPT(_small_calculator_cfg(mode="off"))

    for name, param in model_a.state_dict().items():
        if name.startswith("calculator_hook."):
            continue
        assert torch.equal(param, model_b.state_dict()[name]), name


def test_calculator_off_forward_matches_model_without_hook() -> None:
    seed = 123
    torch.manual_seed(seed)
    model_a = TinyGPT(_small_cfg())
    torch.manual_seed(seed)
    model_b = TinyGPT(_small_calculator_cfg(mode="off"))
    x = torch.tensor([[1, 2, EQ_ID, 3, 4, 5, 6, 7]])

    with torch.no_grad():
        logits_a = model_a(x)
        logits_b = model_b(x)

    assert torch.equal(logits_a, logits_b)


def test_invalid_calculator_injection_mode_raises() -> None:
    cfg = _small_calculator_cfg(injection_mode="middle")

    with pytest.raises(ValueError, match="calculator_injection_mode"):
        TinyGPT(cfg)


def test_invalid_calculator_bottleneck_mode_raises() -> None:
    cfg = _small_calculator_cfg(bottleneck_mode="middle")

    with pytest.raises(ValueError, match="calculator_bottleneck_mode"):
        TinyGPT(cfg)


def test_invalid_calculator_output_format_raises() -> None:
    cfg = _small_calculator_cfg(output_format="middle")

    with pytest.raises(ValueError, match="calculator_output_format"):
        TinyGPT(cfg)


def test_invalid_answer_decoder_interaction_raises() -> None:
    cfg = _small_calculator_cfg(answer_decoder_interaction="middle")

    with pytest.raises(ValueError, match="answer_decoder_interaction"):
        TinyGPT(cfg)


def test_default_calculator_output_format_preserves_projection_width() -> None:
    hook = CalculatorHook(_small_calculator_cfg())

    assert hook.output_proj.in_features == hook.result_vocab_size
    assert hook.output_format == "sum"


def test_sum_left_operand_output_projection_width_includes_operand() -> None:
    hook = CalculatorHook(_small_calculator_cfg(output_format="sum_left_operand"))

    assert hook.output_proj.in_features == (
        hook.result_vocab_size + hook.operand_vocab_size
    )


def test_add_calculator_injection_mode_adds_residual() -> None:
    model = TinyGPT(_small_calculator_cfg(injection_mode="add"))
    h = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
    injection = torch.ones_like(h)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    updated = model._apply_calculator_injection(h, injection, tokens)

    assert torch.equal(updated, h + injection)


def test_replace_calculator_injection_mode_only_replaces_equals_positions() -> None:
    model = TinyGPT(_small_calculator_cfg(injection_mode="replace"))
    h = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
    injection = torch.full_like(h, -1.0)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    updated = model._apply_calculator_injection(h, injection, tokens)

    assert torch.equal(updated[0, :2], h[0, :2])
    assert torch.equal(updated[0, 3:], h[0, 3:])
    assert torch.equal(updated[0, 2], injection[0, 2])


def test_calculator_off_replace_mode_zeros_equals_residual_only() -> None:
    model = TinyGPT(_small_calculator_cfg(mode="off", injection_mode="replace"))
    h = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])
    assert model.calculator_hook is not None

    injection, trace = model.calculator_hook(h, tokens, return_trace=True)
    updated = model._apply_calculator_injection(h, injection, tokens)

    assert torch.all(injection == 0)
    assert trace["injection_norm"][0, 2].item() == 0
    assert torch.equal(updated[0, :2], h[0, :2])
    assert torch.equal(updated[0, 3:], h[0, 3:])
    assert torch.equal(updated[0, 2], torch.zeros_like(updated[0, 2]))


def test_answer_decoder_bottleneck_blocks_operand_bypass_with_zero_calculator() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(mode="off", bottleneck_mode="answer_decoder")
    )
    model.eval()
    x1 = torch.tensor([[1, 2, PLUS_ID, 3, 4, EQ_ID, 5, 6]])
    x2 = torch.tensor([[7, 8, PLUS_ID, 9, 0, EQ_ID, 5, 6]])

    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)

    assert not torch.allclose(logits1[:, :5], logits2[:, :5])
    assert torch.equal(logits1[:, 5:], logits2[:, 5:])


def test_answer_decoder_bottleneck_uses_forced_calculator_result() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(mode="add", bottleneck_mode="answer_decoder")
    )
    assert model.calculator_hook is not None
    assert model.answer_decoder is not None
    with torch.no_grad():
        model.calculator_hook.input_proj.weight.zero_()
        model.calculator_hook.input_proj.bias.fill_(-10.0)
        model.calculator_hook.input_proj.bias[2] = 10.0
        model.calculator_hook.input_proj.bias[10 + 3] = 10.0
        model.calculator_hook.output_proj.weight.zero_()
        model.calculator_hook.output_proj.weight[0, 5] = 10.0
        model.answer_decoder.weight.zero_()
        model.answer_decoder.weight[0, 0] = 1.0

    x = torch.tensor([[1, 2, PLUS_ID, 3, 4, EQ_ID, 5, 6]])

    with torch.no_grad():
        zero_logits = model(x, forced_calculator_result_class=0)
        five_logits = model(x, forced_calculator_result_class=5)

    assert five_logits[0, 5, 0].item() > zero_logits[0, 5, 0].item() + 1.0


def test_sum_only_answer_decoder_same_sum_oracle_prompts_are_indistinguishable() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(
            bottleneck_mode="answer_decoder",
            output_format="sum",
        )
    )
    assert model.calculator_hook is not None
    assert model.answer_decoder is not None
    with torch.no_grad():
        model.calculator_hook.output_proj.weight.zero_()
        model.calculator_hook.output_proj.weight[0, 2] = 1.0
        model.answer_offset_emb.weight.zero_()
        model.answer_decoder.weight.zero_()
        model.answer_decoder.weight[0, 0] = 1.0

    x = torch.tensor(
        [
            [0, PLUS_ID, 2, EQ_ID],
            [1, PLUS_ID, 1, EQ_ID],
        ]
    )
    oracle = torch.zeros((*x.shape, 2), dtype=torch.long)
    oracle[0, :, 0] = 0
    oracle[0, :, 1] = 2
    oracle[1, :, 0] = 1
    oracle[1, :, 1] = 1

    with torch.no_grad():
        logits = model(x, oracle_operands=oracle)

    assert torch.equal(logits[0, 3], logits[1, 3])


def test_default_sum_only_answer_decoder_interaction_is_additive() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(
            bottleneck_mode="answer_decoder",
            output_format="sum",
        )
    )
    assert model.answer_offset_emb is not None
    assert model.answer_decoder is not None
    with torch.no_grad():
        model.answer_offset_emb.weight.zero_()
        model.answer_offset_emb.weight[0, 0] = 2.0
        model.answer_decoder.weight.zero_()
        model.answer_decoder.weight[0, 0] = 1.0
    base_logits = torch.zeros((1, 4, VOCAB_SIZE))
    calculator_signal = torch.zeros((1, 4, model.cfg.n_embd))
    calculator_signal[0, 3, 0] = 3.0
    tokens = torch.tensor([[0, PLUS_ID, 1, EQ_ID]])

    logits = model._answer_bottleneck_logits(base_logits, calculator_signal, tokens)

    assert logits[0, 3, 0].item() == pytest.approx(5.0)


def test_sum_only_product_interaction_changes_answer_decoder_hidden_state() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(
            bottleneck_mode="answer_decoder",
            output_format="sum",
            answer_decoder_interaction="product",
        )
    )
    assert model.answer_offset_emb is not None
    assert model.answer_decoder is not None
    with torch.no_grad():
        model.answer_offset_emb.weight.zero_()
        model.answer_offset_emb.weight[0, 0] = 2.0
        model.answer_decoder.weight.zero_()
        model.answer_decoder.weight[0, 0] = 1.0
    base_logits = torch.zeros((1, 4, VOCAB_SIZE))
    calculator_signal = torch.zeros((1, 4, model.cfg.n_embd))
    calculator_signal[0, 3, 0] = 3.0
    tokens = torch.tensor([[0, PLUS_ID, 1, EQ_ID]])

    logits = model._answer_bottleneck_logits(base_logits, calculator_signal, tokens)

    assert logits[0, 3, 0].item() == pytest.approx(11.0)


def test_sum_left_operand_answer_decoder_distinguishes_same_sum_oracle_prompts() -> None:
    torch.manual_seed(0)
    model = TinyGPT(
        _small_calculator_cfg(
            bottleneck_mode="answer_decoder",
            output_format="sum_left_operand",
        )
    )
    assert model.calculator_hook is not None
    assert model.answer_decoder is not None
    result_vocab_size = model.calculator_hook.result_vocab_size
    with torch.no_grad():
        model.calculator_hook.output_proj.weight.zero_()
        model.calculator_hook.output_proj.weight[0, result_vocab_size + 0] = 1.0
        model.calculator_hook.output_proj.weight[0, result_vocab_size + 1] = 2.0
        model.answer_offset_emb.weight.zero_()
        model.answer_decoder.weight.zero_()
        model.answer_decoder.weight[0, 0] = 1.0

    x = torch.tensor(
        [
            [0, PLUS_ID, 2, EQ_ID],
            [1, PLUS_ID, 1, EQ_ID],
        ]
    )
    oracle = torch.zeros((*x.shape, 2), dtype=torch.long)
    oracle[0, :, 0] = 0
    oracle[0, :, 1] = 2
    oracle[1, :, 0] = 1
    oracle[1, :, 1] = 1

    with torch.no_grad():
        logits = model(x, oracle_operands=oracle)

    assert logits[1, 3, 0].item() > logits[0, 3, 0].item() + 0.5


def test_checkpoint_without_answer_decoder_interaction_loads_with_old_default(
    tmp_path: Path,
) -> None:
    script_path = Path("scripts/diagnose_calculator_protocol.py")
    spec = importlib.util.spec_from_file_location("diagnose_load_checkpoint", script_path)
    assert spec is not None
    assert spec.loader is not None
    diagnose_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnose_script)

    cfg = _small_calculator_cfg(
        bottleneck_mode="answer_decoder",
        output_format="sum",
    )
    source = TinyGPT(cfg)
    model_config = vars(cfg).copy()
    model_config.pop("answer_decoder_interaction")
    checkpoint_path = tmp_path / "old_checkpoint.pt"
    torch.save(
        {"config": {"model": model_config}, "model_state_dict": source.state_dict()},
        checkpoint_path,
    )

    loaded, _ = diagnose_script.load_checkpoint(
        checkpoint_path, device="cpu", injection_scale=None
    )

    assert loaded.cfg.answer_decoder_interaction == "none"


def test_hard_add_ste_forward_returns_sum_class() -> None:
    a_logits = torch.full((1, 10), -10.0)
    b_logits = torch.full((1, 10), -10.0)
    a_logits[0, 3] = 10.0
    b_logits[0, 4] = 10.0

    result = HardAddSTE.apply(a_logits, b_logits)

    assert result.shape == (1, 19)
    assert result.argmax(dim=-1).item() == 7
    assert result[0, 7].item() == 1.0


def test_hard_add_ste_backward_routes_sum_gradients_to_operand_logits() -> None:
    a_logits = torch.full((1, 10), -10.0, requires_grad=True)
    b_logits = torch.full((1, 10), -10.0, requires_grad=True)
    with torch.no_grad():
        a_logits[0, 3] = 10.0
        b_logits[0, 4] = 10.0

    result = HardAddSTE.apply(a_logits, b_logits)
    weights = torch.arange(19, dtype=result.dtype)
    (result * weights).sum().backward()

    assert torch.equal(a_logits.grad[0], torch.arange(4, 14, dtype=result.dtype))
    assert torch.equal(b_logits.grad[0], torch.arange(3, 13, dtype=result.dtype))


def test_gumbel_concrete_interface_uses_hard_forward_soft_backward_signal() -> None:
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_output_format="sum_left_operand",
        relaxed_calculator_temperature=2.0,
    )
    hook = CalculatorHook(cfg)
    a_logits = torch.tensor([[[0.0, 3.0, 1.0, -2.0]]], requires_grad=True)
    b_logits = torch.tensor([[[0.0, -1.0, 4.0, 2.0]]], requires_grad=True)

    flat_result, a_pred, b_pred, signal = hook._relaxed_calculator_output_signal(
        a_logits=a_logits,
        b_logits=b_logits,
        dtype=torch.float32,
    )

    assert a_pred.item() == 1
    assert b_pred.item() == 2
    assert flat_result.argmax(dim=-1).item() == 3
    assert torch.equal(
        signal.detach()[0, 0],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )

    loss = (signal * torch.arange(signal.shape[-1], dtype=signal.dtype)).sum()
    loss.backward()

    assert a_logits.grad is not None
    assert b_logits.grad is not None
    assert a_logits.grad.abs().sum().item() > 0
    assert b_logits.grad.abs().sum().item() > 0


def test_joint_pair_gumbel_concrete_uses_result_group_soft_backward_signal() -> None:
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="joint_pair",
        relaxed_calculator_temperature=2.0,
    )
    hook = CalculatorHook(cfg)
    pair_logits = torch.full((1, 1, 16), -3.0, requires_grad=True)
    with torch.no_grad():
        pair_logits[0, 0, 1 * 4 + 2] = 5.0

    flat_result, a_pred, b_pred, signal = (
        hook._relaxed_joint_pair_calculator_output_signal(
            pair_logits=pair_logits,
            dtype=torch.float32,
        )
    )

    assert a_pred.item() == 1
    assert b_pred.item() == 2
    assert flat_result.argmax(dim=-1).item() == 3
    assert torch.allclose(
        signal.detach()[0, 0],
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    )

    loss = (signal * torch.arange(signal.shape[-1], dtype=signal.dtype)).sum()
    loss.backward()

    assert pair_logits.grad is not None
    assert pair_logits.grad.abs().sum().item() > 0


def test_result_space_gumbel_concrete_uses_result_soft_backward_signal() -> None:
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        relaxed_calculator_temperature=2.0,
    )
    hook = CalculatorHook(cfg)
    result_logits = torch.full((1, 1, 7), -3.0, requires_grad=True)
    with torch.no_grad():
        result_logits[0, 0, 5] = 5.0

    flat_result, a_pred, b_pred, signal = (
        hook._relaxed_result_space_calculator_output_signal(
            result_logits=result_logits,
            dtype=torch.float32,
        )
    )

    assert a_pred.item() == 3
    assert b_pred.item() == 2
    assert flat_result.argmax(dim=-1).item() == 5
    assert torch.allclose(
        signal.detach()[0, 0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )

    loss = (signal * torch.arange(signal.shape[-1], dtype=signal.dtype)).sum()
    loss.backward()

    assert result_logits.grad is not None
    assert result_logits.grad.abs().sum().item() > 0


def test_calculator_injection_is_localized_to_equals_positions() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.fill_(-10.0)
        hook.input_proj.bias[3] = 10.0
        hook.input_proj.bias[10 + 4] = 10.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    injection = hook(h, tokens)

    assert torch.all(injection[0, :2] == 0)
    assert torch.all(injection[0, 3:] == 0)
    assert torch.all(injection[0, 2] != 0)


def test_calculator_hook_after_layer_zero_runs_at_embedding_stream() -> None:
    torch.manual_seed(0)
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_hook_after_layer = 0
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    with torch.no_grad():
        model.calculator_hook.input_proj.weight.zero_()
        model.calculator_hook.input_proj.bias.fill_(-10.0)
        model.calculator_hook.input_proj.bias[3] = 10.0
        model.calculator_hook.input_proj.bias[10 + 4] = 10.0
        model.calculator_hook.output_proj.weight.fill_(1.0)

    x = torch.tensor([[1, 2, EQ_ID, 3, 4, 5, 6, 7]])

    _, diagnostics = model(x, return_diagnostics=True)
    trace = diagnostics["calculator_trace"]

    assert trace["a_pred"][0, 2].item() == 3
    assert trace["b_pred"][0, 2].item() == 4
    assert trace["result_pred"][0, 2].item() == 7
    assert trace["injection_norm"][0, 2].item() > 0


def test_calculator_trace_records_shapes_values_and_equals_positions() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.fill_(-10.0)
        hook.input_proj.bias[3] = 10.0
        hook.input_proj.bias[10 + 4] = 10.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    injection, trace = hook(h, tokens, return_trace=True)

    assert injection.shape == h.shape
    assert trace["eq_mask"].shape == tokens.shape
    assert trace["a_pred"][0, 2].item() == 3
    assert trace["b_pred"][0, 2].item() == 4
    assert trace["result_pred"][0, 2].item() == 7
    assert trace["eq_mask"][0].tolist() == [False, False, True, False, False]
    assert trace["injection_norm"][0, 2].item() > 0
    assert trace["injection_norm"][0, 0].item() == 0
    assert trace["calculator_read_position_id"][0, 2].item() == 0
    assert trace["a_read_position"][0, 2].item() == 2
    assert trace["b_read_position"][0, 2].item() == 2
    assert trace["eq_read_position"][0, 2].item() == 2


def test_calculator_operands_read_position_reads_operand_tokens_and_injects_at_equals() -> None:
    torch.manual_seed(0)
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_read_position = "operands"
    hook = CalculatorHook(cfg)
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.zero_()
        hook.input_proj.weight[3, 0] = 1.0
        hook.input_proj.weight[10 + 4, 1] = 1.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.zeros(1, 8, 32)
    h[0, 1, 0] = 10.0
    h[0, 4, 1] = 10.0
    tokens = torch.tensor([[0, 7, PLUS_ID, 0, 5, EQ_ID, 1, 2]])

    injection, trace = hook(h, tokens, return_trace=True)

    assert trace["a_pred"][0, 5].item() == 3
    assert trace["b_pred"][0, 5].item() == 4
    assert trace["result_pred"][0, 5].item() == 7
    assert trace["calculator_read_position_id"][0, 5].item() == 1
    assert trace["a_read_position"][0, 5].item() == 1
    assert trace["b_read_position"][0, 5].item() == 4
    assert trace["eq_read_position"][0, 5].item() == 5
    assert torch.all(injection[0, :5] == 0)
    assert torch.all(injection[0, 6:] == 0)
    assert torch.all(injection[0, 5] != 0)


def test_calculator_operand_spans_read_full_fixed_width_operands() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=32,
        n_layer=1,
        n_head=1,
        block_size=8,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
    )
    hook = CalculatorHook(cfg)
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.zero_()
        hook.input_proj.weight[7, 32] = 1.0
        hook.input_proj.weight[20 + 5, 32] = 1.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.zeros(1, 8, 32)
    h[0, 1, 0] = 10.0
    h[0, 4, 0] = 10.0
    tokens = torch.tensor([[0, 7, PLUS_ID, 0, 5, EQ_ID, 1, 2]])

    injection, trace = hook(h, tokens, return_trace=True)

    assert trace["a_pred"][0, 5].item() == 7
    assert trace["b_pred"][0, 5].item() == 5
    assert trace["result_pred"][0, 5].item() == 12
    assert trace["calculator_read_position_id"][0, 5].item() == 2
    assert trace["a_read_position"][0, 5].item() == 1
    assert trace["b_read_position"][0, 5].item() == 4
    assert trace["eq_read_position"][0, 5].item() == 5
    assert torch.all(injection[0, :5] == 0)
    assert torch.all(injection[0, 6:] == 0)
    assert torch.all(injection[0, 5] != 0)


def test_calculator_operand_read_positions_ignore_longer_answer_format() -> None:
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_read_position = "operands"
    hook = CalculatorHook(cfg)
    h = torch.randn(1, 12, cfg.n_embd)
    tokens = torch.tensor([tokenize("07+12=01907<eos>")])

    _, trace = hook(h, tokens, return_trace=True)

    assert trace["a_read_position"][0, 5].item() == 1
    assert trace["b_read_position"][0, 5].item() == 4
    assert trace["eq_read_position"][0, 5].item() == 5


def test_calculator_operands_read_position_uses_first_equals_as_prompt_anchor() -> None:
    torch.manual_seed(0)
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_read_position = "operands"
    hook = CalculatorHook(cfg)
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.zero_()
        hook.input_proj.weight[3, 0] = 1.0
        hook.input_proj.weight[10 + 4, 1] = 1.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.zeros(1, 8, 32)
    h[0, 1, 0] = 10.0
    h[0, 4, 1] = 10.0
    tokens = torch.tensor([[0, 7, PLUS_ID, 0, 5, EQ_ID, 1, EQ_ID]])

    injection, trace = hook(h, tokens, return_trace=True)

    assert trace["a_pred"][0, 5].item() == 3
    assert trace["b_pred"][0, 5].item() == 4
    assert trace["eq_read_position"][0, 7].item() == 5
    assert torch.all(injection[0, 5] != 0)
    assert torch.all(injection[0, 7] != 0)


def test_invalid_calculator_read_position_raises() -> None:
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_read_position = "middle"

    with pytest.raises(ValueError, match="calculator_read_position"):
        CalculatorHook(cfg)


def test_reinforce_calculator_trace_records_sample_logprob() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add", estimator="reinforce"))
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.fill_(-10.0)
        hook.input_proj.bias[3] = 10.0
        hook.input_proj.bias[10 + 4] = 10.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    _, trace = hook(h, tokens, return_trace=True)

    assert trace["a_pred"][0, 2].item() == 3
    assert trace["b_pred"][0, 2].item() == 4
    assert trace["result_pred"][0, 2].item() == 7
    assert trace["sampled_logp"][0, 2].item() <= 0
    assert torch.isfinite(trace["sampled_logp"][0, 2])


def test_oracle_operands_force_calculator_result_class() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])
    oracle = torch.zeros(1, 5, 2, dtype=torch.long)
    oracle[..., 0] = 2
    oracle[..., 1] = 5

    _, trace = hook(h, tokens, oracle_operands=oracle, return_trace=True)

    assert trace["a_pred"][0, 2].item() == 2
    assert trace["b_pred"][0, 2].item() == 5
    assert trace["result_pred"][0, 2].item() == 7
    assert trace["oracle_used"][0, 2].item() is True


def test_calculator_result_override_changes_result_class() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])
    oracle = torch.zeros(1, 5, 2, dtype=torch.long)
    oracle[..., 0] = 3
    oracle[..., 1] = 4

    _, zero_trace = hook(
        h,
        tokens,
        oracle_operands=oracle,
        result_override="zero",
        return_trace=True,
    )
    _, plus_one_trace = hook(
        h,
        tokens,
        oracle_operands=oracle,
        result_override="plus_one",
        return_trace=True,
    )

    assert zero_trace["result_pred"][0, 2].item() == 0
    assert plus_one_trace["result_pred"][0, 2].item() == 8


def test_forced_calculator_result_class_overrides_learned_sum() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])
    oracle = torch.zeros(1, 5, 2, dtype=torch.long)
    oracle[..., 0] = 3
    oracle[..., 1] = 4

    _, trace = hook(
        h,
        tokens,
        oracle_operands=oracle,
        forced_result_class=12,
        return_trace=True,
    )

    assert trace["a_pred"][0, 2].item() == 3
    assert trace["b_pred"][0, 2].item() == 4
    assert trace["result_pred"][0, 2].item() == 12


def test_tensor_forced_calculator_result_class_overrides_per_example() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(2, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4], [4, 3, EQ_ID, 2, 1]])
    oracle = torch.zeros(2, 5, 2, dtype=torch.long)
    oracle[..., 0] = 3
    oracle[..., 1] = 4

    _, trace = hook(
        h,
        tokens,
        oracle_operands=oracle,
        forced_result_class=torch.tensor([5, 12]),
        return_trace=True,
    )

    assert trace["result_pred"][0, 2].item() == 5
    assert trace["result_pred"][1, 2].item() == 12


def test_invalid_forced_calculator_result_class_raises() -> None:
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    with pytest.raises(ValueError, match="forced_result_class"):
        hook(h, tokens, forced_result_class=19)


def test_invalid_tensor_forced_calculator_result_class_raises() -> None:
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(2, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4], [4, 3, EQ_ID, 2, 1]])

    with pytest.raises(ValueError, match="forced_result_class tensor values"):
        hook(h, tokens, forced_result_class=torch.tensor([1, 19]))


def test_tiny_gpt_forwards_forced_calculator_result_class() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="add"))
    assert model.calculator_hook is not None
    x = torch.tensor([[1, 2, EQ_ID, 3, 4, 5, 6, 7]])

    _, diagnostics = model(
        x, return_diagnostics=True, forced_calculator_result_class=5
    )

    assert diagnostics["calculator_trace"]["result_pred"][0, 2].item() == 5


def test_read_site_swap_and_corrupt_change_only_calculator_read_input() -> None:
    torch.manual_seed(0)
    cfg = _small_calculator_cfg(mode="add")
    cfg.calculator_read_position = "operands"
    model = TinyGPT(cfg)
    x = torch.tensor(
        [
            [1, 2, PLUS_ID, 3, 4, EQ_ID, 5, 6],
            [7, 8, PLUS_ID, 9, 0, EQ_ID, 1, 2],
        ]
    )

    _, normal = model(x, return_diagnostics=True)
    _, swapped = model(
        x,
        return_diagnostics=True,
        calculator_read_intervention="swap_a_read_vector",
    )
    _, corrupted = model(
        x,
        return_diagnostics=True,
        calculator_read_intervention="corrupt_b_read_vector",
    )

    normal_read = normal["calculator_read_residual"]
    swapped_read = swapped["calculator_read_residual_intervened"]
    corrupted_read = corrupted["calculator_read_residual_intervened"]
    assert torch.equal(normal_read, swapped["calculator_read_residual"])
    assert torch.equal(normal_read[:, 4], swapped_read[:, 4])
    assert torch.equal(swapped_read[0, 1], normal_read[1, 1])
    assert torch.equal(swapped_read[1, 1], normal_read[0, 1])
    assert torch.equal(normal_read[:, 1], corrupted_read[:, 1])
    assert torch.all(corrupted_read[:, 4] == 0)


def test_oracle_operands_with_replace_mode_record_expected_result() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="add", injection_mode="replace"))
    x = torch.tensor([[1, 2, EQ_ID, 3, 4, 5, 6, 7]])
    oracle = torch.zeros(1, 8, 2, dtype=torch.long)
    oracle[..., 0] = 2
    oracle[..., 1] = 5

    _, diagnostics = model(x, return_diagnostics=True, oracle_operands=oracle)
    trace = diagnostics["calculator_trace"]

    assert trace["a_pred"][0, 2].item() == 2
    assert trace["b_pred"][0, 2].item() == 5
    assert trace["result_pred"][0, 2].item() == 7
    assert trace["oracle_used"][0, 2].item() is True


def test_calculator_injection_scale_zero_removes_active_injection() -> None:
    torch.manual_seed(0)
    hook_zero = CalculatorHook(_small_scaled_calculator_cfg(scale=0.0))
    hook_one = CalculatorHook(_small_scaled_calculator_cfg(scale=1.0))
    hook_one.load_state_dict(hook_zero.state_dict())
    with torch.no_grad():
        hook_zero.output_proj.weight.fill_(1.0)
        hook_one.output_proj.weight.fill_(1.0)

    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])
    oracle = torch.zeros(1, 5, 2, dtype=torch.long)
    oracle[..., 0] = 3
    oracle[..., 1] = 4

    injection_zero, trace_zero = hook_zero(
        h, tokens, oracle_operands=oracle, return_trace=True
    )
    injection_one, trace_one = hook_one(
        h, tokens, oracle_operands=oracle, return_trace=True
    )

    assert torch.all(injection_zero == 0)
    assert torch.all(injection_one[0, 2] != 0)
    assert trace_zero["injection_norm"][0, 2].item() == 0
    assert trace_one["injection_norm"][0, 2].item() > 0


def test_causal_mask_does_not_leak_future_tokens_with_calculator_enabled() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="add"))
    model.eval()

    a = torch.randint(0, VOCAB_SIZE, (1, 8))
    a[0, 3] = EQ_ID
    b = a.clone()
    b[0, -1] = (a[0, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        la = model(a)
        lb = model(b)

    assert torch.allclose(la[:, :-1, :], lb[:, :-1, :], atol=1e-6)


def test_masked_cross_entropy_ignores_unmasked_positions() -> None:
    B, T, V = 1, 3, VOCAB_SIZE
    logits = torch.full((B, T, V), -10.0)
    # position 0: perfect prediction for class 1
    logits[0, 0, 1] = 10.0
    # positions 1 and 2: very wrong (would dominate loss if not masked)
    logits[0, 1, 0] = 10.0
    logits[0, 2, 0] = 10.0

    targets = torch.tensor([[1, 5, 7]])
    mask = torch.tensor([[True, False, False]])

    loss = masked_cross_entropy(logits, targets, mask)

    assert loss.item() < 1e-3


def test_diagnostic_cli_smoke(tmp_path, monkeypatch) -> None:
    script_path = Path("scripts/diagnose_calculator_protocol.py")
    spec = importlib.util.spec_from_file_location("diagnose_cli", script_path)
    assert spec is not None
    assert spec.loader is not None
    diagnose_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnose_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--digits",
            "1",
            "--steps",
            "0",
            "--samples",
            "8",
            "--batch-size",
            "4",
            "--operand-max",
            "2",
            "--calculator-read-position",
            "operands",
            "--calculator-bottleneck-mode",
            "answer_decoder",
            "--probe",
            "--probe-steps",
            "2",
            "--forced-result-batch-size",
            "7",
            "--output-dir",
            str(tmp_path),
        ],
    )

    diagnose_cli.main()

    rows_path = tmp_path / "calculator_trace_rows.csv"
    summary_path = tmp_path / "diagnostic_summary.json"
    assert rows_path.exists()
    assert summary_path.exists()
    assert (tmp_path / "result_codebook.csv").exists()
    assert (tmp_path / "operand_codebook.csv").exists()
    assert (tmp_path / "counterfactual_exact_match.csv").exists()
    summary = json.loads(summary_path.read_text())
    assert summary["samples"] == 8
    assert summary["operand_max"] == 2
    assert summary["calculator_read_position"] == "operands"
    assert summary["calculator_bottleneck_mode"] == "answer_decoder"
    assert summary["classification"]["bottleneck_classification"] in {
        "calculator_required_bottleneck",
        "strict_bottleneck_unvalidated",
    }
    assert "mutual_information_bits" in summary
    assert "counterfactual_exact_match" in summary
    assert "classification" in summary
    assert "probe" in summary


def test_diagnostic_cli_forced_result_sweep_writes_outputs(
    tmp_path, monkeypatch
) -> None:
    script_path = Path("scripts/diagnose_calculator_protocol.py")
    spec = importlib.util.spec_from_file_location("diagnose_cli_sweep", script_path)
    assert spec is not None
    assert spec.loader is not None
    diagnose_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnose_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--digits",
            "1",
            "--steps",
            "0",
            "--samples",
            "2",
            "--batch-size",
            "4",
            "--operand-max",
            "2",
            "--forced-result-sweep",
            "--forced-result-batch-size",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
    )

    diagnose_cli.main()

    sweep_path = tmp_path / "forced_result_sweep.csv"
    codebook_path = tmp_path / "result_codebook.csv"
    summary_path = tmp_path / "forced_result_summary.json"
    assert sweep_path.exists()
    assert codebook_path.exists()
    assert summary_path.exists()
    with sweep_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 38
    assert {
        "forced_result_class",
        "forced_matches_learned",
        "correct_first_token_prob",
        "target_logprob",
    }.issubset(rows[0])
    summary = json.loads(summary_path.read_text())
    assert summary["samples"] == 2
    assert summary["result_vocab_size"] == 19
    assert summary["forced_result_batch_size"] == 4


def test_full_enum_exhaustive_grid_helpers_cover_each_pair_once() -> None:
    script_path = Path("scripts/run_full_enum_action_loss_diagnostic.py")
    spec = importlib.util.spec_from_file_location("full_enum_grid", script_path)
    assert spec is not None
    assert spec.loader is not None
    full_enum = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full_enum)

    specs = full_enum.all_pair_specs(2)
    assert specs == [
        {"sample": 0, "true_a": 0, "true_b": 0},
        {"sample": 1, "true_a": 0, "true_b": 1},
        {"sample": 2, "true_a": 0, "true_b": 2},
        {"sample": 3, "true_a": 1, "true_b": 0},
        {"sample": 4, "true_a": 1, "true_b": 1},
        {"sample": 5, "true_a": 1, "true_b": 2},
        {"sample": 6, "true_a": 2, "true_b": 0},
        {"sample": 7, "true_a": 2, "true_b": 1},
        {"sample": 8, "true_a": 2, "true_b": 2},
    ]
    batch = full_enum.batch_from_specs(
        specs,
        num_digits=1,
        fixed_width=True,
        device="cpu",
        answer_format="sum",
    )
    assert batch.x.shape[0] == 9
    assert full_enum.format_prompt(batch.x[0]) == "0+0="
    assert full_enum.format_prompt(batch.x[-1]) == "2+2="


def test_track4_action_loss_diagnostic_reports_operand_action_gaps() -> None:
    script_path = Path("scripts/run_track4_action_loss_diagnostic.py")
    spec = importlib.util.spec_from_file_location("track4_action_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    track4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(track4)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_read_position="operands",
    )
    model = TinyGPT(cfg)

    action_rows, prompt_rows, summary = track4.action_loss_diagnostic(
        model,
        sample_specs=[
            {"sample": 0, "true_a": 1, "true_b": 2},
            {"sample": 1, "true_a": 2, "true_b": 0},
        ],
        num_digits=1,
        operand_max=2,
        random_actions=2,
        seed=0,
        device="cpu",
        oracle_base=False,
    )

    assert len(action_rows) == 12
    assert len(prompt_rows) == 2
    assert summary["samples"] == 2
    assert summary["random_actions_per_prompt"] == 2
    assert {
        "mean_random_minus_true_gap",
        "mean_action_loss_std",
        "operand_exact_match",
        "calculator_result_accuracy",
    }.issubset(summary)


def test_full_enum_action_loss_builds_soft_marginals() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_full_enum", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    losses = torch.tensor([[0.0, 2.0, 1.0, 4.0]])
    weights = overfit_script.action_loss_weights_from_losses(
        losses, temperature=1.0, min_probability_floor=0.0
    )
    candidates = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])
    logits = torch.zeros(1, 2)

    target_a, target_b = overfit_script.action_loss_soft_targets(
        logits, logits, candidates, weights
    )

    assert weights.shape == (1, 4)
    assert weights.sum().item() == pytest.approx(1.0)
    assert target_a[0, 0].item() == pytest.approx(
        weights[0, 0].item() + weights[0, 1].item()
    )
    assert target_a[0, 1].item() == pytest.approx(
        weights[0, 2].item() + weights[0, 3].item()
    )
    assert target_b[0, 0].item() == pytest.approx(
        weights[0, 0].item() + weights[0, 2].item()
    )
    assert target_b[0, 1].item() == pytest.approx(
        weights[0, 1].item() + weights[0, 3].item()
    )


def test_exhaustive_range_batch_covers_ordered_pairs_once() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_exhaustive_batch", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    batch = overfit_script.make_exhaustive_range_batch(
        num_digits=1,
        operand_max=2,
        fixed_width=True,
        device="cpu",
    )
    true_a, true_b = overfit_script.fixed_width_operands_from_batch(
        batch.x, num_digits=1
    )
    pairs = list(zip(true_a.tolist(), true_b.tolist()))

    assert batch.x.shape[0] == 9
    assert pairs == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    assert len(set(pairs)) == 9


def test_exhaustive_range_batch_matches_range_padding_and_masks() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_exhaustive_padding", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    class OrderedPairRng:
        def __init__(self) -> None:
            self.values = iter([0, 0, 0, 1, 1, 0, 1, 1])

        def randint(self, low: int, high: int) -> int:
            assert low == 0
            assert high == 1
            return next(self.values)

    exhaustive = overfit_script.make_exhaustive_range_batch(
        num_digits=1,
        operand_max=1,
        fixed_width=True,
        device="cpu",
        answer_format="sum",
    )
    sampled = overfit_script.make_range_batch(
        batch_size=4,
        num_digits=1,
        operand_max=1,
        rng=OrderedPairRng(),
        fixed_width=True,
        device="cpu",
        answer_format="sum",
    )

    assert torch.equal(exhaustive.x, sampled.x)
    assert torch.equal(exhaustive.y, sampled.y)
    assert torch.equal(exhaustive.loss_mask, sampled.loss_mask)


def test_full_enum_interface_loss_updates_input_projection_only() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_full_enum_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="action_loss_full_enum_interface",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    batch = overfit_script.make_range_batch(
        batch_size=3,
        num_digits=1,
        operand_max=2,
        rng=__import__("random").Random(2),
        fixed_width=True,
        device="cpu",
    )

    loss, metrics = overfit_script.action_loss_full_enum_interface_loss(
        model,
        batch,
        num_digits=1,
        temperature=1.0,
        min_probability_floor=0.0,
        chunk_size=4,
    )
    loss.backward()

    assert loss.item() > 0
    assert metrics["action_loss_full_enum_chunk_size"] == 4
    assert metrics["action_loss_full_enum_effective_pairs"] > 0
    assert model.calculator_hook is not None
    assert model.calculator_hook.input_proj.weight.grad is not None
    assert model.tok_emb.weight.grad is None


def test_joint_pair_head_traces_pair_action() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="action_loss_full_enum_joint_interface",
        calculator_action_head="joint_pair",
        calculator_read_position="operands",
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.calculator_hook.pair_proj is not None
    with torch.no_grad():
        model.calculator_hook.pair_proj.weight.zero_()
        model.calculator_hook.pair_proj.bias.fill_(-10.0)
        model.calculator_hook.pair_proj.bias[1 * 3 + 2] = 10.0

    x = torch.tensor([[0, 1, PLUS_ID, 0, 2, EQ_ID]])

    _, diagnostics = model(x, return_diagnostics=True)
    trace = diagnostics["calculator_trace"]

    assert trace["pair_pred"][0, 5].item() == 5
    assert trace["a_pred"][0, 5].item() == 1
    assert trace["b_pred"][0, 5].item() == 2
    assert trace["result_pred"][0, 5].item() == 3
    assert torch.isfinite(trace["pair_logp"][0, 5])


def test_joint_pair_operand_spans_projection_shape_and_relaxed_gradient() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="joint_pair",
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
        relaxed_calculator_temperature=2.0,
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.calculator_hook.pair_proj is not None
    assert model.calculator_hook.pair_proj.in_features == 2 * 2 * cfg.n_embd

    for name, param in model.named_parameters():
        if not name.startswith("calculator_hook.pair_proj."):
            param.requires_grad = False

    x = torch.tensor([[0, 3, PLUS_ID, 0, 7, EQ_ID]])
    logits, diagnostics = model(x, return_diagnostics=True)
    trace = diagnostics["calculator_trace"]
    assert trace["result_pred"][0, 5].item() == (
        trace["a_pred"][0, 5].item() + trace["b_pred"][0, 5].item()
    )

    loss = logits[:, -1].sum()
    loss.backward()

    assert model.calculator_hook.pair_proj.weight.grad is not None
    assert model.calculator_hook.pair_proj.weight.grad.abs().sum().item() > 0
    assert model.tok_emb.weight.grad is None
    assert model.calculator_hook.output_proj.weight.grad is None


def test_result_space_forward_traces_valid_canonical_pair_for_every_result() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
        relaxed_calculator_temperature=2.0,
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    x = torch.tensor([[0, 3, PLUS_ID, 0, 7, EQ_ID]])

    for result_class in range(cfg.calculator_result_vocab_size):
        with torch.no_grad():
            model.calculator_hook.result_proj.weight.zero_()
            model.calculator_hook.result_proj.bias.fill_(-10.0)
            model.calculator_hook.result_proj.bias[result_class] = 10.0

        _, diagnostics = model(x, return_diagnostics=True)
        trace = diagnostics["calculator_trace"]
        a_pred = trace["a_pred"][0, 5].item()
        b_pred = trace["b_pred"][0, 5].item()

        assert trace["result_pred"][0, 5].item() == result_class
        assert a_pred == min(result_class, cfg.calculator_operand_vocab_size - 1)
        assert b_pred == result_class - a_pred
        assert 0 <= a_pred < cfg.calculator_operand_vocab_size
        assert 0 <= b_pred < cfg.calculator_operand_vocab_size
        assert torch.isfinite(trace["result_confidence"][0, 5])
        assert torch.isfinite(trace["result_entropy"][0, 5])


def test_result_space_reinforce_trace_uses_sampled_result_logprob() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_estimator="reinforce",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    with torch.no_grad():
        model.calculator_hook.result_proj.weight.zero_()
        model.calculator_hook.result_proj.bias.zero_()

    x = torch.tensor([[0, 3, PLUS_ID, 0, 4, EQ_ID]])
    _, diagnostics = model(x, return_diagnostics=True)
    trace = diagnostics["calculator_trace"]

    result_pred = trace["result_pred"][0, 5].item()
    assert trace["a_pred"][0, 5].item() + trace["b_pred"][0, 5].item() == result_pred
    assert torch.isfinite(trace["result_logp"][0, 5])
    assert trace["sampled_logp"][0, 5].item() == pytest.approx(
        trace["result_logp"][0, 5].item()
    )
    assert trace["sampled_logp"][0, 5].item() != pytest.approx(
        (trace["a_logp"][0, 5] + trace["b_logp"][0, 5]).item()
    )


def test_result_space_operand_spans_answer_loss_updates_result_projection_only() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_result_space_grad", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
        relaxed_calculator_temperature=2.0,
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    assert model.calculator_hook.result_proj.in_features == 2 * 2 * cfg.n_embd

    x = torch.tensor([[0, 3, PLUS_ID, 0, 7, EQ_ID]])
    logits = model(x)
    loss = logits[:, -1].sum()
    loss.backward()

    assert model.calculator_hook.result_proj.weight.grad is not None
    assert model.calculator_hook.result_proj.weight.grad.abs().sum().item() > 0
    assert model.calculator_hook.input_proj.weight.grad is None
    assert model.calculator_hook.output_proj.weight.grad is None
    assert model.answer_decoder is not None
    assert model.answer_decoder.weight.grad is None
    assert model.tok_emb.weight.grad is None


def test_result_space_hidden_result_head_preserves_logit_contract() -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        calculator_result_head_hidden_size=16,
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert isinstance(model.calculator_hook.result_proj, torch.nn.Sequential)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    positions = model._calculator_read_positions(batch.x)
    residual = model.tok_emb(batch.x) + model.pos_emb(
        torch.arange(batch.x.shape[1])
    )
    residual = model.blocks[0](residual)
    logits = model.calculator_hook._result_space_logits(residual, positions)

    assert logits.shape == (2, 4, 7)


def test_result_boundary_target_uses_lowest_nll_result(monkeypatch) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_result_boundary", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    result_logits = torch.zeros((2, 7), requires_grad=True)
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )

    monkeypatch.setattr(
        overfit_script,
        "calculator_read_result_logits",
        lambda model_arg, batch_arg: (result_logits, None, None, None),
    )
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses[
            : batch_arg.x.shape[0]
        ],
    )

    loss, metrics = overfit_script.result_boundary_target_loss(
        model,
        batch,
        num_digits=1,
        target_mode="hard_best_result",
        temperature=0.25,
        min_probability_floor=0.0,
        chunk_size=4,
    )

    assert loss.item() == pytest.approx(
        torch.nn.functional.cross_entropy(
            result_logits, torch.tensor([3, 3])
        ).item()
    )
    assert metrics["result_boundary_target_hard_best_equals_true_sum"] == pytest.approx(
        1.0
    )
    assert metrics[
        "result_boundary_target_tie_aware_true_result_best_fraction"
    ] == pytest.approx(1.0)


def test_result_boundary_target_updates_result_projection_only(monkeypatch) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_result_boundary_grad", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses[
            : batch_arg.x.shape[0]
        ],
    )

    loss, _ = overfit_script.result_boundary_target_loss(
        model,
        batch,
        num_digits=1,
        target_mode="hard_best_result",
        temperature=0.25,
        min_probability_floor=0.0,
        chunk_size=4,
    )
    loss.backward()

    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    assert model.calculator_hook.result_proj.weight.grad is not None
    assert model.calculator_hook.result_proj.weight.grad.abs().sum().item() > 0
    assert model.calculator_hook.output_proj.weight.grad is None
    assert model.answer_decoder is not None
    assert model.answer_decoder.weight.grad is None
    assert model.tok_emb.weight.grad is None


def test_result_space_expected_answer_loss_uses_result_marginal(monkeypatch) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_result_expected_loss", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="full_enum_expected_answer_loss",
        calculator_action_head="result_space",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    result_logits = torch.zeros((2, 7), requires_grad=True)
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )

    monkeypatch.setattr(
        overfit_script,
        "calculator_read_result_logits",
        lambda model_arg, batch_arg: (result_logits, None, None, None),
    )
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses[
            : batch_arg.x.shape[0]
        ],
    )

    loss, metrics = overfit_script.full_enum_expected_answer_loss(
        model,
        batch,
        num_digits=1,
        policy_temperature=1.0,
        cost_normalization="none",
        entropy_weight=0.0,
        chunk_size=4,
    )

    assert loss.item() == pytest.approx(forced_losses.mean(dim=1).mean().item())
    assert metrics["expected_answer_loss_best_nll"] == pytest.approx(0.375)
    assert metrics["expected_answer_loss_true_nll"] == pytest.approx(0.375)
    assert metrics["expected_answer_loss_hard_learned_calc_accuracy"] == pytest.approx(
        0.0
    )
    loss.backward()
    assert result_logits.grad is not None
    assert result_logits.grad.abs().sum().item() > 0


def test_result_space_expected_answer_loss_updates_result_projection_only(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_result_expected_loss_grad", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="full_enum_expected_answer_loss",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )
    def fake_forced_losses(model_arg, batch_arg, *, chunk_size):
        return forced_losses[: batch_arg.x.shape[0]]

    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        fake_forced_losses,
    )

    loss, _ = overfit_script.full_enum_expected_answer_loss(
        model,
        batch,
        num_digits=1,
        policy_temperature=1.0,
        cost_normalization="none",
        entropy_weight=0.0,
        chunk_size=4,
    )
    loss.backward()

    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    assert model.calculator_hook.result_proj.weight.grad is not None
    assert model.calculator_hook.result_proj.weight.grad.abs().sum().item() > 0
    assert model.calculator_hook.output_proj.weight.grad is None
    assert model.answer_decoder is not None
    assert model.answer_decoder.weight.grad is None
    assert model.tok_emb.weight.grad is None


def test_boundary_feedback_updates_result_projection_and_open_upstream() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_boundary_feedback_grad", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="direct_feedback_alignment",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        ),
    )

    loss, metrics = overfit_script.boundary_feedback_alignment_loss(
        model,
        batch,
        num_digits=1,
        feedback_mode="output_proj_transpose",
        feedback_seed=0,
    )
    loss.backward()

    assert metrics["boundary_feedback_signal_l2"] > 0.0
    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    assert model.calculator_hook.result_proj.weight.grad is not None
    assert model.calculator_hook.result_proj.weight.grad.abs().sum().item() > 0
    assert model.answer_decoder is not None
    assert model.answer_decoder.weight.grad is None
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum().item() > 0


def test_boundary_feedback_gradient_diagnostic_reports_alignment(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_boundary_feedback_diag", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="direct_feedback_alignment",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        ),
    )
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses,
    )

    summary = overfit_script.run_boundary_feedback_gradient_diagnostic(
        model,
        batch,
        num_digits=1,
        feedback_mode="output_proj_transpose",
        feedback_seed=0,
        result_boundary_target_mode="hard_best_result",
        result_boundary_target_temperature=1.0,
        result_boundary_target_min_probability_floor=0.0,
        result_boundary_target_chunk_size=4,
    )

    assert summary["feedback_result_proj_grad_l2"] > 0.0
    assert summary["feedback_upstream_grad_l2"] > 0.0
    assert summary["feedback_semantic_decoder_grad_l2"] == pytest.approx(0.0)
    assert "feedback_vs_boundary_result_proj_cosine" in summary
    assert "feedback_vs_boundary_upstream_cosine" in summary


def test_linear_shadow_feedback_diagnostic_matches_boundary_gradient(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_shadow_feedback_diag", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="direct_feedback_alignment",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        ),
    )
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )
    def fake_shadow_forced_losses(model_arg, batch_arg, *, chunk_size):
        return forced_losses[: batch_arg.x.shape[0]]

    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        fake_shadow_forced_losses,
    )

    summary = overfit_script.run_shadow_feedback_gradient_diagnostic(
        model,
        batch,
        num_digits=1,
        ridge=1e-3,
        heldout_fraction=0.0,
        result_boundary_target_mode="hard_best_result",
        result_boundary_target_temperature=1.0,
        result_boundary_target_min_probability_floor=0.0,
        result_boundary_target_chunk_size=4,
    )

    assert summary["shadow_feedback_fit_cosine"] > 0.99
    assert summary["shadow_result_proj_grad_l2"] > 0.0
    assert summary["shadow_upstream_grad_l2"] > 0.0
    assert summary["shadow_semantic_decoder_grad_l2"] == pytest.approx(0.0)
    assert summary["shadow_vs_boundary_result_proj_cosine"] > 0.99

    heldout_summary = overfit_script.run_shadow_feedback_gradient_diagnostic(
        model,
        batch,
        num_digits=1,
        ridge=1e-3,
        heldout_fraction=0.5,
        result_boundary_target_mode="hard_best_result",
        result_boundary_target_temperature=1.0,
        result_boundary_target_min_probability_floor=0.0,
        result_boundary_target_chunk_size=4,
    )
    assert heldout_summary["shadow_feedback_fit_batch_size"] == 1
    assert heldout_summary["shadow_feedback_heldout_batch_size"] == 1
    assert "heldout_shadow_vs_boundary_result_proj_cosine" in heldout_summary


def test_online_shadow_feedback_diagnostic_uses_heldout_model_gradients(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_online_shadow_feedback_diag", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="direct_feedback_alignment",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    batch = ArithmeticBatch(
        x=torch.tensor(
            [
                [1, PLUS_ID, 2, EQ_ID],
                [0, PLUS_ID, 3, EQ_ID],
                [2, PLUS_ID, 1, EQ_ID],
                [1, PLUS_ID, 3, EQ_ID],
            ]
        ),
        y=torch.zeros((4, 4), dtype=torch.long),
        loss_mask=torch.tensor(
            [
                [False, False, False, True],
                [False, False, False, True],
                [False, False, False, True],
                [False, False, False, True],
            ]
        ),
    )

    def fake_online_forced_losses(model_arg, batch_arg, *, chunk_size):
        del model_arg, chunk_size
        result_ids = torch.arange(7, dtype=torch.float).unsqueeze(0)
        target = (batch_arg.x[:, 0] + batch_arg.x[:, 2]).float().unsqueeze(1)
        return (result_ids - target).abs()

    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        fake_online_forced_losses,
    )
    before_params = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    summary = overfit_script.run_online_shadow_feedback_gradient_diagnostic(
        model,
        batch,
        num_digits=1,
        heldout_fraction=0.5,
        hidden_size=8,
        learning_rate=1e-3,
        warmup_steps=3,
        updates_per_step=1,
        validation_fraction=0.25,
        validation_every=1,
        result_boundary_target_mode="hard_best_result",
        result_boundary_target_temperature=1.0,
        result_boundary_target_min_probability_floor=0.0,
        result_boundary_target_chunk_size=4,
    )

    assert summary["diagnostic"] == (
        "online_mlp_shadow_feedback_heldout_gradient_agreement"
    )
    assert summary["shadow_feedback_fit_batch_size"] == 1
    assert summary["shadow_feedback_heldout_batch_size"] == 2
    assert summary["shadow_feedback_validation_batch_size"] == 1
    assert summary["shadow_feedback_best_state_restored"] is True
    assert summary["shadow_feedback_best_step"] >= 0
    assert summary["shadow_feedback_validation_history"]
    assert summary["heldout_shadow_result_proj_grad_l2"] > 0.0
    assert summary["heldout_shadow_upstream_grad_l2"] > 0.0
    assert summary["heldout_shadow_semantic_decoder_grad_l2"] == pytest.approx(0.0)
    assert "validation_shadow_vs_boundary_result_proj_cosine" in summary
    assert "heldout_shadow_vs_boundary_result_proj_cosine" in summary
    assert "heldout_shadow_vs_boundary_upstream_cosine" in summary
    assert "shadow_feedback_train_heldout_result_proj_cosine_gap" in summary
    assert "shadow_feedback_validation_test_result_proj_cosine_gap" in summary
    for name, param in model.named_parameters():
        assert torch.allclose(param, before_params[name])


def test_exhaustive_grid_boundary_target_smoke_updates_open_upstream(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_exhaustive_boundary_smoke", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    batch = overfit_script.make_exhaustive_range_batch(
        num_digits=1,
        operand_max=3,
        fixed_width=True,
        device="cpu",
    )
    true_a, true_b = overfit_script.fixed_width_operands_from_batch(
        batch.x, num_digits=1
    )
    forced_losses = torch.full((batch.x.shape[0], 7), 10.0)
    forced_losses.scatter_(1, (true_a + true_b).unsqueeze(-1), 0.0)
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses,
    )
    semantic_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if (
            name.startswith("calculator_hook.output_proj.")
            or name.startswith("calculator_hook.input_proj.")
            or name.startswith("answer_decoder.")
            or name.startswith("answer_offset_emb.")
        )
    }

    loss, metrics = overfit_script.result_boundary_target_loss(
        model,
        batch,
        num_digits=1,
        target_mode="hard_best_result",
        temperature=0.25,
        min_probability_floor=0.0,
        chunk_size=4,
    )
    optim = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad], lr=0.01
    )
    optim.zero_grad(set_to_none=True)
    loss.backward()

    assert metrics["result_boundary_target_hard_best_equals_true_sum"] == pytest.approx(
        1.0
    )
    assert model.calculator_hook is not None
    assert model.calculator_hook.result_proj is not None
    assert model.calculator_hook.result_proj.weight.grad is not None
    assert model.calculator_hook.result_proj.weight.grad.norm().item() > 0
    upstream_grad_l2 = sum(
        param.grad.detach().norm().item()
        for name, param in model.named_parameters()
        if not name.startswith("calculator_hook.") and param.grad is not None
    )
    assert upstream_grad_l2 > 0
    optim.step()

    for name, before in semantic_before.items():
        after = dict(model.named_parameters())[name].detach()
        assert torch.equal(after, before)


def test_result_boundary_hard_best_ce_matches_true_sum_only_after_parity(
    monkeypatch,
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_result_boundary_parity", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = ArithmeticBatch(
        x=torch.tensor([[1, PLUS_ID, 2, EQ_ID], [0, PLUS_ID, 3, EQ_ID]]),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    result_logits = torch.randn((2, 7), requires_grad=True)
    forced_losses = torch.tensor(
        [
            [5.0, 4.0, 3.0, 0.5, 3.5, 4.5, 5.5],
            [7.0, 6.0, 5.0, 0.25, 4.0, 3.0, 2.0],
        ]
    )
    monkeypatch.setattr(
        overfit_script,
        "calculator_read_result_logits",
        lambda model_arg, batch_arg: (result_logits, None, None, None),
    )
    monkeypatch.setattr(
        overfit_script,
        "score_forced_result_classes_chunked",
        lambda model_arg, batch_arg, *, chunk_size: forced_losses,
    )

    boundary_loss, metrics = overfit_script.result_boundary_target_loss(
        model,
        batch,
        num_digits=1,
        target_mode="hard_best_result",
        temperature=0.25,
        min_probability_floor=0.0,
        chunk_size=4,
    )
    true_a, true_b = overfit_script.fixed_width_operands_from_batch(
        batch.x, num_digits=1
    )
    direct_true_ce = torch.nn.functional.cross_entropy(result_logits, true_a + true_b)

    assert metrics["result_boundary_target_hard_best_equals_true_sum"] == pytest.approx(
        1.0
    )
    assert boundary_loss.item() == pytest.approx(direct_true_ce.item())


def test_result_feature_probe_extracts_exact_result_projection_shape() -> None:
    script_path = Path("scripts/run_phase7_result_feature_separability.py")
    spec = importlib.util.spec_from_file_location(
        "phase7_result_feature_probe_shape", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    probe_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=16,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=20,
        calculator_result_vocab_size=39,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        answer_decoder_interaction="product",
    )
    model = TinyGPT(cfg)
    batch = probe_script.exhaustive_natural_batch(
        operand_max=1,
        num_digits=2,
        fixed_width=True,
        answer_format="sum",
        device="cpu",
    )

    features = probe_script.collect_probe_features(model, batch)

    assert features["exact_result_proj_input"].shape == (4, 2 * 2 * cfg.n_embd)
    assert features["operand_a_span"].shape == (4, 2 * cfg.n_embd)
    assert features["operand_b_span"].shape == (4, 2 * cfg.n_embd)


def test_result_feature_probe_target_parity_is_checked_after_target_construction() -> None:
    script_path = Path("scripts/run_phase7_result_feature_separability.py")
    spec = importlib.util.spec_from_file_location(
        "phase7_result_feature_probe_parity", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    probe_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_script)

    batch = probe_script.exhaustive_natural_batch(
        operand_max=3,
        num_digits=1,
        fixed_width=True,
        answer_format="sum",
        device="cpu",
    )
    true_a, true_b = probe_script.fixed_width_operands_from_batch(batch.x, num_digits=1)
    true_sum = true_a + true_b
    forced_losses = torch.full((batch.x.shape[0], 7), 10.0)
    forced_losses.scatter_(1, true_sum.unsqueeze(-1), 0.0)
    target = forced_losses.argmin(dim=-1)

    assert torch.equal(target, true_sum)


def test_result_feature_linear_probe_overfits_linearly_separable_fixture() -> None:
    script_path = Path("scripts/run_phase7_result_feature_separability.py")
    spec = importlib.util.spec_from_file_location(
        "phase7_result_feature_probe_linear", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    probe_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_script)

    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    features = torch.nn.functional.one_hot(labels, num_classes=3).float()
    indices = torch.arange(labels.numel(), dtype=torch.long)

    result, _ = probe_script.train_probe(
        features,
        labels,
        train_indices=indices,
        eval_indices=indices,
        head_kind="linear",
        hidden_size=0,
        seed=2,
        steps=200,
        lr=0.1,
        weight_decay=0.0,
        device="cpu",
    )

    assert result.eval_accuracy == pytest.approx(1.0)


def test_result_feature_probe_cli_validation_rejects_bad_values() -> None:
    script_path = Path("scripts/run_phase7_result_feature_separability.py")
    spec = importlib.util.spec_from_file_location(
        "phase7_result_feature_probe_validate", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    probe_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_script)

    base = dict(
        checkpoint=None,
        probe_heads="linear,mlp",
        mlp_hidden_sizes="64,128",
        linear_steps=10,
        mlp_steps=10,
        folds=5,
        result_boundary_target_chunk_size=64,
    )
    for override, message in [
        ({"probe_heads": "linear,wide"}, "unsupported probe head"),
        ({"mlp_hidden_sizes": "64,0"}, "positive"),
        ({"linear_steps": 0}, "positive"),
        ({"mlp_steps": 0}, "positive"),
        ({"folds": 1}, "at least 2"),
    ]:
        kwargs = dict(base)
        kwargs.update(override)
        with pytest.raises(ValueError, match=message):
            probe_script.validate_args(argparse.Namespace(**kwargs))


def test_result_boundary_cli_validation(monkeypatch, tmp_path) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location(
        "overfit_result_boundary_cli", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    with pytest.raises(SystemExit):
        monkeypatch.setattr(
            sys,
            "argv",
            [str(script_path), "--result-boundary-target-mode", "bad_mode"],
        )
        overfit_script.parse_args()

    validation_cases = [
        ["--result-boundary-target-loss-weight", "-0.1"],
        ["--result-boundary-target-temperature", "0"],
        ["--result-boundary-target-chunk-size", "0"],
        ["--exhaustive-grid-batch"],
    ]
    for extra in validation_cases:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(script_path),
                "--steps",
                "0",
                "--run-root",
                str(tmp_path),
                *extra,
            ],
        )
        with pytest.raises(ValueError):
            overfit_script.main()


def test_joint_full_enum_interface_loss_updates_pair_projection_only() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_joint_full_enum_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="action_loss_full_enum_joint_interface",
        calculator_action_head="joint_pair",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    batch = overfit_script.make_range_batch(
        batch_size=3,
        num_digits=1,
        operand_max=2,
        rng=__import__("random").Random(2),
        fixed_width=True,
        device="cpu",
    )

    loss, metrics = overfit_script.action_loss_full_enum_joint_interface_loss(
        model,
        batch,
        num_digits=1,
        temperature=1.0,
        min_probability_floor=0.0,
        chunk_size=4,
    )
    loss.backward()

    assert loss.item() > 0
    assert metrics["action_loss_full_enum_joint_target_loss"] == pytest.approx(
        loss.item()
    )
    assert model.calculator_hook is not None
    assert model.calculator_hook.pair_proj is not None
    assert model.calculator_hook.pair_proj.weight.grad is not None
    assert model.calculator_hook.input_proj.weight.grad is None
    assert model.tok_emb.weight.grad is None


def test_joint_pair_relaxed_metrics_report_soft_result_hardening(monkeypatch) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_relaxed_metrics", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="joint_pair",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = ArithmeticBatch(
        x=torch.tensor(
            [
                [1, PLUS_ID, 2, EQ_ID],
                [0, PLUS_ID, 3, EQ_ID],
            ]
        ),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    pair_logits = torch.full((2, 16), -10.0)
    pair_logits[0, 1 * 4 + 2] = 10.0
    pair_logits[1, 0 * 4 + 0] = 10.0
    pair_logits[1, 0 * 4 + 3] = 9.0

    def fake_pair_logits(model_arg, batch_arg):
        assert model_arg is model
        assert batch_arg is batch
        return pair_logits, None, None, None

    monkeypatch.setattr(overfit_script, "calculator_read_pair_logits", fake_pair_logits)

    _, metrics = overfit_script.relaxed_calculator_policy_metrics(
        model,
        batch,
        num_digits=1,
        temperature=1.0,
        entropy_weight=0.0,
    )

    assert metrics["relaxed_calculator_argmax_result_accuracy"] == pytest.approx(0.5)
    assert metrics["relaxed_calculator_top3_result_accuracy"] == pytest.approx(1.0)
    assert metrics["relaxed_calculator_hard_learned_calc_accuracy"] == pytest.approx(0.5)
    assert 0.5 < metrics["relaxed_calculator_true_result_probability"] < 1.0
    assert metrics["relaxed_calculator_result_entropy"] > 0.0
    assert metrics["relaxed_calculator_effective_results"] > 1.0


def test_result_space_relaxed_metrics_and_cli_validation(monkeypatch) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_result_space_metrics", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=4,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=4,
        calculator_result_vocab_size=7,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = ArithmeticBatch(
        x=torch.tensor(
            [
                [1, PLUS_ID, 2, EQ_ID],
                [0, PLUS_ID, 3, EQ_ID],
            ]
        ),
        y=torch.zeros((2, 4), dtype=torch.long),
        loss_mask=torch.zeros((2, 4), dtype=torch.bool),
    )
    result_logits = torch.full((2, 7), -10.0)
    result_logits[0, 3] = 10.0
    result_logits[1, 0] = 10.0
    result_logits[1, 3] = 9.0

    def fake_result_logits(model_arg, batch_arg):
        assert model_arg is model
        assert batch_arg is batch
        return result_logits, None, None, None

    monkeypatch.setattr(
        overfit_script, "calculator_read_result_logits", fake_result_logits
    )

    _, metrics = overfit_script.relaxed_calculator_policy_metrics(
        model,
        batch,
        num_digits=1,
        temperature=1.0,
        entropy_weight=0.0,
    )

    assert metrics["relaxed_calculator_argmax_result_accuracy"] == pytest.approx(0.5)
    assert metrics["relaxed_calculator_top3_result_accuracy"] == pytest.approx(1.0)
    assert metrics["relaxed_calculator_hard_learned_calc_accuracy"] == pytest.approx(0.5)
    assert 0.5 < metrics["relaxed_calculator_true_result_probability"] < 1.0
    assert metrics["relaxed_calculator_result_entropy"] > 0.0
    assert metrics["relaxed_calculator_effective_results"] > 1.0

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--calculator-estimator",
            "gumbel_concrete_interface",
            "--calculator-action-head",
            "result_space",
        ],
    )
    parsed = overfit_script.parse_args()
    assert parsed.calculator_action_head == "result_space"
    assert parsed.calculator_estimator == "gumbel_concrete_interface"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--calculator-estimator",
            "direct_feedback_alignment",
            "--calculator-action-head",
            "result_space",
            "--boundary-feedback-weight",
            "1.0",
        ],
    )
    parsed = overfit_script.parse_args()
    assert parsed.calculator_estimator == "direct_feedback_alignment"
    assert parsed.boundary_feedback_mode == "output_proj_transpose"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--calculator-estimator",
            "direct_feedback_alignment",
            "--calculator-action-head",
            "result_space",
            "--shadow-feedback-gradient-diagnostic-only",
            "--shadow-feedback-weight",
            "0.5",
            "--shadow-feedback-heldout-fraction",
            "0.2",
            "--shadow-feedback-hidden-size",
            "32",
            "--shadow-feedback-online-lr",
            "0.002",
            "--shadow-feedback-warmup-steps",
            "7",
            "--shadow-feedback-updates-per-step",
            "2",
            "--shadow-feedback-validation-fraction",
            "0.1",
            "--shadow-feedback-validation-every",
            "5",
        ],
    )
    parsed = overfit_script.parse_args()
    assert parsed.shadow_feedback_gradient_diagnostic_only
    assert parsed.shadow_feedback_mode == "fit_once_linear"
    assert parsed.shadow_feedback_ridge == pytest.approx(1e-3)
    assert parsed.shadow_feedback_weight == pytest.approx(0.5)
    assert parsed.shadow_feedback_heldout_fraction == pytest.approx(0.2)
    assert parsed.shadow_feedback_hidden_size == 32
    assert parsed.shadow_feedback_online_lr == pytest.approx(0.002)
    assert parsed.shadow_feedback_warmup_steps == 7
    assert parsed.shadow_feedback_updates_per_step == 2
    assert parsed.shadow_feedback_validation_fraction == pytest.approx(0.1)
    assert parsed.shadow_feedback_validation_every == 5

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--calculator-estimator",
            "direct_feedback_alignment",
            "--calculator-action-head",
            "result_space",
            "--shadow-feedback-gradient-diagnostic-only",
            "--shadow-feedback-mode",
            "online_mlp",
            "--shadow-feedback-heldout-fraction",
            "0.5",
        ],
    )
    parsed = overfit_script.parse_args()
    assert parsed.shadow_feedback_mode == "online_mlp"
    assert parsed.shadow_feedback_hidden_size == 64
    assert parsed.shadow_feedback_warmup_steps == 100
    assert parsed.shadow_feedback_validation_fraction == pytest.approx(0.0)
    assert parsed.shadow_feedback_validation_every == 0

    with pytest.raises(ValueError, match="result_space.*gumbel_concrete_interface"):
        CalculatorHook(
            GPTConfig(
                calculator_enabled=True,
                calculator_mode="add",
                calculator_estimator="ste",
                calculator_action_head="result_space",
            )
        )
    with pytest.raises(ValueError, match="result_space.*calculator_output_format"):
        CalculatorHook(
            GPTConfig(
                calculator_enabled=True,
                calculator_mode="add",
                calculator_estimator="gumbel_concrete_interface",
                calculator_action_head="result_space",
                calculator_output_format="sum_left_operand",
            )
        )


def test_joint_auxiliary_operand_loss_updates_pair_projection_only() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_joint_aux_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="action_loss_full_enum_joint_interface",
        calculator_action_head="joint_pair",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    overfit_script.freeze_semantic_decoder_parameters(model)
    overfit_script.freeze_upstream_encoder_parameters(model)
    batch = overfit_script.make_range_batch(
        batch_size=4,
        num_digits=1,
        operand_max=2,
        rng=__import__("random").Random(3),
        fixed_width=True,
        device="cpu",
    )

    loss = overfit_script.auxiliary_operand_loss(
        model,
        batch,
        num_digits=1,
        grad_upstream=False,
    )
    loss.backward()

    assert loss.item() > 0
    assert model.calculator_hook is not None
    assert model.calculator_hook.pair_proj is not None
    assert model.calculator_hook.pair_proj.weight.grad is not None
    assert model.calculator_hook.input_proj.weight.grad is None
    assert model.tok_emb.weight.grad is None


def test_training_oracle_operand_extraction_from_fixed_width_batch() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    x = torch.tensor(
        [
            [0, 7, 10, 0, 5, EQ_ID, 1, 2],
            [4, 2, 10, 9, 9, EQ_ID, 1, 4],
        ]
    )

    oracle = overfit_script.make_oracle_operands_from_batch(x, num_digits=2)

    assert oracle.shape == (2, 8, 2)
    assert oracle[0, 0].tolist() == [7, 5]
    assert oracle[0, -1].tolist() == [7, 5]
    assert oracle[1, 0].tolist() == [42, 99]
    assert oracle[1, -1].tolist() == [42, 99]


def test_training_script_builds_legal_one_layer_calculator_config() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_one_layer", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = overfit_script.make_model_config(
        1,
        "model-c",
        n_layer=1,
        n_head=2,
        n_embd=32,
        mlp_expansion=1,
    )

    assert cfg.n_layer == 1
    assert cfg.n_head == 2
    assert cfg.n_embd == 32
    assert cfg.mlp_expansion == 1
    assert cfg.calculator_hook_after_layer == 1
    assert TinyGPT(cfg).num_params() < TinyGPT(GPTConfig()).num_params()


def test_training_aux_operand_weight_respects_floor() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_aux", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    assert overfit_script.auxiliary_operand_weight(
        initial_weight=0.1, decay_steps=1000, floor=0.01, step=0
    ) == pytest.approx(0.1)
    assert overfit_script.auxiliary_operand_weight(
        initial_weight=0.1, decay_steps=1000, floor=0.01, step=500
    ) == pytest.approx(0.05)
    assert overfit_script.auxiliary_operand_weight(
        initial_weight=0.1, decay_steps=1000, floor=0.01, step=1000
    ) == pytest.approx(0.01)
    assert overfit_script.auxiliary_operand_weight(
        initial_weight=0.1, decay_steps=0, floor=0.01, step=1000
    ) == pytest.approx(0.1)


def test_training_adaptive_interface_weight_schedule_respects_floor() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_iface", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    assert overfit_script.adaptive_interface_weight(
        initial_weight=1.0, decay_steps=150, floor=0.0, step=0
    ) == pytest.approx(1.0)
    assert overfit_script.adaptive_interface_weight(
        initial_weight=1.0, decay_steps=150, floor=0.0, step=75
    ) == pytest.approx(0.5)
    assert overfit_script.adaptive_interface_weight(
        initial_weight=1.0, decay_steps=150, floor=0.0, step=150
    ) == pytest.approx(0.0)
    assert overfit_script.adaptive_interface_weight(
        initial_weight=1.0, decay_steps=150, floor=0.25, step=150
    ) == pytest.approx(0.25)
    assert overfit_script.adaptive_interface_weight(
        initial_weight=1.0, decay_steps=0, floor=0.0, step=150
    ) == pytest.approx(1.0)
    assert overfit_script.adaptive_interface_weight(
        initial_weight=0.0, decay_steps=150, floor=0.25, step=150
    ) == pytest.approx(0.0)


def test_adaptive_interface_selects_high_probability_operand_pair() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_select", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    a_logits = torch.tensor([[0.0, 4.0, 1.0], [5.0, 0.0, 3.0]])
    b_logits = torch.tensor([[3.0, 0.0, 2.0], [0.0, 1.0, 4.0]])
    result_targets = torch.tensor([2, 2])

    a_target, b_target = overfit_script.select_adaptive_operand_targets(
        a_logits, b_logits, result_targets
    )

    assert a_target.tolist() == [1, 0]
    assert b_target.tolist() == [1, 2]


def test_adaptive_soft_result_loss_rewards_total_valid_pair_mass() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_soft", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    result_targets = torch.tensor([2])
    high_valid_a = torch.tensor([[10.0, 10.0, -10.0]])
    high_valid_b = torch.tensor([[-10.0, 10.0, 10.0]])
    high_invalid_a = torch.tensor([[10.0, -10.0, -10.0]])
    high_invalid_b = torch.tensor([[10.0, -10.0, -10.0]])

    valid_loss, valid_mass = overfit_script.adaptive_soft_result_loss(
        high_valid_a, high_valid_b, result_targets
    )
    invalid_loss, invalid_mass = overfit_script.adaptive_soft_result_loss(
        high_invalid_a, high_invalid_b, result_targets
    )

    assert valid_mass.item() > invalid_mass.item()
    assert valid_loss.item() < invalid_loss.item()


def test_adaptive_interface_loss_updates_input_interface_and_upstream() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = overfit_script.make_range_batch(
        batch_size=4,
        num_digits=1,
        operand_max=2,
        rng=__import__("random").Random(0),
        fixed_width=True,
        device="cpu",
    )

    assert model.calculator_hook is not None
    before = model.calculator_hook.input_proj.weight.detach().clone()
    loss, metrics = overfit_script.adaptive_interface_loss(
        model, batch, num_digits=1, target_mode="hard_pair"
    )
    loss.backward()

    assert loss.item() > 0
    assert "adaptive_target_result_accuracy" in metrics
    assert model.calculator_hook.input_proj.weight.grad is not None
    assert model.tok_emb.weight.grad is not None

    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    optim.step()

    assert not torch.equal(before, model.calculator_hook.input_proj.weight)


def test_adaptive_interface_entropy_term_produces_finite_input_gradients() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_entropy", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    torch.manual_seed(0)
    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_read_position="operands",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    batch = overfit_script.make_range_batch(
        batch_size=4,
        num_digits=1,
        operand_max=2,
        rng=__import__("random").Random(1),
        fixed_width=True,
        device="cpu",
    )

    assert model.calculator_hook is not None
    loss, metrics = overfit_script.adaptive_interface_loss(
        model,
        batch,
        num_digits=1,
        target_mode="soft_result",
        entropy_weight=0.01,
    )
    loss.backward()

    grad = model.calculator_hook.input_proj.weight.grad
    assert metrics["adaptive_interface_entropy"] > 0
    assert grad is not None
    assert torch.isfinite(grad).all()


def test_adaptive_optimizer_groups_assign_lrs_and_exclude_frozen_decoder() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_groups", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.answer_decoder is not None
    overfit_script.freeze_semantic_decoder_parameters(model)

    groups = overfit_script.adaptive_optimizer_param_groups(
        model,
        lr=3e-3,
        input_proj_lr=3e-4,
        upstream_lr=1e-4,
        weight_decay=0.0,
    )
    group_by_name = {group["name"]: group for group in groups}
    grouped_params = {
        id(param)
        for group in groups
        for param in group["params"]
    }

    assert group_by_name["calculator_hook.input_proj"]["lr"] == pytest.approx(3e-4)
    assert group_by_name["upstream"]["lr"] == pytest.approx(1e-4)
    assert id(model.calculator_hook.input_proj.weight) in grouped_params
    assert id(model.answer_decoder.weight) not in grouped_params
    assert id(model.calculator_hook.output_proj.weight) not in grouped_params


def test_freeze_semantic_decoder_preserves_decoder_but_not_interface() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_adaptive_freeze", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    assert model.answer_decoder is not None

    overfit_script.freeze_semantic_decoder_parameters(model)

    assert model.calculator_hook.input_proj.weight.requires_grad
    assert not model.calculator_hook.output_proj.weight.requires_grad
    assert not model.answer_decoder.weight.requires_grad


def test_semantic_decoder_checkpoint_load_scope_is_opt_in(tmp_path: Path) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_load_scope", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_bottleneck_mode="answer_decoder",
    )
    source = TinyGPT(cfg)
    target = TinyGPT(cfg)
    original_target = {
        name: tensor.detach().clone()
        for name, tensor in target.state_dict().items()
    }
    checkpoint_state = {
        name: torch.full_like(tensor, fill_value=(idx + 1) / 100.0)
        for idx, (name, tensor) in enumerate(source.state_dict().items())
    }
    checkpoint_path = tmp_path / "semantic_seed.pt"
    torch.save({"model_state_dict": checkpoint_state}, checkpoint_path)

    overfit_script.load_semantic_decoder_checkpoint(
        target, checkpoint_path, load_scope="semantic_decoder_only"
    )
    loaded_state = target.state_dict()
    assert torch.equal(
        loaded_state["answer_decoder.weight"],
        checkpoint_state["answer_decoder.weight"],
    )
    assert torch.equal(
        loaded_state["calculator_hook.output_proj.weight"],
        checkpoint_state["calculator_hook.output_proj.weight"],
    )
    assert torch.equal(
        loaded_state["calculator_hook.input_proj.weight"],
        original_target["calculator_hook.input_proj.weight"],
    )
    assert torch.equal(loaded_state["tok_emb.weight"], original_target["tok_emb.weight"])

    full_target = TinyGPT(cfg)
    overfit_script.load_semantic_decoder_checkpoint(
        full_target, checkpoint_path, load_scope="full_model"
    )
    full_state = full_target.state_dict()
    assert torch.equal(
        full_state["calculator_hook.input_proj.weight"],
        checkpoint_state["calculator_hook.input_proj.weight"],
    )
    assert torch.equal(full_state["tok_emb.weight"], checkpoint_state["tok_emb.weight"])


def test_strict_phase6_runner_threads_semantic_decoder_only_scope(tmp_path: Path) -> None:
    script_path = Path("scripts/run_phase6_strict_random_upstream_local_target.py")
    spec = importlib.util.spec_from_file_location("strict_phase6_runner", script_path)
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    command = runner.phase6_train_command(
        checkpoint=tmp_path / "seed.pt",
        run_root=tmp_path / "runs",
        estimator="identifiable_full_enum_local_target",
        answer_loss_weight=0.0,
        local_target_loss_weight=1.0,
        input_proj_lr=0.03,
        upstream_lr=0.003,
        steps=300,
        snapshot_every=25,
        checkpoint_every=25,
        target_mode="hard_best_pair",
        freeze_upstream=True,
        seed=0,
        load_scope="semantic_decoder_only",
    )

    scope_flag = command.index("--semantic-decoder-checkpoint-load-scope")
    assert command[scope_flag + 1] == "semantic_decoder_only"


def test_phase6_decay_runner_threads_scope_and_decay_flags(tmp_path: Path) -> None:
    script_path = Path("scripts/run_phase6_strict_local_target_decay_boundary.py")
    spec = importlib.util.spec_from_file_location("phase6_decay_runner", script_path)
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    command = runner.phase6_train_command(
        checkpoint=tmp_path / "seed.pt",
        run_root=tmp_path / "runs",
        estimator="identifiable_full_enum_local_target",
        answer_loss_weight=1.0,
        local_target_loss_weight=1.0,
        local_target_decay_steps=75,
        local_target_floor=0.0,
        input_proj_lr=0.03,
        upstream_lr=0.003,
        steps=300,
        snapshot_every=25,
        checkpoint_every=25,
        target_mode="hard_best_pair",
        freeze_upstream=True,
        seed=0,
        load_scope="semantic_decoder_only",
    )

    scope_flag = command.index("--semantic-decoder-checkpoint-load-scope")
    decay_flag = command.index("--adaptive-interface-loss-decay-steps")
    floor_flag = command.index("--adaptive-interface-loss-floor")
    assert command[scope_flag + 1] == "semantic_decoder_only"
    assert command[decay_flag + 1] == "75"
    assert command[floor_flag + 1] == "0.0"


def test_input_proj_anchor_loss_and_decay() -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_anchor", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)

    cfg = GPTConfig(
        n_embd=8,
        n_layer=1,
        n_head=1,
        block_size=6,
        mlp_expansion=1,
        calculator_enabled=True,
        calculator_mode="add",
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=3,
        calculator_result_vocab_size=5,
        calculator_estimator="adaptive_interface",
        calculator_bottleneck_mode="answer_decoder",
    )
    model = TinyGPT(cfg)
    assert model.calculator_hook is not None
    anchor = {
        "weight": model.calculator_hook.input_proj.weight.detach().clone(),
        "bias": model.calculator_hook.input_proj.bias.detach().clone(),
    }

    assert overfit_script.input_proj_anchor_weight(
        initial_weight=0.01, decay_steps=100, step=50
    ) == pytest.approx(0.005)
    assert overfit_script.input_proj_anchor_loss(model, anchor).item() == pytest.approx(0.0)

    with torch.no_grad():
        model.calculator_hook.input_proj.bias.add_(1.0)

    loss = overfit_script.input_proj_anchor_loss(model, anchor)
    delta = overfit_script.input_proj_anchor_delta_summary(model, anchor)

    assert loss.item() > 0.0
    assert delta["bias_l2"] > 0.0
    assert delta["weight_l2"] == pytest.approx(0.0)


def test_training_cli_supports_oracle_warmup_and_snapshots(
    tmp_path, monkeypatch
) -> None:
    script_path = Path("scripts/overfit_one_batch.py")
    spec = importlib.util.spec_from_file_location("overfit_script_cli", script_path)
    assert spec is not None
    assert spec.loader is not None
    overfit_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overfit_script)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script_path),
            "--variant",
            "model-c",
            "--digits",
            "1",
            "--steps",
            "1",
            "--batch-size",
            "4",
            "--eval-samples",
            "4",
            "--operand-max",
            "2",
            "--calculator-operand-vocab-size",
            "3",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "8",
            "--mlp-expansion",
            "1",
            "--calculator-hook-after-layer",
            "1",
            "--calculator-read-position",
            "operands",
            "--calculator-injection-mode",
            "replace",
            "--calculator-bottleneck-mode",
            "answer_decoder",
            "--calculator-output-format",
            "sum_left_operand",
            "--calculator-estimator",
            "adaptive_interface",
            "--semantic-decoder-checkpoint",
            str(tmp_path / "seed.pt"),
            "--input-proj-anchor-checkpoint",
            str(tmp_path / "seed.pt"),
            "--input-proj-anchor-weight",
            "0.01",
            "--input-proj-anchor-decay-steps",
            "1",
            "--input-proj-lr",
            "0.0003",
            "--upstream-lr",
            "0.0001",
            "--adaptive-interface-target-mode",
            "soft_result",
            "--adaptive-interface-entropy-weight",
            "0.003",
            "--adaptive-interface-loss-decay-steps",
            "1",
            "--oracle-warmup-steps",
            "1",
            "--aux-operand-loss-weight",
            "0.1",
            "--aux-operand-loss-decay-steps",
            "1",
            "--aux-operand-loss-floor",
            "0.01",
            "--snapshot-every",
            "1",
            "--snapshot-samples",
            "2",
            "--log-every",
            "1",
            "--run-root",
            str(tmp_path),
        ],
    )
    torch.manual_seed(0)
    seed_model = TinyGPT(
        GPTConfig(
            n_embd=8,
            n_layer=1,
            n_head=1,
            block_size=6,
            mlp_expansion=1,
            calculator_enabled=True,
            calculator_mode="add",
            calculator_hook_after_layer=1,
            calculator_operand_vocab_size=3,
            calculator_result_vocab_size=5,
            calculator_estimator="adaptive_interface",
            calculator_read_position="operands",
            calculator_bottleneck_mode="answer_decoder",
            calculator_output_format="sum_left_operand",
        )
    )
    torch.save({"model_state_dict": seed_model.state_dict()}, tmp_path / "seed.pt")

    overfit_script.main()

    run_dirs = [path for path in tmp_path.glob("*") if path.is_dir()]
    assert len(run_dirs) == 1
    child_dirs = list(run_dirs[0].glob("model-c-1digit-seed1"))
    assert len(child_dirs) == 1
    run_dir = child_dirs[0]
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert config["oracle_warmup_steps"] == 1
    assert config["answer_loss_weight"] == 1.0
    assert config["calculator_read_position"] == "operands"
    assert config["calculator_injection_mode"] == "replace"
    assert config["calculator_bottleneck_mode"] == "answer_decoder"
    assert config["calculator_output_format"] == "sum_left_operand"
    assert config["calculator_estimator"] == "adaptive_interface"
    assert config["adaptive_interface_target_mode"] == "soft_result"
    assert config["adaptive_interface_entropy_weight"] == 0.003
    assert config["adaptive_interface_loss_decay_steps"] == 1
    assert config["adaptive_interface_loss_floor"] == 0.0
    assert config["input_proj_anchor_checkpoint"] == str(tmp_path / "seed.pt")
    assert config["input_proj_anchor_weight"] == 0.01
    assert config["input_proj_anchor_decay_steps"] == 1
    assert config["input_proj_lr"] == 0.0003
    assert config["upstream_lr"] == 0.0001
    assert config["model"]["calculator_read_position"] == "operands"
    assert config["model"]["calculator_injection_mode"] == "replace"
    assert config["model"]["calculator_bottleneck_mode"] == "answer_decoder"
    assert config["model"]["calculator_output_format"] == "sum_left_operand"
    assert config["aux_operand_loss_floor"] == 0.01
    assert config["snapshot_every"] == 1
    assert config["trainable_parameter_groups"]
    assert (run_dir / "diagnostic_snapshots.csv").exists()
    assert "counterfactuals" in metrics
    assert metrics["answer_loss_weight"] == 1.0
    assert metrics["calculator_injection_mode"] == "replace"
    assert metrics["calculator_bottleneck_mode"] == "answer_decoder"
    assert metrics["calculator_output_format"] == "sum_left_operand"
    assert metrics["adaptive_interface_target_mode"] == "soft_result"
    assert metrics["adaptive_interface_entropy_weight"] == 0.003
    assert metrics["adaptive_interface_loss_decay_steps"] == 1
    assert metrics["adaptive_interface_loss_floor"] == 0.0
    assert metrics["final_adaptive_interface_loss_weight"] == 0.0
    assert metrics["input_proj_anchor_checkpoint"] == str(tmp_path / "seed.pt")
    assert metrics["input_proj_anchor_weight"] == 0.01
    assert metrics["final_input_proj_anchor_weight"] == 0.0
    assert metrics["final_input_proj_anchor_loss"] >= 0.0
    assert "input_proj_anchor_delta" in metrics
    assert metrics["input_proj_lr"] == 0.0003
    assert metrics["upstream_lr"] == 0.0001
    assert metrics["final_aux_operand_loss_weight"] == 0.01
    assert metrics["final_aux_operand_loss"] >= 0.0
    assert metrics["trainable_parameter_groups"] == config["trainable_parameter_groups"]
    curve_rows = list(csv.DictReader((run_dir / "training_curve.csv").open()))
    assert curve_rows[-1]["adaptive_interface_loss_weight"] == "0.0"
