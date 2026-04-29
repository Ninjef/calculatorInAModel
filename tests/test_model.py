import importlib.util
import json
import sys
from pathlib import Path

import torch

from src.data import EQ_ID, VOCAB_SIZE
from src.model import CalculatorHook, GPTConfig, HardAddSTE, TinyGPT, masked_cross_entropy


def _small_cfg() -> GPTConfig:
    return GPTConfig(n_embd=32, n_layer=2, n_head=2, block_size=8)


def _small_calculator_cfg(mode: str = "add", estimator: str = "ste") -> GPTConfig:
    return GPTConfig(
        n_embd=32,
        n_layer=2,
        n_head=2,
        block_size=8,
        calculator_enabled=True,
        calculator_mode=mode,
        calculator_estimator=estimator,
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=10,
        calculator_result_vocab_size=19,
    )


def _small_scaled_calculator_cfg(scale: float, mode: str = "add") -> GPTConfig:
    cfg = _small_calculator_cfg(mode=mode)
    cfg.calculator_injection_scale = scale
    return cfg


def test_forward_shape() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_cfg())
    x = torch.randint(0, VOCAB_SIZE, (2, 8))

    logits = model(x)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert logits.dtype == torch.float32


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
            "--probe",
            "--probe-steps",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    diagnose_cli.main()

    rows_path = tmp_path / "calculator_trace_rows.csv"
    summary_path = tmp_path / "diagnostic_summary.json"
    assert rows_path.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["samples"] == 8
    assert summary["operand_max"] == 2
    assert "probe" in summary


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
