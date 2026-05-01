import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from src.data import EQ_ID, PLUS_ID, VOCAB_SIZE
from src.model import CalculatorHook, GPTConfig, HardAddSTE, TinyGPT, masked_cross_entropy


def _small_cfg() -> GPTConfig:
    return GPTConfig(n_embd=32, n_layer=2, n_head=2, block_size=8)


def _small_calculator_cfg(
    mode: str = "add",
    estimator: str = "ste",
    injection_mode: str = "add",
    bottleneck_mode: str = "none",
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
            "--calculator-estimator",
            "adaptive_interface",
            "--semantic-decoder-checkpoint",
            str(tmp_path / "seed.pt"),
            "--input-proj-lr",
            "0.0003",
            "--upstream-lr",
            "0.0001",
            "--adaptive-interface-target-mode",
            "soft_result",
            "--adaptive-interface-entropy-weight",
            "0.003",
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
    assert config["calculator_estimator"] == "adaptive_interface"
    assert config["adaptive_interface_target_mode"] == "soft_result"
    assert config["adaptive_interface_entropy_weight"] == 0.003
    assert config["input_proj_lr"] == 0.0003
    assert config["upstream_lr"] == 0.0001
    assert config["model"]["calculator_read_position"] == "operands"
    assert config["model"]["calculator_injection_mode"] == "replace"
    assert config["model"]["calculator_bottleneck_mode"] == "answer_decoder"
    assert config["aux_operand_loss_floor"] == 0.01
    assert config["snapshot_every"] == 1
    assert config["trainable_parameter_groups"]
    assert (run_dir / "diagnostic_snapshots.csv").exists()
    assert "counterfactuals" in metrics
    assert metrics["answer_loss_weight"] == 1.0
    assert metrics["calculator_injection_mode"] == "replace"
    assert metrics["calculator_bottleneck_mode"] == "answer_decoder"
    assert metrics["adaptive_interface_target_mode"] == "soft_result"
    assert metrics["adaptive_interface_entropy_weight"] == 0.003
    assert metrics["input_proj_lr"] == 0.0003
    assert metrics["upstream_lr"] == 0.0001
    assert metrics["final_aux_operand_loss_weight"] == 0.01
    assert metrics["final_aux_operand_loss"] >= 0.0
    assert metrics["trainable_parameter_groups"] == config["trainable_parameter_groups"]
