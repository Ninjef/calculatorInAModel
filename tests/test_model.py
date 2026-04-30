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


def test_invalid_forced_calculator_result_class_raises() -> None:
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    with pytest.raises(ValueError, match="forced_result_class"):
        hook(h, tokens, forced_result_class=19)


def test_tiny_gpt_forwards_forced_calculator_result_class() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="add"))
    assert model.calculator_hook is not None
    x = torch.tensor([[1, 2, EQ_ID, 3, 4, 5, 6, 7]])

    _, diagnostics = model(
        x, return_diagnostics=True, forced_calculator_result_class=5
    )

    assert diagnostics["calculator_trace"]["result_pred"][0, 2].item() == 5


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
    assert summary["calculator_read_position"] == "operands"
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

    overfit_script.main()

    run_dirs = list(tmp_path.glob("*"))
    assert len(run_dirs) == 1
    child_dirs = list(run_dirs[0].glob("model-c-1digit-seed1"))
    assert len(child_dirs) == 1
    run_dir = child_dirs[0]
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert config["oracle_warmup_steps"] == 1
    assert config["calculator_read_position"] == "operands"
    assert config["model"]["calculator_read_position"] == "operands"
    assert config["aux_operand_loss_floor"] == 0.01
    assert config["snapshot_every"] == 1
    assert (run_dir / "diagnostic_snapshots.csv").exists()
    assert "counterfactuals" in metrics
