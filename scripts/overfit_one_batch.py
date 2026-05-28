import argparse
import csv
import json
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import (
    ANSWER_FORMATS,
    AnswerFormat,
    EOS_ID,
    EQ_ID,
    ID_TO_TOKEN,
    ArithmeticBatch,
    answer_target,
    make_loss_mask,
    detokenize,
    make_batch,
    max_answer_tokens,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import GPTConfig, TinyGPT, masked_cross_entropy

DEFAULT_DIGITS = (1, 2, 3)
DEFAULT_STEPS = 1000
DEFAULT_EVAL_SAMPLES = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-3
DEFAULT_SEED = 0
LOG_EVERY = 50
SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS = 1e-6
SHADOW_FEEDBACK_FEATURE_NORMALIZATION_EPS = 1e-6


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    run_name: str
    seed: int
    num_digits: int
    steps: int
    batch_size: int
    eval_samples: int
    lr: float
    answer_loss_weight: float
    answer_format: str
    weight_decay: float
    grad_clip: float
    fixed_width: bool
    operand_max: int | None
    exhaustive_grid_batch: bool
    exhaustive_grid_size: int | None
    calculator_operand_vocab_size: int
    oracle_train: bool
    oracle_warmup_steps: int
    aux_operand_loss_weight: float
    aux_operand_loss_decay_steps: int
    aux_operand_loss_floor: float
    aux_operand_loss_grad_upstream: bool
    snapshot_every: int
    snapshot_samples: int
    checkpoint_every: int
    calculator_estimator: str
    calculator_action_head: str
    calculator_read_position: str
    calculator_read_span_width: int
    calculator_injection_mode: str
    calculator_bottleneck_mode: str
    calculator_output_format: str
    answer_decoder_interaction: str
    semantic_decoder_checkpoint: str | None
    semantic_decoder_checkpoint_load_scope: str
    adaptive_interface_loss_weight: float
    adaptive_interface_loss_decay_steps: int
    adaptive_interface_loss_floor: float
    adaptive_interface_target_mode: str
    adaptive_interface_entropy_weight: float
    action_loss_candidate_random: int
    action_loss_candidate_topk: int
    action_loss_candidate_local_radius: int
    action_loss_candidate_temperature: float
    action_loss_candidate_refresh_every: int
    action_loss_candidate_ema_beta: float
    action_loss_full_enum_temperature: float
    action_loss_full_enum_min_probability_floor: float
    action_loss_full_enum_chunk_size: int
    action_loss_full_enum_target_mode: str
    expected_answer_loss_weight: float
    expected_answer_loss_policy_temperature: float
    expected_answer_loss_cost_normalization: str
    expected_answer_loss_entropy_weight: float
    expected_answer_loss_entropy_decay_steps: int
    expected_answer_loss_chunk_size: int
    expected_answer_loss_gradient_diagnostic_only: bool
    result_boundary_target_loss_weight: float
    result_boundary_target_mode: str
    result_boundary_target_temperature: float
    result_boundary_target_min_probability_floor: float
    result_boundary_target_chunk_size: int
    result_policy_entropy_weight: float
    result_policy_batch_diversity_weight: float
    result_policy_improvement_assignment_weight: float
    result_policy_improvement_assignment_min_improvement: float
    result_policy_improvement_assignment_quota_multiplier: float
    result_policy_stabilization_temperature: float
    result_policy_stabilization_decay_steps: int
    calculator_causal_gap_weight: float
    calculator_causal_gap_margin: float
    boundary_feedback_weight: float
    boundary_feedback_mode: str
    boundary_feedback_seed: int
    boundary_feedback_gradient_diagnostic_only: bool
    shadow_feedback_mode: str
    shadow_feedback_ridge: float
    shadow_feedback_weight: float
    shadow_feedback_heldout_fraction: float
    shadow_feedback_hidden_size: int
    shadow_feedback_dropout: float
    shadow_feedback_online_lr: float
    shadow_feedback_weight_decay: float
    shadow_feedback_warmup_steps: int
    shadow_feedback_updates_per_step: int
    shadow_feedback_apply_max_norm: float
    shadow_feedback_refresh_every: int
    shadow_feedback_validation_fraction: float
    shadow_feedback_validation_every: int
    shadow_feedback_validation_loss_weight: float
    shadow_feedback_validation_gradient_loss_weight: float
    shadow_feedback_validation_gradient_norm_weight: float
    shadow_feedback_target_normalization: str
    shadow_feedback_target_transform: str
    shadow_feedback_feature_mode: str
    shadow_feedback_feature_normalization: str
    shadow_feedback_loss_mode: str
    shadow_feedback_selection_score_mode: str
    shadow_feedback_selection_gap_penalty: float
    shadow_feedback_gradient_diagnostic_only: bool
    calculator_result_head_hidden_size: int
    relaxed_calculator_temperature: float
    relaxed_calculator_final_temperature: float
    relaxed_calculator_temperature_decay_steps: int
    relaxed_calculator_mode: str
    relaxed_calculator_hard_forward: bool
    relaxed_calculator_entropy_weight: float
    relaxed_calculator_entropy_decay_steps: int
    input_proj_anchor_checkpoint: str | None
    input_proj_anchor_weight: float
    input_proj_anchor_decay_steps: int
    input_proj_lr: float
    upstream_lr: float
    optimizer_step_max_delta_norm: float
    optimizer_step_acceptance_mode: str
    optimizer_step_acceptance_tolerance: float
    optimizer_step_line_search_scales: str
    freeze_semantic_decoder: bool
    freeze_upstream_encoder: bool
    trainable_parameter_groups: list[dict[str, object]]
    reinforce_baseline_mode: str
    reinforce_num_samples_per_prompt: int
    reinforce_baseline_beta: float
    reinforce_entropy_weight: float
    reinforce_entropy_decay_steps: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_expansion: int
    calculator_hook_after_layer: int
    model: dict[str, object]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def make_problem(
    a: int,
    b: int,
    num_digits: int,
    fixed_width: bool,
    answer_format: AnswerFormat = "sum",
) -> tuple[list[int], str]:
    if fixed_width:
        prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
    else:
        prompt = f"{a}+{b}="
    return tokenize(prompt), answer_target(
        a,
        b,
        num_digits,
        answer_format=answer_format,
        fixed_width=fixed_width,
    )


@contextmanager
def temporary_calculator_injection_scale(
    model: TinyGPT, scale: float | None
) -> object:
    if scale is None or model.calculator_hook is None:
        yield
        return
    old_scale = model.calculator_hook.injection_scale
    model.calculator_hook.injection_scale = scale
    try:
        yield
    finally:
        model.calculator_hook.injection_scale = old_scale


def generate_answer(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str | torch.device,
    oracle_operands: tuple[int, int] | None = None,
    calculator_result_override: str = "add",
) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.cfg.block_size :]
        oracle_tensor = None
        if oracle_operands is not None:
            oracle_tensor = make_oracle_operands_from_values(
                a=oracle_operands[0],
                b=oracle_operands[1],
                shape=ids_cond.shape,
                device=device,
            )
        logits = model(
            ids_cond,
            oracle_operands=oracle_tensor,
            calculator_result_override=calculator_result_override,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt_ids) :].tolist()


def trim_after_eos(ids: list[int]) -> list[int]:
    if EOS_ID in ids:
        return ids[: ids.index(EOS_ID) + 1]
    return ids


def evaluate(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    fixed_width: bool,
    answer_format: AnswerFormat,
    device: str | torch.device,
    oracle_train: bool,
    calculator_result_override: str = "add",
    injection_scale: float | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    answer_tokens = max_answer_tokens(num_digits, answer_format=answer_format)
    exact = 0
    examples: list[dict[str, str | bool]] = []

    was_training = model.training
    model.eval()
    with temporary_calculator_injection_scale(model, injection_scale):
        for i in range(samples):
            a = rng.randint(0, operand_max)
            b = rng.randint(0, operand_max)
            prompt_ids, target = make_problem(
                a,
                b,
                num_digits,
                fixed_width=fixed_width,
                answer_format=answer_format,
            )
            oracle_operands = (a, b) if oracle_train else None
            pred_ids = trim_after_eos(
                generate_answer(
                    model,
                    prompt_ids,
                    answer_tokens,
                    device,
                    oracle_operands=oracle_operands,
                    calculator_result_override=calculator_result_override,
                )
            )
            pred = decode_tokens(pred_ids)
            ok = pred == target
            exact += int(ok)
            if i < 8:
                examples.append(
                    {
                        "prompt": detokenize(prompt_ids),
                        "target": target,
                        "prediction": pred,
                        "correct": ok,
                    }
                )
    if was_training:
        model.train()

    return {
        "num_digits": num_digits,
        "samples": samples,
        "exact_match": exact / samples,
        "correct": exact,
        "examples": examples,
    }


@torch.no_grad()
def calculator_trace_rows(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
    oracle_train: bool,
    answer_format: AnswerFormat,
    calculator_result_override: str = "add",
    injection_scale: float | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rows: list[dict[str, object]] = []
    answer_tokens = max_answer_tokens(num_digits, answer_format=answer_format)
    was_training = model.training
    model.eval()
    with temporary_calculator_injection_scale(model, injection_scale):
        for i in range(samples):
            a = rng.randint(0, operand_max)
            b = rng.randint(0, operand_max)
            prompt_ids, target = make_problem(
                a,
                b,
                num_digits,
                fixed_width=True,
                answer_format=answer_format,
            )
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            oracle_operands = None
            if oracle_train:
                oracle_operands = make_oracle_operands_from_values(
                    a=a, b=b, shape=x.shape, device=device
                )
            logits, diagnostics = model(
                x,
                return_diagnostics=True,
                oracle_operands=oracle_operands,
                calculator_result_override=calculator_result_override,
            )
            probs = logits[:, -1, :].softmax(dim=-1)
            pred_ids = trim_after_eos(
                generate_answer(
                    model,
                    prompt_ids,
                    answer_tokens,
                    device,
                    oracle_operands=(a, b) if oracle_train else None,
                    calculator_result_override=calculator_result_override,
                )
            )
            pred = decode_tokens(pred_ids)
            eq_pos = prompt_ids.index(EQ_ID)
            trace = diagnostics.get("calculator_trace", {})

            def trace_value(name: str, default: float | int | bool) -> float | int | bool:
                if name not in trace:
                    return default
                value = trace[name][0, eq_pos]
                if value.dtype == torch.bool:
                    return bool(value.item())
                if value.dtype.is_floating_point:
                    return float(value.item())
                return int(value.item())

            rows.append(
                {
                    "sample": i,
                    "prompt": decode_tokens(prompt_ids),
                    "true_a": a,
                    "true_b": b,
                    "true_sum": a + b,
                    "target_answer": target,
                    "prediction": pred,
                    "correct": pred == target,
                    "first_token_confidence": float(probs.max().item()),
                    "a_pred": trace_value("a_pred", -1),
                    "b_pred": trace_value("b_pred", -1),
                    "pair_pred": trace_value("pair_pred", -1),
                    "calculator_result": trace_value("result_pred", -1),
                    "a_confidence": trace_value("a_confidence", float("nan")),
                    "b_confidence": trace_value("b_confidence", float("nan")),
                    "pair_confidence": trace_value("pair_confidence", float("nan")),
                    "a_entropy": trace_value("a_entropy", float("nan")),
                    "b_entropy": trace_value("b_entropy", float("nan")),
                    "pair_entropy": trace_value("pair_entropy", float("nan")),
                    "a_logp": trace_value("a_logp", float("nan")),
                    "b_logp": trace_value("b_logp", float("nan")),
                    "pair_logp": trace_value("pair_logp", float("nan")),
                    "sampled_logp": trace_value("sampled_logp", float("nan")),
                    "injection_norm": trace_value("injection_norm", float("nan")),
                    "calculator_read_position_id": trace_value(
                        "calculator_read_position_id", -1
                    ),
                    "a_read_position": trace_value("a_read_position", -1),
                    "b_read_position": trace_value("b_read_position", -1),
                    "eq_read_position": trace_value("eq_read_position", -1),
                    "oracle_used": trace_value("oracle_used", False),
                }
            )
    if was_training:
        model.train()
    return rows


def summarize_trace_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    answer_correct = sum(int(row["correct"]) for row in rows)
    operand_rows = [row for row in rows if row["a_pred"] >= 0 and row["b_pred"] >= 0]
    operand_correct = sum(
        int(row["a_pred"] == row["true_a"] and row["b_pred"] == row["true_b"])
        for row in operand_rows
    )
    result_correct = sum(
        int(row["calculator_result"] == row["true_sum"]) for row in operand_rows
    )
    pair_correct = sum(
        int(row["a_pred"] == row["true_a"] and row["b_pred"] == row["true_b"])
        for row in operand_rows
        if row.get("pair_pred", -1) >= 0
    )
    pair_rows = [row for row in operand_rows if row.get("pair_pred", -1) >= 0]

    def mean_field(name: str) -> float:
        finite = [
            float(row[name])
            for row in operand_rows
            if isinstance(row[name], float) and row[name] == row[name]
        ]
        return sum(finite) / len(finite) if finite else float("nan")

    return {
        "samples": len(rows),
        "exact_match": answer_correct / max(len(rows), 1),
        "correct": answer_correct,
        "operand_exact_match": operand_correct / max(len(operand_rows), 1),
        "calculator_result_accuracy": result_correct / max(len(operand_rows), 1),
        "pair_exact_match": pair_correct / max(len(pair_rows), 1),
        "mean_a_confidence": mean_field("a_confidence"),
        "mean_b_confidence": mean_field("b_confidence"),
        "mean_pair_confidence": mean_field("pair_confidence"),
        "mean_a_entropy": mean_field("a_entropy"),
        "mean_b_entropy": mean_field("b_entropy"),
        "mean_pair_entropy": mean_field("pair_entropy"),
        "mean_sampled_logp": mean_field("sampled_logp"),
    }


def compact_distribution(values: list[object], *, limit: int = 12) -> str:
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return json.dumps(dict(ordered[:limit]), sort_keys=True)


def snapshot_row_from_model(
    model: TinyGPT,
    *,
    step: int,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
    answer_format: AnswerFormat,
) -> dict[str, object]:
    normal_rows = calculator_trace_rows(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        device=device,
        oracle_train=False,
        answer_format=answer_format,
    )
    normal = summarize_trace_rows(normal_rows)

    injection_zero = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        answer_format=answer_format,
        device=device,
        oracle_train=False,
        injection_scale=0.0,
    )
    oracle = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        answer_format=answer_format,
        device=device,
        oracle_train=True,
    )
    forced_zero = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        answer_format=answer_format,
        device=device,
        oracle_train=False,
        calculator_result_override="zero",
    )
    forced_random = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        answer_format=answer_format,
        device=device,
        oracle_train=False,
        calculator_result_override="random",
    )

    return {
        "step": step,
        "samples": samples,
        "normal_exact_match": normal["exact_match"],
        "injection_zero_exact_match": injection_zero["exact_match"],
        "oracle_exact_match": oracle["exact_match"],
        "forced_zero_exact_match": forced_zero["exact_match"],
        "forced_random_exact_match": forced_random["exact_match"],
        "operand_exact_match": normal["operand_exact_match"],
        "pair_exact_match": normal["pair_exact_match"],
        "calculator_result_accuracy": normal["calculator_result_accuracy"],
        "mean_a_confidence": normal["mean_a_confidence"],
        "mean_b_confidence": normal["mean_b_confidence"],
        "mean_pair_confidence": normal["mean_pair_confidence"],
        "mean_a_entropy": normal["mean_a_entropy"],
        "mean_b_entropy": normal["mean_b_entropy"],
        "mean_pair_entropy": normal["mean_pair_entropy"],
        "learned_result_distribution": compact_distribution(
            [row["calculator_result"] for row in normal_rows]
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    preferred = [
        "step",
        "loss",
        "answer_loss",
        "answer_loss_weight",
        "aux_operand_loss",
        "aux_operand_loss_weight",
        "policy_loss",
        "policy_baseline",
        "policy_advantage_mean",
        "policy_advantage_std",
        "sampled_logp",
        "result_entropy",
        "sampled_result_accuracy",
        "operand_entropy",
        "entropy_weight",
        "adaptive_interface_loss",
        "adaptive_interface_loss_weight",
        "adaptive_interface_target_loss",
        "adaptive_interface_objective",
        "adaptive_interface_entropy",
        "adaptive_interface_entropy_weight",
        "adaptive_target_result_accuracy",
        "adaptive_learned_target_agreement",
        "adaptive_target_operand_exact_match",
        "adaptive_target_pair_mass",
        "action_loss_interface_loss",
        "action_loss_interface_objective",
        "action_loss_interface_target_loss",
        "action_loss_candidate_count",
        "action_loss_candidate_temperature",
        "action_loss_candidate_refresh_every",
        "action_loss_candidate_ema_beta",
        "action_loss_replay_cache_size",
        "action_loss_replay_refresh_fraction",
        "action_loss_candidate_best_improvement",
        "action_loss_candidate_better_fraction",
        "action_loss_candidate_best_matches_true_operands",
        "action_loss_candidate_best_result_accuracy",
        "action_loss_candidate_learned_result_accuracy",
        "action_loss_candidate_soft_target_true_a_mass",
        "action_loss_candidate_soft_target_true_b_mass",
        "action_loss_full_enum_temperature",
        "action_loss_full_enum_min_probability_floor",
        "action_loss_full_enum_chunk_size",
        "action_loss_full_enum_target_mode",
        "action_loss_full_enum_best_nll",
        "action_loss_full_enum_learned_nll",
        "action_loss_full_enum_true_nll",
        "action_loss_full_enum_learned_minus_true_gap",
        "action_loss_full_enum_learned_minus_best_gap",
        "action_loss_full_enum_true_best_fraction",
        "action_loss_full_enum_learned_best_fraction",
        "action_loss_full_enum_joint_target_loss",
        "action_loss_full_enum_pair_entropy",
        "action_loss_full_enum_pair_logit_entropy",
        "action_loss_full_enum_pair_logit_effective_pairs",
        "action_loss_full_enum_pair_exact_match",
        "action_loss_full_enum_result_equivalent_pair_accuracy",
        "action_loss_full_enum_true_pair_probability",
        "action_loss_full_enum_true_pair_rank",
        "action_loss_full_enum_learned_pair_nll",
        "action_loss_full_enum_true_pair_nll",
        "action_loss_full_enum_best_pair_nll",
        "action_loss_full_enum_soft_target_true_a_mass",
        "action_loss_full_enum_soft_target_true_b_mass",
        "action_loss_full_enum_entropy",
        "action_loss_full_enum_effective_pairs",
        "action_loss_full_enum_true_pair_probability",
        "action_loss_full_enum_true_pair_rank",
        "action_loss_full_enum_best_left_operand_accuracy",
        "action_loss_full_enum_top1_mass",
        "action_loss_full_enum_top3_mass",
        "action_loss_full_enum_top5_mass",
        "expected_answer_loss",
        "expected_answer_loss_objective",
        "expected_answer_loss_weight",
        "expected_answer_loss_policy_temperature",
        "expected_answer_loss_cost_normalization",
        "expected_answer_loss_entropy_weight",
        "expected_answer_loss_entropy",
        "expected_answer_loss_effective_pairs",
        "expected_answer_loss_effective_results",
        "expected_answer_loss_best_nll",
        "expected_answer_loss_true_nll",
        "expected_answer_loss_learned_nll",
        "expected_answer_loss_raw_expected_nll",
        "expected_answer_loss_expected_minus_best_gap",
        "expected_answer_loss_learned_minus_best_gap",
        "expected_answer_loss_learned_minus_true_gap",
        "expected_answer_loss_best_pair_probability",
        "expected_answer_loss_true_pair_probability",
        "expected_answer_loss_learned_pair_probability",
        "expected_answer_loss_best_result_probability",
        "expected_answer_loss_true_result_probability",
        "expected_answer_loss_learned_result_probability",
        "expected_answer_loss_hard_learned_best_fraction",
        "expected_answer_loss_hard_learned_pair_exact",
        "expected_answer_loss_hard_learned_result_accuracy",
        "expected_answer_loss_hard_learned_calc_accuracy",
        "result_boundary_target_loss",
        "result_boundary_target_objective",
        "result_boundary_target_loss_weight",
        "result_boundary_target_mode",
        "result_boundary_target_temperature",
        "result_boundary_target_min_probability_floor",
        "result_boundary_target_chunk_size",
        "result_boundary_target_best_nll",
        "result_boundary_target_true_nll",
        "result_boundary_target_learned_nll",
        "result_boundary_target_learned_minus_best_gap",
        "result_boundary_target_learned_minus_true_gap",
        "result_boundary_target_hard_best_equals_true_sum",
        "result_boundary_target_tie_aware_true_result_best_fraction",
        "result_boundary_target_true_result_probability",
        "result_boundary_target_entropy",
        "result_boundary_target_effective_results",
        "result_boundary_target_learned_best_fraction",
        "result_boundary_target_hard_learned_calc_accuracy",
        "result_policy_stabilization_objective",
        "result_policy_entropy_weight",
        "result_policy_batch_diversity_weight",
        "result_policy_improvement_assignment_weight",
        "result_policy_improvement_assignment_objective",
        "result_policy_improvement_assignment_min_improvement",
        "result_policy_improvement_assignment_quota_multiplier",
        "result_policy_improvement_assignment_quota",
        "result_policy_improvement_assignment_fraction",
        "result_policy_improvement_assignment_mean_improvement",
        "result_policy_improvement_assignment_unique_results",
        "result_policy_improvement_assignment_target_accuracy",
        "result_policy_improvement_assignment_learned_target_fraction",
        "result_policy_stabilization_temperature",
        "result_policy_entropy",
        "result_policy_effective_results",
        "result_policy_marginal_entropy",
        "result_policy_marginal_effective_results",
        "result_policy_hard_marginal_entropy",
        "result_policy_hard_marginal_effective_results",
        "result_policy_argmax_result_accuracy",
        "result_policy_top3_result_accuracy",
        "calculator_causal_gap_weight",
        "calculator_causal_gap_margin",
        "calculator_causal_gap_objective",
        "calculator_causal_gap",
        "calculator_causal_gap_zero_loss",
        "calculator_causal_gap_normal_loss",
        "optimizer_step_delta_l2",
        "optimizer_step_unclamped_delta_l2",
        "optimizer_step_trust_scale",
        "optimizer_step_max_delta_norm",
        "optimizer_step_acceptance_mode",
        "optimizer_step_acceptance_before_answer_loss",
        "optimizer_step_acceptance_after_answer_loss",
        "optimizer_step_acceptance_delta",
        "optimizer_step_acceptance_tolerance",
        "optimizer_step_accepted",
        "optimizer_step_acceptance_attempts",
        "optimizer_step_acceptance_accepted",
        "optimizer_step_acceptance_rate",
        "optimizer_step_line_search_scales",
        "optimizer_step_selected_scale",
        "relaxed_calculator_temperature",
        "relaxed_calculator_final_temperature",
        "relaxed_calculator_mode",
        "relaxed_calculator_hard_forward",
        "relaxed_calculator_entropy_weight",
        "relaxed_calculator_entropy",
        "relaxed_calculator_effective_pairs",
        "relaxed_calculator_result_entropy",
        "relaxed_calculator_effective_results",
        "relaxed_calculator_true_result_probability",
        "relaxed_calculator_argmax_result_accuracy",
        "relaxed_calculator_top3_result_accuracy",
        "relaxed_calculator_hard_learned_pair_exact",
        "relaxed_calculator_hard_learned_calc_accuracy",
    ]
    fieldnames = preferred + sorted(
        {key for row in curve for key in row.keys()} - set(preferred)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)


def create_unique_dir(path: Path) -> Path:
    for attempt in range(100):
        candidate = path if attempt == 0 else path.with_name(f"{path.name}-{attempt}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique run directory for {path}")


def masked_cross_entropy_per_example(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mask_f = mask.to(loss.dtype)
    return (loss * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)


def single_eq_trace_values(
    trace: dict[str, torch.Tensor], batch: ArithmeticBatch, key: str
) -> torch.Tensor:
    eq_mask = trace["eq_mask"]
    eq_counts = eq_mask.long().sum(dim=-1)
    if not torch.all(eq_counts == 1):
        raise ValueError("REINFORCE training expects one '=' token per example")
    eq_pos = eq_mask.float().argmax(dim=-1).long()
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    return trace[key][batch_idx, eq_pos]


def reinforce_entropy_from_trace(
    trace: dict[str, torch.Tensor],
    batch: ArithmeticBatch,
    *,
    action_head: str,
) -> torch.Tensor:
    if action_head == "result_space":
        return single_eq_trace_values(trace, batch, "result_entropy")
    return single_eq_trace_values(trace, batch, "a_entropy") + single_eq_trace_values(
        trace, batch, "b_entropy"
    )


def reinforce_advantages(
    sample_losses: torch.Tensor,
    *,
    baseline_mode: str,
    global_baseline: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if baseline_mode == "global_ema":
        if global_baseline is None:
            baseline = sample_losses.detach().mean()
        else:
            baseline = sample_losses.new_tensor(float(global_baseline))
        advantages = sample_losses.detach() - baseline
        return advantages, baseline.expand_as(sample_losses)
    if baseline_mode == "per_prompt_mean":
        baseline = sample_losses.detach().mean(dim=0, keepdim=True)
        return sample_losses.detach() - baseline, baseline.expand_as(sample_losses)
    if baseline_mode == "leave_one_out":
        if sample_losses.shape[0] < 2:
            raise ValueError("leave_one_out REINFORCE baseline requires K >= 2")
        detached = sample_losses.detach()
        baseline = (detached.sum(dim=0, keepdim=True) - detached) / (
            sample_losses.shape[0] - 1
        )
        return detached - baseline, baseline
    raise ValueError(f"unknown reinforce baseline mode: {baseline_mode}")


def reinforce_policy_gradient_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    baseline_mode: str,
    num_samples_per_prompt: int,
    global_baseline: float | None,
    entropy_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if num_samples_per_prompt < 1:
        raise ValueError("REINFORCE samples per prompt must be positive")
    sample_losses: list[torch.Tensor] = []
    sample_logps: list[torch.Tensor] = []
    sample_entropies: list[torch.Tensor] = []
    sampled_result_accs: list[torch.Tensor] = []
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    for _ in range(num_samples_per_prompt):
        logits, diagnostics = model(batch.x, return_diagnostics=True)
        per_example_loss = masked_cross_entropy_per_example(
            logits, batch.y, batch.loss_mask
        )
        trace = diagnostics["calculator_trace"]
        sample_losses.append(per_example_loss)
        sample_logps.append(single_eq_trace_values(trace, batch, "sampled_logp"))
        sample_entropies.append(
            reinforce_entropy_from_trace(
                trace, batch, action_head=model.cfg.calculator_action_head
            )
        )
        sampled_result = single_eq_trace_values(trace, batch, "result_pred")
        sampled_result_accs.append((sampled_result == true_sum).float())

    losses = torch.stack(sample_losses, dim=0)
    logps = torch.stack(sample_logps, dim=0)
    entropies = torch.stack(sample_entropies, dim=0)
    result_acc = torch.stack(sampled_result_accs, dim=0)
    advantages, baselines = reinforce_advantages(
        losses,
        baseline_mode=baseline_mode,
        global_baseline=global_baseline,
    )
    policy_loss = (advantages * logps).mean()
    entropy_loss = -entropy_weight * entropies.mean()
    objective = policy_loss + entropy_loss
    answer_loss = losses.mean()
    metrics = {
        "policy_loss": float(policy_loss.detach().item()),
        "policy_objective": float(objective.detach().item()),
        "policy_baseline": float(baselines.mean().detach().item()),
        "policy_advantage_mean": float(advantages.mean().detach().item()),
        "policy_advantage_std": float(advantages.std(unbiased=False).detach().item()),
        "sampled_logp": float(logps.mean().detach().item()),
        "operand_entropy": float(entropies.mean().detach().item()),
        "result_entropy": float(entropies.mean().detach().item())
        if model.cfg.calculator_action_head == "result_space"
        else float("nan"),
        "sampled_result_accuracy": float(result_acc.mean().detach().item()),
    }
    return objective, answer_loss, metrics


def make_oracle_operands_from_values(
    *,
    a: int,
    b: int,
    shape: tuple[int, int],
    device: str | torch.device,
) -> torch.Tensor:
    oracle = torch.zeros((*shape, 2), dtype=torch.long, device=device)
    oracle[..., 0] = a
    oracle[..., 1] = b
    return oracle


def make_oracle_operands_from_batch(
    x: torch.Tensor, *, num_digits: int
) -> torch.Tensor:
    powers = torch.tensor(
        [10**i for i in range(num_digits - 1, -1, -1)],
        dtype=torch.long,
        device=x.device,
    )
    a = (x[:, :num_digits].long() * powers).sum(dim=-1)
    b_start = num_digits + 1
    b_end = b_start + num_digits
    b = (x[:, b_start:b_end].long() * powers).sum(dim=-1)
    oracle = torch.zeros((*x.shape, 2), dtype=torch.long, device=x.device)
    oracle[..., 0] = a.unsqueeze(-1)
    oracle[..., 1] = b.unsqueeze(-1)
    return oracle


def fixed_width_operands_from_batch(
    x: torch.Tensor, *, num_digits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    powers = torch.tensor(
        [10**i for i in range(num_digits - 1, -1, -1)],
        dtype=torch.long,
        device=x.device,
    )
    a = (x[:, :num_digits].long() * powers).sum(dim=-1)
    b_start = num_digits + 1
    b_end = b_start + num_digits
    b = (x[:, b_start:b_end].long() * powers).sum(dim=-1)
    return a, b


@torch.no_grad()
def counterfactual_result_targets(
    model: TinyGPT, batch: ArithmeticBatch, *, forced_result_batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("adaptive interface targets require a calculator hook")
    was_training = model.training
    model.eval()
    result_losses: list[torch.Tensor] = []
    result_vocab_size = model.cfg.calculator_result_vocab_size
    for start in range(0, result_vocab_size, forced_result_batch_size):
        forced_classes = torch.arange(
            start,
            min(start + forced_result_batch_size, result_vocab_size),
            device=batch.x.device,
        )
        expanded_x = batch.x.repeat_interleave(len(forced_classes), dim=0)
        expanded_y = batch.y.repeat_interleave(len(forced_classes), dim=0)
        expanded_mask = batch.loss_mask.repeat_interleave(len(forced_classes), dim=0)
        forced = forced_classes.repeat(batch.x.shape[0])
        logits = model(expanded_x, forced_calculator_result_class=forced)
        losses = masked_cross_entropy_per_example(
            logits, expanded_y, expanded_mask
        ).reshape(batch.x.shape[0], len(forced_classes))
        result_losses.append(losses)
    if was_training:
        model.train()
    losses = torch.cat(result_losses, dim=-1)
    targets = losses.argmin(dim=-1)
    return targets, losses


def select_adaptive_operand_targets(
    a_logits: torch.Tensor, b_logits: torch.Tensor, result_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if a_logits.shape != b_logits.shape:
        raise ValueError("a_logits and b_logits must have the same shape")
    if a_logits.ndim != 2:
        raise ValueError("operand target selection expects [batch, classes] logits")
    classes = a_logits.shape[-1]
    a_idx = torch.arange(classes, device=a_logits.device).view(1, classes, 1)
    b_idx = torch.arange(classes, device=a_logits.device).view(1, 1, classes)
    sum_idx = a_idx + b_idx
    valid = sum_idx == result_targets.view(-1, 1, 1)
    pair_scores = (
        a_logits.log_softmax(dim=-1).unsqueeze(-1)
        + b_logits.log_softmax(dim=-1).unsqueeze(-2)
    )
    pair_scores = pair_scores.masked_fill(~valid, float("-inf"))
    best_pair = pair_scores.reshape(a_logits.shape[0], -1).argmax(dim=-1)
    a_target = best_pair // classes
    b_target = best_pair % classes
    return a_target, b_target


def adaptive_soft_result_loss(
    a_logits: torch.Tensor, b_logits: torch.Tensor, result_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if a_logits.shape != b_logits.shape:
        raise ValueError("a_logits and b_logits must have the same shape")
    if a_logits.ndim != 2:
        raise ValueError("soft result loss expects [batch, classes] logits")
    classes = a_logits.shape[-1]
    a_idx = torch.arange(classes, device=a_logits.device).view(1, classes, 1)
    b_idx = torch.arange(classes, device=b_logits.device).view(1, 1, classes)
    valid = (a_idx + b_idx) == result_targets.view(-1, 1, 1)
    pair_logp = (
        a_logits.log_softmax(dim=-1).unsqueeze(-1)
        + b_logits.log_softmax(dim=-1).unsqueeze(-2)
    )
    valid_logp = pair_logp.masked_fill(~valid, float("-inf"))
    log_mass = torch.logsumexp(valid_logp.reshape(a_logits.shape[0], -1), dim=-1)
    return -log_mass.mean(), log_mass.exp()


def operand_distribution_entropy(
    a_logits: torch.Tensor, b_logits: torch.Tensor
) -> torch.Tensor:
    a_probs = a_logits.softmax(dim=-1)
    b_probs = b_logits.softmax(dim=-1)
    a_entropy = -(a_probs * a_probs.clamp_min(1e-12).log()).sum(dim=-1)
    b_entropy = -(b_probs * b_probs.clamp_min(1e-12).log()).sum(dim=-1)
    return a_entropy + b_entropy


def relaxed_calculator_temperature(
    *,
    initial_temperature: float,
    final_temperature: float,
    decay_steps: int,
    step: int,
) -> float:
    if decay_steps <= 0:
        return initial_temperature
    progress = min(max(step / decay_steps, 0.0), 1.0)
    return initial_temperature + progress * (final_temperature - initial_temperature)


def relaxed_calculator_entropy_weight(
    *, initial_weight: float, decay_steps: int, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    return initial_weight * max(0.0, 1.0 - (step / decay_steps))


def result_policy_stabilization_weight(
    *, initial_weight: float, decay_steps: int, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    return initial_weight * max(0.0, 1.0 - (step / decay_steps))


def hard_improvement_assignment_targets(
    full_losses: torch.Tensor,
    learned_result: torch.Tensor,
    *,
    min_improvement: float,
    quota_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if min_improvement < 0:
        raise ValueError("assignment min improvement must be non-negative")
    if quota_multiplier <= 0:
        raise ValueError("assignment quota multiplier must be positive")
    batch_size, result_count = full_losses.shape
    quota = max(1, math.ceil((batch_size / result_count) * quota_multiplier))
    learned_losses = full_losses.gather(1, learned_result.unsqueeze(-1)).squeeze(-1)
    improvements = learned_losses.unsqueeze(-1) - full_losses
    flat_order = torch.argsort(improvements.reshape(-1), descending=True)
    targets = learned_result.new_full((batch_size,), -1)
    result_counts = learned_result.new_zeros((result_count,))
    assigned_improvements: list[float] = []
    with torch.no_grad():
        for flat_index_tensor in flat_order:
            flat_index = int(flat_index_tensor.item())
            example_index = flat_index // result_count
            result_index = flat_index % result_count
            improvement = float(improvements[example_index, result_index].item())
            if improvement <= min_improvement:
                break
            if int(targets[example_index].item()) >= 0:
                continue
            if int(result_counts[result_index].item()) >= quota:
                continue
            targets[example_index] = result_index
            result_counts[result_index] += 1
            assigned_improvements.append(improvement)
            if len(assigned_improvements) == batch_size:
                break
    assigned_mask = targets >= 0
    assigned_count = int(assigned_mask.sum().item())
    metrics: dict[str, float | int] = {
        "result_policy_improvement_assignment_quota": int(quota),
        "result_policy_improvement_assignment_fraction": (
            assigned_count / max(batch_size, 1)
        ),
        "result_policy_improvement_assignment_mean_improvement": (
            float(sum(assigned_improvements) / assigned_count)
            if assigned_count > 0
            else 0.0
        ),
        "result_policy_improvement_assignment_unique_results": int(
            result_counts.gt(0).sum().item()
        ),
    }
    return targets, metrics


def result_policy_stabilization_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    temperature: float,
    entropy_weight: float,
    batch_diversity_weight: float,
    improvement_assignment_weight: float,
    improvement_assignment_min_improvement: float,
    improvement_assignment_quota_multiplier: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("result policy stabilization temperature must be positive")
    if improvement_assignment_weight < 0:
        raise ValueError(
            "result policy improvement assignment weight must be non-negative"
        )
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("result policy stabilization requires result-space logits")
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    result_probs = torch.softmax(result_logits / temperature, dim=-1)
    result_entropy = -(
        result_probs * result_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    marginal_probs = result_probs.mean(dim=0)
    marginal_entropy = -(
        marginal_probs * marginal_probs.clamp_min(1e-12).log()
    ).sum()
    result_pred = result_logits.argmax(dim=-1)
    hard_marginal = torch.nn.functional.one_hot(
        result_pred,
        num_classes=result_logits.shape[-1],
    ).to(result_probs.dtype).mean(dim=0)
    hard_marginal_entropy = -(
        hard_marginal * hard_marginal.clamp_min(1e-12).log()
    ).sum()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    topk = min(3, result_probs.shape[-1])
    topk_results = result_probs.topk(k=topk, dim=-1).indices
    objective = (
        -(entropy_weight * result_entropy.mean())
        - (batch_diversity_weight * marginal_entropy)
    )
    assignment_metrics: dict[str, float | int] = {}
    assignment_objective = result_logits.new_zeros(())
    if improvement_assignment_weight > 0:
        full_losses = score_forced_result_classes_chunked(
            model, batch, chunk_size=chunk_size
        ).detach()
        assignment_targets, assignment_metrics = hard_improvement_assignment_targets(
            full_losses,
            result_pred.detach(),
            min_improvement=improvement_assignment_min_improvement,
            quota_multiplier=improvement_assignment_quota_multiplier,
        )
        assignment_mask = assignment_targets >= 0
        if bool(assignment_mask.any().item()):
            assignment_objective = torch.nn.functional.cross_entropy(
                result_logits[assignment_mask] / temperature,
                assignment_targets[assignment_mask],
            )
            objective = objective + (
                improvement_assignment_weight * assignment_objective
            )
            assigned_targets = assignment_targets[assignment_mask]
            assignment_metrics[
                "result_policy_improvement_assignment_target_accuracy"
            ] = float(
                (assigned_targets == true_sum[assignment_mask])
                .float()
                .mean()
                .detach()
                .item()
            )
            assignment_metrics[
                "result_policy_improvement_assignment_learned_target_fraction"
            ] = float(
                (assigned_targets == result_pred.detach()[assignment_mask])
                .float()
                .mean()
                .detach()
                .item()
            )
        else:
            assignment_metrics[
                "result_policy_improvement_assignment_target_accuracy"
            ] = 0.0
            assignment_metrics[
                "result_policy_improvement_assignment_learned_target_fraction"
            ] = 0.0
    metrics = {
        "result_policy_entropy_weight": float(entropy_weight),
        "result_policy_batch_diversity_weight": float(batch_diversity_weight),
        "result_policy_improvement_assignment_weight": float(
            improvement_assignment_weight
        ),
        "result_policy_improvement_assignment_objective": float(
            assignment_objective.detach().item()
        ),
        "result_policy_improvement_assignment_min_improvement": float(
            improvement_assignment_min_improvement
        ),
        "result_policy_improvement_assignment_quota_multiplier": float(
            improvement_assignment_quota_multiplier
        ),
        "result_policy_stabilization_temperature": float(temperature),
        "result_policy_entropy": float(result_entropy.mean().detach().item()),
        "result_policy_effective_results": float(
            result_entropy.exp().mean().detach().item()
        ),
        "result_policy_marginal_entropy": float(
            marginal_entropy.detach().item()
        ),
        "result_policy_marginal_effective_results": float(
            marginal_entropy.exp().detach().item()
        ),
        "result_policy_hard_marginal_entropy": float(
            hard_marginal_entropy.detach().item()
        ),
        "result_policy_hard_marginal_effective_results": float(
            hard_marginal_entropy.exp().detach().item()
        ),
        "result_policy_argmax_result_accuracy": float(
            (result_pred == true_sum).float().mean().detach().item()
        ),
        "result_policy_top3_result_accuracy": float(
            (topk_results == true_sum.unsqueeze(1))
            .any(dim=-1)
            .float()
            .mean()
            .detach()
            .item()
        ),
    }
    metrics.update(assignment_metrics)
    return objective, metrics


def snapshot_trainable_parameters(
    model: TinyGPT,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    return [
        (param, param.detach().clone())
        for param in model.parameters()
        if param.requires_grad
    ]


def restore_trainable_parameters(
    before: list[tuple[torch.nn.Parameter, torch.Tensor]]
) -> None:
    with torch.no_grad():
        for param, old_value in before:
            param.copy_(old_value)


def apply_scaled_parameter_delta(
    before: list[tuple[torch.nn.Parameter, torch.Tensor]],
    after: list[tuple[torch.nn.Parameter, torch.Tensor]],
    *,
    scale: float,
) -> None:
    with torch.no_grad():
        for (param, old_value), (_after_param, proposed_value) in zip(before, after):
            delta = proposed_value - old_value
            param.copy_(old_value + (delta * scale))


def parse_optimizer_step_line_search_scales(text: str) -> list[float]:
    values = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0:
            raise ValueError("--optimizer-step-line-search-scales must be non-negative")
        values.append(value)
    if not values:
        raise ValueError("--optimizer-step-line-search-scales must not be empty")
    return values


def apply_optimizer_step_trust_region(
    before: list[tuple[torch.nn.Parameter, torch.Tensor]],
    *,
    max_delta_norm: float,
) -> dict[str, float]:
    if max_delta_norm < 0:
        raise ValueError("optimizer step max delta norm must be non-negative")
    delta_sq = 0.0
    for param, old_value in before:
        delta = param.detach() - old_value
        delta_sq += float(delta.pow(2).sum().item())
    unclamped_delta = math.sqrt(delta_sq)
    scale = 1.0
    if max_delta_norm > 0 and unclamped_delta > max_delta_norm:
        scale = max_delta_norm / max(unclamped_delta, 1e-12)
        with torch.no_grad():
            for param, old_value in before:
                param.copy_(old_value + ((param - old_value) * scale))
    return {
        "optimizer_step_delta_l2": float(
            unclamped_delta * scale
            if max_delta_norm > 0 and unclamped_delta > max_delta_norm
            else unclamped_delta
        ),
        "optimizer_step_unclamped_delta_l2": float(unclamped_delta),
        "optimizer_step_trust_scale": float(scale),
        "optimizer_step_max_delta_norm": float(max_delta_norm),
    }


def hard_path_answer_loss_metric(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    oracle_operands: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(batch.x, oracle_operands=oracle_operands)
        loss = masked_cross_entropy_per_example(
            logits,
            batch.y,
            batch.loss_mask,
        ).mean()
    if was_training:
        model.train()
    return float(loss.item())


def relaxed_calculator_policy_metrics(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    temperature: float,
    entropy_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("relaxed calculator temperature must be positive")
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    if model.cfg.calculator_action_head == "result_space":
        result_logits, _, _, _ = calculator_read_result_logits(model, batch)
        result_probs = torch.softmax(result_logits / temperature, dim=-1)
        result_entropy = -(
            result_probs * result_probs.clamp_min(1e-12).log()
        ).sum(dim=-1)
        true_result_probability = result_probs.gather(1, true_sum.unsqueeze(1)).squeeze(1)
        argmax_result = result_probs.argmax(dim=-1)
        topk = min(3, result_probs.shape[-1])
        topk_results = result_probs.topk(k=topk, dim=-1).indices
        max_operand = model.cfg.calculator_operand_vocab_size - 1
        learned_a = torch.minimum(
            argmax_result, torch.full_like(argmax_result, max_operand)
        )
        learned_b = argmax_result - learned_a
        entropy_objective = -entropy_weight * result_entropy.mean()
        metrics = {
            "relaxed_calculator_temperature": float(temperature),
            "relaxed_calculator_entropy_weight": float(entropy_weight),
            "relaxed_calculator_entropy": float(result_entropy.mean().item()),
            "relaxed_calculator_effective_pairs": float("nan"),
            "relaxed_calculator_result_entropy": float(result_entropy.mean().item()),
            "relaxed_calculator_effective_results": float(
                result_entropy.exp().mean().item()
            ),
            "relaxed_calculator_true_result_probability": float(
                true_result_probability.mean().item()
            ),
            "relaxed_calculator_argmax_result_accuracy": float(
                (argmax_result == true_sum).float().mean().item()
            ),
            "relaxed_calculator_top3_result_accuracy": float(
                (topk_results == true_sum.unsqueeze(1)).any(dim=-1).float().mean().item()
            ),
            "relaxed_calculator_hard_learned_pair_exact": float(
                ((learned_a == true_a) & (learned_b == true_b)).float().mean().item()
            ),
            "relaxed_calculator_hard_learned_calc_accuracy": float(
                (argmax_result == true_sum).float().mean().item()
            ),
        }
        return entropy_objective, metrics
    if model.cfg.calculator_action_head == "joint_pair":
        pair_logits, _, _, _ = calculator_read_pair_logits(model, batch)
        pair_probs = torch.softmax(pair_logits / temperature, dim=-1)
        pair_entropy = -(pair_probs * pair_probs.clamp_min(1e-12).log()).sum(dim=-1)
        classes = model.cfg.calculator_operand_vocab_size
        learned_pair = pair_logits.argmax(dim=-1)
        learned_a = learned_pair // classes
        learned_b = learned_pair % classes
        values = torch.arange(classes, device=batch.x.device)
        sum_idx = (values.view(-1, 1) + values.view(1, -1)).reshape(1, -1)
        sum_idx = sum_idx.expand(pair_probs.shape[0], -1)
        result_probs = pair_probs.new_zeros(
            (pair_probs.shape[0], model.cfg.calculator_result_vocab_size)
        )
        result_probs.scatter_add_(1, sum_idx, pair_probs)
        result_entropy = -(
            result_probs * result_probs.clamp_min(1e-12).log()
        ).sum(dim=-1)
        true_result_probability = result_probs.gather(1, true_sum.unsqueeze(1)).squeeze(1)
        argmax_result = result_probs.argmax(dim=-1)
        topk = min(3, result_probs.shape[-1])
        topk_results = result_probs.topk(k=topk, dim=-1).indices
        entropy = pair_entropy
        entropy_objective = -entropy_weight * entropy.mean()
        learned_sum = learned_a + learned_b
        metrics = {
            "relaxed_calculator_temperature": float(temperature),
            "relaxed_calculator_entropy_weight": float(entropy_weight),
            "relaxed_calculator_entropy": float(entropy.mean().item()),
            "relaxed_calculator_effective_pairs": float(entropy.exp().mean().item()),
            "relaxed_calculator_result_entropy": float(result_entropy.mean().item()),
            "relaxed_calculator_effective_results": float(
                result_entropy.exp().mean().item()
            ),
            "relaxed_calculator_true_result_probability": float(
                true_result_probability.mean().item()
            ),
            "relaxed_calculator_argmax_result_accuracy": float(
                (argmax_result == true_sum).float().mean().item()
            ),
            "relaxed_calculator_top3_result_accuracy": float(
                (topk_results == true_sum.unsqueeze(1)).any(dim=-1).float().mean().item()
            ),
            "relaxed_calculator_hard_learned_pair_exact": float(
                ((learned_a == true_a) & (learned_b == true_b)).float().mean().item()
            ),
            "relaxed_calculator_hard_learned_calc_accuracy": float(
                (learned_sum == true_sum).float().mean().item()
            ),
        }
        return entropy_objective, metrics

    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    a_probs = torch.softmax(a_logits / temperature, dim=-1)
    b_probs = torch.softmax(b_logits / temperature, dim=-1)
    a_entropy = -(a_probs * a_probs.clamp_min(1e-12).log()).sum(dim=-1)
    b_entropy = -(b_probs * b_probs.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = a_entropy + b_entropy
    entropy_objective = -entropy_weight * entropy.mean()
    classes = model.cfg.calculator_operand_vocab_size
    pair_probs = (a_probs.unsqueeze(2) * b_probs.unsqueeze(1)).reshape(a_probs.shape[0], -1)
    values = torch.arange(classes, device=batch.x.device)
    sum_idx = (values.view(-1, 1) + values.view(1, -1)).reshape(1, -1)
    sum_idx = sum_idx.expand(pair_probs.shape[0], -1)
    result_probs = pair_probs.new_zeros(
        (pair_probs.shape[0], model.cfg.calculator_result_vocab_size)
    )
    result_probs.scatter_add_(1, sum_idx, pair_probs)
    result_entropy = -(result_probs * result_probs.clamp_min(1e-12).log()).sum(dim=-1)
    true_result_probability = result_probs.gather(1, true_sum.unsqueeze(1)).squeeze(1)
    argmax_result = result_probs.argmax(dim=-1)
    topk = min(3, result_probs.shape[-1])
    topk_results = result_probs.topk(k=topk, dim=-1).indices

    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_sum = learned_a + learned_b
    metrics = {
        "relaxed_calculator_temperature": float(temperature),
        "relaxed_calculator_entropy_weight": float(entropy_weight),
        "relaxed_calculator_entropy": float(entropy.mean().item()),
        "relaxed_calculator_effective_pairs": float(entropy.exp().mean().item()),
        "relaxed_calculator_result_entropy": float(result_entropy.mean().item()),
        "relaxed_calculator_effective_results": float(
            result_entropy.exp().mean().item()
        ),
        "relaxed_calculator_true_result_probability": float(
            true_result_probability.mean().item()
        ),
        "relaxed_calculator_argmax_result_accuracy": float(
            (argmax_result == true_sum).float().mean().item()
        ),
        "relaxed_calculator_top3_result_accuracy": float(
            (topk_results == true_sum.unsqueeze(1)).any(dim=-1).float().mean().item()
        ),
        "relaxed_calculator_hard_learned_pair_exact": float(
            ((learned_a == true_a) & (learned_b == true_b)).float().mean().item()
        ),
        "relaxed_calculator_hard_learned_calc_accuracy": float(
            (learned_sum == true_sum).float().mean().item()
        ),
    }
    return entropy_objective, metrics


def calculator_read_operand_logits(
    model: TinyGPT, batch: ArithmeticBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("calculator operand logits require a calculator hook")
    B, T = batch.x.shape
    assert T <= model.cfg.block_size, (
        f"sequence length {T} > block_size {model.cfg.block_size}"
    )
    pos = torch.arange(T, device=batch.x.device)
    residual = model.tok_emb(batch.x) + model.pos_emb(pos)
    if model.cfg.calculator_hook_after_layer > 0:
        for i, block in enumerate(model.blocks, start=1):
            residual = block(residual)
            if i == model.cfg.calculator_hook_after_layer:
                break
    positions = model._calculator_read_positions(batch.x)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    a_pos = positions["a"]
    b_pos = positions["b"]
    if model.cfg.calculator_read_position == "operand_spans":
        a_input, b_input = model.calculator_hook._operand_span_inputs(
            residual, positions
        )
        a_logits, _ = model.calculator_hook.input_proj(a_input).split(
            model.cfg.calculator_operand_vocab_size, dim=-1
        )
        _, b_logits = model.calculator_hook.input_proj(b_input).split(
            model.cfg.calculator_operand_vocab_size, dim=-1
        )
        return a_logits, b_logits, a_pos, b_pos
    operand_logits = model.calculator_hook.input_proj(residual)
    a_logits_all, b_logits_all = operand_logits.split(
        model.cfg.calculator_operand_vocab_size, dim=-1
    )
    return (
        a_logits_all[batch_idx, a_pos],
        b_logits_all[batch_idx, b_pos],
        a_pos,
        b_pos,
    )


def calculator_read_pair_logits(
    model: TinyGPT, batch: ArithmeticBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("calculator pair logits require a calculator hook")
    if model.cfg.calculator_action_head != "joint_pair":
        raise ValueError("calculator pair logits require calculator_action_head=joint_pair")
    B, T = batch.x.shape
    assert T <= model.cfg.block_size, (
        f"sequence length {T} > block_size {model.cfg.block_size}"
    )
    pos = torch.arange(T, device=batch.x.device)
    residual = model.tok_emb(batch.x) + model.pos_emb(pos)
    if model.cfg.calculator_hook_after_layer > 0:
        for i, block in enumerate(model.blocks, start=1):
            residual = block(residual)
            if i == model.cfg.calculator_hook_after_layer:
                break
    positions = model._calculator_read_positions(batch.x)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    if model.cfg.calculator_read_position == "operand_spans":
        pair_input = torch.cat(
            model.calculator_hook._operand_span_inputs(residual, positions),
            dim=-1,
        )
    else:
        pair_input = torch.cat(
            [
                residual[batch_idx, positions["a"]],
                residual[batch_idx, positions["b"]],
            ],
            dim=-1,
        )
    if model.calculator_hook.pair_proj is None:
        raise ValueError("calculator pair logits require a joint pair projection")
    return (
        model.calculator_hook.pair_proj(pair_input),
        positions["a"],
        positions["b"],
        positions["eq"],
    )


def calculator_read_result_logits(
    model: TinyGPT, batch: ArithmeticBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    result_logits, _, positions = calculator_read_result_logits_and_input(
        model,
        batch,
    )
    return (
        result_logits,
        positions["a"],
        positions["b"],
        positions["eq"],
    )


def calculator_read_result_logits_and_input(
    model: TinyGPT, batch: ArithmeticBatch
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if model.calculator_hook is None:
        raise ValueError("calculator result logits require a calculator hook")
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError(
            "calculator result logits require calculator_action_head=result_space"
        )
    B, T = batch.x.shape
    assert T <= model.cfg.block_size, (
        f"sequence length {T} > block_size {model.cfg.block_size}"
    )
    pos = torch.arange(T, device=batch.x.device)
    residual = model.tok_emb(batch.x) + model.pos_emb(pos)
    if model.cfg.calculator_hook_after_layer > 0:
        for i, block in enumerate(model.blocks, start=1):
            residual = block(residual)
            if i == model.cfg.calculator_hook_after_layer:
                break
    positions = model._calculator_read_positions(batch.x)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    if model.cfg.calculator_read_position == "operand_spans":
        result_input = torch.cat(
            model.calculator_hook._operand_span_inputs(residual, positions),
            dim=-1,
        )
    else:
        result_input = torch.cat(
            [
                residual[batch_idx, positions["a"]],
                residual[batch_idx, positions["b"]],
            ],
            dim=-1,
        )
    if model.calculator_hook.result_proj is None:
        raise ValueError("calculator result logits require a result projection")
    return model.calculator_hook.result_proj(result_input), result_input, positions


def adaptive_interface_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    target_mode: str,
    entropy_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_mode not in {"hard_pair", "soft_result"}:
        raise ValueError("adaptive target mode must be hard_pair or soft_result")
    result_targets, _ = counterfactual_result_targets(model, batch)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    a_targets, b_targets = select_adaptive_operand_targets(
        a_logits, b_logits, result_targets
    )
    if target_mode == "hard_pair":
        target_loss = (
            torch.nn.functional.cross_entropy(a_logits, a_targets)
            + torch.nn.functional.cross_entropy(b_logits, b_targets)
        ) / 2
        target_pair_mass = torch.full_like(result_targets, float("nan"), dtype=torch.float)
    else:
        target_loss, target_pair_mass = adaptive_soft_result_loss(
            a_logits, b_logits, result_targets
        )
    operand_entropy = operand_distribution_entropy(a_logits, b_logits)
    objective_loss = target_loss - (entropy_weight * operand_entropy.mean())
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_sum = learned_a + learned_b
    metrics = {
        "adaptive_target_result_accuracy": float(
            (result_targets == true_sum).float().mean().item()
        ),
        "adaptive_learned_target_agreement": float(
            (learned_sum == result_targets).float().mean().item()
        ),
        "adaptive_target_operand_exact_match": float(
            ((a_targets == true_a) & (b_targets == true_b)).float().mean().item()
        ),
        "adaptive_interface_target_loss": float(target_loss.item()),
        "adaptive_interface_entropy": float(operand_entropy.mean().item()),
        "adaptive_target_pair_mass": float(target_pair_mass.nanmean().item()),
    }
    return objective_loss, metrics


def action_loss_candidate_pairs(
    a_logits: torch.Tensor,
    b_logits: torch.Tensor,
    *,
    random_actions: int,
    topk: int,
    local_radius: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if a_logits.shape != b_logits.shape:
        raise ValueError("a_logits and b_logits must have the same shape")
    if a_logits.ndim != 2:
        raise ValueError("action-loss candidates expect [batch, classes] logits")
    batch_size, classes = a_logits.shape
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    candidate_chunks = [
        torch.stack([learned_a, learned_b], dim=-1).unsqueeze(1)
    ]
    if topk > 0:
        k = min(topk, classes)
        top_a = a_logits.topk(k, dim=-1).indices
        top_b = b_logits.topk(k, dim=-1).indices
        pairs = torch.stack(
            [
                top_a.unsqueeze(2).expand(batch_size, k, k),
                top_b.unsqueeze(1).expand(batch_size, k, k),
            ],
            dim=-1,
        ).reshape(batch_size, k * k, 2)
        candidate_chunks.append(pairs)
    if local_radius > 0:
        offsets = torch.arange(
            -local_radius, local_radius + 1, device=a_logits.device
        )
        local_a = (learned_a.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            min=0, max=classes - 1
        )
        local_b = (learned_b.unsqueeze(1) + offsets.unsqueeze(0)).clamp(
            min=0, max=classes - 1
        )
        candidate_chunks.extend(
            [
                torch.stack(
                    [local_a, learned_b.unsqueeze(1).expand_as(local_a)], dim=-1
                ),
                torch.stack(
                    [learned_a.unsqueeze(1).expand_as(local_b), local_b], dim=-1
                ),
            ]
        )
    if random_actions > 0:
        random_pairs = torch.randint(
            low=0,
            high=classes,
            size=(batch_size, random_actions, 2),
            device=a_logits.device,
            generator=generator,
        )
        candidate_chunks.append(random_pairs)
    return torch.cat(candidate_chunks, dim=1)


def score_action_loss_candidates(
    model: TinyGPT,
    batch: ArithmeticBatch,
    candidates: torch.Tensor,
) -> torch.Tensor:
    if candidates.ndim != 3 or candidates.shape[-1] != 2:
        raise ValueError("candidates must have shape [batch, candidates, 2]")
    batch_size, candidate_count, _ = candidates.shape
    expanded_x = batch.x.repeat_interleave(candidate_count, dim=0)
    expanded_y = batch.y.repeat_interleave(candidate_count, dim=0)
    expanded_mask = batch.loss_mask.repeat_interleave(candidate_count, dim=0)
    forced_pairs = candidates.reshape(batch_size * candidate_count, 2)
    oracle_operands = torch.zeros(
        (*expanded_x.shape, 2), dtype=torch.long, device=expanded_x.device
    )
    oracle_operands[..., 0] = forced_pairs[:, 0].unsqueeze(-1)
    oracle_operands[..., 1] = forced_pairs[:, 1].unsqueeze(-1)
    logits = model(expanded_x, oracle_operands=oracle_operands)
    return masked_cross_entropy_per_example(
        logits, expanded_y, expanded_mask
    ).reshape(batch_size, candidate_count)


def score_action_loss_candidates_chunked(
    model: TinyGPT,
    batch: ArithmeticBatch,
    candidates: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("action-loss candidate chunk size must be positive")
    losses = []
    for start in range(0, candidates.shape[1], chunk_size):
        losses.append(
            score_action_loss_candidates(
                model, batch, candidates[:, start : start + chunk_size]
            )
        )
    return torch.cat(losses, dim=-1)


def subset_arithmetic_batch(batch: ArithmeticBatch, indices: torch.Tensor) -> ArithmeticBatch:
    return ArithmeticBatch(
        x=batch.x.index_select(0, indices),
        y=batch.y.index_select(0, indices),
        loss_mask=batch.loss_mask.index_select(0, indices),
    )


def action_loss_soft_targets(
    a_logits: torch.Tensor,
    b_logits: torch.Tensor,
    candidates: torch.Tensor,
    candidate_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_a = torch.zeros_like(a_logits)
    target_b = torch.zeros_like(b_logits)
    target_a.scatter_add_(1, candidates[..., 0], candidate_weights)
    target_b.scatter_add_(1, candidates[..., 1], candidate_weights)
    return target_a, target_b


def action_loss_target_ce(
    a_logits: torch.Tensor,
    b_logits: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
) -> torch.Tensor:
    return (
        -(target_a * a_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
        + -(target_b * b_logits.log_softmax(dim=-1)).sum(dim=-1).mean()
    ) / 2


def full_enum_action_pairs(*, classes: int, device: str | torch.device) -> torch.Tensor:
    values = torch.arange(classes, device=device)
    grid_a, grid_b = torch.meshgrid(values, values, indexing="ij")
    return torch.stack([grid_a.reshape(-1), grid_b.reshape(-1)], dim=-1)


def action_loss_weights_from_losses(
    losses: torch.Tensor,
    *,
    temperature: float,
    min_probability_floor: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("full-enum action-loss temperature must be positive")
    if min_probability_floor < 0:
        raise ValueError("full-enum min probability floor must be non-negative")
    if losses.ndim != 2:
        raise ValueError("full-enum action losses must have shape [batch, actions]")
    action_count = losses.shape[-1]
    if min_probability_floor * action_count >= 1.0:
        raise ValueError(
            "full-enum min probability floor times action count must be < 1"
        )
    weights = torch.softmax(-losses / temperature, dim=-1)
    if min_probability_floor > 0:
        weights = weights.clamp_min(min_probability_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights


@torch.no_grad()
def score_forced_result_classes_chunked(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("result-boundary target chunk size must be positive")
    if model.calculator_hook is None:
        raise ValueError("result-boundary target requires a calculator hook")
    result_vocab_size = model.cfg.calculator_result_vocab_size
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    try:
        for start in range(0, result_vocab_size, chunk_size):
            forced_classes = torch.arange(
                start,
                min(start + chunk_size, result_vocab_size),
                device=batch.x.device,
            )
            expanded_x = batch.x.repeat_interleave(len(forced_classes), dim=0)
            expanded_y = batch.y.repeat_interleave(len(forced_classes), dim=0)
            expanded_mask = batch.loss_mask.repeat_interleave(
                len(forced_classes), dim=0
            )
            forced = forced_classes.repeat(batch.x.shape[0])
            logits = model(expanded_x, forced_calculator_result_class=forced)
            losses.append(
                masked_cross_entropy_per_example(
                    logits, expanded_y, expanded_mask
                ).reshape(batch.x.shape[0], len(forced_classes))
            )
    finally:
        if was_training:
            model.train()
    return torch.cat(losses, dim=-1)


def result_boundary_target_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    target_mode: str,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_mode not in {"hard_best_result", "soft_result"}:
        raise ValueError(
            "result-boundary target mode must be 'hard_best_result' or 'soft_result'"
        )
    if temperature <= 0:
        raise ValueError("result-boundary target temperature must be positive")
    if min_probability_floor < 0:
        raise ValueError(
            "result-boundary target min probability floor must be non-negative"
        )
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    full_losses = score_forced_result_classes_chunked(
        model, batch, chunk_size=chunk_size
    )
    target_weights = action_loss_weights_from_losses(
        full_losses,
        temperature=temperature,
        min_probability_floor=min_probability_floor,
    )
    best_result = full_losses.argmin(dim=-1)
    if target_mode == "hard_best_result":
        target_loss = torch.nn.functional.cross_entropy(result_logits, best_result)
    else:
        target_loss = -(
            target_weights * result_logits.log_softmax(dim=-1)
        ).sum(dim=-1).mean()

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_result = result_logits.argmax(dim=-1)
    best_losses = full_losses.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    true_losses = full_losses.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    learned_losses = full_losses.gather(1, learned_result.unsqueeze(-1)).squeeze(-1)
    target_entropy = -(
        target_weights * target_weights.clamp_min(1e-12).log()
    ).sum(dim=-1)
    true_result_probability = target_weights.gather(
        1, true_sum.unsqueeze(-1)
    ).squeeze(-1)
    tie_tolerance = 1e-6
    true_result_strictly_beaten = full_losses < (
        true_losses.unsqueeze(-1) - tie_tolerance
    )
    metrics = {
        "result_boundary_target_loss": float(target_loss.item()),
        "result_boundary_target_mode": target_mode,
        "result_boundary_target_temperature": float(temperature),
        "result_boundary_target_min_probability_floor": float(min_probability_floor),
        "result_boundary_target_chunk_size": int(chunk_size),
        "result_boundary_target_best_nll": float(best_losses.mean().item()),
        "result_boundary_target_true_nll": float(true_losses.mean().item()),
        "result_boundary_target_learned_nll": float(learned_losses.mean().item()),
        "result_boundary_target_learned_minus_best_gap": float(
            (learned_losses - best_losses).mean().item()
        ),
        "result_boundary_target_learned_minus_true_gap": float(
            (learned_losses - true_losses).mean().item()
        ),
        "result_boundary_target_hard_best_equals_true_sum": float(
            (best_result == true_sum).float().mean().item()
        ),
        "result_boundary_target_tie_aware_true_result_best_fraction": float(
            (~true_result_strictly_beaten).float().mean().item()
        ),
        "result_boundary_target_true_result_probability": float(
            true_result_probability.mean().item()
        ),
        "result_boundary_target_entropy": float(target_entropy.mean().item()),
        "result_boundary_target_effective_results": float(
            target_entropy.exp().mean().item()
        ),
        "result_boundary_target_learned_best_fraction": float(
            (learned_result == best_result).float().mean().item()
        ),
        "result_boundary_target_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    return target_loss, metrics


def _selected_eq_rows(tensor: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    eq_mask = tokens == EQ_ID
    if not bool(eq_mask.any().item()):
        raise ValueError("boundary feedback requires an '=' token per example")
    eq_pos = eq_mask.float().argmax(dim=-1).long()
    batch_idx = torch.arange(tokens.shape[0], device=tokens.device)
    return tensor[batch_idx, eq_pos]


def boundary_feedback_matrix(
    *,
    model: TinyGPT,
    mode: str,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if model.calculator_hook is None:
        raise ValueError("boundary feedback requires a calculator hook")
    if mode == "output_proj_transpose":
        return model.calculator_hook.output_proj.weight.detach().to(
            device=device,
            dtype=dtype,
        )
    if mode == "direct_random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        matrix = torch.randn(
            model.cfg.n_embd,
            model.cfg.calculator_result_vocab_size,
            generator=generator,
            dtype=torch.float32,
        ) / math.sqrt(model.cfg.n_embd)
        return matrix.to(device=device, dtype=dtype)
    raise ValueError(
        "boundary feedback mode must be 'output_proj_transpose' or 'direct_random'"
    )


def boundary_feedback_alignment_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    feedback_mode: str,
    feedback_seed: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("boundary feedback requires result_space head")
    if model.calculator_hook is None:
        raise ValueError("boundary feedback requires a calculator hook")
    if model.cfg.calculator_bottleneck_mode != "answer_decoder":
        raise ValueError("boundary feedback currently requires answer_decoder bottleneck")

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, diagnostics = model(batch.x, return_diagnostics=True)
    injection = diagnostics["calculator_injection"].detach().requires_grad_(True)
    base_logits = injection.new_zeros(
        (*batch.x.shape, model.cfg.vocab_size),
    )
    logits = model._answer_bottleneck_logits(base_logits, injection, batch.x)
    answer_loss = masked_cross_entropy_per_example(
        logits,
        batch.y,
        batch.loss_mask,
    ).mean()
    answer_loss.backward()
    if injection.grad is None:
        raise RuntimeError("calculator injection gradient was not populated")
    injection_grad = injection.grad.detach()
    feedback_input = _selected_eq_rows(injection_grad, batch.x)
    feedback = feedback_input @ boundary_feedback_matrix(
        model=model,
        mode=feedback_mode,
        seed=feedback_seed,
        device=feedback_input.device,
        dtype=feedback_input.dtype,
    )
    model.zero_grad(set_to_none=True)

    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    objective = (result_logits * feedback.detach()).sum(dim=-1).mean()

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_result = result_logits.argmax(dim=-1)
    result_probs = result_logits.softmax(dim=-1)
    result_entropy = -(
        result_probs * result_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    metrics: dict[str, float | int | str] = {
        "boundary_feedback_loss": float(objective.detach().item()),
        "boundary_feedback_answer_loss": float(answer_loss.detach().item()),
        "boundary_feedback_mode": feedback_mode,
        "boundary_feedback_seed": int(feedback_seed),
        "boundary_feedback_input_grad_l2": float(feedback_input.norm().item()),
        "boundary_feedback_signal_l2": float(feedback.norm().item()),
        "boundary_feedback_signal_abs_mean": float(feedback.abs().mean().item()),
        "boundary_feedback_result_entropy": float(result_entropy.mean().item()),
        "boundary_feedback_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    return objective, metrics


def _answer_loss_injection_gradient(
    model: TinyGPT,
    batch: ArithmeticBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.cfg.calculator_bottleneck_mode != "answer_decoder":
        raise ValueError("shadow feedback currently requires answer_decoder bottleneck")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, diagnostics = model(batch.x, return_diagnostics=True)
    injection = diagnostics["calculator_injection"].detach().requires_grad_(True)
    base_logits = injection.new_zeros(
        (*batch.x.shape, model.cfg.vocab_size),
    )
    logits = model._answer_bottleneck_logits(base_logits, injection, batch.x)
    answer_loss = masked_cross_entropy_per_example(
        logits,
        batch.y,
        batch.loss_mask,
    ).mean()
    answer_loss.backward()
    if injection.grad is None:
        raise RuntimeError("calculator injection gradient was not populated")
    feedback_input = _selected_eq_rows(injection.grad.detach(), batch.x)
    model.zero_grad(set_to_none=True)
    return feedback_input, answer_loss.detach()


def boundary_result_logit_gradient_target(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    target_mode: str,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    target_grad, _best_result, metrics = (
        boundary_result_logit_gradient_target_with_classes(
            model,
            batch,
            num_digits=num_digits,
            target_mode=target_mode,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
            chunk_size=chunk_size,
        )
    )
    return target_grad, metrics


def boundary_result_logit_gradient_target_with_classes(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    target_mode: str,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    if target_mode not in {"hard_best_result", "soft_result"}:
        raise ValueError(
            "shadow feedback target mode must be 'hard_best_result' or 'soft_result'"
        )
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    full_losses = score_forced_result_classes_chunked(
        model, batch, chunk_size=chunk_size
    )
    target_weights = action_loss_weights_from_losses(
        full_losses,
        temperature=temperature,
        min_probability_floor=min_probability_floor,
    )
    best_result = full_losses.argmin(dim=-1)
    if target_mode == "hard_best_result":
        target_loss = torch.nn.functional.cross_entropy(result_logits, best_result)
    else:
        target_loss = -(
            target_weights * result_logits.log_softmax(dim=-1)
        ).sum(dim=-1).mean()
    (target_grad,) = torch.autograd.grad(target_loss, result_logits)

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_result = result_logits.argmax(dim=-1)
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_target_loss": float(target_loss.detach().item()),
        "shadow_feedback_target_mode": target_mode,
        "shadow_feedback_target_grad_l2": float(target_grad.detach().norm().item()),
        "shadow_feedback_target_best_equals_true_sum": float(
            (best_result == true_sum).float().mean().item()
        ),
        "shadow_feedback_target_learned_best_fraction": float(
            (learned_result == best_result).float().mean().item()
        ),
        "shadow_feedback_target_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    return target_grad.detach(), best_result.detach(), metrics


def linear_shadow_feedback_alignment_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    ridge: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    weights, fit_metrics = fit_linear_shadow_feedback_weights(
        model,
        batch,
        num_digits=num_digits,
        ridge=ridge,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    objective, apply_metrics = fixed_linear_shadow_feedback_alignment_loss(
        model,
        batch,
        num_digits=num_digits,
        weights=weights,
        ridge=ridge,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    metrics = dict(fit_metrics)
    metrics.update(apply_metrics)
    return objective, metrics


def fit_linear_shadow_feedback_weights(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    ridge: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if ridge < 0:
        raise ValueError("shadow feedback ridge must be non-negative")
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("shadow feedback requires result_space head")
    if model.calculator_hook is None:
        raise ValueError("shadow feedback requires a calculator hook")

    feedback_input, answer_loss = _answer_loss_injection_gradient(model, batch)
    target_grad, target_metrics = boundary_result_logit_gradient_target(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    ones = torch.ones(
        (feedback_input.shape[0], 1),
        device=feedback_input.device,
        dtype=feedback_input.dtype,
    )
    features = torch.cat([feedback_input, ones], dim=-1)
    target = target_grad.to(device=features.device, dtype=features.dtype)
    xtx = features.transpose(0, 1) @ features
    eye = torch.eye(xtx.shape[0], device=features.device, dtype=features.dtype)
    eye[-1, -1] = 0.0
    rhs = features.transpose(0, 1) @ target
    weights = torch.linalg.solve(xtx + (ridge * eye), rhs)
    predicted_feedback = features @ weights

    target_norm = target.reshape(-1).norm()
    pred_norm = predicted_feedback.reshape(-1).norm()
    denom = target_norm * pred_norm
    fit_cosine = (
        float(torch.dot(target.reshape(-1), predicted_feedback.reshape(-1)).div(denom).item())
        if float(denom.item()) > 0.0
        else float("nan")
    )
    fit_mse = (predicted_feedback - target).pow(2).mean()
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_answer_loss": float(answer_loss.item()),
        "shadow_feedback_ridge": float(ridge),
        "shadow_feedback_input_grad_l2": float(feedback_input.norm().item()),
        "shadow_feedback_predicted_l2": float(predicted_feedback.norm().item()),
        "shadow_feedback_fit_mse": float(fit_mse.item()),
        "shadow_feedback_fit_cosine": fit_cosine,
        "shadow_feedback_weight_l2": float(weights.norm().item()),
    }
    metrics.update(target_metrics)
    return weights.detach(), metrics


def fixed_linear_shadow_feedback_alignment_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    weights: torch.Tensor,
    ridge: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("shadow feedback requires result_space head")
    feedback_input, answer_loss = _answer_loss_injection_gradient(model, batch)
    ones = torch.ones(
        (feedback_input.shape[0], 1),
        device=feedback_input.device,
        dtype=feedback_input.dtype,
    )
    features = torch.cat([feedback_input, ones], dim=-1)
    predicted_feedback = features @ weights.to(
        device=features.device,
        dtype=features.dtype,
    )

    model.zero_grad(set_to_none=True)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    objective = (result_logits * predicted_feedback.detach()).sum()
    learned_result = result_logits.argmax(dim=-1)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_loss": float(objective.detach().item()),
        "shadow_feedback_answer_loss": float(answer_loss.item()),
        "shadow_feedback_ridge": float(ridge),
        "shadow_feedback_target_mode": result_boundary_target_mode,
        "shadow_feedback_target_temperature": float(
            result_boundary_target_temperature
        ),
        "shadow_feedback_target_min_probability_floor": float(
            result_boundary_target_min_probability_floor
        ),
        "shadow_feedback_target_chunk_size": int(
            result_boundary_target_chunk_size
        ),
        "shadow_feedback_input_grad_l2": float(feedback_input.norm().item()),
        "shadow_feedback_predicted_l2": float(predicted_feedback.norm().item()),
        "shadow_feedback_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    return objective, metrics


class ShadowFeedbackMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("shadow feedback hidden size must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("shadow feedback dropout must be in [0, 1)")
        layers: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.GELU(),
        ]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_size, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def shadow_feedback_mlp_features(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    feature_mode: str,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("shadow feedback requires result_space head")
    feedback_input, answer_loss = _answer_loss_injection_gradient(model, batch)
    with torch.no_grad():
        result_logits, result_input, _ = calculator_read_result_logits_and_input(
            model,
            batch,
        )
    batch_scale = float(batch.x.shape[0])
    scaled_feedback_input = feedback_input.detach() * batch_scale
    result_logits = result_logits.detach().to(
        device=feedback_input.device,
        dtype=feedback_input.dtype,
    )
    result_input = result_input.detach().to(
        device=feedback_input.device,
        dtype=feedback_input.dtype,
    )
    result_probs = result_logits.softmax(dim=-1)
    result_log_probs = result_logits.log_softmax(dim=-1)
    result_entropy = -(
        result_probs * result_log_probs
    ).sum(dim=-1, keepdim=True)
    output_jacobian_scores = (
        scaled_feedback_input
        @ model.calculator_hook.output_proj.weight[
            :, : model.cfg.calculator_result_vocab_size
        ].to(device=feedback_input.device, dtype=feedback_input.dtype)
    )
    if feature_mode == "injection_grad_logits":
        feature_chunks = [scaled_feedback_input, result_logits]
    elif feature_mode == "injection_grad_logits_output_jacobian":
        feature_chunks = [
            scaled_feedback_input,
            result_logits,
            output_jacobian_scores.detach(),
        ]
    elif feature_mode == "injection_grad_logits_result_input":
        feature_chunks = [scaled_feedback_input, result_logits, result_input]
    elif feature_mode == "injection_grad_policy_state":
        feature_chunks = [
            scaled_feedback_input,
            result_logits,
            result_probs,
            result_log_probs,
            result_entropy,
        ]
    else:
        raise ValueError(
            "shadow feedback feature mode must be injection_grad_logits "
            "injection_grad_logits_output_jacobian, "
            "injection_grad_logits_result_input, or injection_grad_policy_state"
        )
    features = torch.cat(feature_chunks, dim=-1)
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_feature_mode": feature_mode,
        "shadow_feedback_answer_loss": float(answer_loss.item()),
        "shadow_feedback_input_grad_l2": float(scaled_feedback_input.norm().item()),
        "shadow_feedback_feature_l2": float(features.norm().item()),
        "shadow_feedback_feature_dim": int(features.shape[-1]),
        "shadow_feedback_feature_input_grad_l2": float(
            scaled_feedback_input.norm().item()
        ),
        "shadow_feedback_feature_logits_l2": float(result_logits.norm().item()),
        "shadow_feedback_feature_output_jacobian_l2": float(
            output_jacobian_scores.norm().item()
        ),
        "shadow_feedback_feature_result_input_l2": float(result_input.norm().item()),
        "shadow_feedback_feature_probs_l2": float(result_probs.norm().item()),
        "shadow_feedback_feature_log_probs_l2": float(
            result_log_probs.norm().item()
        ),
        "shadow_feedback_feature_entropy_mean": float(result_entropy.mean().item()),
        "shadow_feedback_feature_entropy_l2": float(result_entropy.norm().item()),
    }
    return features, metrics


def shadow_feedback_mlp_examples(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    feature_mode: str,
    target_transform: str,
    target_prototypes: torch.Tensor | None = None,
    target_prototype_counts: torch.Tensor | None = None,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    features, metrics = shadow_feedback_mlp_features(
        model,
        batch,
        feature_mode=feature_mode,
    )
    target_grad, target_class_ids, target_metrics = (
        boundary_result_logit_gradient_target_with_classes(
            model,
            batch,
            num_digits=num_digits,
            target_mode=result_boundary_target_mode,
            temperature=result_boundary_target_temperature,
            min_probability_floor=result_boundary_target_min_probability_floor,
            chunk_size=result_boundary_target_chunk_size,
        )
    )
    target = target_grad.detach().to(device=features.device, dtype=features.dtype)
    target = target * float(batch.x.shape[0])
    target, target_transform_metrics = transform_shadow_feedback_target(
        target,
        mode=target_transform,
        class_ids=target_class_ids,
        prototypes=target_prototypes,
        prototype_counts=target_prototype_counts,
    )
    metrics.update(target_metrics)
    metrics.update(target_transform_metrics)
    return features, target, metrics


def transform_shadow_feedback_target(
    target: torch.Tensor,
    *,
    mode: str,
    class_ids: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    prototype_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if mode == "none":
        return target, {
            "shadow_feedback_target_transform": mode,
            "shadow_feedback_target_transform_epsilon": (
                SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS
            ),
            "shadow_feedback_target_transform_clamped_count": 0,
        }
    if mode == "fit_result_prototype":
        if class_ids is None or prototypes is None or prototype_counts is None:
            raise ValueError(
                "fit_result_prototype target transform requires class ids, "
                "prototypes, and prototype counts"
            )
        class_ids = class_ids.to(device=target.device, dtype=torch.long)
        prototypes = prototypes.to(device=target.device, dtype=target.dtype)
        prototype_counts = prototype_counts.to(device=target.device)
        transformed = prototypes.index_select(0, class_ids)
        missing_mask = prototype_counts.index_select(0, class_ids) <= 0
        if bool(missing_mask.any().item()):
            transformed = transformed.clone()
            transformed[missing_mask] = target[missing_mask]
        return transformed, {
            "shadow_feedback_target_transform": mode,
            "shadow_feedback_target_transform_epsilon": (
                SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS
            ),
            "shadow_feedback_target_transform_missing_class_examples": int(
                missing_mask.sum().item()
            ),
            "shadow_feedback_target_transform_nonempty_classes": int(
                (prototype_counts > 0).sum().item()
            ),
            "shadow_feedback_target_transform_class_count_min": int(
                prototype_counts[prototype_counts > 0].min().item()
            )
            if bool((prototype_counts > 0).any().item())
            else 0,
            "shadow_feedback_target_transform_class_count_max": int(
                prototype_counts.max().item()
            ),
            "shadow_feedback_transformed_target_l2": float(transformed.norm().item()),
        }
    if mode != "unit_norm_per_example":
        raise ValueError(
            "shadow feedback target transform must be none, "
            "unit_norm_per_example, or fit_result_prototype"
        )
    raw_norm = target.norm(dim=-1, keepdim=True)
    clamped_count = int(
        (raw_norm < SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS).sum().item()
    )
    scale = raw_norm.clamp_min(SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS)
    transformed = target / scale
    return transformed, {
        "shadow_feedback_target_transform": mode,
        "shadow_feedback_target_transform_epsilon": (
            SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS
        ),
        "shadow_feedback_target_transform_raw_norm_min": float(raw_norm.min().item()),
        "shadow_feedback_target_transform_raw_norm_median": float(
            raw_norm.median().item()
        ),
        "shadow_feedback_target_transform_raw_norm_max": float(raw_norm.max().item()),
        "shadow_feedback_target_transform_raw_norm_mean": float(raw_norm.mean().item()),
        "shadow_feedback_target_transform_clamped_count": clamped_count,
        "shadow_feedback_transformed_target_l2": float(transformed.norm().item()),
    }


def fit_shadow_feedback_target_prototypes(
    target: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    if target.ndim != 2:
        raise ValueError("shadow feedback prototype target must be rank 2")
    if class_ids.ndim != 1 or class_ids.shape[0] != target.shape[0]:
        raise ValueError("shadow feedback prototype class ids must match target rows")
    if num_classes < 1:
        raise ValueError("shadow feedback prototype class count must be positive")
    class_ids = class_ids.to(device=target.device, dtype=torch.long)
    prototypes = torch.zeros(
        num_classes,
        target.shape[-1],
        device=target.device,
        dtype=target.dtype,
    )
    counts = torch.zeros(num_classes, device=target.device, dtype=target.dtype)
    prototypes.index_add_(0, class_ids, target)
    counts.index_add_(0, class_ids, torch.ones_like(class_ids, dtype=target.dtype))
    nonempty = counts > 0
    prototypes[nonempty] = prototypes[nonempty] / counts[nonempty].unsqueeze(-1)
    count_int = counts.to(dtype=torch.long)
    return prototypes.detach(), count_int.detach(), {
        "shadow_feedback_target_transform": "fit_result_prototype",
        "shadow_feedback_target_prototype_classes": int(num_classes),
        "shadow_feedback_target_prototype_nonempty_classes": int(
            nonempty.sum().item()
        ),
        "shadow_feedback_target_prototype_empty_classes": int(
            (~nonempty).sum().item()
        ),
        "shadow_feedback_target_prototype_count_min": int(
            count_int[nonempty].min().item()
        )
        if bool(nonempty.any().item())
        else 0,
        "shadow_feedback_target_prototype_count_max": int(count_int.max().item()),
        "shadow_feedback_target_prototype_l2": float(prototypes.norm().item()),
    }


def fit_shadow_feedback_target_normalizer(
    target: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float | int | str]]:
    if mode == "none":
        return None, None, {
            "shadow_feedback_target_normalization": mode,
            "shadow_feedback_target_normalization_epsilon": (
                SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS
            ),
        }
    if mode != "fit_zscore_per_result":
        raise ValueError(
            "shadow feedback target normalization must be none or fit_zscore_per_result"
        )
    mean = target.mean(dim=0, keepdim=True).detach()
    raw_scale = target.std(dim=0, unbiased=False, keepdim=True).detach()
    clamped_count = int(
        (raw_scale < SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS).sum().item()
    )
    scale = raw_scale.clamp_min(SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS)
    normalized = (target - mean) / scale
    return mean, scale, {
        "shadow_feedback_target_normalization": mode,
        "shadow_feedback_target_normalization_epsilon": (
            SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS
        ),
        "shadow_feedback_target_mean_l2": float(mean.norm().item()),
        "shadow_feedback_target_scale_min": float(scale.min().item()),
        "shadow_feedback_target_scale_median": float(scale.median().item()),
        "shadow_feedback_target_scale_max": float(scale.max().item()),
        "shadow_feedback_target_scale_mean": float(scale.mean().item()),
        "shadow_feedback_target_scale_clamped_count": clamped_count,
        "shadow_feedback_normalized_target_l2": float(normalized.norm().item()),
    }


def normalize_shadow_feedback_target(
    target: torch.Tensor,
    *,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or scale is None:
        return target
    return (target - mean.to(device=target.device, dtype=target.dtype)) / scale.to(
        device=target.device,
        dtype=target.dtype,
    )


def denormalize_shadow_feedback_prediction(
    prediction: torch.Tensor,
    *,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or scale is None:
        return prediction
    return (
        prediction
        * scale.to(device=prediction.device, dtype=prediction.dtype)
        + mean.to(device=prediction.device, dtype=prediction.dtype)
    )


def fit_shadow_feedback_feature_normalizer(
    features: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float | int | str]]:
    if mode == "none":
        return None, None, {
            "shadow_feedback_feature_normalization": mode,
            "shadow_feedback_feature_normalization_epsilon": (
                SHADOW_FEEDBACK_FEATURE_NORMALIZATION_EPS
            ),
        }
    if mode != "fit_zscore_per_feature":
        raise ValueError(
            "shadow feedback feature normalization must be none or "
            "fit_zscore_per_feature"
        )
    mean = features.mean(dim=0, keepdim=True).detach()
    raw_scale = features.std(dim=0, unbiased=False, keepdim=True).detach()
    clamped_count = int(
        (raw_scale < SHADOW_FEEDBACK_FEATURE_NORMALIZATION_EPS).sum().item()
    )
    scale = raw_scale.clamp_min(SHADOW_FEEDBACK_FEATURE_NORMALIZATION_EPS)
    normalized = (features - mean) / scale
    return mean, scale, {
        "shadow_feedback_feature_normalization": mode,
        "shadow_feedback_feature_normalization_epsilon": (
            SHADOW_FEEDBACK_FEATURE_NORMALIZATION_EPS
        ),
        "shadow_feedback_feature_mean_l2": float(mean.norm().item()),
        "shadow_feedback_feature_scale_min": float(scale.min().item()),
        "shadow_feedback_feature_scale_median": float(scale.median().item()),
        "shadow_feedback_feature_scale_max": float(scale.max().item()),
        "shadow_feedback_feature_scale_mean": float(scale.mean().item()),
        "shadow_feedback_feature_scale_clamped_count": clamped_count,
        "shadow_feedback_normalized_feature_l2": float(normalized.norm().item()),
    }


def normalize_shadow_feedback_features(
    features: torch.Tensor,
    *,
    mean: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if mean is None or scale is None:
        return features
    return (features - mean.to(device=features.device, dtype=features.dtype)) / scale.to(
        device=features.device,
        dtype=features.dtype,
    )


def shadow_feedback_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    mse = torch.nn.functional.mse_loss(predicted, target)
    if mode == "mse":
        return mse
    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    denom = pred_flat.norm() * target_flat.norm()
    if float(denom.detach().item()) == 0.0:
        cosine_loss = torch.ones((), device=predicted.device, dtype=predicted.dtype)
    else:
        cosine_loss = 1.0 - torch.dot(pred_flat, target_flat).div(denom)
    if mode == "cosine":
        return cosine_loss
    if mode == "mse_plus_cosine":
        return mse + cosine_loss
    raise ValueError("shadow feedback loss mode must be mse, cosine, or mse_plus_cosine")


def shadow_feedback_feature_dim(model: TinyGPT, *, feature_mode: str) -> int:
    if feature_mode == "injection_grad_logits":
        return model.cfg.n_embd + model.cfg.calculator_result_vocab_size
    if feature_mode == "injection_grad_logits_output_jacobian":
        return model.cfg.n_embd + (2 * model.cfg.calculator_result_vocab_size)
    result_input_dim = (
        2 * model.cfg.calculator_read_span_width * model.cfg.n_embd
        if model.cfg.calculator_read_position == "operand_spans"
        else 2 * model.cfg.n_embd
    )
    if feature_mode == "injection_grad_logits_result_input":
        return (
            model.cfg.n_embd
            + model.cfg.calculator_result_vocab_size
            + result_input_dim
        )
    if feature_mode == "injection_grad_policy_state":
        return model.cfg.n_embd + (3 * model.cfg.calculator_result_vocab_size) + 1
    raise ValueError(
        "shadow feedback feature mode must be injection_grad_logits "
        "injection_grad_logits_output_jacobian, "
        "injection_grad_logits_result_input, or injection_grad_policy_state"
    )


def online_shadow_feedback_alignment_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    shadow_module: ShadowFeedbackMLP,
    num_digits: int,
    feature_mode: str,
    target_transform: str,
    target_prototypes: torch.Tensor | None = None,
    target_prototype_counts: torch.Tensor | None = None,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
    target_mean: torch.Tensor | None = None,
    target_scale: torch.Tensor | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    features, target, example_metrics = shadow_feedback_mlp_examples(
        model,
        batch,
        num_digits=num_digits,
        feature_mode=feature_mode,
        target_transform=target_transform,
        target_prototypes=target_prototypes,
        target_prototype_counts=target_prototype_counts,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    normalized_features = normalize_shadow_feedback_features(
        features,
        mean=feature_mean,
        scale=feature_scale,
    )
    shadow_module.eval()
    with torch.no_grad():
        normalized_prediction = shadow_module(normalized_features).detach()
        predicted_feedback = denormalize_shadow_feedback_prediction(
            normalized_prediction,
            mean=target_mean,
            scale=target_scale,
        )

    model.zero_grad(set_to_none=True)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    objective = (result_logits * predicted_feedback).sum(dim=-1).mean()
    learned_result = result_logits.argmax(dim=-1)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    pred_norm = predicted_feedback.reshape(-1).norm()
    target_norm = target.reshape(-1).norm()
    denom = pred_norm * target_norm
    pred_cosine = (
        float(
            torch.dot(predicted_feedback.reshape(-1), target.reshape(-1))
            .div(denom)
            .item()
        )
        if float(denom.item()) > 0.0
        else float("nan")
    )
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_mode": "online_mlp",
        "shadow_feedback_loss": float(objective.detach().item()),
        "shadow_feedback_normalized_predicted_l2": float(
            normalized_prediction.norm().item()
        ),
        "shadow_feedback_predicted_l2": float(predicted_feedback.norm().item()),
        "shadow_feedback_normalized_feature_l2": float(
            normalized_features.norm().item()
        ),
        "shadow_feedback_prediction_mse": float(
            (predicted_feedback - target).pow(2).mean().item()
        ),
        "shadow_feedback_prediction_cosine": pred_cosine,
        "shadow_feedback_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    metrics.update(example_metrics)
    return objective, metrics


def online_shadow_feedback_fixed_module_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    shadow_module: ShadowFeedbackMLP,
    num_digits: int,
    feature_mode: str,
    target_mean: torch.Tensor | None = None,
    target_scale: torch.Tensor | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_scale: torch.Tensor | None = None,
    max_predicted_norm: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    if max_predicted_norm < 0:
        raise ValueError("shadow feedback apply max norm must be non-negative")
    features, feature_metrics = shadow_feedback_mlp_features(
        model,
        batch,
        feature_mode=feature_mode,
    )
    normalized_features = normalize_shadow_feedback_features(
        features,
        mean=feature_mean,
        scale=feature_scale,
    )
    shadow_module.eval()
    with torch.no_grad():
        normalized_prediction = shadow_module(normalized_features).detach()
        predicted_feedback = denormalize_shadow_feedback_prediction(
            normalized_prediction,
            mean=target_mean,
            scale=target_scale,
        )
        unclamped_predicted_feedback_l2 = float(predicted_feedback.norm().item())
        predicted_feedback_scale = 1.0
        if max_predicted_norm > 0:
            predicted_norm = predicted_feedback.norm()
            if float(predicted_norm.item()) > max_predicted_norm:
                scale = predicted_norm.new_tensor(max_predicted_norm) / predicted_norm
                predicted_feedback = predicted_feedback * scale
                predicted_feedback_scale = float(scale.item())

    model.zero_grad(set_to_none=True)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    objective = (result_logits * predicted_feedback).sum(dim=-1).mean()
    learned_result = result_logits.argmax(dim=-1)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    metrics: dict[str, float | int | str] = {
        "shadow_feedback_mode": "online_mlp",
        "shadow_feedback_loss": float(objective.detach().item()),
        "shadow_feedback_normalized_predicted_l2": float(
            normalized_prediction.norm().item()
        ),
        "shadow_feedback_predicted_l2": float(predicted_feedback.norm().item()),
        "shadow_feedback_unclamped_predicted_l2": unclamped_predicted_feedback_l2,
        "shadow_feedback_apply_max_norm": float(max_predicted_norm),
        "shadow_feedback_apply_norm_scale": predicted_feedback_scale,
        "shadow_feedback_normalized_feature_l2": float(
            normalized_features.norm().item()
        ),
        "shadow_feedback_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    metrics.update(feature_metrics)
    return objective, metrics


def online_shadow_feedback_validation_gradient_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    shadow_module: ShadowFeedbackMLP,
    num_digits: int,
    feature_mode: str,
    target_transform: str,
    target_prototypes: torch.Tensor | None = None,
    target_prototype_counts: torch.Tensor | None = None,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
    target_mean: torch.Tensor | None = None,
    target_scale: torch.Tensor | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_scale: torch.Tensor | None = None,
    norm_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if norm_weight < 0:
        raise ValueError(
            "shadow feedback validation gradient norm weight must be non-negative"
        )
    gradient_groups = {"calculator_hook.result_proj", "upstream"}
    named_params = named_parameters_for_gradient_groups(model, gradient_groups)
    if not named_params:
        raise ValueError(
            "shadow feedback validation gradient loss found no trainable parameters"
        )

    features, _target, _example_metrics = shadow_feedback_mlp_examples(
        model,
        batch,
        num_digits=num_digits,
        feature_mode=feature_mode,
        target_transform=target_transform,
        target_prototypes=target_prototypes,
        target_prototype_counts=target_prototype_counts,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    normalized_prediction = shadow_module(
        normalize_shadow_feedback_features(
            features,
            mean=feature_mean,
            scale=feature_scale,
        )
    )
    predicted_feedback = denormalize_shadow_feedback_prediction(
        normalized_prediction,
        mean=target_mean,
        scale=target_scale,
    )

    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    shadow_objective = (result_logits * predicted_feedback).sum(dim=-1).mean()
    shadow_grads = differentiable_group_gradients(
        shadow_objective,
        named_params,
        create_graph=True,
    )
    boundary_loss, _boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_grads = {
        group: grad.detach()
        for group, grad in differentiable_group_gradients(
            boundary_loss,
            named_params,
            create_graph=False,
        ).items()
    }
    result_loss = differentiable_gradient_cosine_loss(
        shadow_grads["calculator_hook.result_proj"],
        boundary_grads["calculator_hook.result_proj"],
    )
    upstream_loss = differentiable_gradient_cosine_loss(
        shadow_grads["upstream"],
        boundary_grads["upstream"],
    )
    result_norm_loss = differentiable_relative_norm_loss(
        shadow_grads["calculator_hook.result_proj"],
        boundary_grads["calculator_hook.result_proj"],
    )
    upstream_norm_loss = differentiable_relative_norm_loss(
        shadow_grads["upstream"],
        boundary_grads["upstream"],
    )
    loss = 0.5 * (result_loss + upstream_loss)
    if norm_weight > 0:
        loss = loss + (0.5 * norm_weight * (result_norm_loss + upstream_norm_loss))
    result_relative_norm = (
        shadow_grads["calculator_hook.result_proj"].detach().norm()
        / boundary_grads["calculator_hook.result_proj"].norm()
    )
    upstream_relative_norm = (
        shadow_grads["upstream"].detach().norm() / boundary_grads["upstream"].norm()
    )
    metrics = {
        "shadow_feedback_validation_gradient_objective": float(loss.detach().item()),
        "shadow_feedback_validation_gradient_result_objective": float(
            result_loss.detach().item()
        ),
        "shadow_feedback_validation_gradient_upstream_objective": float(
            upstream_loss.detach().item()
        ),
        "shadow_feedback_validation_gradient_result_norm_objective": float(
            result_norm_loss.detach().item()
        ),
        "shadow_feedback_validation_gradient_upstream_norm_objective": float(
            upstream_norm_loss.detach().item()
        ),
        "shadow_feedback_validation_gradient_result_relative_norm": float(
            result_relative_norm.detach().item()
        ),
        "shadow_feedback_validation_gradient_upstream_relative_norm": float(
            upstream_relative_norm.detach().item()
        ),
    }
    return loss, metrics


def parameter_group_name(name: str) -> str:
    if name.startswith("calculator_hook.result_proj."):
        return "calculator_hook.result_proj"
    if name.startswith("calculator_hook.input_proj."):
        return "calculator_hook.input_proj"
    if name.startswith("calculator_hook.pair_proj."):
        return "calculator_hook.pair_proj"
    if name.startswith(SEMANTIC_DECODER_CHECKPOINT_PREFIXES):
        return "semantic_decoder"
    return "upstream"


def flattened_group_gradients(model: TinyGPT) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        group = parameter_group_name(name)
        if param.grad is None:
            grad = torch.zeros_like(param.detach()).reshape(-1)
        else:
            grad = param.grad.detach().reshape(-1)
        chunks.setdefault(group, []).append(grad.cpu())
    return {
        group: torch.cat(group_chunks) if group_chunks else torch.tensor([])
        for group, group_chunks in chunks.items()
    }


def gradient_l2(gradients: dict[str, torch.Tensor], group: str) -> float:
    grad = gradients.get(group)
    if grad is None or grad.numel() == 0:
        return 0.0
    return float(grad.norm().item())


def gradient_cosine(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], group: str
) -> float:
    left_grad = left.get(group)
    right_grad = right.get(group)
    if left_grad is None or right_grad is None or left_grad.numel() == 0:
        return float("nan")
    denom = left_grad.norm() * right_grad.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float(torch.dot(left_grad, right_grad).div(denom).item())


def differentiable_gradient_cosine_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    denom = predicted.norm() * target.norm()
    if float(denom.detach().item()) == 0.0:
        return torch.ones((), device=predicted.device, dtype=predicted.dtype)
    return 1.0 - torch.dot(predicted, target).div(denom)


def differentiable_relative_norm_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    eps = torch.tensor(
        SHADOW_FEEDBACK_TARGET_NORMALIZATION_EPS,
        device=predicted.device,
        dtype=predicted.dtype,
    )
    relative_norm = (predicted.norm() + eps) / (target.norm() + eps)
    return relative_norm.log().pow(2)


def named_parameters_for_gradient_groups(
    model: TinyGPT,
    groups: set[str],
) -> list[tuple[str, torch.nn.Parameter]]:
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and parameter_group_name(name) in groups
    ]


def differentiable_group_gradients(
    loss: torch.Tensor,
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    params = [param for _name, param in named_params]
    grads = torch.autograd.grad(
        loss,
        params,
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    chunks: dict[str, list[torch.Tensor]] = {}
    for (name, param), grad in zip(named_params, grads):
        group = parameter_group_name(name)
        if grad is None:
            flat_grad = torch.zeros_like(param).reshape(-1)
        else:
            flat_grad = grad.reshape(-1)
        chunks.setdefault(group, []).append(flat_grad)
    return {
        group: torch.cat(group_chunks) if group_chunks else loss.new_empty(0)
        for group, group_chunks in chunks.items()
    }


@contextmanager
def temporary_calculator_estimator(model: TinyGPT, estimator: str):
    old_cfg_estimator = model.cfg.calculator_estimator
    old_hook_estimator = (
        model.calculator_hook.estimator
        if model.calculator_hook is not None
        else None
    )
    model.cfg.calculator_estimator = estimator
    if model.calculator_hook is not None:
        model.calculator_hook.estimator = estimator
    try:
        yield
    finally:
        model.cfg.calculator_estimator = old_cfg_estimator
        if model.calculator_hook is not None and old_hook_estimator is not None:
            model.calculator_hook.estimator = old_hook_estimator


def run_reinforce_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    baseline_mode: str,
    num_samples_per_prompt: int,
    global_baseline: float | None,
    entropy_weight: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> dict[str, float | int | str]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("REINFORCE gradient diagnostic requires result_space head")
    model.train()
    baseline_summaries: dict[str, dict[str, float]] = {}
    for mode in ["global_ema", "per_prompt_mean", "leave_one_out"]:
        if mode == "leave_one_out" and num_samples_per_prompt < 2:
            continue
        model.zero_grad(set_to_none=True)
        _, _, mode_metrics = reinforce_policy_gradient_loss(
            model,
            batch,
            num_digits=num_digits,
            baseline_mode=mode,
            num_samples_per_prompt=num_samples_per_prompt,
            global_baseline=global_baseline,
            entropy_weight=entropy_weight,
        )
        baseline_summaries[mode] = {
            "advantage_mean": mode_metrics["policy_advantage_mean"],
            "advantage_std": mode_metrics["policy_advantage_std"],
            "policy_objective": mode_metrics["policy_objective"],
            "result_entropy": mode_metrics["result_entropy"],
            "sampled_result_accuracy": mode_metrics["sampled_result_accuracy"],
        }

    model.zero_grad(set_to_none=True)
    pg_objective, pg_answer_loss, pg_metrics = reinforce_policy_gradient_loss(
        model,
        batch,
        num_digits=num_digits,
        baseline_mode=baseline_mode,
        num_samples_per_prompt=num_samples_per_prompt,
        global_baseline=global_baseline,
        entropy_weight=entropy_weight,
    )
    pg_objective.backward()
    pg_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    result_pg_norm = gradient_l2(pg_grads, "calculator_hook.result_proj")
    result_boundary_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    upstream_pg_norm = gradient_l2(pg_grads, "upstream")
    upstream_boundary_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "reinforce_result_space_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "reinforce_baseline_mode": baseline_mode,
        "reinforce_num_samples_per_prompt": int(num_samples_per_prompt),
        "answer_loss": float(pg_answer_loss.detach().item()),
        "policy_gradient_objective": pg_metrics["policy_objective"],
        "policy_loss": pg_metrics["policy_loss"],
        "advantage_mean": pg_metrics["policy_advantage_mean"],
        "advantage_std": pg_metrics["policy_advantage_std"],
        "sampled_logp": pg_metrics["sampled_logp"],
        "result_entropy": pg_metrics["result_entropy"],
        "sampled_result_accuracy": pg_metrics["sampled_result_accuracy"],
        "pg_result_proj_grad_l2": result_pg_norm,
        "pg_upstream_grad_l2": upstream_pg_norm,
        "pg_semantic_decoder_grad_l2": gradient_l2(pg_grads, "semantic_decoder"),
        "boundary_result_proj_grad_l2": result_boundary_norm,
        "boundary_upstream_grad_l2": upstream_boundary_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads, "semantic_decoder"
        ),
        "pg_vs_boundary_result_proj_cosine": gradient_cosine(
            pg_grads, boundary_grads, "calculator_hook.result_proj"
        ),
        "pg_vs_boundary_upstream_cosine": gradient_cosine(
            pg_grads, boundary_grads, "upstream"
        ),
        "pg_vs_boundary_result_proj_relative_norm": (
            result_pg_norm / result_boundary_norm
            if result_boundary_norm > 0
            else float("nan")
        ),
        "pg_vs_boundary_upstream_relative_norm": (
            upstream_pg_norm / upstream_boundary_norm
            if upstream_boundary_norm > 0
            else float("nan")
        ),
        "global_ema_advantage_std": baseline_summaries.get(
            "global_ema", {}
        ).get("advantage_std", float("nan")),
        "per_prompt_mean_advantage_std": baseline_summaries.get(
            "per_prompt_mean", {}
        ).get("advantage_std", float("nan")),
        "leave_one_out_advantage_std": baseline_summaries.get(
            "leave_one_out", {}
        ).get("advantage_std", float("nan")),
    }
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def run_expected_answer_loss_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    policy_temperature: float,
    cost_normalization: str,
    entropy_weight: float,
    expected_answer_loss_chunk_size: int,
    baseline_mode: str,
    num_samples_per_prompt: int,
    global_baseline: float | None,
    reinforce_entropy_weight: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> dict[str, float | int | str]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError(
            "expected answer-loss gradient diagnostic requires result_space head"
        )
    model.train()
    baseline_summaries: dict[str, dict[str, float]] = {}
    with temporary_calculator_estimator(model, "reinforce"):
        for mode in ["global_ema", "per_prompt_mean", "leave_one_out"]:
            if mode == "leave_one_out" and num_samples_per_prompt < 2:
                continue
            model.zero_grad(set_to_none=True)
            _, _, mode_metrics = reinforce_policy_gradient_loss(
                model,
                batch,
                num_digits=num_digits,
                baseline_mode=mode,
                num_samples_per_prompt=num_samples_per_prompt,
                global_baseline=global_baseline,
                entropy_weight=reinforce_entropy_weight,
            )
            baseline_summaries[mode] = {
                "advantage_mean": mode_metrics["policy_advantage_mean"],
                "advantage_std": mode_metrics["policy_advantage_std"],
                "policy_objective": mode_metrics["policy_objective"],
                "result_entropy": mode_metrics["result_entropy"],
                "sampled_result_accuracy": mode_metrics["sampled_result_accuracy"],
            }

    model.zero_grad(set_to_none=True)
    exact_objective, exact_metrics = full_enum_expected_answer_loss(
        model,
        batch,
        num_digits=num_digits,
        policy_temperature=policy_temperature,
        cost_normalization=cost_normalization,
        entropy_weight=entropy_weight,
        chunk_size=expected_answer_loss_chunk_size,
    )
    exact_objective.backward()
    exact_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    with temporary_calculator_estimator(model, "reinforce"):
        pg_objective, pg_answer_loss, pg_metrics = reinforce_policy_gradient_loss(
            model,
            batch,
            num_digits=num_digits,
            baseline_mode=baseline_mode,
            num_samples_per_prompt=num_samples_per_prompt,
            global_baseline=global_baseline,
            entropy_weight=reinforce_entropy_weight,
        )
    pg_objective.backward()
    pg_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    exact_result_norm = gradient_l2(exact_grads, "calculator_hook.result_proj")
    exact_upstream_norm = gradient_l2(exact_grads, "upstream")
    pg_result_norm = gradient_l2(pg_grads, "calculator_hook.result_proj")
    pg_upstream_norm = gradient_l2(pg_grads, "upstream")
    boundary_result_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    boundary_upstream_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "result_space_expected_answer_loss_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "expected_answer_loss_policy_temperature": float(policy_temperature),
        "expected_answer_loss_cost_normalization": cost_normalization,
        "expected_answer_loss_entropy_weight": float(entropy_weight),
        "expected_answer_loss_chunk_size": int(expected_answer_loss_chunk_size),
        "exact_expected_answer_loss_objective": float(
            exact_objective.detach().item()
        ),
        "exact_result_proj_grad_l2": exact_result_norm,
        "exact_upstream_grad_l2": exact_upstream_norm,
        "exact_semantic_decoder_grad_l2": gradient_l2(
            exact_grads, "semantic_decoder"
        ),
        "reinforce_baseline_mode": baseline_mode,
        "reinforce_num_samples_per_prompt": int(num_samples_per_prompt),
        "answer_loss": float(pg_answer_loss.detach().item()),
        "policy_gradient_objective": pg_metrics["policy_objective"],
        "policy_loss": pg_metrics["policy_loss"],
        "advantage_mean": pg_metrics["policy_advantage_mean"],
        "advantage_std": pg_metrics["policy_advantage_std"],
        "sampled_logp": pg_metrics["sampled_logp"],
        "result_entropy": pg_metrics["result_entropy"],
        "sampled_result_accuracy": pg_metrics["sampled_result_accuracy"],
        "pg_result_proj_grad_l2": pg_result_norm,
        "pg_upstream_grad_l2": pg_upstream_norm,
        "pg_semantic_decoder_grad_l2": gradient_l2(pg_grads, "semantic_decoder"),
        "boundary_result_proj_grad_l2": boundary_result_norm,
        "boundary_upstream_grad_l2": boundary_upstream_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads, "semantic_decoder"
        ),
        "exact_vs_boundary_result_proj_cosine": gradient_cosine(
            exact_grads, boundary_grads, "calculator_hook.result_proj"
        ),
        "exact_vs_boundary_upstream_cosine": gradient_cosine(
            exact_grads, boundary_grads, "upstream"
        ),
        "pg_vs_exact_result_proj_cosine": gradient_cosine(
            pg_grads, exact_grads, "calculator_hook.result_proj"
        ),
        "pg_vs_exact_upstream_cosine": gradient_cosine(
            pg_grads, exact_grads, "upstream"
        ),
        "pg_vs_boundary_result_proj_cosine": gradient_cosine(
            pg_grads, boundary_grads, "calculator_hook.result_proj"
        ),
        "pg_vs_boundary_upstream_cosine": gradient_cosine(
            pg_grads, boundary_grads, "upstream"
        ),
        "exact_vs_boundary_result_proj_relative_norm": (
            exact_result_norm / boundary_result_norm
            if boundary_result_norm > 0
            else float("nan")
        ),
        "exact_vs_boundary_upstream_relative_norm": (
            exact_upstream_norm / boundary_upstream_norm
            if boundary_upstream_norm > 0
            else float("nan")
        ),
        "pg_vs_exact_result_proj_relative_norm": (
            pg_result_norm / exact_result_norm
            if exact_result_norm > 0
            else float("nan")
        ),
        "pg_vs_exact_upstream_relative_norm": (
            pg_upstream_norm / exact_upstream_norm
            if exact_upstream_norm > 0
            else float("nan")
        ),
        "global_ema_advantage_std": baseline_summaries.get(
            "global_ema", {}
        ).get("advantage_std", float("nan")),
        "per_prompt_mean_advantage_std": baseline_summaries.get(
            "per_prompt_mean", {}
        ).get("advantage_std", float("nan")),
        "leave_one_out_advantage_std": baseline_summaries.get(
            "leave_one_out", {}
        ).get("advantage_std", float("nan")),
    }
    for key, value in exact_metrics.items():
        summary[key] = value
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def run_boundary_feedback_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    feedback_mode: str,
    feedback_seed: int,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> dict[str, float | int | str]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("boundary feedback diagnostic requires result_space head")
    model.train()

    feedback_objective, feedback_metrics = boundary_feedback_alignment_loss(
        model,
        batch,
        num_digits=num_digits,
        feedback_mode=feedback_mode,
        feedback_seed=feedback_seed,
    )
    feedback_objective.backward()
    feedback_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    feedback_result_norm = gradient_l2(
        feedback_grads,
        "calculator_hook.result_proj",
    )
    feedback_upstream_norm = gradient_l2(feedback_grads, "upstream")
    boundary_result_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    boundary_upstream_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "boundary_feedback_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "boundary_feedback_mode": feedback_mode,
        "boundary_feedback_seed": int(feedback_seed),
        "feedback_result_proj_grad_l2": feedback_result_norm,
        "feedback_upstream_grad_l2": feedback_upstream_norm,
        "feedback_semantic_decoder_grad_l2": gradient_l2(
            feedback_grads,
            "semantic_decoder",
        ),
        "boundary_result_proj_grad_l2": boundary_result_norm,
        "boundary_upstream_grad_l2": boundary_upstream_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads,
            "semantic_decoder",
        ),
        "feedback_vs_boundary_result_proj_cosine": gradient_cosine(
            feedback_grads,
            boundary_grads,
            "calculator_hook.result_proj",
        ),
        "feedback_vs_boundary_upstream_cosine": gradient_cosine(
            feedback_grads,
            boundary_grads,
            "upstream",
        ),
        "feedback_vs_boundary_result_proj_relative_norm": (
            feedback_result_norm / boundary_result_norm
            if boundary_result_norm > 0
            else float("nan")
        ),
        "feedback_vs_boundary_upstream_relative_norm": (
            feedback_upstream_norm / boundary_upstream_norm
            if boundary_upstream_norm > 0
            else float("nan")
        ),
    }
    summary.update(feedback_metrics)
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def shadow_feedback_heldout_split(
    batch: ArithmeticBatch,
    *,
    heldout_fraction: float,
) -> tuple[ArithmeticBatch, ArithmeticBatch]:
    if heldout_fraction <= 0 or heldout_fraction >= 1:
        raise ValueError("shadow feedback heldout fraction must be in (0, 1)")
    total = int(batch.x.shape[0])
    heldout_count = max(1, int(round(total * heldout_fraction)))
    if heldout_count >= total:
        raise ValueError("shadow feedback heldout split leaves no fit examples")
    stride = max(1, int(round(1.0 / heldout_fraction)))
    all_indices = torch.arange(total, device=batch.x.device)
    heldout_mask = (all_indices % stride) == 0
    if int(heldout_mask.sum().item()) != heldout_count:
        heldout_mask = torch.zeros(total, dtype=torch.bool, device=batch.x.device)
        heldout_positions = (
            torch.linspace(0, total - 1, steps=heldout_count, device=batch.x.device)
            .round()
            .long()
            .unique()
        )
        heldout_mask[heldout_positions] = True
    if int(heldout_mask.sum().item()) >= total:
        raise ValueError("shadow feedback heldout split leaves no fit examples")
    return (
        subset_arithmetic_batch(batch, all_indices[~heldout_mask]),
        subset_arithmetic_batch(batch, all_indices[heldout_mask]),
    )


def _shadow_prediction_cosine(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    pred_flat = predicted.reshape(-1)
    target_flat = target.reshape(-1)
    denom = pred_flat.norm() * target_flat.norm()
    if float(denom.item()) == 0.0:
        return float("nan")
    return float(torch.dot(pred_flat, target_flat).div(denom).item())


def online_shadow_feedback_selection_score(
    validation_summary: dict[str, float | int | str],
    *,
    train_summary: dict[str, float | int | str] | None = None,
    mode: str,
    gap_penalty: float,
) -> float:
    result_cosine = float(validation_summary["shadow_vs_boundary_result_proj_cosine"])
    upstream_cosine = float(validation_summary["shadow_vs_boundary_upstream_cosine"])
    if math.isnan(result_cosine) or math.isnan(upstream_cosine):
        return float("-inf")
    base_score = min(result_cosine, upstream_cosine)
    if mode == "min_result_upstream_cosine":
        return base_score
    if mode != "gap_penalized_min_cosine":
        raise ValueError(
            "shadow feedback selection score mode must be "
            "min_result_upstream_cosine or gap_penalized_min_cosine"
        )
    if train_summary is None:
        return float("-inf")
    train_result = float(train_summary["shadow_vs_boundary_result_proj_cosine"])
    train_upstream = float(train_summary["shadow_vs_boundary_upstream_cosine"])
    result_gap = max(0.0, train_result - result_cosine)
    upstream_gap = max(0.0, train_upstream - upstream_cosine)
    return base_score - (gap_penalty * max(result_gap, upstream_gap))


def run_online_shadow_feedback_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    heldout_fraction: float,
    hidden_size: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    updates_per_step: int,
    validation_fraction: float,
    validation_every: int,
    validation_loss_weight: float,
    validation_gradient_loss_weight: float,
    validation_gradient_norm_weight: float,
    target_normalization: str,
    target_transform: str,
    feature_mode: str,
    feature_normalization: str,
    loss_mode: str,
    selection_score_mode: str,
    selection_gap_penalty: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
    return_artifacts: bool = False,
) -> dict[str, float | int | str] | tuple[dict[str, float | int | str], dict[str, object]]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("online shadow feedback diagnostic requires result_space head")
    if heldout_fraction <= 0 or heldout_fraction >= 1:
        raise ValueError("online shadow feedback diagnostic requires heldout fraction in (0, 1)")
    if hidden_size < 1:
        raise ValueError("shadow feedback hidden size must be positive")
    if dropout < 0 or dropout >= 1:
        raise ValueError("shadow feedback dropout must be in [0, 1)")
    if learning_rate <= 0:
        raise ValueError("shadow feedback online lr must be positive")
    if weight_decay < 0:
        raise ValueError("shadow feedback weight decay must be non-negative")
    if warmup_steps < 1:
        raise ValueError("shadow feedback warmup steps must be positive")
    if updates_per_step < 1:
        raise ValueError("shadow feedback updates per step must be positive")
    if validation_fraction < 0 or validation_fraction >= 1:
        raise ValueError("shadow feedback validation fraction must be in [0, 1)")
    if heldout_fraction + validation_fraction >= 1:
        raise ValueError(
            "shadow feedback validation plus heldout fractions must leave fit examples"
        )
    if validation_every < 0:
        raise ValueError("shadow feedback validation cadence must be non-negative")
    if validation_fraction > 0 and validation_every < 1:
        raise ValueError(
            "shadow feedback validation fraction requires validation_every > 0"
        )
    if validation_loss_weight < 0:
        raise ValueError("shadow feedback validation loss weight must be non-negative")
    if validation_gradient_loss_weight < 0:
        raise ValueError(
            "shadow feedback validation gradient loss weight must be non-negative"
        )
    if validation_gradient_norm_weight < 0:
        raise ValueError(
            "shadow feedback validation gradient norm weight must be non-negative"
        )
    if target_normalization not in {"none", "fit_zscore_per_result"}:
        raise ValueError(
            "shadow feedback target normalization must be none or fit_zscore_per_result"
        )
    if target_transform not in {
        "none",
        "unit_norm_per_example",
        "fit_result_prototype",
    }:
        raise ValueError(
            "shadow feedback target transform must be none, "
            "unit_norm_per_example, or fit_result_prototype"
        )
    if feature_mode not in {
        "injection_grad_logits",
        "injection_grad_logits_output_jacobian",
        "injection_grad_logits_result_input",
        "injection_grad_policy_state",
    }:
        raise ValueError(
            "shadow feedback feature mode must be injection_grad_logits "
            "injection_grad_logits_output_jacobian, "
            "injection_grad_logits_result_input, or injection_grad_policy_state"
        )
    if feature_normalization not in {"none", "fit_zscore_per_feature"}:
        raise ValueError(
            "shadow feedback feature normalization must be none or "
            "fit_zscore_per_feature"
        )
    if loss_mode not in {"mse", "cosine", "mse_plus_cosine"}:
        raise ValueError(
            "shadow feedback loss mode must be mse, cosine, or mse_plus_cosine"
        )
    if selection_score_mode not in {
        "min_result_upstream_cosine",
        "gap_penalized_min_cosine",
    }:
        raise ValueError(
            "shadow feedback selection score mode must be "
            "min_result_upstream_cosine or gap_penalized_min_cosine"
        )
    if selection_gap_penalty < 0:
        raise ValueError("shadow feedback selection gap penalty must be non-negative")
    model.train()

    fit_batch, heldout_batch = shadow_feedback_heldout_split(
        batch,
        heldout_fraction=heldout_fraction,
    )
    validation_batch: ArithmeticBatch | None = None
    if validation_fraction > 0:
        fit_fraction = 1.0 - heldout_fraction
        validation_within_fit_fraction = validation_fraction / fit_fraction
        fit_batch, validation_batch = shadow_feedback_heldout_split(
            fit_batch,
            heldout_fraction=validation_within_fit_fraction,
        )
    if validation_loss_weight > 0 and validation_batch is None:
        raise ValueError(
            "shadow feedback validation loss weight requires validation_fraction > 0"
        )
    if validation_gradient_loss_weight > 0 and validation_batch is None:
        raise ValueError(
            "shadow feedback validation gradient loss weight requires "
            "validation_fraction > 0"
        )
    feature_dim = shadow_feedback_feature_dim(model, feature_mode=feature_mode)
    shadow_module = ShadowFeedbackMLP(
        input_dim=feature_dim,
        hidden_size=hidden_size,
        output_dim=model.cfg.calculator_result_vocab_size,
        dropout=dropout,
    ).to(device=batch.x.device)
    optimizer = torch.optim.AdamW(
        shadow_module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    target_mean: torch.Tensor | None = None
    target_scale: torch.Tensor | None = None
    feature_mean: torch.Tensor | None = None
    feature_scale: torch.Tensor | None = None
    target_normalization_metrics: dict[str, float | int | str] = {}
    target_transform_metrics: dict[str, float | int | str] = {}
    feature_normalization_metrics: dict[str, float | int | str] = {}
    fit_features_for_normalizer: torch.Tensor | None = None
    target_prototypes: torch.Tensor | None = None
    target_prototype_counts: torch.Tensor | None = None
    if target_transform == "fit_result_prototype":
        (
            fit_target_for_prototypes,
            fit_target_classes,
            _,
        ) = boundary_result_logit_gradient_target_with_classes(
            model,
            fit_batch,
            num_digits=num_digits,
            target_mode=result_boundary_target_mode,
            temperature=result_boundary_target_temperature,
            min_probability_floor=result_boundary_target_min_probability_floor,
            chunk_size=result_boundary_target_chunk_size,
        )
        fit_target_for_prototypes = fit_target_for_prototypes.detach().to(
            device=batch.x.device
        ) * float(fit_batch.x.shape[0])
        (
            target_prototypes,
            target_prototype_counts,
            target_transform_metrics,
        ) = fit_shadow_feedback_target_prototypes(
            fit_target_for_prototypes,
            fit_target_classes,
            num_classes=model.cfg.calculator_result_vocab_size,
        )
    if target_normalization != "none":
        (
            fit_features_for_normalizer,
            fit_target_for_normalizer,
            _,
        ) = shadow_feedback_mlp_examples(
            model,
            fit_batch,
            num_digits=num_digits,
            feature_mode=feature_mode,
            target_transform=target_transform,
            target_prototypes=target_prototypes,
            target_prototype_counts=target_prototype_counts,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        )
        (
            target_mean,
            target_scale,
            target_normalization_metrics,
        ) = fit_shadow_feedback_target_normalizer(
            fit_target_for_normalizer,
            mode=target_normalization,
        )
    else:
        (
            target_mean,
            target_scale,
            target_normalization_metrics,
        ) = fit_shadow_feedback_target_normalizer(
            torch.empty(0, device=batch.x.device),
            mode=target_normalization,
        )
    if feature_normalization != "none":
        if fit_features_for_normalizer is None:
            fit_features_for_normalizer, _, _ = shadow_feedback_mlp_examples(
                model,
                fit_batch,
                num_digits=num_digits,
                feature_mode=feature_mode,
                target_transform=target_transform,
                target_prototypes=target_prototypes,
                target_prototype_counts=target_prototype_counts,
                result_boundary_target_mode=result_boundary_target_mode,
                result_boundary_target_temperature=(
                    result_boundary_target_temperature
                ),
                result_boundary_target_min_probability_floor=(
                    result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=result_boundary_target_chunk_size,
            )
        (
            feature_mean,
            feature_scale,
            feature_normalization_metrics,
        ) = fit_shadow_feedback_feature_normalizer(
            fit_features_for_normalizer,
            mode=feature_normalization,
        )
    else:
        (
            feature_mean,
            feature_scale,
            feature_normalization_metrics,
        ) = fit_shadow_feedback_feature_normalizer(
            torch.empty(0, device=batch.x.device),
            mode=feature_normalization,
        )

    last_train_metrics: dict[str, float | int | str] = {}
    last_train_loss = float("nan")
    last_train_objective = float("nan")
    last_train_cosine = float("nan")
    last_normalized_train_loss = float("nan")
    last_normalized_train_cosine = float("nan")
    last_validation_regularization_objective = float("nan")
    last_validation_gradient_regularization_objective = float("nan")
    last_validation_gradient_metrics: dict[str, float] = {}
    last_total_train_objective = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_summary: dict[str, float | int | str] = {}
    best_validation_step = -1
    best_validation_score = float("-inf")
    validation_history: list[dict[str, float | int]] = []

    def maybe_update_best(step: int) -> None:
        nonlocal best_state
        nonlocal best_validation_summary
        nonlocal best_validation_step
        nonlocal best_validation_score
        if validation_batch is None:
            return
        validation_summary = run_online_shadow_feedback_module_gradient_diagnostic(
            model,
            validation_batch,
            shadow_module=shadow_module,
            num_digits=num_digits,
            feature_mode=feature_mode,
            target_transform=target_transform,
            target_prototypes=target_prototypes,
            target_prototype_counts=target_prototype_counts,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
            target_mean=target_mean,
            target_scale=target_scale,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        selection_train_summary: dict[str, float | int | str] | None = None
        result_gap = float("nan")
        upstream_gap = float("nan")
        if selection_score_mode == "gap_penalized_min_cosine":
            selection_train_summary = run_online_shadow_feedback_module_gradient_diagnostic(
                model,
                fit_batch,
                shadow_module=shadow_module,
                num_digits=num_digits,
                feature_mode=feature_mode,
                target_transform=target_transform,
                target_prototypes=target_prototypes,
                target_prototype_counts=target_prototype_counts,
                result_boundary_target_mode=result_boundary_target_mode,
                result_boundary_target_temperature=result_boundary_target_temperature,
                result_boundary_target_min_probability_floor=(
                    result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=result_boundary_target_chunk_size,
                target_mean=target_mean,
                target_scale=target_scale,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
            )
            result_gap = max(
                0.0,
                float(
                    selection_train_summary[
                        "shadow_vs_boundary_result_proj_cosine"
                    ]
                )
                - float(validation_summary["shadow_vs_boundary_result_proj_cosine"]),
            )
            upstream_gap = max(
                0.0,
                float(selection_train_summary["shadow_vs_boundary_upstream_cosine"])
                - float(validation_summary["shadow_vs_boundary_upstream_cosine"]),
            )
        score = online_shadow_feedback_selection_score(
            validation_summary,
            train_summary=selection_train_summary,
            mode=selection_score_mode,
            gap_penalty=selection_gap_penalty,
        )
        validation_history.append(
            {
                "step": int(step),
                "update": int(step * updates_per_step),
                "score": float(score),
                "result_proj_cosine": float(
                    validation_summary["shadow_vs_boundary_result_proj_cosine"]
                ),
                "upstream_cosine": float(
                    validation_summary["shadow_vs_boundary_upstream_cosine"]
                ),
                "train_validation_result_proj_cosine_gap": float(result_gap),
                "train_validation_upstream_cosine_gap": float(upstream_gap),
            }
        )
        if score > best_validation_score:
            best_validation_score = score
            best_validation_step = int(step)
            best_validation_summary = validation_summary
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in shadow_module.state_dict().items()
            }

    maybe_update_best(0)
    for warmup_step in range(1, warmup_steps + 1):
        for _update in range(updates_per_step):
            shadow_module.train()
            features, target, train_metrics = shadow_feedback_mlp_examples(
                model,
                fit_batch,
                num_digits=num_digits,
                feature_mode=feature_mode,
                target_transform=target_transform,
                target_prototypes=target_prototypes,
                target_prototype_counts=target_prototype_counts,
                result_boundary_target_mode=result_boundary_target_mode,
                result_boundary_target_temperature=(
                    result_boundary_target_temperature
                ),
                result_boundary_target_min_probability_floor=(
                    result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=result_boundary_target_chunk_size,
            )
            normalized_features = normalize_shadow_feedback_features(
                features,
                mean=feature_mean,
                scale=feature_scale,
            )
            predicted = shadow_module(normalized_features)
            normalized_target = normalize_shadow_feedback_target(
                target,
                mean=target_mean,
                scale=target_scale,
            )
            fit_loss = shadow_feedback_prediction_loss(
                predicted,
                normalized_target,
                mode=loss_mode,
            )
            total_loss = fit_loss
            if validation_loss_weight > 0 and validation_batch is not None:
                val_features, val_target, _ = shadow_feedback_mlp_examples(
                    model,
                    validation_batch,
                    num_digits=num_digits,
                    feature_mode=feature_mode,
                    target_transform=target_transform,
                    target_prototypes=target_prototypes,
                    target_prototype_counts=target_prototype_counts,
                    result_boundary_target_mode=result_boundary_target_mode,
                    result_boundary_target_temperature=(
                        result_boundary_target_temperature
                    ),
                    result_boundary_target_min_probability_floor=(
                        result_boundary_target_min_probability_floor
                    ),
                    result_boundary_target_chunk_size=(
                        result_boundary_target_chunk_size
                    ),
                )
                val_predicted = shadow_module(
                    normalize_shadow_feedback_features(
                        val_features,
                        mean=feature_mean,
                        scale=feature_scale,
                    )
                )
                val_normalized_target = normalize_shadow_feedback_target(
                    val_target,
                    mean=target_mean,
                    scale=target_scale,
                )
                validation_regularization_loss = shadow_feedback_prediction_loss(
                    val_predicted,
                    val_normalized_target,
                    mode=loss_mode,
                )
                total_loss = (
                    fit_loss + validation_loss_weight * validation_regularization_loss
                )
                last_validation_regularization_objective = float(
                    validation_regularization_loss.detach().item()
                )
            if (
                validation_gradient_loss_weight > 0
                and validation_batch is not None
            ):
                (
                    validation_gradient_regularization_loss,
                    validation_gradient_metrics,
                ) = online_shadow_feedback_validation_gradient_loss(
                    model,
                    validation_batch,
                    shadow_module=shadow_module,
                    num_digits=num_digits,
                    feature_mode=feature_mode,
                    target_transform=target_transform,
                    target_prototypes=target_prototypes,
                    target_prototype_counts=target_prototype_counts,
                    result_boundary_target_mode=result_boundary_target_mode,
                    result_boundary_target_temperature=(
                        result_boundary_target_temperature
                    ),
                    result_boundary_target_min_probability_floor=(
                        result_boundary_target_min_probability_floor
                    ),
                    result_boundary_target_chunk_size=(
                        result_boundary_target_chunk_size
                    ),
                    target_mean=target_mean,
                    target_scale=target_scale,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    norm_weight=validation_gradient_norm_weight,
                )
                total_loss = total_loss + (
                    validation_gradient_loss_weight
                    * validation_gradient_regularization_loss
                )
                last_validation_gradient_regularization_objective = float(
                    validation_gradient_regularization_loss.detach().item()
                )
                last_validation_gradient_metrics = validation_gradient_metrics
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            denormalized_prediction = denormalize_shadow_feedback_prediction(
                predicted.detach(),
                mean=target_mean,
                scale=target_scale,
            )
            last_train_loss = float(
                (denormalized_prediction - target.detach()).pow(2).mean().item()
            )
            last_train_cosine = _shadow_prediction_cosine(
                denormalized_prediction,
                target.detach(),
            )
            last_train_objective = float(fit_loss.detach().item())
            last_total_train_objective = float(total_loss.detach().item())
            last_normalized_train_loss = float(
                torch.nn.functional.mse_loss(
                    predicted.detach(),
                    normalized_target.detach(),
                ).item()
            )
            last_normalized_train_cosine = _shadow_prediction_cosine(
                predicted.detach(),
                normalized_target.detach(),
            )
            last_train_metrics = train_metrics
            last_train_metrics["shadow_feedback_normalized_feature_l2"] = float(
                normalized_features.detach().norm().item()
            )
        if validation_every > 0 and warmup_step % validation_every == 0:
            maybe_update_best(warmup_step)
    if validation_every > 0 and warmup_steps % validation_every != 0:
        maybe_update_best(warmup_steps)

    train_summary = run_online_shadow_feedback_module_gradient_diagnostic(
        model,
        fit_batch,
        shadow_module=shadow_module,
        num_digits=num_digits,
        feature_mode=feature_mode,
        target_transform=target_transform,
        target_prototypes=target_prototypes,
        target_prototype_counts=target_prototype_counts,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        target_mean=target_mean,
        target_scale=target_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    heldout_summary = run_online_shadow_feedback_module_gradient_diagnostic(
        model,
        heldout_batch,
        shadow_module=shadow_module,
        num_digits=num_digits,
        feature_mode=feature_mode,
        target_transform=target_transform,
        target_prototypes=target_prototypes,
        target_prototype_counts=target_prototype_counts,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        target_mean=target_mean,
        target_scale=target_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    result_gap = (
        float(train_summary["shadow_vs_boundary_result_proj_cosine"])
        - float(heldout_summary["shadow_vs_boundary_result_proj_cosine"])
    )
    upstream_gap = (
        float(train_summary["shadow_vs_boundary_upstream_cosine"])
        - float(heldout_summary["shadow_vs_boundary_upstream_cosine"])
    )
    summary: dict[str, float | int | str] = {
        "diagnostic": "online_mlp_shadow_feedback_heldout_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "shadow_feedback_mode": "online_mlp",
        "shadow_feedback_heldout_fraction": float(heldout_fraction),
        "shadow_feedback_fit_batch_size": int(fit_batch.x.shape[0]),
        "shadow_feedback_heldout_batch_size": int(heldout_batch.x.shape[0]),
        "shadow_feedback_hidden_size": int(hidden_size),
        "shadow_feedback_dropout": float(dropout),
        "shadow_feedback_online_lr": float(learning_rate),
        "shadow_feedback_weight_decay": float(weight_decay),
        "shadow_feedback_warmup_steps": int(warmup_steps),
        "shadow_feedback_updates_per_step": int(updates_per_step),
        "shadow_feedback_validation_fraction": float(validation_fraction),
        "shadow_feedback_test_fraction": float(heldout_fraction),
        "shadow_feedback_validation_every": int(validation_every),
        "shadow_feedback_validation_loss_weight": float(validation_loss_weight),
        "shadow_feedback_validation_gradient_loss_weight": float(
            validation_gradient_loss_weight
        ),
        "shadow_feedback_validation_gradient_norm_weight": float(
            validation_gradient_norm_weight
        ),
        "shadow_feedback_validation_batch_size": (
            int(validation_batch.x.shape[0]) if validation_batch is not None else 0
        ),
        "shadow_feedback_selection_mode": (
            "validation_best_checkpoint" if validation_batch is not None else "final"
        ),
        "shadow_feedback_selection_metric": selection_score_mode,
        "shadow_feedback_selection_metric_formula": (
            "min(validation_result_cosine, validation_upstream_cosine)"
            if selection_score_mode == "min_result_upstream_cosine"
            else (
                "min(validation_result_cosine, validation_upstream_cosine) - "
                "gap_penalty * max(train_validation_result_gap, "
                "train_validation_upstream_gap)"
            )
        ),
        "shadow_feedback_selection_gap_penalty": float(selection_gap_penalty),
        "shadow_feedback_validation_history": validation_history,
        "shadow_feedback_best_state_restored": bool(best_state is not None),
        "shadow_feedback_target_normalization": target_normalization,
        "shadow_feedback_target_transform": target_transform,
        "shadow_feedback_feature_mode": feature_mode,
        "shadow_feedback_feature_normalization": feature_normalization,
        "shadow_feedback_loss_mode": loss_mode,
        "shadow_feedback_feature_dim": int(feature_dim),
        "shadow_feedback_final_fit_mse": last_train_loss,
        "shadow_feedback_final_fit_objective": last_train_objective,
        "shadow_feedback_final_total_objective": last_total_train_objective,
        "shadow_feedback_final_validation_regularization_objective": (
            last_validation_regularization_objective
        ),
        "shadow_feedback_final_validation_gradient_regularization_objective": (
            last_validation_gradient_regularization_objective
        ),
        "shadow_feedback_final_fit_prediction_cosine": last_train_cosine,
        "shadow_feedback_final_normalized_fit_mse": last_normalized_train_loss,
        "shadow_feedback_final_normalized_fit_prediction_cosine": (
            last_normalized_train_cosine
        ),
    }
    summary.update(target_normalization_metrics)
    summary.update(target_transform_metrics)
    summary.update(feature_normalization_metrics)
    summary.update(last_validation_gradient_metrics)
    if best_state is not None and validation_batch is not None:
        summary["shadow_feedback_best_step"] = int(best_validation_step)
        summary["shadow_feedback_best_update"] = int(
            best_validation_step * updates_per_step
        )
        summary["shadow_feedback_best_validation_score"] = float(
            best_validation_score
        )
        for key, value in train_summary.items():
            if key != "diagnostic":
                summary[f"final_train_{key}"] = value
        for key, value in heldout_summary.items():
            if key != "diagnostic":
                summary[f"final_heldout_{key}"] = value
        shadow_module.load_state_dict(
            {
                name: value.to(device=batch.x.device)
                for name, value in best_state.items()
            }
        )
        train_summary = run_online_shadow_feedback_module_gradient_diagnostic(
            model,
            fit_batch,
            shadow_module=shadow_module,
            num_digits=num_digits,
            feature_mode=feature_mode,
            target_transform=target_transform,
            target_prototypes=target_prototypes,
            target_prototype_counts=target_prototype_counts,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
            target_mean=target_mean,
            target_scale=target_scale,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        heldout_summary = run_online_shadow_feedback_module_gradient_diagnostic(
            model,
            heldout_batch,
            shadow_module=shadow_module,
            num_digits=num_digits,
            feature_mode=feature_mode,
            target_transform=target_transform,
            target_prototypes=target_prototypes,
            target_prototype_counts=target_prototype_counts,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
            target_mean=target_mean,
            target_scale=target_scale,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )
        result_gap = (
            float(train_summary["shadow_vs_boundary_result_proj_cosine"])
            - float(heldout_summary["shadow_vs_boundary_result_proj_cosine"])
        )
        upstream_gap = (
            float(train_summary["shadow_vs_boundary_upstream_cosine"])
            - float(heldout_summary["shadow_vs_boundary_upstream_cosine"])
        )
        for key, value in best_validation_summary.items():
            if key != "diagnostic":
                summary[f"validation_{key}"] = value
        summary["shadow_feedback_train_validation_result_proj_cosine_gap"] = (
            float(train_summary["shadow_vs_boundary_result_proj_cosine"])
            - float(best_validation_summary["shadow_vs_boundary_result_proj_cosine"])
        )
        summary["shadow_feedback_train_validation_upstream_cosine_gap"] = (
            float(train_summary["shadow_vs_boundary_upstream_cosine"])
            - float(best_validation_summary["shadow_vs_boundary_upstream_cosine"])
        )
        summary["shadow_feedback_validation_test_result_proj_cosine_gap"] = (
            float(best_validation_summary["shadow_vs_boundary_result_proj_cosine"])
            - float(heldout_summary["shadow_vs_boundary_result_proj_cosine"])
        )
        summary["shadow_feedback_validation_test_upstream_cosine_gap"] = (
            float(best_validation_summary["shadow_vs_boundary_upstream_cosine"])
            - float(heldout_summary["shadow_vs_boundary_upstream_cosine"])
        )
    for key, value in last_train_metrics.items():
        summary[f"fit_{key}"] = value
    for key, value in train_summary.items():
        if key != "diagnostic":
            summary[f"train_{key}"] = value
    for key, value in heldout_summary.items():
        if key != "diagnostic":
            summary[f"heldout_{key}"] = value
    summary["shadow_feedback_train_heldout_result_proj_cosine_gap"] = result_gap
    summary["shadow_feedback_train_heldout_upstream_cosine_gap"] = upstream_gap
    if return_artifacts:
        artifacts: dict[str, object] = {
            "shadow_module": shadow_module,
            "feature_mode": feature_mode,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }
        return summary, artifacts
    return summary


def run_online_shadow_feedback_module_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    shadow_module: ShadowFeedbackMLP,
    num_digits: int,
    feature_mode: str,
    target_transform: str,
    target_prototypes: torch.Tensor | None = None,
    target_prototype_counts: torch.Tensor | None = None,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
    target_mean: torch.Tensor | None = None,
    target_scale: torch.Tensor | None = None,
    feature_mean: torch.Tensor | None = None,
    feature_scale: torch.Tensor | None = None,
) -> dict[str, float | int | str]:
    model.train()
    shadow_objective, shadow_metrics = online_shadow_feedback_alignment_loss(
        model,
        batch,
        shadow_module=shadow_module,
        num_digits=num_digits,
        feature_mode=feature_mode,
        target_transform=target_transform,
        target_prototypes=target_prototypes,
        target_prototype_counts=target_prototype_counts,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        target_mean=target_mean,
        target_scale=target_scale,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )
    shadow_objective.backward()
    shadow_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    shadow_result_norm = gradient_l2(shadow_grads, "calculator_hook.result_proj")
    shadow_upstream_norm = gradient_l2(shadow_grads, "upstream")
    boundary_result_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    boundary_upstream_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "online_mlp_shadow_feedback_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "shadow_result_proj_grad_l2": shadow_result_norm,
        "shadow_upstream_grad_l2": shadow_upstream_norm,
        "shadow_semantic_decoder_grad_l2": gradient_l2(
            shadow_grads,
            "semantic_decoder",
        ),
        "boundary_result_proj_grad_l2": boundary_result_norm,
        "boundary_upstream_grad_l2": boundary_upstream_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads,
            "semantic_decoder",
        ),
        "shadow_vs_boundary_result_proj_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "calculator_hook.result_proj",
        ),
        "shadow_vs_boundary_upstream_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "upstream",
        ),
        "shadow_vs_boundary_result_proj_relative_norm": (
            shadow_result_norm / boundary_result_norm
            if boundary_result_norm > 0
            else float("nan")
        ),
        "shadow_vs_boundary_upstream_relative_norm": (
            shadow_upstream_norm / boundary_upstream_norm
            if boundary_upstream_norm > 0
            else float("nan")
        ),
    }
    summary.update(shadow_metrics)
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def run_shadow_feedback_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    ridge: float,
    heldout_fraction: float,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> dict[str, float | int | str]:
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("shadow feedback diagnostic requires result_space head")
    model.train()
    if heldout_fraction < 0 or heldout_fraction >= 1:
        raise ValueError("shadow feedback heldout fraction must be in [0, 1)")

    if heldout_fraction > 0:
        total = int(batch.x.shape[0])
        heldout_count = max(1, int(round(total * heldout_fraction)))
        if heldout_count >= total:
            raise ValueError("shadow feedback heldout split leaves no fit examples")
        stride = max(1, int(round(1.0 / heldout_fraction)))
        all_indices = torch.arange(total, device=batch.x.device)
        heldout_mask = (all_indices % stride) == 0
        if int(heldout_mask.sum().item()) != heldout_count:
            heldout_mask = torch.zeros(total, dtype=torch.bool, device=batch.x.device)
            heldout_positions = torch.linspace(
                0,
                total - 1,
                steps=heldout_count,
                device=batch.x.device,
            ).round().long().unique()
            heldout_mask[heldout_positions] = True
        if int(heldout_mask.sum().item()) >= total:
            raise ValueError("shadow feedback heldout split leaves no fit examples")
        fit_batch = subset_arithmetic_batch(batch, all_indices[~heldout_mask])
        heldout_batch = subset_arithmetic_batch(batch, all_indices[heldout_mask])
        weights, fit_metrics = fit_linear_shadow_feedback_weights(
            model,
            fit_batch,
            num_digits=num_digits,
            ridge=ridge,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        )
        train_summary = run_fixed_shadow_feedback_gradient_diagnostic(
            model,
            fit_batch,
            num_digits=num_digits,
            ridge=ridge,
            weights=weights,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        )
        heldout_summary = run_fixed_shadow_feedback_gradient_diagnostic(
            model,
            heldout_batch,
            num_digits=num_digits,
            ridge=ridge,
            weights=weights,
            result_boundary_target_mode=result_boundary_target_mode,
            result_boundary_target_temperature=result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=result_boundary_target_chunk_size,
        )
        summary: dict[str, float | int | str] = {
            "diagnostic": "linear_shadow_feedback_heldout_gradient_agreement",
            "batch_size": total,
            "shadow_feedback_ridge": float(ridge),
            "shadow_feedback_heldout_fraction": float(heldout_fraction),
            "shadow_feedback_fit_batch_size": int(fit_batch.x.shape[0]),
            "shadow_feedback_heldout_batch_size": int(heldout_batch.x.shape[0]),
        }
        for key, value in fit_metrics.items():
            summary[f"fit_{key}"] = value
        for key, value in train_summary.items():
            if key not in {"diagnostic"}:
                summary[f"train_{key}"] = value
        for key, value in heldout_summary.items():
            if key not in {"diagnostic"}:
                summary[f"heldout_{key}"] = value
        result_gap = (
            float(train_summary["shadow_vs_boundary_result_proj_cosine"])
            - float(heldout_summary["shadow_vs_boundary_result_proj_cosine"])
        )
        upstream_gap = (
            float(train_summary["shadow_vs_boundary_upstream_cosine"])
            - float(heldout_summary["shadow_vs_boundary_upstream_cosine"])
        )
        summary["shadow_feedback_train_heldout_result_proj_cosine_gap"] = result_gap
        summary["shadow_feedback_train_heldout_upstream_cosine_gap"] = upstream_gap
        return summary

    shadow_objective, shadow_metrics = linear_shadow_feedback_alignment_loss(
        model,
        batch,
        num_digits=num_digits,
        ridge=ridge,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    shadow_objective.backward()
    shadow_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    shadow_result_norm = gradient_l2(shadow_grads, "calculator_hook.result_proj")
    shadow_upstream_norm = gradient_l2(shadow_grads, "upstream")
    boundary_result_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    boundary_upstream_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "linear_shadow_feedback_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "shadow_feedback_ridge": float(ridge),
        "shadow_result_proj_grad_l2": shadow_result_norm,
        "shadow_upstream_grad_l2": shadow_upstream_norm,
        "shadow_semantic_decoder_grad_l2": gradient_l2(
            shadow_grads,
            "semantic_decoder",
        ),
        "boundary_result_proj_grad_l2": boundary_result_norm,
        "boundary_upstream_grad_l2": boundary_upstream_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads,
            "semantic_decoder",
        ),
        "shadow_vs_boundary_result_proj_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "calculator_hook.result_proj",
        ),
        "shadow_vs_boundary_upstream_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "upstream",
        ),
        "shadow_vs_boundary_result_proj_relative_norm": (
            shadow_result_norm / boundary_result_norm
            if boundary_result_norm > 0
            else float("nan")
        ),
        "shadow_vs_boundary_upstream_relative_norm": (
            shadow_upstream_norm / boundary_upstream_norm
            if boundary_upstream_norm > 0
            else float("nan")
        ),
    }
    summary.update(shadow_metrics)
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def run_fixed_shadow_feedback_gradient_diagnostic(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    ridge: float,
    weights: torch.Tensor,
    result_boundary_target_mode: str,
    result_boundary_target_temperature: float,
    result_boundary_target_min_probability_floor: float,
    result_boundary_target_chunk_size: int,
) -> dict[str, float | int | str]:
    model.train()
    shadow_objective, shadow_metrics = fixed_linear_shadow_feedback_alignment_loss(
        model,
        batch,
        num_digits=num_digits,
        weights=weights,
        ridge=ridge,
        result_boundary_target_mode=result_boundary_target_mode,
        result_boundary_target_temperature=result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=result_boundary_target_chunk_size,
    )
    shadow_objective.backward()
    shadow_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode=result_boundary_target_mode,
        temperature=result_boundary_target_temperature,
        min_probability_floor=result_boundary_target_min_probability_floor,
        chunk_size=result_boundary_target_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    shadow_result_norm = gradient_l2(shadow_grads, "calculator_hook.result_proj")
    shadow_upstream_norm = gradient_l2(shadow_grads, "upstream")
    boundary_result_norm = gradient_l2(boundary_grads, "calculator_hook.result_proj")
    boundary_upstream_norm = gradient_l2(boundary_grads, "upstream")
    summary: dict[str, float | int | str] = {
        "diagnostic": "fixed_linear_shadow_feedback_gradient_agreement",
        "batch_size": int(batch.x.shape[0]),
        "shadow_feedback_ridge": float(ridge),
        "shadow_result_proj_grad_l2": shadow_result_norm,
        "shadow_upstream_grad_l2": shadow_upstream_norm,
        "shadow_semantic_decoder_grad_l2": gradient_l2(
            shadow_grads,
            "semantic_decoder",
        ),
        "boundary_result_proj_grad_l2": boundary_result_norm,
        "boundary_upstream_grad_l2": boundary_upstream_norm,
        "boundary_semantic_decoder_grad_l2": gradient_l2(
            boundary_grads,
            "semantic_decoder",
        ),
        "shadow_vs_boundary_result_proj_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "calculator_hook.result_proj",
        ),
        "shadow_vs_boundary_upstream_cosine": gradient_cosine(
            shadow_grads,
            boundary_grads,
            "upstream",
        ),
        "shadow_vs_boundary_result_proj_relative_norm": (
            shadow_result_norm / boundary_result_norm
            if boundary_result_norm > 0
            else float("nan")
        ),
        "shadow_vs_boundary_upstream_relative_norm": (
            shadow_upstream_norm / boundary_upstream_norm
            if boundary_upstream_norm > 0
            else float("nan")
        ),
    }
    summary.update(shadow_metrics)
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    return summary


def action_loss_full_enum_interface_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
    target_mode: str = "soft_pair",
) -> tuple[torch.Tensor, dict[str, float]]:
    if target_mode not in {"soft_pair", "hard_best_pair"}:
        raise ValueError(
            "full-enum local target mode must be 'soft_pair' or 'hard_best_pair'"
        )
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    classes = a_logits.shape[-1]
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    candidates = pairs.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    with torch.no_grad():
        full_losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
        pair_weights = action_loss_weights_from_losses(
            full_losses,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
        )
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_idx = learned_a * classes + learned_b
    true_idx = true_a * classes + true_b
    best_idx = full_losses.argmin(dim=-1)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    best_pairs = pairs.index_select(0, best_idx)
    if target_mode == "hard_best_pair":
        target_loss = (
            torch.nn.functional.cross_entropy(a_logits, best_pairs[:, 0])
            + torch.nn.functional.cross_entropy(b_logits, best_pairs[:, 1])
        ) / 2
        target_a = torch.nn.functional.one_hot(
            best_pairs[:, 0], num_classes=classes
        ).to(a_logits.dtype)
        target_b = torch.nn.functional.one_hot(
            best_pairs[:, 1], num_classes=classes
        ).to(b_logits.dtype)
    else:
        target_a, target_b = action_loss_soft_targets(
            a_logits, b_logits, candidates, pair_weights
        )
        target_loss = action_loss_target_ce(a_logits, b_logits, target_a, target_b)

    learned_losses = full_losses.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
    true_losses = full_losses.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    best_losses = full_losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    entropy = -(pair_weights * pair_weights.clamp_min(1e-12).log()).sum(dim=-1)
    true_pair_probs = pair_weights.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    true_pair_ranks = (full_losses < true_losses.unsqueeze(-1)).sum(dim=-1) + 1
    sorted_weights = pair_weights.sort(dim=-1, descending=True).values
    best_left_matches_true = best_pairs[:, 0] == true_a
    metrics = {
        "action_loss_full_enum_temperature": float(temperature),
        "action_loss_full_enum_min_probability_floor": float(min_probability_floor),
        "action_loss_full_enum_chunk_size": int(chunk_size),
        "action_loss_full_enum_target_mode": target_mode,
        "action_loss_full_enum_best_nll": float(best_losses.mean().item()),
        "action_loss_full_enum_learned_nll": float(learned_losses.mean().item()),
        "action_loss_full_enum_true_nll": float(true_losses.mean().item()),
        "action_loss_full_enum_learned_minus_true_gap": float(
            (learned_losses - true_losses).mean().item()
        ),
        "action_loss_full_enum_learned_minus_best_gap": float(
            (learned_losses - best_losses).mean().item()
        ),
        "action_loss_full_enum_true_best_fraction": float(
            (best_idx == true_idx).float().mean().item()
        ),
        "action_loss_full_enum_learned_best_fraction": float(
            (best_idx == learned_idx).float().mean().item()
        ),
        "action_loss_full_enum_soft_target_true_a_mass": float(
            target_a.gather(1, true_a.unsqueeze(-1)).mean().item()
        ),
        "action_loss_full_enum_soft_target_true_b_mass": float(
            target_b.gather(1, true_b.unsqueeze(-1)).mean().item()
        ),
        "action_loss_full_enum_entropy": float(entropy.mean().item()),
        "action_loss_full_enum_effective_pairs": float(entropy.exp().mean().item()),
        "action_loss_full_enum_true_pair_probability": float(
            true_pair_probs.mean().item()
        ),
        "action_loss_full_enum_true_pair_rank": float(
            true_pair_ranks.float().mean().item()
        ),
        "action_loss_full_enum_best_left_operand_accuracy": float(
            best_left_matches_true.float().mean().item()
        ),
        "action_loss_full_enum_top1_mass": float(
            sorted_weights[:, :1].sum(dim=-1).mean().item()
        ),
        "action_loss_full_enum_top3_mass": float(
            sorted_weights[:, :3].sum(dim=-1).mean().item()
        ),
        "action_loss_full_enum_top5_mass": float(
            sorted_weights[:, :5].sum(dim=-1).mean().item()
        ),
    }
    return target_loss, metrics


def action_loss_full_enum_joint_interface_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    pair_logits, _, _, _ = calculator_read_pair_logits(model, batch)
    classes = model.cfg.calculator_operand_vocab_size
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    candidates = pairs.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    with torch.no_grad():
        full_losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
        pair_weights = action_loss_weights_from_losses(
            full_losses,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
        )
    logp = pair_logits.log_softmax(dim=-1)
    target_loss = -(pair_weights * logp).sum(dim=-1).mean()

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_idx = pair_logits.argmax(dim=-1)
    learned_a = learned_idx // classes
    learned_b = learned_idx % classes
    true_idx = true_a * classes + true_b
    best_idx = full_losses.argmin(dim=-1)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    learned_losses = full_losses.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
    true_losses = full_losses.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    best_losses = full_losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    target_entropy = -(pair_weights * pair_weights.clamp_min(1e-12).log()).sum(dim=-1)
    learned_probs = pair_logits.softmax(dim=-1)
    learned_entropy = -(
        learned_probs * learned_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    true_pair_probs = pair_weights.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    true_pair_ranks = (
        (full_losses < true_losses.unsqueeze(-1)).sum(dim=-1) + 1
    )
    true_sum = true_a + true_b
    learned_sum = learned_a + learned_b
    metrics = {
        "action_loss_full_enum_temperature": float(temperature),
        "action_loss_full_enum_min_probability_floor": float(min_probability_floor),
        "action_loss_full_enum_chunk_size": int(chunk_size),
        "action_loss_full_enum_joint_target_loss": float(target_loss.item()),
        "action_loss_full_enum_best_nll": float(best_losses.mean().item()),
        "action_loss_full_enum_learned_nll": float(learned_losses.mean().item()),
        "action_loss_full_enum_true_nll": float(true_losses.mean().item()),
        "action_loss_full_enum_learned_pair_nll": float(learned_losses.mean().item()),
        "action_loss_full_enum_true_pair_nll": float(true_losses.mean().item()),
        "action_loss_full_enum_best_pair_nll": float(best_losses.mean().item()),
        "action_loss_full_enum_learned_minus_true_gap": float(
            (learned_losses - true_losses).mean().item()
        ),
        "action_loss_full_enum_learned_minus_best_gap": float(
            (learned_losses - best_losses).mean().item()
        ),
        "action_loss_full_enum_true_best_fraction": float(
            (best_idx == true_idx).float().mean().item()
        ),
        "action_loss_full_enum_learned_best_fraction": float(
            (best_idx == learned_idx).float().mean().item()
        ),
        "action_loss_full_enum_pair_exact_match": float(
            (learned_idx == true_idx).float().mean().item()
        ),
        "action_loss_full_enum_result_equivalent_pair_accuracy": float(
            (learned_sum == true_sum).float().mean().item()
        ),
        "action_loss_full_enum_true_pair_probability": float(
            true_pair_probs.mean().item()
        ),
        "action_loss_full_enum_true_pair_rank": float(
            true_pair_ranks.float().mean().item()
        ),
        "action_loss_full_enum_pair_entropy": float(target_entropy.mean().item()),
        "action_loss_full_enum_entropy": float(target_entropy.mean().item()),
        "action_loss_full_enum_effective_pairs": float(
            target_entropy.exp().mean().item()
        ),
        "action_loss_full_enum_pair_logit_entropy": float(
            learned_entropy.mean().item()
        ),
        "action_loss_full_enum_pair_logit_effective_pairs": float(
            learned_entropy.exp().mean().item()
        ),
    }
    return target_loss, metrics


def normalize_expected_answer_costs(
    costs: torch.Tensor, *, mode: str
) -> torch.Tensor:
    if mode == "none":
        return costs
    if mode == "center":
        return costs - costs.mean(dim=-1, keepdim=True)
    if mode == "zscore":
        centered = costs - costs.mean(dim=-1, keepdim=True)
        scale = costs.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return centered / scale
    raise ValueError(
        "expected answer-loss cost normalization must be one of "
        "{'none', 'center', 'zscore'}"
    )


def full_enum_expected_answer_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    policy_temperature: float,
    cost_normalization: str,
    entropy_weight: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if policy_temperature <= 0:
        raise ValueError("expected answer-loss policy temperature must be positive")
    if model.cfg.calculator_action_head == "result_space":
        return result_space_expected_answer_loss(
            model,
            batch,
            num_digits=num_digits,
            policy_temperature=policy_temperature,
            cost_normalization=cost_normalization,
            entropy_weight=entropy_weight,
            chunk_size=chunk_size,
        )
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    classes = a_logits.shape[-1]
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    candidates = pairs.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    with torch.no_grad():
        full_costs = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
        objective_costs = normalize_expected_answer_costs(
            full_costs, mode=cost_normalization
        )

    a_probs = torch.softmax(a_logits / policy_temperature, dim=-1)
    b_probs = torch.softmax(b_logits / policy_temperature, dim=-1)
    pair_probs = (a_probs.unsqueeze(-1) * b_probs.unsqueeze(-2)).reshape(
        batch.x.shape[0], classes * classes
    )
    expected_loss = (pair_probs * objective_costs.detach()).sum(dim=-1).mean()
    entropy = -(pair_probs * pair_probs.clamp_min(1e-12).log()).sum(dim=-1)
    objective = expected_loss - (entropy_weight * entropy.mean())

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_idx = learned_a * classes + learned_b
    true_idx = true_a * classes + true_b
    best_idx = full_costs.argmin(dim=-1)
    learned_costs = full_costs.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
    true_costs = full_costs.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    best_costs = full_costs.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    expected_raw = (pair_probs.detach() * full_costs).sum(dim=-1)
    best_probs = pair_probs.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    true_probs = pair_probs.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    learned_probs = pair_probs.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
    true_sum = true_a + true_b
    learned_sum = learned_a + learned_b
    metrics = {
        "expected_answer_loss": float(expected_loss.item()),
        "expected_answer_loss_policy_temperature": float(policy_temperature),
        "expected_answer_loss_cost_normalization": cost_normalization,
        "expected_answer_loss_entropy_weight": float(entropy_weight),
        "expected_answer_loss_entropy": float(entropy.mean().item()),
        "expected_answer_loss_effective_pairs": float(entropy.exp().mean().item()),
        "expected_answer_loss_best_nll": float(best_costs.mean().item()),
        "expected_answer_loss_true_nll": float(true_costs.mean().item()),
        "expected_answer_loss_learned_nll": float(learned_costs.mean().item()),
        "expected_answer_loss_raw_expected_nll": float(expected_raw.mean().item()),
        "expected_answer_loss_expected_minus_best_gap": float(
            (expected_raw - best_costs).mean().item()
        ),
        "expected_answer_loss_learned_minus_best_gap": float(
            (learned_costs - best_costs).mean().item()
        ),
        "expected_answer_loss_learned_minus_true_gap": float(
            (learned_costs - true_costs).mean().item()
        ),
        "expected_answer_loss_best_pair_probability": float(best_probs.mean().item()),
        "expected_answer_loss_true_pair_probability": float(true_probs.mean().item()),
        "expected_answer_loss_learned_pair_probability": float(
            learned_probs.mean().item()
        ),
        "expected_answer_loss_hard_learned_best_fraction": float(
            (learned_idx == best_idx).float().mean().item()
        ),
        "expected_answer_loss_hard_learned_pair_exact": float(
            (learned_idx == true_idx).float().mean().item()
        ),
        "expected_answer_loss_hard_learned_calc_accuracy": float(
            (learned_sum == true_sum).float().mean().item()
        ),
    }
    return objective, metrics


def result_space_expected_answer_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    policy_temperature: float,
    cost_normalization: str,
    entropy_weight: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if policy_temperature <= 0:
        raise ValueError("expected answer-loss policy temperature must be positive")
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    with torch.no_grad():
        full_costs = score_forced_result_classes_chunked(
            model, batch, chunk_size=chunk_size
        )
        objective_costs = normalize_expected_answer_costs(
            full_costs, mode=cost_normalization
        )

    result_probs = torch.softmax(result_logits / policy_temperature, dim=-1)
    expected_loss = (result_probs * objective_costs.detach()).sum(dim=-1).mean()
    entropy = -(
        result_probs * result_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    objective = expected_loss - (entropy_weight * entropy.mean())

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_result = result_logits.argmax(dim=-1)
    best_result = full_costs.argmin(dim=-1)
    learned_costs = full_costs.gather(
        1, learned_result.unsqueeze(-1)
    ).squeeze(-1)
    true_costs = full_costs.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    best_costs = full_costs.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    expected_raw = (result_probs.detach() * full_costs).sum(dim=-1)
    best_probs = result_probs.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    true_probs = result_probs.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    learned_probs = result_probs.gather(
        1, learned_result.unsqueeze(-1)
    ).squeeze(-1)
    metrics = {
        "expected_answer_loss": float(expected_loss.item()),
        "expected_answer_loss_policy_temperature": float(policy_temperature),
        "expected_answer_loss_cost_normalization": cost_normalization,
        "expected_answer_loss_entropy_weight": float(entropy_weight),
        "expected_answer_loss_entropy": float(entropy.mean().item()),
        "expected_answer_loss_effective_results": float(entropy.exp().mean().item()),
        "expected_answer_loss_best_nll": float(best_costs.mean().item()),
        "expected_answer_loss_true_nll": float(true_costs.mean().item()),
        "expected_answer_loss_learned_nll": float(learned_costs.mean().item()),
        "expected_answer_loss_raw_expected_nll": float(expected_raw.mean().item()),
        "expected_answer_loss_expected_minus_best_gap": float(
            (expected_raw - best_costs).mean().item()
        ),
        "expected_answer_loss_learned_minus_best_gap": float(
            (learned_costs - best_costs).mean().item()
        ),
        "expected_answer_loss_learned_minus_true_gap": float(
            (learned_costs - true_costs).mean().item()
        ),
        "expected_answer_loss_best_result_probability": float(
            best_probs.mean().item()
        ),
        "expected_answer_loss_true_result_probability": float(
            true_probs.mean().item()
        ),
        "expected_answer_loss_learned_result_probability": float(
            learned_probs.mean().item()
        ),
        "expected_answer_loss_hard_learned_best_fraction": float(
            (learned_result == best_result).float().mean().item()
        ),
        "expected_answer_loss_hard_learned_result_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
        "expected_answer_loss_hard_learned_calc_accuracy": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }
    return objective, metrics


class ActionLossReplayCache:
    def __init__(self) -> None:
        self.entries: dict[tuple[int, ...], dict[str, object]] = {}

    @staticmethod
    def keys_from_batch(batch: ArithmeticBatch) -> list[tuple[int, ...]]:
        return [tuple(int(token) for token in row.tolist()) for row in batch.x.detach().cpu()]

    def stale_indices(
        self,
        keys: list[tuple[int, ...]],
        *,
        step: int,
        refresh_every: int,
    ) -> list[int]:
        stale: list[int] = []
        for idx, key in enumerate(keys):
            entry = self.entries.get(key)
            if entry is None or step - int(entry["refresh_step"]) >= refresh_every:
                stale.append(idx)
        return stale

    def update(
        self,
        *,
        keys: list[tuple[int, ...]],
        indices: list[int],
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        step: int,
        ema_beta: float,
    ) -> None:
        for local_idx, batch_idx in enumerate(indices):
            key = keys[batch_idx]
            new_a = target_a[local_idx].detach().cpu()
            new_b = target_b[local_idx].detach().cpu()
            old = self.entries.get(key)
            if old is not None and ema_beta > 0:
                new_a = (ema_beta * old["target_a"]) + ((1.0 - ema_beta) * new_a)
                new_b = (ema_beta * old["target_b"]) + ((1.0 - ema_beta) * new_b)
            self.entries[key] = {
                "target_a": new_a,
                "target_b": new_b,
                "refresh_step": step,
            }

    def targets_for(
        self,
        keys: list[tuple[int, ...]],
        *,
        like_a: torch.Tensor,
        like_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_a_rows = []
        target_b_rows = []
        for key in keys:
            entry = self.entries[key]
            target_a_rows.append(entry["target_a"].to(device=like_a.device, dtype=like_a.dtype))
            target_b_rows.append(entry["target_b"].to(device=like_b.device, dtype=like_b.dtype))
        return torch.stack(target_a_rows), torch.stack(target_b_rows)


def action_loss_weighted_interface_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    random_actions: int,
    topk: int,
    local_radius: int,
    temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("action-loss candidate temperature must be positive")
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    candidates = action_loss_candidate_pairs(
        a_logits.detach(),
        b_logits.detach(),
        random_actions=random_actions,
        topk=topk,
        local_radius=local_radius,
        generator=generator,
    )
    with torch.no_grad():
        candidate_losses = score_action_loss_candidates(model, batch, candidates)
        candidate_weights = torch.softmax(-candidate_losses / temperature, dim=-1)
    classes = a_logits.shape[-1]
    target_a, target_b = action_loss_soft_targets(
        a_logits, b_logits, candidates, candidate_weights
    )
    target_loss = action_loss_target_ce(a_logits, b_logits, target_a, target_b)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_idx = torch.zeros(
        (batch.x.shape[0],), dtype=torch.long, device=batch.x.device
    )
    best_idx = candidate_losses.argmin(dim=-1)
    best_pairs = candidates[torch.arange(batch.x.shape[0], device=batch.x.device), best_idx]
    learned_losses = candidate_losses.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
    best_losses = candidate_losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    true_sum = true_a + true_b
    best_sum = best_pairs[:, 0] + best_pairs[:, 1]
    learned_sum = learned_a + learned_b
    metrics = {
        "action_loss_interface_target_loss": float(target_loss.item()),
        "action_loss_candidate_count": int(candidates.shape[1]),
        "action_loss_candidate_temperature": float(temperature),
        "action_loss_candidate_best_improvement": float(
            (learned_losses - best_losses).mean().item()
        ),
        "action_loss_candidate_better_fraction": float(
            (best_losses < learned_losses - 1e-8).float().mean().item()
        ),
        "action_loss_candidate_best_matches_true_operands": float(
            ((best_pairs[:, 0] == true_a) & (best_pairs[:, 1] == true_b))
            .float()
            .mean()
            .item()
        ),
        "action_loss_candidate_best_result_accuracy": float(
            (best_sum == true_sum).float().mean().item()
        ),
        "action_loss_candidate_learned_result_accuracy": float(
            (learned_sum == true_sum).float().mean().item()
        ),
        "action_loss_candidate_soft_target_true_a_mass": float(
            target_a.gather(1, true_a.clamp(max=classes - 1).unsqueeze(-1))
            .mean()
            .item()
        ),
        "action_loss_candidate_soft_target_true_b_mass": float(
            target_b.gather(1, true_b.clamp(max=classes - 1).unsqueeze(-1))
            .mean()
            .item()
        ),
    }
    return target_loss, metrics


def action_loss_replay_interface_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    random_actions: int,
    topk: int,
    local_radius: int,
    temperature: float,
    generator: torch.Generator,
    cache: ActionLossReplayCache,
    step: int,
    refresh_every: int,
    ema_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("action-loss candidate temperature must be positive")
    if refresh_every < 1:
        raise ValueError("action-loss replay refresh interval must be positive")
    if not 0 <= ema_beta < 1:
        raise ValueError("action-loss candidate EMA beta must be in [0, 1)")
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    keys = cache.keys_from_batch(batch)
    refresh_indices = cache.stale_indices(
        keys, step=step, refresh_every=refresh_every
    )
    refreshed_metrics: dict[str, float] = {}
    if refresh_indices:
        refresh_tensor = torch.tensor(
            refresh_indices, dtype=torch.long, device=batch.x.device
        )
        refresh_batch = subset_arithmetic_batch(batch, refresh_tensor)
        refresh_a_logits = a_logits.index_select(0, refresh_tensor)
        refresh_b_logits = b_logits.index_select(0, refresh_tensor)
        candidates = action_loss_candidate_pairs(
            refresh_a_logits.detach(),
            refresh_b_logits.detach(),
            random_actions=random_actions,
            topk=topk,
            local_radius=local_radius,
            generator=generator,
        )
        with torch.no_grad():
            candidate_losses = score_action_loss_candidates(
                model, refresh_batch, candidates
            )
            candidate_weights = torch.softmax(-candidate_losses / temperature, dim=-1)
        refresh_target_a, refresh_target_b = action_loss_soft_targets(
            refresh_a_logits, refresh_b_logits, candidates, candidate_weights
        )
        cache.update(
            keys=keys,
            indices=refresh_indices,
            target_a=refresh_target_a,
            target_b=refresh_target_b,
            step=step,
            ema_beta=ema_beta,
        )
        true_a, true_b = fixed_width_operands_from_batch(
            refresh_batch.x, num_digits=num_digits
        )
        learned_a = refresh_a_logits.argmax(dim=-1)
        learned_b = refresh_b_logits.argmax(dim=-1)
        learned_idx = torch.zeros(
            (refresh_batch.x.shape[0],), dtype=torch.long, device=batch.x.device
        )
        best_idx = candidate_losses.argmin(dim=-1)
        best_pairs = candidates[
            torch.arange(refresh_batch.x.shape[0], device=batch.x.device), best_idx
        ]
        learned_losses = candidate_losses.gather(
            1, learned_idx.unsqueeze(-1)
        ).squeeze(-1)
        best_losses = candidate_losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        true_sum = true_a + true_b
        best_sum = best_pairs[:, 0] + best_pairs[:, 1]
        learned_sum = learned_a + learned_b
        classes = a_logits.shape[-1]
        refreshed_metrics = {
            "action_loss_candidate_best_improvement": float(
                (learned_losses - best_losses).mean().item()
            ),
            "action_loss_candidate_better_fraction": float(
                (best_losses < learned_losses - 1e-8).float().mean().item()
            ),
            "action_loss_candidate_best_matches_true_operands": float(
                ((best_pairs[:, 0] == true_a) & (best_pairs[:, 1] == true_b))
                .float()
                .mean()
                .item()
            ),
            "action_loss_candidate_best_result_accuracy": float(
                (best_sum == true_sum).float().mean().item()
            ),
            "action_loss_candidate_learned_result_accuracy": float(
                (learned_sum == true_sum).float().mean().item()
            ),
            "action_loss_candidate_soft_target_true_a_mass": float(
                refresh_target_a.gather(1, true_a.clamp(max=classes - 1).unsqueeze(-1))
                .mean()
                .item()
            ),
            "action_loss_candidate_soft_target_true_b_mass": float(
                refresh_target_b.gather(1, true_b.clamp(max=classes - 1).unsqueeze(-1))
                .mean()
                .item()
            ),
        }
    target_a, target_b = cache.targets_for(keys, like_a=a_logits, like_b=b_logits)
    target_loss = action_loss_target_ce(a_logits, b_logits, target_a, target_b)
    metrics = {
        "action_loss_interface_target_loss": float(target_loss.item()),
        "action_loss_candidate_count": int(
            1
            + (min(topk, a_logits.shape[-1]) ** 2 if topk > 0 else 0)
            + (2 * ((2 * local_radius) + 1) if local_radius > 0 else 0)
            + random_actions
        ),
        "action_loss_candidate_temperature": float(temperature),
        "action_loss_candidate_refresh_every": int(refresh_every),
        "action_loss_candidate_ema_beta": float(ema_beta),
        "action_loss_replay_cache_size": int(len(cache.entries)),
        "action_loss_replay_refresh_fraction": float(
            len(refresh_indices) / max(1, batch.x.shape[0])
        ),
        "action_loss_candidate_best_improvement": float("nan"),
        "action_loss_candidate_better_fraction": float("nan"),
        "action_loss_candidate_best_matches_true_operands": float("nan"),
        "action_loss_candidate_best_result_accuracy": float("nan"),
        "action_loss_candidate_learned_result_accuracy": float("nan"),
        "action_loss_candidate_soft_target_true_a_mass": float("nan"),
        "action_loss_candidate_soft_target_true_b_mass": float("nan"),
    }
    metrics.update(refreshed_metrics)
    return target_loss, metrics


@torch.no_grad()
def adaptive_interface_trace_rows(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
    target_mode: str,
    answer_format: AnswerFormat = "sum",
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    batch = make_range_batch(
        batch_size=samples,
        num_digits=num_digits,
        operand_max=operand_max,
        rng=rng,
        fixed_width=True,
        device=device,
        answer_format=answer_format,
    )
    was_training = model.training
    model.eval()
    result_targets, result_losses = counterfactual_result_targets(model, batch)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    a_targets, b_targets = select_adaptive_operand_targets(
        a_logits, b_logits, result_targets
    )
    if target_mode == "soft_result":
        _, target_pair_mass = adaptive_soft_result_loss(
            a_logits, b_logits, result_targets
        )
    else:
        target_pair_mass = torch.full(
            result_targets.shape, float("nan"), dtype=torch.float, device=batch.x.device
        )
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    prompts = [decode_tokens(row.tolist()).split("=")[0] + "=" for row in batch.x]
    rows: list[dict[str, object]] = []
    for i in range(samples):
        rows.append(
            {
                "sample": i,
                "prompt": prompts[i],
                "true_a": int(true_a[i].item()),
                "true_b": int(true_b[i].item()),
                "true_sum": int((true_a[i] + true_b[i]).item()),
                "target_result": int(result_targets[i].item()),
                "target_a": int(a_targets[i].item()),
                "target_b": int(b_targets[i].item()),
                "learned_a": int(learned_a[i].item()),
                "learned_b": int(learned_b[i].item()),
                "learned_result": int((learned_a[i] + learned_b[i]).item()),
                "target_result_loss": float(
                    result_losses[i, result_targets[i]].item()
                ),
                "target_pair_mass": float(target_pair_mass[i].item()),
                "target_matches_true_sum": bool(
                    result_targets[i].item() == (true_a[i] + true_b[i]).item()
                ),
                "learned_matches_target_result": bool(
                    (learned_a[i] + learned_b[i]).item() == result_targets[i].item()
                ),
                "target_operands_match_true": bool(
                    a_targets[i].item() == true_a[i].item()
                    and b_targets[i].item() == true_b[i].item()
                ),
            }
        )
    if was_training:
        model.train()
    return rows


def summarize_adaptive_interface_rows(
    rows: list[dict[str, object]]
) -> dict[str, float | int]:
    if not rows:
        return {
            "samples": 0,
            "target_result_accuracy": 0.0,
            "learned_target_result_agreement": 0.0,
            "target_operand_exact_match": 0.0,
            "target_pair_mass": float("nan"),
        }
    pair_masses = [
        float(row["target_pair_mass"])
        for row in rows
        if isinstance(row.get("target_pair_mass"), float)
        and row["target_pair_mass"] == row["target_pair_mass"]
    ]
    return {
        "samples": len(rows),
        "target_result_accuracy": sum(
            int(row["target_matches_true_sum"]) for row in rows
        )
        / len(rows),
        "learned_target_result_agreement": sum(
            int(row["learned_matches_target_result"]) for row in rows
        )
        / len(rows),
        "target_operand_exact_match": sum(
            int(row["target_operands_match_true"]) for row in rows
        )
        / len(rows),
        "target_pair_mass": sum(pair_masses) / len(pair_masses)
        if pair_masses
        else float("nan"),
    }


SEMANTIC_DECODER_CHECKPOINT_PREFIXES = (
    "answer_offset_emb.",
    "answer_decoder.",
    "calculator_hook.output_proj.",
)


def load_semantic_decoder_checkpoint(
    model: TinyGPT, checkpoint_path: Path, *, load_scope: str = "full_model"
) -> None:
    if load_scope not in {"full_model", "semantic_decoder_only"}:
        raise ValueError(
            "semantic decoder checkpoint load scope must be "
            "'full_model' or 'semantic_decoder_only'"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    if load_scope == "semantic_decoder_only":
        state_dict = {
            name: tensor
            for name, tensor in state_dict.items()
            if name.startswith(SEMANTIC_DECODER_CHECKPOINT_PREFIXES)
        }
    state_dict = {
        name: tensor
        for name, tensor in state_dict.items()
        if not (
            name.startswith("calculator_hook.input_proj.")
            and name in model_state
            and tensor.shape != model_state[name].shape
        )
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "calculator_hook.input_proj.weight",
        "calculator_hook.input_proj.bias",
        "calculator_hook.pair_proj.weight",
        "calculator_hook.pair_proj.bias",
        "calculator_hook.result_proj.weight",
        "calculator_hook.result_proj.bias",
    }
    unexpected_nonempty = [name for name in unexpected]
    if load_scope == "semantic_decoder_only":
        allowed_missing.update(
            name
            for name in model_state
            if not name.startswith(SEMANTIC_DECODER_CHECKPOINT_PREFIXES)
        )
    disallowed_missing = [name for name in missing if name not in allowed_missing]
    if disallowed_missing or unexpected_nonempty:
        raise ValueError(
            "semantic decoder checkpoint had incompatible keys: "
            f"missing={disallowed_missing}, unexpected={unexpected_nonempty}"
        )


def load_input_proj_anchor(
    model: TinyGPT, checkpoint_path: Path, *, device: str | torch.device
) -> dict[str, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("input-proj anchor requires a calculator hook")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    required = {
        "weight": "calculator_hook.input_proj.weight",
        "bias": "calculator_hook.input_proj.bias",
    }
    anchor: dict[str, torch.Tensor] = {}
    model_state = model.state_dict()
    for short_name, full_name in required.items():
        if full_name not in state_dict:
            raise ValueError(f"anchor checkpoint missing {full_name}")
        if state_dict[full_name].shape != model_state[full_name].shape:
            raise ValueError(f"anchor checkpoint has incompatible {full_name} shape")
        anchor[short_name] = state_dict[full_name].detach().clone().to(device)
    return anchor


def input_proj_anchor_weight(
    *, initial_weight: float, decay_steps: int, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    return initial_weight * max(0.0, 1.0 - (step / decay_steps))


def input_proj_anchor_loss(
    model: TinyGPT, anchor: dict[str, torch.Tensor]
) -> torch.Tensor:
    if model.calculator_hook is None:
        raise ValueError("input-proj anchor loss requires a calculator hook")
    weight_delta = model.calculator_hook.input_proj.weight - anchor["weight"]
    bias_delta = model.calculator_hook.input_proj.bias - anchor["bias"]
    return 0.5 * (weight_delta.pow(2).mean() + bias_delta.pow(2).mean())


def input_proj_anchor_delta_summary(
    model: TinyGPT, anchor: dict[str, torch.Tensor]
) -> dict[str, float]:
    if model.calculator_hook is None:
        raise ValueError("input-proj anchor delta requires a calculator hook")
    weight_delta = model.calculator_hook.input_proj.weight.detach() - anchor["weight"]
    bias_delta = model.calculator_hook.input_proj.bias.detach() - anchor["bias"]
    return {
        "weight_l2": float(weight_delta.norm().item()),
        "weight_max_abs": float(weight_delta.abs().max().item()),
        "bias_l2": float(bias_delta.norm().item()),
        "bias_max_abs": float(bias_delta.abs().max().item()),
    }


def freeze_semantic_decoder_parameters(model: TinyGPT) -> None:
    if model.calculator_hook is not None:
        for param in model.calculator_hook.output_proj.parameters():
            param.requires_grad = False
    if model.answer_offset_emb is not None:
        for param in model.answer_offset_emb.parameters():
            param.requires_grad = False
    if model.answer_decoder is not None:
        for param in model.answer_decoder.parameters():
            param.requires_grad = False
    if (
        model.calculator_hook is not None
        and model.cfg.calculator_action_head in {"joint_pair", "result_space"}
    ):
        for param in model.calculator_hook.input_proj.parameters():
            param.requires_grad = False


def freeze_upstream_encoder_parameters(model: TinyGPT) -> None:
    for module in [model.tok_emb, model.pos_emb, model.blocks, model.ln_f, model.lm_head]:
        for param in module.parameters():
            param.requires_grad = False


def adaptive_optimizer_param_groups(
    model: TinyGPT,
    *,
    lr: float,
    input_proj_lr: float | None,
    upstream_lr: float | None,
    weight_decay: float,
) -> list[dict[str, object]]:
    effective_input_lr = lr if input_proj_lr is None else input_proj_lr
    effective_upstream_lr = lr if upstream_lr is None else upstream_lr
    input_params: list[torch.nn.Parameter] = []
    pair_params: list[torch.nn.Parameter] = []
    result_params: list[torch.nn.Parameter] = []
    upstream_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("calculator_hook.input_proj."):
            input_params.append(param)
        elif name.startswith("calculator_hook.pair_proj."):
            pair_params.append(param)
        elif name.startswith("calculator_hook.result_proj."):
            result_params.append(param)
        else:
            upstream_params.append(param)

    groups: list[dict[str, object]] = []
    if input_params:
        groups.append(
            {
                "params": input_params,
                "lr": effective_input_lr,
                "weight_decay": weight_decay,
                "name": "calculator_hook.input_proj",
            }
        )
    if pair_params:
        groups.append(
            {
                "params": pair_params,
                "lr": effective_input_lr,
                "weight_decay": weight_decay,
                "name": "calculator_hook.pair_proj",
            }
        )
    if result_params:
        groups.append(
            {
                "params": result_params,
                "lr": effective_input_lr,
                "weight_decay": weight_decay,
                "name": "calculator_hook.result_proj",
            }
        )
    if upstream_params:
        groups.append(
            {
                "params": upstream_params,
                "lr": effective_upstream_lr,
                "weight_decay": weight_decay,
                "name": "upstream",
            }
        )
    if not groups:
        raise ValueError("no trainable parameters for adaptive optimizer")
    return groups


def trainable_parameter_summary(model: TinyGPT) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = (
            "calculator_hook.input_proj"
            if name.startswith("calculator_hook.input_proj.")
            else "calculator_hook.pair_proj"
            if name.startswith("calculator_hook.pair_proj.")
            else "calculator_hook.result_proj"
            if name.startswith("calculator_hook.result_proj.")
            else "upstream"
        )
        group = groups.setdefault(
            group_name,
            {"name": group_name, "parameter_count": 0, "parameters": []},
        )
        group["parameter_count"] = int(group["parameter_count"]) + param.numel()
        group["parameters"].append(name)
    return [groups[name] for name in sorted(groups)]


def make_range_batch(
    *,
    batch_size: int,
    num_digits: int,
    operand_max: int,
    rng: random.Random,
    fixed_width: bool,
    device: str | torch.device,
    answer_format: AnswerFormat = "sum",
) -> ArithmeticBatch:
    seq_len = max_sequence_length(num_digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for _ in range(batch_size):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        if fixed_width:
            prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
        else:
            prompt = f"{a}+{b}="
        ids = tokenize(
            prompt
            + answer_target(
                a,
                b,
                num_digits,
                answer_format=answer_format,
                fixed_width=fixed_width,
            )
        )
        samples.append(pad_sequence(ids, seq_len))
        masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))

    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def make_exhaustive_range_batch(
    *,
    num_digits: int,
    operand_max: int,
    fixed_width: bool,
    device: str | torch.device,
    answer_format: AnswerFormat = "sum",
) -> ArithmeticBatch:
    seq_len = max_sequence_length(num_digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            if fixed_width:
                prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
            else:
                prompt = f"{a}+{b}="
            ids = tokenize(
                prompt
                + answer_target(
                    a,
                    b,
                    num_digits,
                    answer_format=answer_format,
                    fixed_width=fixed_width,
                )
            )
            samples.append(pad_sequence(ids, seq_len))
            masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))

    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def auxiliary_operand_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    num_digits: int,
    *,
    grad_upstream: bool = False,
) -> torch.Tensor:
    if model.calculator_hook is None:
        raise ValueError("auxiliary operand loss requires a calculator hook")
    with torch.no_grad():
        targets = make_oracle_operands_from_batch(batch.x, num_digits=num_digits)
        eq_pos = (batch.x == EQ_ID).float().argmax(dim=-1).long()
        batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
        if model.cfg.calculator_read_position in {"operands", "operand_spans"}:
            a_pos = torch.full_like(eq_pos, num_digits - 1)
            b_pos = torch.full_like(eq_pos, (num_digits + 1) + (num_digits - 1))
        else:
            a_pos = eq_pos
            b_pos = eq_pos
        target_a = targets[batch_idx, a_pos, 0]
        target_b = targets[batch_idx, b_pos, 1]

    if model.cfg.calculator_action_head == "joint_pair":
        classes = model.cfg.calculator_operand_vocab_size
        target_pair = target_a * classes + target_b
        if grad_upstream:
            pair_logits, _, _, _ = calculator_read_pair_logits(model, batch)
        else:
            with torch.no_grad():
                B, T = batch.x.shape
                assert T <= model.cfg.block_size, (
                    f"sequence length {T} > block_size {model.cfg.block_size}"
                )
                pos = torch.arange(T, device=batch.x.device)
                residual = model.tok_emb(batch.x) + model.pos_emb(pos)
                if model.cfg.calculator_hook_after_layer > 0:
                    for i, block in enumerate(model.blocks, start=1):
                        residual = block(residual)
                        if i == model.cfg.calculator_hook_after_layer:
                            break
                positions = model._calculator_read_positions(batch.x)
                if model.cfg.calculator_read_position == "operand_spans":
                    pair_input = torch.cat(
                        model.calculator_hook._operand_span_inputs(residual, positions),
                        dim=-1,
                    ).detach()
                else:
                    pair_input = torch.cat(
                        [
                            residual[batch_idx, positions["a"]],
                            residual[batch_idx, positions["b"]],
                        ],
                        dim=-1,
                    ).detach()
        if not grad_upstream:
            if model.calculator_hook.pair_proj is None:
                raise ValueError("joint auxiliary loss requires a pair projection")
            pair_logits = model.calculator_hook.pair_proj(pair_input)
        return torch.nn.functional.cross_entropy(pair_logits, target_pair)

    if grad_upstream:
        a_eq_logits, b_eq_logits, _, _ = calculator_read_operand_logits(model, batch)
    else:
        _, diagnostics = model(batch.x, return_diagnostics=True)
        residual = diagnostics["calculator_read_residual"]
        if model.cfg.calculator_read_position == "operand_spans":
            positions = model._calculator_read_positions(batch.x)
            a_input, b_input = model.calculator_hook._operand_span_inputs(
                residual, positions
            )
            a_eq_logits, _ = model.calculator_hook.input_proj(a_input).split(
                model.cfg.calculator_operand_vocab_size, dim=-1
            )
            _, b_eq_logits = model.calculator_hook.input_proj(b_input).split(
                model.cfg.calculator_operand_vocab_size, dim=-1
            )
        else:
            operand_logits = model.calculator_hook.input_proj(residual)
            a_logits, b_logits = operand_logits.split(
                model.cfg.calculator_operand_vocab_size, dim=-1
            )
            a_eq_logits = a_logits[batch_idx, a_pos]
            b_eq_logits = b_logits[batch_idx, b_pos]
    return (
        torch.nn.functional.cross_entropy(a_eq_logits, target_a)
        + torch.nn.functional.cross_entropy(b_eq_logits, target_b)
    ) / 2


def auxiliary_operand_weight(
    *, initial_weight: float, decay_steps: int, floor: float, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    decayed = initial_weight * max(0.0, 1.0 - (step / decay_steps))
    return max(floor, decayed)


def adaptive_interface_weight(
    *, initial_weight: float, decay_steps: int, floor: float, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    decayed = initial_weight * max(0.0, 1.0 - (step / decay_steps))
    return max(floor, decayed)


def make_model_config(
    num_digits: int,
    variant: str,
    *,
    injection_scale: float = 1.0,
    operand_vocab_size: int | None = None,
    calculator_estimator: str = "ste",
    calculator_action_head: str = "independent_operands",
    calculator_read_position: str = "eq",
    calculator_read_span_width: int = 1,
    calculator_injection_mode: str = "add",
    calculator_bottleneck_mode: str = "none",
    calculator_output_format: str = "sum",
    answer_decoder_interaction: str | None = None,
    calculator_result_head_hidden_size: int = 0,
    relaxed_calculator_temperature: float = 1.0,
    relaxed_calculator_mode: str = "deterministic",
    relaxed_calculator_hard_forward: bool = True,
    answer_format: AnswerFormat = "sum",
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    mlp_expansion: int = 4,
    calculator_hook_after_layer: int | None = None,
) -> GPTConfig:
    operand_vocab_size = operand_vocab_size or 10**num_digits
    calculator_enabled = variant in {"model-b", "model-c"}
    calculator_mode = "add" if variant == "model-c" else "off"
    if calculator_hook_after_layer is None:
        calculator_hook_after_layer = min(2, n_layer)
    if answer_decoder_interaction is None:
        answer_decoder_interaction = (
            "product" if calculator_output_format == "sum_left_operand" else "none"
        )
    return GPTConfig(
        block_size=max_sequence_length(num_digits, answer_format=answer_format) - 1,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_expansion=mlp_expansion,
        calculator_enabled=calculator_enabled,
        calculator_mode=calculator_mode,
        calculator_hook_after_layer=calculator_hook_after_layer,
        calculator_operand_vocab_size=operand_vocab_size,
        calculator_result_vocab_size=(2 * operand_vocab_size) - 1,
        calculator_injection_scale=injection_scale,
        calculator_injection_mode=calculator_injection_mode,
        calculator_estimator=calculator_estimator,
        calculator_action_head=calculator_action_head,
        calculator_read_position=calculator_read_position,
        calculator_read_span_width=calculator_read_span_width,
        calculator_bottleneck_mode=calculator_bottleneck_mode,
        calculator_output_format=calculator_output_format,
        answer_decoder_interaction=answer_decoder_interaction,
        calculator_result_head_hidden_size=calculator_result_head_hidden_size,
        relaxed_calculator_temperature=relaxed_calculator_temperature,
        relaxed_calculator_mode=relaxed_calculator_mode,
        relaxed_calculator_hard_forward=relaxed_calculator_hard_forward,
    )


def run_variant(
    *,
    num_digits: int,
    args: argparse.Namespace,
    base_run_dir: Path,
    device: str,
) -> dict[str, object]:
    seed = args.seed + num_digits
    torch.manual_seed(seed)
    rng = random.Random(seed)
    candidate_generator = torch.Generator(device=device)
    candidate_generator.manual_seed(seed + 90_000)

    operand_max = args.operand_max
    if operand_max is None:
        operand_max = 10**num_digits - 1
    if operand_max >= 10**num_digits:
        raise ValueError("--operand-max must fit inside --digits")
    calculator_operand_vocab_size = args.calculator_operand_vocab_size
    if calculator_operand_vocab_size is None:
        calculator_operand_vocab_size = 10**num_digits
    if operand_max >= calculator_operand_vocab_size:
        raise ValueError(
            "--calculator-operand-vocab-size must be greater than --operand-max"
        )

    cfg = make_model_config(
        num_digits,
        args.variant,
        injection_scale=args.injection_scale,
        operand_vocab_size=calculator_operand_vocab_size,
        calculator_estimator=args.calculator_estimator,
        calculator_action_head=args.calculator_action_head,
        calculator_read_position=args.calculator_read_position,
        calculator_read_span_width=args.calculator_read_span_width,
        calculator_injection_mode=args.calculator_injection_mode,
        calculator_bottleneck_mode=args.calculator_bottleneck_mode,
        calculator_output_format=args.calculator_output_format,
        answer_decoder_interaction=args.answer_decoder_interaction,
        calculator_result_head_hidden_size=args.calculator_result_head_hidden_size,
        relaxed_calculator_temperature=args.relaxed_calculator_temperature,
        relaxed_calculator_mode=args.relaxed_calculator_mode,
        relaxed_calculator_hard_forward=args.relaxed_calculator_hard_forward,
        answer_format=args.answer_format,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=args.calculator_hook_after_layer,
    )
    model = TinyGPT(cfg).to(device)
    use_result_space_reinforce = (
        args.calculator_estimator == "reinforce"
        and args.calculator_action_head == "result_space"
    )
    if args.semantic_decoder_checkpoint is not None:
        load_semantic_decoder_checkpoint(
            model,
            args.semantic_decoder_checkpoint,
            load_scope=args.semantic_decoder_checkpoint_load_scope,
        )
    input_proj_anchor = None
    if args.input_proj_anchor_checkpoint is not None:
        input_proj_anchor = load_input_proj_anchor(
            model, args.input_proj_anchor_checkpoint, device=device
        )
    if (
        args.freeze_semantic_decoder
        and (
            args.calculator_estimator
            in {
                "adaptive_interface",
                "action_loss_weighted_interface",
                "action_loss_replay_interface",
                "action_loss_full_enum_interface",
                "action_loss_full_enum_joint_interface",
                "identifiable_full_enum_local_target",
                "full_enum_expected_answer_loss",
                "gumbel_concrete_interface",
                "direct_feedback_alignment",
            }
            or use_result_space_reinforce
        )
    ):
        freeze_semantic_decoder_parameters(model)
    if (
        args.freeze_upstream_encoder
        and (
            args.calculator_estimator
            in {
                "adaptive_interface",
                "action_loss_weighted_interface",
                "action_loss_replay_interface",
                "action_loss_full_enum_interface",
                "action_loss_full_enum_joint_interface",
                "identifiable_full_enum_local_target",
                "full_enum_expected_answer_loss",
                "gumbel_concrete_interface",
                "direct_feedback_alignment",
            }
            or use_result_space_reinforce
        )
    ):
        freeze_upstream_encoder_parameters(model)
    trainable_groups = trainable_parameter_summary(model)
    exhaustive_grid_batch = None
    exhaustive_grid_size = None
    if args.exhaustive_grid_batch:
        exhaustive_grid_batch = make_exhaustive_range_batch(
            num_digits=num_digits,
            operand_max=operand_max,
            fixed_width=True,
            device=device,
            answer_format=args.answer_format,
        )
        exhaustive_grid_size = int(exhaustive_grid_batch.x.shape[0])
    if (
        args.calculator_estimator
        in {
            "adaptive_interface",
            "action_loss_weighted_interface",
            "action_loss_replay_interface",
            "action_loss_full_enum_interface",
            "action_loss_full_enum_joint_interface",
            "identifiable_full_enum_local_target",
            "full_enum_expected_answer_loss",
            "gumbel_concrete_interface",
            "direct_feedback_alignment",
        }
        or use_result_space_reinforce
    ):
        optim = torch.optim.AdamW(
            adaptive_optimizer_param_groups(
                model,
                lr=args.lr,
                input_proj_lr=args.input_proj_lr,
                upstream_lr=args.upstream_lr,
                weight_decay=args.weight_decay,
            ),
            betas=(0.9, 0.95),
        )
    else:
        optim = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )

    run_name = f"{args.variant}-{num_digits}digit-seed{seed}"
    run_dir = base_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_cfg = TrainConfig(
        variant=args.variant,
        run_name=run_name,
        seed=seed,
        num_digits=num_digits,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_samples=args.eval_samples,
        lr=args.lr,
        answer_loss_weight=args.answer_loss_weight,
        answer_format=args.answer_format,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        fixed_width=True,
        operand_max=args.operand_max,
        exhaustive_grid_batch=args.exhaustive_grid_batch,
        exhaustive_grid_size=exhaustive_grid_size,
        calculator_operand_vocab_size=calculator_operand_vocab_size,
        oracle_train=args.oracle_train,
        oracle_warmup_steps=args.oracle_warmup_steps,
        aux_operand_loss_weight=args.aux_operand_loss_weight,
        aux_operand_loss_decay_steps=args.aux_operand_loss_decay_steps,
        aux_operand_loss_floor=args.aux_operand_loss_floor,
        aux_operand_loss_grad_upstream=args.aux_operand_loss_grad_upstream,
        snapshot_every=args.snapshot_every,
        snapshot_samples=args.snapshot_samples,
        checkpoint_every=args.checkpoint_every,
        calculator_estimator=args.calculator_estimator,
        calculator_action_head=args.calculator_action_head,
        calculator_read_position=args.calculator_read_position,
        calculator_read_span_width=args.calculator_read_span_width,
        calculator_injection_mode=args.calculator_injection_mode,
        calculator_bottleneck_mode=args.calculator_bottleneck_mode,
        calculator_output_format=args.calculator_output_format,
        answer_decoder_interaction=cfg.answer_decoder_interaction,
        semantic_decoder_checkpoint=(
            str(args.semantic_decoder_checkpoint)
            if args.semantic_decoder_checkpoint is not None
            else None
        ),
        semantic_decoder_checkpoint_load_scope=args.semantic_decoder_checkpoint_load_scope,
        adaptive_interface_loss_weight=args.adaptive_interface_loss_weight,
        adaptive_interface_loss_decay_steps=args.adaptive_interface_loss_decay_steps,
        adaptive_interface_loss_floor=args.adaptive_interface_loss_floor,
        adaptive_interface_target_mode=args.adaptive_interface_target_mode,
        adaptive_interface_entropy_weight=args.adaptive_interface_entropy_weight,
        action_loss_candidate_random=args.action_loss_candidate_random,
        action_loss_candidate_topk=args.action_loss_candidate_topk,
        action_loss_candidate_local_radius=args.action_loss_candidate_local_radius,
        action_loss_candidate_temperature=args.action_loss_candidate_temperature,
        action_loss_candidate_refresh_every=args.action_loss_candidate_refresh_every,
        action_loss_candidate_ema_beta=args.action_loss_candidate_ema_beta,
        action_loss_full_enum_temperature=args.action_loss_full_enum_temperature,
        action_loss_full_enum_min_probability_floor=(
            args.action_loss_full_enum_min_probability_floor
        ),
        action_loss_full_enum_chunk_size=args.action_loss_full_enum_chunk_size,
        action_loss_full_enum_target_mode=args.action_loss_full_enum_target_mode,
        expected_answer_loss_weight=args.expected_answer_loss_weight,
        expected_answer_loss_policy_temperature=(
            args.expected_answer_loss_policy_temperature
        ),
        expected_answer_loss_cost_normalization=(
            args.expected_answer_loss_cost_normalization
        ),
        expected_answer_loss_entropy_weight=args.expected_answer_loss_entropy_weight,
        expected_answer_loss_entropy_decay_steps=(
            args.expected_answer_loss_entropy_decay_steps
        ),
        expected_answer_loss_chunk_size=args.expected_answer_loss_chunk_size,
        expected_answer_loss_gradient_diagnostic_only=(
            args.expected_answer_loss_gradient_diagnostic_only
        ),
        result_boundary_target_loss_weight=args.result_boundary_target_loss_weight,
        result_boundary_target_mode=args.result_boundary_target_mode,
        result_boundary_target_temperature=args.result_boundary_target_temperature,
        result_boundary_target_min_probability_floor=(
            args.result_boundary_target_min_probability_floor
        ),
        result_boundary_target_chunk_size=args.result_boundary_target_chunk_size,
        result_policy_entropy_weight=args.result_policy_entropy_weight,
        result_policy_batch_diversity_weight=(
            args.result_policy_batch_diversity_weight
        ),
        result_policy_improvement_assignment_weight=(
            args.result_policy_improvement_assignment_weight
        ),
        result_policy_improvement_assignment_min_improvement=(
            args.result_policy_improvement_assignment_min_improvement
        ),
        result_policy_improvement_assignment_quota_multiplier=(
            args.result_policy_improvement_assignment_quota_multiplier
        ),
        result_policy_stabilization_temperature=(
            args.result_policy_stabilization_temperature
        ),
        result_policy_stabilization_decay_steps=(
            args.result_policy_stabilization_decay_steps
        ),
        calculator_causal_gap_weight=args.calculator_causal_gap_weight,
        calculator_causal_gap_margin=args.calculator_causal_gap_margin,
        boundary_feedback_weight=args.boundary_feedback_weight,
        boundary_feedback_mode=args.boundary_feedback_mode,
        boundary_feedback_seed=args.boundary_feedback_seed,
        boundary_feedback_gradient_diagnostic_only=(
            args.boundary_feedback_gradient_diagnostic_only
        ),
        shadow_feedback_mode=args.shadow_feedback_mode,
        shadow_feedback_ridge=args.shadow_feedback_ridge,
        shadow_feedback_weight=args.shadow_feedback_weight,
        shadow_feedback_heldout_fraction=args.shadow_feedback_heldout_fraction,
        shadow_feedback_hidden_size=args.shadow_feedback_hidden_size,
        shadow_feedback_dropout=args.shadow_feedback_dropout,
        shadow_feedback_online_lr=args.shadow_feedback_online_lr,
        shadow_feedback_weight_decay=args.shadow_feedback_weight_decay,
        shadow_feedback_warmup_steps=args.shadow_feedback_warmup_steps,
        shadow_feedback_updates_per_step=args.shadow_feedback_updates_per_step,
        shadow_feedback_apply_max_norm=args.shadow_feedback_apply_max_norm,
        shadow_feedback_refresh_every=args.shadow_feedback_refresh_every,
        shadow_feedback_validation_fraction=(
            args.shadow_feedback_validation_fraction
        ),
        shadow_feedback_validation_every=args.shadow_feedback_validation_every,
        shadow_feedback_validation_loss_weight=(
            args.shadow_feedback_validation_loss_weight
        ),
        shadow_feedback_validation_gradient_loss_weight=(
            args.shadow_feedback_validation_gradient_loss_weight
        ),
        shadow_feedback_validation_gradient_norm_weight=(
            args.shadow_feedback_validation_gradient_norm_weight
        ),
        shadow_feedback_target_normalization=(
            args.shadow_feedback_target_normalization
        ),
        shadow_feedback_target_transform=args.shadow_feedback_target_transform,
        shadow_feedback_feature_mode=args.shadow_feedback_feature_mode,
        shadow_feedback_feature_normalization=(
            args.shadow_feedback_feature_normalization
        ),
        shadow_feedback_loss_mode=args.shadow_feedback_loss_mode,
        shadow_feedback_selection_score_mode=(
            args.shadow_feedback_selection_score_mode
        ),
        shadow_feedback_selection_gap_penalty=(
            args.shadow_feedback_selection_gap_penalty
        ),
        shadow_feedback_gradient_diagnostic_only=(
            args.shadow_feedback_gradient_diagnostic_only
        ),
        calculator_result_head_hidden_size=args.calculator_result_head_hidden_size,
        relaxed_calculator_temperature=args.relaxed_calculator_temperature,
        relaxed_calculator_final_temperature=(
            args.relaxed_calculator_final_temperature
        ),
        relaxed_calculator_temperature_decay_steps=(
            args.relaxed_calculator_temperature_decay_steps
        ),
        relaxed_calculator_mode=args.relaxed_calculator_mode,
        relaxed_calculator_hard_forward=args.relaxed_calculator_hard_forward,
        relaxed_calculator_entropy_weight=args.relaxed_calculator_entropy_weight,
        relaxed_calculator_entropy_decay_steps=(
            args.relaxed_calculator_entropy_decay_steps
        ),
        input_proj_anchor_checkpoint=(
            str(args.input_proj_anchor_checkpoint)
            if args.input_proj_anchor_checkpoint is not None
            else None
        ),
        input_proj_anchor_weight=args.input_proj_anchor_weight,
        input_proj_anchor_decay_steps=args.input_proj_anchor_decay_steps,
        input_proj_lr=(
            args.lr if args.input_proj_lr is None else args.input_proj_lr
        ),
        upstream_lr=args.lr if args.upstream_lr is None else args.upstream_lr,
        optimizer_step_max_delta_norm=args.optimizer_step_max_delta_norm,
        optimizer_step_acceptance_mode=args.optimizer_step_acceptance_mode,
        optimizer_step_acceptance_tolerance=(
            args.optimizer_step_acceptance_tolerance
        ),
        optimizer_step_line_search_scales=args.optimizer_step_line_search_scales,
        freeze_semantic_decoder=args.freeze_semantic_decoder,
        freeze_upstream_encoder=args.freeze_upstream_encoder,
        trainable_parameter_groups=trainable_groups,
        reinforce_baseline_mode=args.reinforce_baseline_mode,
        reinforce_num_samples_per_prompt=args.reinforce_num_samples_per_prompt,
        reinforce_baseline_beta=args.reinforce_baseline_beta,
        reinforce_entropy_weight=args.reinforce_entropy_weight,
        reinforce_entropy_decay_steps=args.reinforce_entropy_decay_steps,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=cfg.calculator_hook_after_layer,
        model=asdict(cfg),
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(train_cfg), indent=2) + "\n"
    )
    if args.reinforce_gradient_diagnostic_only:
        if exhaustive_grid_batch is not None:
            diagnostic_batch = exhaustive_grid_batch
        else:
            diagnostic_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        diagnostic_summary = run_reinforce_gradient_diagnostic(
            model,
            diagnostic_batch,
            num_digits=num_digits,
            baseline_mode=args.reinforce_baseline_mode,
            num_samples_per_prompt=args.reinforce_num_samples_per_prompt,
            global_baseline=None,
            entropy_weight=args.reinforce_entropy_weight,
            result_boundary_target_mode=args.result_boundary_target_mode,
            result_boundary_target_temperature=args.result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                args.result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=args.result_boundary_target_chunk_size,
        )
        diagnostic_summary["run_dir"] = str(run_dir)
        diagnostic_summary["num_digits"] = int(num_digits)
        (run_dir / "reinforce_gradient_diagnostic_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2) + "\n"
        )
        print(
            "reinforce gradient diagnostic: "
            f"pg_result_grad={diagnostic_summary['pg_result_proj_grad_l2']:.6f} "
            f"cos={diagnostic_summary['pg_vs_boundary_result_proj_cosine']:.6f} "
            f"adv_std={diagnostic_summary['advantage_std']:.6f} "
            f"sample_result_acc={diagnostic_summary['sampled_result_accuracy']:.4f}"
        )
        return {
            "num_digits": num_digits,
            "exact_match": float("nan"),
            "final_loss": float("nan"),
            "run_dir": str(run_dir),
            "reinforce_gradient_diagnostic": diagnostic_summary,
        }
    if args.expected_answer_loss_gradient_diagnostic_only:
        if exhaustive_grid_batch is not None:
            diagnostic_batch = exhaustive_grid_batch
        else:
            diagnostic_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        diagnostic_summary = run_expected_answer_loss_gradient_diagnostic(
            model,
            diagnostic_batch,
            num_digits=num_digits,
            policy_temperature=args.expected_answer_loss_policy_temperature,
            cost_normalization=args.expected_answer_loss_cost_normalization,
            entropy_weight=args.expected_answer_loss_entropy_weight,
            expected_answer_loss_chunk_size=args.expected_answer_loss_chunk_size,
            baseline_mode=args.reinforce_baseline_mode,
            num_samples_per_prompt=args.reinforce_num_samples_per_prompt,
            global_baseline=None,
            reinforce_entropy_weight=args.reinforce_entropy_weight,
            result_boundary_target_mode=args.result_boundary_target_mode,
            result_boundary_target_temperature=args.result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                args.result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=args.result_boundary_target_chunk_size,
        )
        diagnostic_summary["run_dir"] = str(run_dir)
        diagnostic_summary["num_digits"] = int(num_digits)
        (run_dir / "expected_answer_loss_gradient_diagnostic_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2) + "\n"
        )
        print(
            "expected answer-loss gradient diagnostic: "
            f"exact_result_grad={diagnostic_summary['exact_result_proj_grad_l2']:.6f} "
            f"exact_boundary_cos={diagnostic_summary['exact_vs_boundary_result_proj_cosine']:.6f} "
            f"pg_exact_cos={diagnostic_summary['pg_vs_exact_result_proj_cosine']:.6f} "
            f"learned_acc={diagnostic_summary['expected_answer_loss_hard_learned_calc_accuracy']:.4f}"
        )
        return {
            "num_digits": num_digits,
            "exact_match": float("nan"),
            "final_loss": float("nan"),
            "run_dir": str(run_dir),
            "expected_answer_loss_gradient_diagnostic": diagnostic_summary,
        }
    if args.boundary_feedback_gradient_diagnostic_only:
        if exhaustive_grid_batch is not None:
            diagnostic_batch = exhaustive_grid_batch
        else:
            diagnostic_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        diagnostic_summary = run_boundary_feedback_gradient_diagnostic(
            model,
            diagnostic_batch,
            num_digits=num_digits,
            feedback_mode=args.boundary_feedback_mode,
            feedback_seed=args.boundary_feedback_seed,
            result_boundary_target_mode=args.result_boundary_target_mode,
            result_boundary_target_temperature=args.result_boundary_target_temperature,
            result_boundary_target_min_probability_floor=(
                args.result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=args.result_boundary_target_chunk_size,
        )
        diagnostic_summary["run_dir"] = str(run_dir)
        diagnostic_summary["num_digits"] = int(num_digits)
        (run_dir / "boundary_feedback_gradient_diagnostic_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2) + "\n"
        )
        print(
            "boundary-feedback gradient diagnostic: "
            f"feedback_result_grad={diagnostic_summary['feedback_result_proj_grad_l2']:.6f} "
            f"result_cos={diagnostic_summary['feedback_vs_boundary_result_proj_cosine']:.6f} "
            f"upstream_cos={diagnostic_summary['feedback_vs_boundary_upstream_cosine']:.6f} "
            f"learned_acc={diagnostic_summary['boundary_feedback_hard_learned_calc_accuracy']:.4f}"
        )
        return {
            "num_digits": num_digits,
            "exact_match": float("nan"),
            "final_loss": float("nan"),
            "run_dir": str(run_dir),
            "boundary_feedback_gradient_diagnostic": diagnostic_summary,
        }
    if args.shadow_feedback_gradient_diagnostic_only:
        if exhaustive_grid_batch is not None:
            diagnostic_batch = exhaustive_grid_batch
        else:
            diagnostic_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        if args.shadow_feedback_mode == "online_mlp":
            diagnostic_summary = run_online_shadow_feedback_gradient_diagnostic(
                model,
                diagnostic_batch,
                num_digits=num_digits,
                heldout_fraction=args.shadow_feedback_heldout_fraction,
                hidden_size=args.shadow_feedback_hidden_size,
                dropout=args.shadow_feedback_dropout,
                learning_rate=args.shadow_feedback_online_lr,
                weight_decay=args.shadow_feedback_weight_decay,
                warmup_steps=args.shadow_feedback_warmup_steps,
                updates_per_step=args.shadow_feedback_updates_per_step,
                validation_fraction=args.shadow_feedback_validation_fraction,
                validation_every=args.shadow_feedback_validation_every,
                validation_loss_weight=args.shadow_feedback_validation_loss_weight,
                validation_gradient_loss_weight=(
                    args.shadow_feedback_validation_gradient_loss_weight
                ),
                validation_gradient_norm_weight=(
                    args.shadow_feedback_validation_gradient_norm_weight
                ),
                target_normalization=args.shadow_feedback_target_normalization,
                target_transform=args.shadow_feedback_target_transform,
                feature_mode=args.shadow_feedback_feature_mode,
                feature_normalization=args.shadow_feedback_feature_normalization,
                loss_mode=args.shadow_feedback_loss_mode,
                selection_score_mode=args.shadow_feedback_selection_score_mode,
                selection_gap_penalty=args.shadow_feedback_selection_gap_penalty,
                result_boundary_target_mode=args.result_boundary_target_mode,
                result_boundary_target_temperature=(
                    args.result_boundary_target_temperature
                ),
                result_boundary_target_min_probability_floor=(
                    args.result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=(
                    args.result_boundary_target_chunk_size
                ),
            )
        else:
            diagnostic_summary = run_shadow_feedback_gradient_diagnostic(
                model,
                diagnostic_batch,
                num_digits=num_digits,
                ridge=args.shadow_feedback_ridge,
                heldout_fraction=args.shadow_feedback_heldout_fraction,
                result_boundary_target_mode=args.result_boundary_target_mode,
                result_boundary_target_temperature=(
                    args.result_boundary_target_temperature
                ),
                result_boundary_target_min_probability_floor=(
                    args.result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=(
                    args.result_boundary_target_chunk_size
                ),
            )
        diagnostic_summary["run_dir"] = str(run_dir)
        diagnostic_summary["num_digits"] = int(num_digits)
        (run_dir / "shadow_feedback_gradient_diagnostic_summary.json").write_text(
            json.dumps(diagnostic_summary, indent=2) + "\n"
        )
        shadow_result_key = (
            "heldout_shadow_result_proj_grad_l2"
            if args.shadow_feedback_heldout_fraction > 0
            else "shadow_result_proj_grad_l2"
        )
        shadow_result_cosine_key = (
            "heldout_shadow_vs_boundary_result_proj_cosine"
            if args.shadow_feedback_heldout_fraction > 0
            else "shadow_vs_boundary_result_proj_cosine"
        )
        shadow_upstream_cosine_key = (
            "heldout_shadow_vs_boundary_upstream_cosine"
            if args.shadow_feedback_heldout_fraction > 0
            else "shadow_vs_boundary_upstream_cosine"
        )
        shadow_fit_cosine_key = (
            "shadow_feedback_final_fit_prediction_cosine"
            if args.shadow_feedback_mode == "online_mlp"
            else "fit_shadow_feedback_fit_cosine"
            if args.shadow_feedback_heldout_fraction > 0
            else "shadow_feedback_fit_cosine"
        )
        print(
            "shadow-feedback gradient diagnostic: "
            f"mode={args.shadow_feedback_mode} "
            f"shadow_result_grad={diagnostic_summary[shadow_result_key]:.6f} "
            f"result_cos={diagnostic_summary[shadow_result_cosine_key]:.6f} "
            f"upstream_cos={diagnostic_summary[shadow_upstream_cosine_key]:.6f} "
            f"fit_cos={diagnostic_summary[shadow_fit_cosine_key]:.6f}"
        )
        return {
            "num_digits": num_digits,
            "exact_match": float("nan"),
            "final_loss": float("nan"),
            "run_dir": str(run_dir),
            "shadow_feedback_gradient_diagnostic": diagnostic_summary,
        }

    shadow_feedback_weights: torch.Tensor | None = None
    online_shadow_feedback_artifacts: dict[str, object] | None = None
    shadow_feedback_calibration_metrics: dict[str, float | int | str] = {}
    online_shadow_feedback_refresh_history: list[dict[str, float | int | str]] = []

    def fit_online_shadow_feedback_artifacts(
        shadow_calibration_batch: ArithmeticBatch,
        *,
        refresh_step: int,
    ) -> tuple[dict[str, float | int | str], dict[str, object]]:
        diagnostic_result = run_online_shadow_feedback_gradient_diagnostic(
            model,
            shadow_calibration_batch,
            num_digits=num_digits,
            heldout_fraction=args.shadow_feedback_heldout_fraction,
            hidden_size=args.shadow_feedback_hidden_size,
            dropout=args.shadow_feedback_dropout,
            learning_rate=args.shadow_feedback_online_lr,
            weight_decay=args.shadow_feedback_weight_decay,
            warmup_steps=args.shadow_feedback_warmup_steps,
            updates_per_step=args.shadow_feedback_updates_per_step,
            validation_fraction=args.shadow_feedback_validation_fraction,
            validation_every=args.shadow_feedback_validation_every,
            validation_loss_weight=args.shadow_feedback_validation_loss_weight,
            validation_gradient_loss_weight=(
                args.shadow_feedback_validation_gradient_loss_weight
            ),
            validation_gradient_norm_weight=(
                args.shadow_feedback_validation_gradient_norm_weight
            ),
            target_normalization=args.shadow_feedback_target_normalization,
            target_transform=args.shadow_feedback_target_transform,
            feature_mode=args.shadow_feedback_feature_mode,
            feature_normalization=args.shadow_feedback_feature_normalization,
            loss_mode=args.shadow_feedback_loss_mode,
            selection_score_mode=args.shadow_feedback_selection_score_mode,
            selection_gap_penalty=args.shadow_feedback_selection_gap_penalty,
            result_boundary_target_mode=args.result_boundary_target_mode,
            result_boundary_target_temperature=(
                args.result_boundary_target_temperature
            ),
            result_boundary_target_min_probability_floor=(
                args.result_boundary_target_min_probability_floor
            ),
            result_boundary_target_chunk_size=(
                args.result_boundary_target_chunk_size
            ),
            return_artifacts=True,
        )
        summary, artifacts = diagnostic_result
        summary["shadow_feedback_refresh_step"] = int(refresh_step)
        online_shadow_feedback_refresh_history.append(
            {
                "step": int(refresh_step),
                "heldout_result_cosine": float(
                    summary["heldout_shadow_vs_boundary_result_proj_cosine"]
                ),
                "heldout_upstream_cosine": float(
                    summary["heldout_shadow_vs_boundary_upstream_cosine"]
                ),
                "train_heldout_result_gap": float(
                    summary[
                        "shadow_feedback_train_heldout_result_proj_cosine_gap"
                    ]
                ),
                "train_heldout_upstream_gap": float(
                    summary["shadow_feedback_train_heldout_upstream_cosine_gap"]
                ),
                "best_step": int(summary.get("shadow_feedback_best_step", -1)),
            }
        )
        return summary, artifacts

    if (
        args.shadow_feedback_weight > 0
        and args.shadow_feedback_mode == "fit_once_linear"
    ):
        if exhaustive_grid_batch is not None:
            shadow_calibration_batch = exhaustive_grid_batch
        else:
            shadow_calibration_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        shadow_feedback_weights, shadow_feedback_calibration_metrics = (
            fit_linear_shadow_feedback_weights(
                model,
                shadow_calibration_batch,
                num_digits=num_digits,
                ridge=args.shadow_feedback_ridge,
                result_boundary_target_mode=args.result_boundary_target_mode,
                result_boundary_target_temperature=(
                    args.result_boundary_target_temperature
                ),
                result_boundary_target_min_probability_floor=(
                    args.result_boundary_target_min_probability_floor
                ),
                result_boundary_target_chunk_size=(
                    args.result_boundary_target_chunk_size
                ),
            )
        )
        torch.save(
            {
                "weights": shadow_feedback_weights.cpu(),
                "metrics": shadow_feedback_calibration_metrics,
                "ridge": args.shadow_feedback_ridge,
            },
            run_dir / "shadow_feedback_weights.pt",
        )
        (run_dir / "shadow_feedback_calibration_summary.json").write_text(
            json.dumps(shadow_feedback_calibration_metrics, indent=2) + "\n"
        )
        print(
            "shadow feedback calibration: "
            f"fit_cos={shadow_feedback_calibration_metrics['shadow_feedback_fit_cosine']:.6f} "
            f"target_grad={shadow_feedback_calibration_metrics['shadow_feedback_target_grad_l2']:.6f} "
            f"pred_norm={shadow_feedback_calibration_metrics['shadow_feedback_predicted_l2']:.6f}"
        )
    if (
        args.shadow_feedback_weight > 0
        and args.shadow_feedback_mode == "online_mlp"
    ):
        if exhaustive_grid_batch is not None:
            shadow_calibration_batch = exhaustive_grid_batch
        else:
            shadow_calibration_batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        (
            shadow_feedback_calibration_metrics,
            online_shadow_feedback_artifacts,
        ) = fit_online_shadow_feedback_artifacts(
            shadow_calibration_batch,
            refresh_step=0,
        )
        shadow_module = online_shadow_feedback_artifacts["shadow_module"]
        assert isinstance(shadow_module, ShadowFeedbackMLP)
        torch.save(
            {
                "state_dict": shadow_module.state_dict(),
                "metrics": shadow_feedback_calibration_metrics,
                "feature_mode": args.shadow_feedback_feature_mode,
                "target_mean": online_shadow_feedback_artifacts["target_mean"],
                "target_scale": online_shadow_feedback_artifacts["target_scale"],
                "feature_mean": online_shadow_feedback_artifacts["feature_mean"],
                "feature_scale": online_shadow_feedback_artifacts["feature_scale"],
            },
            run_dir / "online_shadow_feedback_module.pt",
        )
        (run_dir / "online_shadow_feedback_calibration_summary.json").write_text(
            json.dumps(shadow_feedback_calibration_metrics, indent=2) + "\n"
        )
        (run_dir / "online_shadow_feedback_refresh_history.json").write_text(
            json.dumps(online_shadow_feedback_refresh_history, indent=2) + "\n"
        )
        print(
            "online shadow feedback calibration: "
            f"result_cos={shadow_feedback_calibration_metrics['heldout_shadow_vs_boundary_result_proj_cosine']:.6f} "
            f"upstream_cos={shadow_feedback_calibration_metrics['heldout_shadow_vs_boundary_upstream_cosine']:.6f} "
            f"result_gap={shadow_feedback_calibration_metrics['shadow_feedback_train_heldout_result_proj_cosine_gap']:.6f}"
        )

    curve: list[dict[str, float | int]] = []
    snapshots: list[dict[str, object]] = []
    final_loss = float("nan")
    policy_baseline: float | None = None
    action_loss_replay_cache = ActionLossReplayCache()
    last_optimizer_step_metrics: dict[str, float] = {}
    optimizer_step_acceptance_attempts = 0
    optimizer_step_acceptance_accepted = 0
    optimizer_step_line_search_scales = parse_optimizer_step_line_search_scales(
        args.optimizer_step_line_search_scales
    )
    model.train()
    for step in range(args.steps + 1):
        if args.operand_max is None:
            batch = make_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                rng=rng,
                fixed_width=True,
                answer_format=args.answer_format,
                device=device,
            )
        elif exhaustive_grid_batch is not None:
            batch = exhaustive_grid_batch
        else:
            batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        if (
            args.shadow_feedback_mode == "online_mlp"
            and args.shadow_feedback_weight > 0
            and args.shadow_feedback_refresh_every > 0
            and step > 0
            and step % args.shadow_feedback_refresh_every == 0
        ):
            (
                shadow_feedback_calibration_metrics,
                online_shadow_feedback_artifacts,
            ) = fit_online_shadow_feedback_artifacts(
                shadow_calibration_batch,
                refresh_step=step,
            )
            shadow_module = online_shadow_feedback_artifacts["shadow_module"]
            assert isinstance(shadow_module, ShadowFeedbackMLP)
            torch.save(
                {
                    "state_dict": shadow_module.state_dict(),
                    "metrics": shadow_feedback_calibration_metrics,
                    "feature_mode": args.shadow_feedback_feature_mode,
                    "target_mean": online_shadow_feedback_artifacts["target_mean"],
                    "target_scale": online_shadow_feedback_artifacts["target_scale"],
                    "feature_mean": online_shadow_feedback_artifacts["feature_mean"],
                    "feature_scale": online_shadow_feedback_artifacts["feature_scale"],
                },
                run_dir / f"online_shadow_feedback_module_step_{step:05d}.pt",
            )
            (
                run_dir / "online_shadow_feedback_refresh_history.json"
            ).write_text(
                json.dumps(online_shadow_feedback_refresh_history, indent=2)
                + "\n"
            )
            print(
                "online shadow feedback refresh: "
                f"step={step} "
                f"result_cos={shadow_feedback_calibration_metrics['heldout_shadow_vs_boundary_result_proj_cosine']:.6f} "
                f"upstream_cos={shadow_feedback_calibration_metrics['heldout_shadow_vs_boundary_upstream_cosine']:.6f} "
                f"result_gap={shadow_feedback_calibration_metrics['shadow_feedback_train_heldout_result_proj_cosine_gap']:.6f}"
            )
        oracle_operands = None
        use_oracle_for_step = args.oracle_train or (
            args.variant == "model-c" and step < args.oracle_warmup_steps
        )
        if use_oracle_for_step:
            oracle_operands = make_oracle_operands_from_batch(
                batch.x, num_digits=num_digits
            )
        use_reinforce = (
            args.variant == "model-c"
            and args.calculator_estimator == "reinforce"
            and not args.oracle_train
        )
        use_adaptive_interface = (
            args.variant == "model-c"
            and args.calculator_estimator == "adaptive_interface"
            and not args.oracle_train
        )
        use_action_loss_weighted_interface = (
            args.variant == "model-c"
            and args.calculator_estimator
            in {
                "action_loss_weighted_interface",
                "action_loss_replay_interface",
                "action_loss_full_enum_interface",
                "action_loss_full_enum_joint_interface",
                "identifiable_full_enum_local_target",
            }
            and not args.oracle_train
        )
        use_expected_answer_loss = (
            args.variant == "model-c"
            and args.calculator_estimator == "full_enum_expected_answer_loss"
            and not args.oracle_train
        )
        use_boundary_feedback = (
            args.variant == "model-c"
            and args.calculator_estimator == "direct_feedback_alignment"
            and args.boundary_feedback_weight > 0
            and shadow_feedback_weights is None
            and not args.oracle_train
        )
        use_shadow_feedback = (
            args.variant == "model-c"
            and args.calculator_estimator == "direct_feedback_alignment"
            and args.shadow_feedback_weight > 0
            and (
                (
                    args.shadow_feedback_mode == "fit_once_linear"
                    and shadow_feedback_weights is not None
                )
                or (
                    args.shadow_feedback_mode == "online_mlp"
                    and online_shadow_feedback_artifacts is not None
                )
            )
            and not args.oracle_train
        )
        use_relaxed_calculator = (
            args.variant == "model-c"
            and args.calculator_estimator == "gumbel_concrete_interface"
            and not args.oracle_train
        )
        use_result_boundary_target = (
            use_relaxed_calculator
            and args.result_boundary_target_loss_weight > 0
        )
        use_result_policy_stabilization = (
            args.variant == "model-c"
            and args.calculator_action_head == "result_space"
            and (
                args.result_policy_entropy_weight > 0
                or args.result_policy_batch_diversity_weight > 0
                or args.result_policy_improvement_assignment_weight > 0
            )
            and not args.oracle_train
        )
        current_relaxed_temperature = relaxed_calculator_temperature(
            initial_temperature=args.relaxed_calculator_temperature,
            final_temperature=args.relaxed_calculator_final_temperature,
            decay_steps=args.relaxed_calculator_temperature_decay_steps,
            step=step,
        )
        if model.calculator_hook is not None:
            model.calculator_hook.relaxed_temperature = current_relaxed_temperature
            model.calculator_hook.relaxed_mode = args.relaxed_calculator_mode
            model.calculator_hook.relaxed_hard_forward = (
                args.relaxed_calculator_hard_forward
            )
        if use_reinforce:
            diagnostics = {}
            per_example_answer_loss = None
            answer_loss = batch.x.new_tensor(0.0, dtype=torch.float)
        else:
            diagnostics = {}
            logits = model(batch.x, oracle_operands=oracle_operands)
            per_example_answer_loss = masked_cross_entropy_per_example(
                logits, batch.y, batch.loss_mask
            )
            answer_loss = per_example_answer_loss.mean()
        loss = args.answer_loss_weight * answer_loss
        policy_loss_value = None
        policy_advantage_mean = None
        policy_advantage_std = None
        sampled_logp_value = None
        operand_entropy_value = None
        result_entropy_value = None
        sampled_result_accuracy_value = None
        entropy_weight = 0.0
        adaptive_interface_loss_value = None
        adaptive_interface_objective_value = None
        scheduled_adaptive_interface_weight = adaptive_interface_weight(
            initial_weight=args.adaptive_interface_loss_weight,
            decay_steps=args.adaptive_interface_loss_decay_steps,
            floor=args.adaptive_interface_loss_floor,
            step=step,
        )
        adaptive_metrics: dict[str, float] = {}
        action_loss_interface_loss_value = None
        action_loss_interface_objective_value = None
        action_loss_metrics: dict[str, float] = {}
        expected_answer_loss_value = None
        expected_answer_loss_objective_value = None
        expected_answer_loss_entropy_weight = 0.0
        expected_answer_loss_metrics: dict[str, float] = {}
        result_boundary_target_loss_value = None
        result_boundary_target_objective_value = None
        result_boundary_target_metrics: dict[str, float] = {}
        result_policy_stabilization_objective_value = None
        result_policy_stabilization_metrics: dict[str, float] = {}
        calculator_causal_gap_objective_value = None
        calculator_causal_gap_metrics: dict[str, float] = {}
        boundary_feedback_loss_value = None
        boundary_feedback_objective_value = None
        boundary_feedback_metrics: dict[str, float | int | str] = {}
        shadow_feedback_loss_value = None
        shadow_feedback_objective_value = None
        shadow_feedback_metrics: dict[str, float | int | str] = {}
        relaxed_calculator_entropy_objective_value = None
        current_relaxed_entropy_weight = 0.0
        relaxed_calculator_metrics: dict[str, float] = {}
        anchor_loss_value = None
        anchor_weight = 0.0
        if use_reinforce:
            if policy_baseline is None:
                policy_baseline = None
            if args.reinforce_entropy_decay_steps > 0:
                entropy_weight = args.reinforce_entropy_weight * max(
                    0.0, 1.0 - (step / args.reinforce_entropy_decay_steps)
                )
            else:
                entropy_weight = args.reinforce_entropy_weight
            policy_objective, answer_loss, policy_metrics = (
                reinforce_policy_gradient_loss(
                    model,
                    batch,
                    num_digits=num_digits,
                    baseline_mode=args.reinforce_baseline_mode,
                    num_samples_per_prompt=args.reinforce_num_samples_per_prompt,
                    global_baseline=policy_baseline,
                    entropy_weight=entropy_weight,
                )
            )
            loss = (args.answer_loss_weight * answer_loss) + policy_objective
            policy_loss_value = policy_metrics["policy_loss"]
            policy_baseline = policy_metrics["policy_baseline"]
            policy_advantage_mean = policy_metrics["policy_advantage_mean"]
            policy_advantage_std = policy_metrics["policy_advantage_std"]
            sampled_logp_value = policy_metrics["sampled_logp"]
            operand_entropy_value = policy_metrics["operand_entropy"]
            result_entropy_value = policy_metrics["result_entropy"]
            sampled_result_accuracy_value = policy_metrics["sampled_result_accuracy"]
        if use_adaptive_interface:
            adaptive_loss, adaptive_metrics = adaptive_interface_loss(
                model,
                batch,
                num_digits=num_digits,
                target_mode=args.adaptive_interface_target_mode,
                entropy_weight=args.adaptive_interface_entropy_weight,
            )
            adaptive_interface_loss_value = adaptive_metrics[
                "adaptive_interface_target_loss"
            ]
            adaptive_objective = scheduled_adaptive_interface_weight * adaptive_loss
            adaptive_interface_objective_value = float(adaptive_objective.item())
            loss = loss + adaptive_objective
        if use_action_loss_weighted_interface:
            if args.calculator_estimator == "action_loss_replay_interface":
                action_loss_interface_loss, action_loss_metrics = (
                    action_loss_replay_interface_loss(
                        model,
                        batch,
                        num_digits=num_digits,
                        random_actions=args.action_loss_candidate_random,
                        topk=args.action_loss_candidate_topk,
                        local_radius=args.action_loss_candidate_local_radius,
                        temperature=args.action_loss_candidate_temperature,
                        generator=candidate_generator,
                        cache=action_loss_replay_cache,
                        step=step,
                        refresh_every=args.action_loss_candidate_refresh_every,
                        ema_beta=args.action_loss_candidate_ema_beta,
                    )
                )
            elif args.calculator_estimator in {
                "action_loss_full_enum_interface",
                "identifiable_full_enum_local_target",
            }:
                action_loss_interface_loss, action_loss_metrics = (
                    action_loss_full_enum_interface_loss(
                        model,
                        batch,
                        num_digits=num_digits,
                        temperature=args.action_loss_full_enum_temperature,
                        min_probability_floor=(
                            args.action_loss_full_enum_min_probability_floor
                        ),
                        chunk_size=args.action_loss_full_enum_chunk_size,
                        target_mode=args.action_loss_full_enum_target_mode,
                    )
                )
            elif args.calculator_estimator == "action_loss_full_enum_joint_interface":
                action_loss_interface_loss, action_loss_metrics = (
                    action_loss_full_enum_joint_interface_loss(
                        model,
                        batch,
                        num_digits=num_digits,
                        temperature=args.action_loss_full_enum_temperature,
                        min_probability_floor=(
                            args.action_loss_full_enum_min_probability_floor
                        ),
                        chunk_size=args.action_loss_full_enum_chunk_size,
                    )
                )
            else:
                action_loss_interface_loss, action_loss_metrics = (
                    action_loss_weighted_interface_loss(
                        model,
                        batch,
                        num_digits=num_digits,
                        random_actions=args.action_loss_candidate_random,
                        topk=args.action_loss_candidate_topk,
                        local_radius=args.action_loss_candidate_local_radius,
                        temperature=args.action_loss_candidate_temperature,
                        generator=candidate_generator,
                    )
                )
            action_loss_interface_loss_value = float(
                action_loss_interface_loss.item()
            )
            action_loss_objective = (
                scheduled_adaptive_interface_weight * action_loss_interface_loss
            )
            action_loss_interface_objective_value = float(
                action_loss_objective.item()
            )
            loss = loss + action_loss_objective
        if use_expected_answer_loss:
            if args.expected_answer_loss_entropy_decay_steps > 0:
                expected_answer_loss_entropy_weight = (
                    args.expected_answer_loss_entropy_weight
                    * max(
                        0.0,
                        1.0
                        - (step / args.expected_answer_loss_entropy_decay_steps),
                    )
                )
            else:
                expected_answer_loss_entropy_weight = (
                    args.expected_answer_loss_entropy_weight
                )
            expected_answer_loss, expected_answer_loss_metrics = (
                full_enum_expected_answer_loss(
                    model,
                    batch,
                    num_digits=num_digits,
                    policy_temperature=(
                        args.expected_answer_loss_policy_temperature
                    ),
                    cost_normalization=(
                        args.expected_answer_loss_cost_normalization
                    ),
                    entropy_weight=expected_answer_loss_entropy_weight,
                    chunk_size=args.expected_answer_loss_chunk_size,
                )
            )
            expected_answer_loss_value = expected_answer_loss_metrics[
                "expected_answer_loss"
            ]
            expected_answer_loss_objective = (
                args.expected_answer_loss_weight * expected_answer_loss
            )
            expected_answer_loss_objective_value = float(
                expected_answer_loss_objective.item()
            )
            loss = loss + expected_answer_loss_objective
        if use_result_boundary_target:
            (
                result_boundary_loss,
                result_boundary_target_metrics,
            ) = result_boundary_target_loss(
                model,
                batch,
                num_digits=num_digits,
                target_mode=args.result_boundary_target_mode,
                temperature=args.result_boundary_target_temperature,
                min_probability_floor=(
                    args.result_boundary_target_min_probability_floor
                ),
                chunk_size=args.result_boundary_target_chunk_size,
            )
            result_boundary_target_loss_value = result_boundary_target_metrics[
                "result_boundary_target_loss"
            ]
            result_boundary_objective = (
                args.result_boundary_target_loss_weight * result_boundary_loss
            )
            result_boundary_target_objective_value = float(
                result_boundary_objective.item()
            )
            loss = loss + result_boundary_objective
        if use_result_policy_stabilization:
            current_result_policy_entropy_weight = (
                result_policy_stabilization_weight(
                    initial_weight=args.result_policy_entropy_weight,
                    decay_steps=args.result_policy_stabilization_decay_steps,
                    step=step,
                )
            )
            current_result_policy_batch_diversity_weight = (
                result_policy_stabilization_weight(
                    initial_weight=args.result_policy_batch_diversity_weight,
                    decay_steps=args.result_policy_stabilization_decay_steps,
                    step=step,
                )
            )
            current_result_policy_improvement_assignment_weight = (
                result_policy_stabilization_weight(
                    initial_weight=args.result_policy_improvement_assignment_weight,
                    decay_steps=args.result_policy_stabilization_decay_steps,
                    step=step,
                )
            )
            (
                result_policy_stabilization_objective,
                result_policy_stabilization_metrics,
            ) = result_policy_stabilization_loss(
                model,
                batch,
                num_digits=num_digits,
                temperature=args.result_policy_stabilization_temperature,
                entropy_weight=current_result_policy_entropy_weight,
                batch_diversity_weight=(
                    current_result_policy_batch_diversity_weight
                ),
                improvement_assignment_weight=(
                    current_result_policy_improvement_assignment_weight
                ),
                improvement_assignment_min_improvement=(
                    args.result_policy_improvement_assignment_min_improvement
                ),
                improvement_assignment_quota_multiplier=(
                    args.result_policy_improvement_assignment_quota_multiplier
                ),
                chunk_size=args.result_boundary_target_chunk_size,
            )
            result_policy_stabilization_objective_value = float(
                result_policy_stabilization_objective.item()
            )
            loss = loss + result_policy_stabilization_objective
        if args.calculator_causal_gap_weight > 0:
            if use_reinforce:
                raise ValueError(
                    "calculator causal-gap objective is not supported with reinforce"
                )
            with temporary_calculator_injection_scale(model, 0.0):
                zero_logits = model(batch.x, oracle_operands=oracle_operands)
            zero_answer_loss = masked_cross_entropy_per_example(
                zero_logits, batch.y, batch.loss_mask
            ).mean()
            causal_gap = zero_answer_loss - answer_loss
            calculator_causal_gap_objective = (
                args.calculator_causal_gap_weight
                * torch.relu(args.calculator_causal_gap_margin - causal_gap)
            )
            calculator_causal_gap_objective_value = float(
                calculator_causal_gap_objective.detach().item()
            )
            calculator_causal_gap_metrics = {
                "calculator_causal_gap_weight": float(
                    args.calculator_causal_gap_weight
                ),
                "calculator_causal_gap_margin": float(
                    args.calculator_causal_gap_margin
                ),
                "calculator_causal_gap_objective": (
                    calculator_causal_gap_objective_value
                ),
                "calculator_causal_gap": float(causal_gap.detach().item()),
                "calculator_causal_gap_zero_loss": float(
                    zero_answer_loss.detach().item()
                ),
                "calculator_causal_gap_normal_loss": float(
                    answer_loss.detach().item()
                ),
            }
            loss = loss + calculator_causal_gap_objective
        if use_boundary_feedback:
            (
                boundary_feedback_loss,
                boundary_feedback_metrics,
            ) = boundary_feedback_alignment_loss(
                model,
                batch,
                num_digits=num_digits,
                feedback_mode=args.boundary_feedback_mode,
                feedback_seed=args.boundary_feedback_seed,
            )
            boundary_feedback_loss_value = float(boundary_feedback_loss.item())
            boundary_feedback_objective = (
                args.boundary_feedback_weight * boundary_feedback_loss
            )
            boundary_feedback_objective_value = float(
                boundary_feedback_objective.item()
            )
            loss = loss + boundary_feedback_objective
        if use_shadow_feedback:
            if args.shadow_feedback_mode == "fit_once_linear":
                assert shadow_feedback_weights is not None
                (
                    shadow_feedback_loss,
                    shadow_feedback_metrics,
                ) = fixed_linear_shadow_feedback_alignment_loss(
                    model,
                    batch,
                    num_digits=num_digits,
                    weights=shadow_feedback_weights,
                    ridge=args.shadow_feedback_ridge,
                    result_boundary_target_mode=args.result_boundary_target_mode,
                    result_boundary_target_temperature=(
                        args.result_boundary_target_temperature
                    ),
                    result_boundary_target_min_probability_floor=(
                        args.result_boundary_target_min_probability_floor
                    ),
                    result_boundary_target_chunk_size=(
                        args.result_boundary_target_chunk_size
                    ),
                )
            else:
                assert online_shadow_feedback_artifacts is not None
                shadow_module = online_shadow_feedback_artifacts["shadow_module"]
                assert isinstance(shadow_module, ShadowFeedbackMLP)
                (
                    shadow_feedback_loss,
                    shadow_feedback_metrics,
                ) = online_shadow_feedback_fixed_module_loss(
                    model,
                    batch,
                    shadow_module=shadow_module,
                    num_digits=num_digits,
                    feature_mode=args.shadow_feedback_feature_mode,
                    target_mean=online_shadow_feedback_artifacts["target_mean"],
                    target_scale=online_shadow_feedback_artifacts["target_scale"],
                    feature_mean=online_shadow_feedback_artifacts["feature_mean"],
                    feature_scale=online_shadow_feedback_artifacts["feature_scale"],
                    max_predicted_norm=args.shadow_feedback_apply_max_norm,
                )
            shadow_feedback_loss_value = float(shadow_feedback_loss.item())
            shadow_feedback_objective = (
                args.shadow_feedback_weight * shadow_feedback_loss
            )
            shadow_feedback_objective_value = float(
                shadow_feedback_objective.item()
            )
            loss = loss + shadow_feedback_objective
        if use_relaxed_calculator:
            current_relaxed_entropy_weight = relaxed_calculator_entropy_weight(
                initial_weight=args.relaxed_calculator_entropy_weight,
                decay_steps=args.relaxed_calculator_entropy_decay_steps,
                step=step,
            )
            entropy_objective, relaxed_calculator_metrics = (
                relaxed_calculator_policy_metrics(
                    model,
                    batch,
                    num_digits=num_digits,
                    temperature=current_relaxed_temperature,
                    entropy_weight=current_relaxed_entropy_weight,
                )
            )
            relaxed_calculator_entropy_objective_value = float(
                entropy_objective.item()
            )
            loss = loss + entropy_objective
        if input_proj_anchor is not None:
            anchor_weight = input_proj_anchor_weight(
                initial_weight=args.input_proj_anchor_weight,
                decay_steps=args.input_proj_anchor_decay_steps,
                step=step,
            )
            anchor_loss = input_proj_anchor_loss(model, input_proj_anchor)
            anchor_loss_value = float(anchor_loss.detach().item())
            loss = loss + (anchor_weight * anchor_loss)
        aux_loss_value = None
        aux_weight = 0.0
        if args.aux_operand_loss_weight > 0:
            if args.variant != "model-c":
                raise ValueError("--aux-operand-loss-weight requires --variant model-c")
            aux_weight = auxiliary_operand_weight(
                initial_weight=args.aux_operand_loss_weight,
                decay_steps=args.aux_operand_loss_decay_steps,
                floor=args.aux_operand_loss_floor,
                step=step,
            )
            aux_loss = auxiliary_operand_loss(
                model,
                batch,
                num_digits,
                grad_upstream=args.aux_operand_loss_grad_upstream,
            )
            aux_loss_value = aux_loss.item()
            loss = loss + (aux_weight * aux_loss)

        if step % args.log_every == 0:
            loss_value = loss.item()
            curve_row: dict[str, float | int] = {
                "step": step,
                "loss": loss_value,
                "answer_loss": answer_loss.item(),
                "answer_loss_weight": args.answer_loss_weight,
                "adaptive_interface_loss_weight": scheduled_adaptive_interface_weight,
                "oracle_operands_used": int(use_oracle_for_step),
            }
            if aux_loss_value is not None:
                curve_row["aux_operand_loss"] = aux_loss_value
                curve_row["aux_operand_loss_weight"] = aux_weight
            if use_reinforce:
                curve_row["policy_loss"] = policy_loss_value
                curve_row["policy_baseline"] = policy_baseline
                curve_row["policy_advantage_mean"] = policy_advantage_mean
                curve_row["policy_advantage_std"] = policy_advantage_std
                curve_row["sampled_logp"] = sampled_logp_value
                curve_row["operand_entropy"] = operand_entropy_value
                curve_row["result_entropy"] = result_entropy_value
                curve_row["sampled_result_accuracy"] = sampled_result_accuracy_value
                curve_row["entropy_weight"] = entropy_weight
            if use_adaptive_interface:
                curve_row["adaptive_interface_loss"] = adaptive_interface_loss_value
                curve_row["adaptive_interface_objective"] = (
                    adaptive_interface_objective_value
                )
                curve_row["adaptive_interface_entropy_weight"] = (
                    args.adaptive_interface_entropy_weight
                )
                curve_row.update(adaptive_metrics)
            if use_action_loss_weighted_interface:
                curve_row["action_loss_interface_loss"] = (
                    action_loss_interface_loss_value
                )
                curve_row["action_loss_interface_objective"] = (
                    action_loss_interface_objective_value
                )
                curve_row.update(action_loss_metrics)
            if use_expected_answer_loss:
                curve_row["expected_answer_loss_weight"] = (
                    args.expected_answer_loss_weight
                )
                curve_row["expected_answer_loss_objective"] = (
                    expected_answer_loss_objective_value
                )
                curve_row.update(expected_answer_loss_metrics)
            if use_result_boundary_target:
                curve_row["result_boundary_target_loss_weight"] = (
                    args.result_boundary_target_loss_weight
                )
                curve_row["result_boundary_target_objective"] = (
                    result_boundary_target_objective_value
                )
                curve_row.update(result_boundary_target_metrics)
            if use_result_policy_stabilization:
                curve_row["result_policy_stabilization_objective"] = (
                    result_policy_stabilization_objective_value
                )
                curve_row.update(result_policy_stabilization_metrics)
            if args.calculator_causal_gap_weight > 0:
                curve_row.update(calculator_causal_gap_metrics)
            if use_boundary_feedback:
                curve_row["boundary_feedback_weight"] = (
                    args.boundary_feedback_weight
                )
                curve_row["boundary_feedback_objective"] = (
                    boundary_feedback_objective_value
                )
                curve_row.update(boundary_feedback_metrics)
            if use_shadow_feedback:
                curve_row["shadow_feedback_weight"] = (
                    args.shadow_feedback_weight
                )
                curve_row["shadow_feedback_objective"] = (
                    shadow_feedback_objective_value
                )
                curve_row.update(shadow_feedback_metrics)
            if use_relaxed_calculator:
                curve_row["relaxed_calculator_entropy_objective"] = (
                    relaxed_calculator_entropy_objective_value
                )
                curve_row["relaxed_calculator_final_temperature"] = (
                    args.relaxed_calculator_final_temperature
                )
                curve_row["relaxed_calculator_hard_forward"] = int(
                    args.relaxed_calculator_hard_forward
                )
                curve_row.update(relaxed_calculator_metrics)
            if anchor_loss_value is not None:
                curve_row["input_proj_anchor_loss"] = anchor_loss_value
                curve_row["input_proj_anchor_weight"] = anchor_weight
            if last_optimizer_step_metrics:
                curve_row.update(last_optimizer_step_metrics)
            curve.append(curve_row)
            print(
                f"variant={args.variant} digits={num_digits} "
                f"step={step:5d} loss={loss_value:.4f} "
                f"answer_loss={answer_loss.item():.4f}"
                + (
                    " oracle_warmup=1"
                    if use_oracle_for_step and not args.oracle_train
                    else ""
                )
                + (
                    f" policy_loss={policy_loss_value:.4f}"
                    f" baseline={policy_baseline:.4f}"
                    f" adv_std={policy_advantage_std:.4f}"
                    f" entropy={operand_entropy_value:.4f}"
                    f" sample_result_acc={sampled_result_accuracy_value:.3f}"
                    if use_reinforce
                    else ""
                )
                + (
                    f" aux_operand_loss={aux_loss_value:.4f}"
                    f" aux_weight={aux_weight:.4f}"
                    if aux_loss_value is not None
                    else ""
                )
                + (
                    f" adaptive_interface_loss={adaptive_interface_loss_value:.4f}"
                    f" iface_weight={scheduled_adaptive_interface_weight:.4f}"
                    f" target_acc={adaptive_metrics['adaptive_target_result_accuracy']:.3f}"
                    f" entropy={adaptive_metrics['adaptive_interface_entropy']:.3f}"
                    if use_adaptive_interface
                    else ""
                )
                + (
                    f" action_loss_iface_weight={scheduled_adaptive_interface_weight:.4f}"
                    if use_action_loss_weighted_interface
                    else ""
                )
                + (
                    f" action_loss_interface_loss={action_loss_interface_loss_value:.4f}"
                    f" better_frac={action_loss_metrics.get('action_loss_candidate_better_fraction', action_loss_metrics.get('action_loss_full_enum_learned_best_fraction', float('nan'))):.3f}"
                    f" best_improve={action_loss_metrics.get('action_loss_candidate_best_improvement', action_loss_metrics.get('action_loss_full_enum_learned_minus_best_gap', float('nan'))):.3f}"
                    if use_action_loss_weighted_interface
                    else ""
                )
                + (
                    f" expected_answer_loss={expected_answer_loss_value:.4f}"
                    f" expected_weight={args.expected_answer_loss_weight:.4f}"
                    f" entropy={expected_answer_loss_metrics['expected_answer_loss_entropy']:.3f}"
                    f" learned_best={expected_answer_loss_metrics['expected_answer_loss_hard_learned_best_fraction']:.3f}"
                    if use_expected_answer_loss
                    else ""
                )
                + (
                    f" result_boundary_loss={result_boundary_target_loss_value:.4f}"
                    f" result_boundary_weight={args.result_boundary_target_loss_weight:.4f}"
                    f" best_true={result_boundary_target_metrics['result_boundary_target_hard_best_equals_true_sum']:.3f}"
                    f" learned_best={result_boundary_target_metrics['result_boundary_target_learned_best_fraction']:.3f}"
                    if use_result_boundary_target
                    else ""
                )
                + (
                    f" result_policy_entropy={result_policy_stabilization_metrics['result_policy_entropy']:.3f}"
                    f" result_policy_marg_eff={result_policy_stabilization_metrics['result_policy_marginal_effective_results']:.3f}"
                    f" result_policy_hard_eff={result_policy_stabilization_metrics['result_policy_hard_marginal_effective_results']:.3f}"
                    f" result_policy_acc={result_policy_stabilization_metrics['result_policy_argmax_result_accuracy']:.3f}"
                    if use_result_policy_stabilization
                    else ""
                )
                + (
                    f" causal_gap={calculator_causal_gap_metrics['calculator_causal_gap']:.4f}"
                    f" causal_obj={calculator_causal_gap_objective_value:.4f}"
                    if args.calculator_causal_gap_weight > 0
                    else ""
                )
                + (
                    f" boundary_feedback_loss={boundary_feedback_loss_value:.4f}"
                    f" boundary_feedback_weight={args.boundary_feedback_weight:.4f}"
                    f" feedback_norm={boundary_feedback_metrics['boundary_feedback_signal_l2']:.3f}"
                    f" learned_calc={boundary_feedback_metrics['boundary_feedback_hard_learned_calc_accuracy']:.3f}"
                    if use_boundary_feedback
                    else ""
                )
                + (
                    f" shadow_feedback_loss={shadow_feedback_loss_value:.4f}"
                    f" shadow_feedback_weight={args.shadow_feedback_weight:.4f}"
                    f" shadow_norm={shadow_feedback_metrics['shadow_feedback_predicted_l2']:.3f}"
                    f" learned_calc={shadow_feedback_metrics['shadow_feedback_hard_learned_calc_accuracy']:.3f}"
                    if use_shadow_feedback
                    else ""
                )
                + (
                    f" relaxed_temp={current_relaxed_temperature:.4f}"
                    f" relaxed_entropy={relaxed_calculator_metrics['relaxed_calculator_entropy']:.3f}"
                    f" relaxed_pair={relaxed_calculator_metrics['relaxed_calculator_hard_learned_pair_exact']:.3f}"
                    f" relaxed_calc={relaxed_calculator_metrics['relaxed_calculator_hard_learned_calc_accuracy']:.3f}"
                    if use_relaxed_calculator
                    else ""
                )
                + (
                    f" input_proj_anchor_loss={anchor_loss_value:.6f}"
                    f" anchor_weight={anchor_weight:.6f}"
                    if anchor_loss_value is not None
                    else ""
                )
            )

        if (
            args.variant == "model-c"
            and args.snapshot_every > 0
            and step % args.snapshot_every == 0
        ):
            snapshot = snapshot_row_from_model(
                model,
                step=step,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=args.snapshot_samples,
                seed=seed + 30_000 + step,
                device=device,
                answer_format=args.answer_format,
            )
            snapshots.append(snapshot)
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                snapshot_dir = run_dir / "checkpoint_snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(train_cfg),
                        "snapshot": snapshot,
                        "step": step,
                    },
                    snapshot_dir / f"step_{step:05d}_weights.pt",
                )
            print(
                f"snapshot step={step:5d} "
                f"normal={snapshot['normal_exact_match']:.3f} "
                f"zero_inj={snapshot['injection_zero_exact_match']:.3f} "
                f"oracle={snapshot['oracle_exact_match']:.3f} "
                f"operand={snapshot['operand_exact_match']:.3f}"
            )
            model.train()

        if step == args.steps:
            final_loss = loss.item()
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        trust_region_before = (
            snapshot_trainable_parameters(model)
            if (
                args.optimizer_step_max_delta_norm > 0
                or args.optimizer_step_acceptance_mode != "none"
            )
            else []
        )
        optimizer_acceptance_before_answer_loss = None
        if args.optimizer_step_acceptance_mode in {
            "answer_loss_decrease",
            "answer_loss_line_search",
        }:
            optimizer_acceptance_before_answer_loss = hard_path_answer_loss_metric(
                model,
                batch,
                oracle_operands=oracle_operands,
            )
        optim.step()
        if trust_region_before and args.optimizer_step_max_delta_norm > 0:
            last_optimizer_step_metrics = apply_optimizer_step_trust_region(
                trust_region_before,
                max_delta_norm=args.optimizer_step_max_delta_norm,
            )
        else:
            last_optimizer_step_metrics = {}
        if args.optimizer_step_acceptance_mode == "answer_loss_decrease":
            assert optimizer_acceptance_before_answer_loss is not None
            optimizer_acceptance_after_answer_loss = hard_path_answer_loss_metric(
                model,
                batch,
                oracle_operands=oracle_operands,
            )
            optimizer_step_accepted = (
                optimizer_acceptance_after_answer_loss
                <= optimizer_acceptance_before_answer_loss
                + args.optimizer_step_acceptance_tolerance
            )
            optimizer_step_acceptance_attempts += 1
            optimizer_step_acceptance_accepted += int(optimizer_step_accepted)
            if not optimizer_step_accepted:
                restore_trainable_parameters(trust_region_before)
            last_optimizer_step_metrics.update(
                {
                    "optimizer_step_acceptance_mode": (
                        args.optimizer_step_acceptance_mode
                    ),
                    "optimizer_step_acceptance_before_answer_loss": (
                        optimizer_acceptance_before_answer_loss
                    ),
                    "optimizer_step_acceptance_after_answer_loss": (
                        optimizer_acceptance_after_answer_loss
                    ),
                    "optimizer_step_acceptance_delta": (
                        optimizer_acceptance_after_answer_loss
                        - optimizer_acceptance_before_answer_loss
                    ),
                    "optimizer_step_acceptance_tolerance": (
                        args.optimizer_step_acceptance_tolerance
                    ),
                    "optimizer_step_accepted": float(optimizer_step_accepted),
                    "optimizer_step_acceptance_attempts": float(
                        optimizer_step_acceptance_attempts
                    ),
                    "optimizer_step_acceptance_accepted": float(
                        optimizer_step_acceptance_accepted
                    ),
                    "optimizer_step_acceptance_rate": (
                        optimizer_step_acceptance_accepted
                        / max(optimizer_step_acceptance_attempts, 1)
                    ),
                }
            )
        elif args.optimizer_step_acceptance_mode == "answer_loss_line_search":
            assert optimizer_acceptance_before_answer_loss is not None
            best_scale = 0.0
            best_after_answer_loss = optimizer_acceptance_before_answer_loss
            line_search_proposed = snapshot_trainable_parameters(model)
            for candidate_scale in optimizer_step_line_search_scales:
                apply_scaled_parameter_delta(
                    trust_region_before,
                    line_search_proposed,
                    scale=candidate_scale,
                )
                candidate_answer_loss = hard_path_answer_loss_metric(
                    model,
                    batch,
                    oracle_operands=oracle_operands,
                )
                if candidate_answer_loss < best_after_answer_loss:
                    best_after_answer_loss = candidate_answer_loss
                    best_scale = candidate_scale
            optimizer_step_accepted = (
                best_scale > 0
                and best_after_answer_loss
                <= optimizer_acceptance_before_answer_loss
                + args.optimizer_step_acceptance_tolerance
            )
            if optimizer_step_accepted:
                apply_scaled_parameter_delta(
                    trust_region_before,
                    line_search_proposed,
                    scale=best_scale,
                )
            else:
                restore_trainable_parameters(trust_region_before)
                best_scale = 0.0
                best_after_answer_loss = optimizer_acceptance_before_answer_loss
            optimizer_step_acceptance_attempts += 1
            optimizer_step_acceptance_accepted += int(optimizer_step_accepted)
            last_optimizer_step_metrics.update(
                {
                    "optimizer_step_acceptance_mode": (
                        args.optimizer_step_acceptance_mode
                    ),
                    "optimizer_step_acceptance_before_answer_loss": (
                        optimizer_acceptance_before_answer_loss
                    ),
                    "optimizer_step_acceptance_after_answer_loss": (
                        best_after_answer_loss
                    ),
                    "optimizer_step_acceptance_delta": (
                        best_after_answer_loss
                        - optimizer_acceptance_before_answer_loss
                    ),
                    "optimizer_step_acceptance_tolerance": (
                        args.optimizer_step_acceptance_tolerance
                    ),
                    "optimizer_step_accepted": float(optimizer_step_accepted),
                    "optimizer_step_acceptance_attempts": float(
                        optimizer_step_acceptance_attempts
                    ),
                    "optimizer_step_acceptance_accepted": float(
                        optimizer_step_acceptance_accepted
                    ),
                    "optimizer_step_acceptance_rate": (
                        optimizer_step_acceptance_accepted
                        / max(optimizer_step_acceptance_attempts, 1)
                    ),
                    "optimizer_step_line_search_scales": (
                        args.optimizer_step_line_search_scales
                    ),
                    "optimizer_step_selected_scale": float(best_scale),
                }
            )
        if use_reinforce and args.reinforce_baseline_mode == "global_ema":
            answer_loss_value = float(answer_loss.detach().item())
            if policy_baseline is None:
                policy_baseline = answer_loss_value
            else:
                policy_baseline = (
                    args.reinforce_baseline_beta * policy_baseline
                    + (1.0 - args.reinforce_baseline_beta) * answer_loss_value
                )

    metrics = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=args.eval_samples,
        seed=seed + 10_000,
        fixed_width=True,
        answer_format=args.answer_format,
        device=device,
        oracle_train=args.oracle_train,
    )
    metrics["final_loss"] = final_loss
    metrics["operand_max"] = operand_max
    metrics["exhaustive_grid_batch"] = args.exhaustive_grid_batch
    metrics["exhaustive_grid_size"] = (
        int(exhaustive_grid_size) if exhaustive_grid_size is not None else None
    )
    metrics["calculator_operand_vocab_size"] = calculator_operand_vocab_size
    metrics["parameter_count"] = model.num_params()
    metrics["run_dir"] = str(run_dir)
    metrics["variant"] = args.variant
    metrics["oracle_train"] = args.oracle_train
    metrics["oracle_warmup_steps"] = args.oracle_warmup_steps
    metrics["answer_loss_weight"] = args.answer_loss_weight
    metrics["answer_format"] = args.answer_format
    metrics["aux_operand_loss_floor"] = args.aux_operand_loss_floor
    metrics["calculator_estimator"] = args.calculator_estimator
    metrics["calculator_action_head"] = args.calculator_action_head
    metrics["calculator_read_position"] = args.calculator_read_position
    metrics["calculator_read_span_width"] = args.calculator_read_span_width
    metrics["calculator_injection_mode"] = args.calculator_injection_mode
    metrics["calculator_bottleneck_mode"] = args.calculator_bottleneck_mode
    metrics["calculator_output_format"] = args.calculator_output_format
    metrics["answer_decoder_interaction"] = cfg.answer_decoder_interaction
    metrics["semantic_decoder_checkpoint"] = (
        str(args.semantic_decoder_checkpoint)
        if args.semantic_decoder_checkpoint is not None
        else None
    )
    metrics["semantic_decoder_checkpoint_load_scope"] = (
        args.semantic_decoder_checkpoint_load_scope
    )
    metrics["adaptive_interface_loss_weight"] = args.adaptive_interface_loss_weight
    metrics["adaptive_interface_loss_decay_steps"] = (
        args.adaptive_interface_loss_decay_steps
    )
    metrics["adaptive_interface_loss_floor"] = args.adaptive_interface_loss_floor
    metrics["final_adaptive_interface_loss_weight"] = adaptive_interface_weight(
        initial_weight=args.adaptive_interface_loss_weight,
        decay_steps=args.adaptive_interface_loss_decay_steps,
        floor=args.adaptive_interface_loss_floor,
        step=args.steps,
    )
    metrics["adaptive_interface_target_mode"] = args.adaptive_interface_target_mode
    metrics["adaptive_interface_entropy_weight"] = args.adaptive_interface_entropy_weight
    metrics["action_loss_candidate_random"] = args.action_loss_candidate_random
    metrics["action_loss_candidate_topk"] = args.action_loss_candidate_topk
    metrics["action_loss_candidate_local_radius"] = (
        args.action_loss_candidate_local_radius
    )
    metrics["action_loss_candidate_temperature"] = (
        args.action_loss_candidate_temperature
    )
    metrics["action_loss_candidate_refresh_every"] = (
        args.action_loss_candidate_refresh_every
    )
    metrics["action_loss_candidate_ema_beta"] = args.action_loss_candidate_ema_beta
    metrics["action_loss_full_enum_temperature"] = (
        args.action_loss_full_enum_temperature
    )
    metrics["action_loss_full_enum_min_probability_floor"] = (
        args.action_loss_full_enum_min_probability_floor
    )
    metrics["action_loss_full_enum_chunk_size"] = (
        args.action_loss_full_enum_chunk_size
    )
    metrics["action_loss_full_enum_target_mode"] = (
        args.action_loss_full_enum_target_mode
    )
    metrics["local_target_loss_weight"] = args.adaptive_interface_loss_weight
    metrics["local_target_loss_decay_steps"] = args.adaptive_interface_loss_decay_steps
    metrics["local_target_loss_floor"] = args.adaptive_interface_loss_floor
    metrics["local_target_mode"] = args.action_loss_full_enum_target_mode
    metrics["final_local_target_loss_weight"] = adaptive_interface_weight(
        initial_weight=args.adaptive_interface_loss_weight,
        decay_steps=args.adaptive_interface_loss_decay_steps,
        floor=args.adaptive_interface_loss_floor,
        step=args.steps,
    )
    metrics["expected_answer_loss_weight"] = args.expected_answer_loss_weight
    metrics["final_expected_answer_loss_weight"] = args.expected_answer_loss_weight
    metrics["expected_answer_loss_policy_temperature"] = (
        args.expected_answer_loss_policy_temperature
    )
    metrics["expected_answer_loss_cost_normalization"] = (
        args.expected_answer_loss_cost_normalization
    )
    metrics["expected_answer_loss_entropy_weight"] = (
        args.expected_answer_loss_entropy_weight
    )
    metrics["expected_answer_loss_entropy_decay_steps"] = (
        args.expected_answer_loss_entropy_decay_steps
    )
    metrics["final_expected_answer_loss_entropy_weight"] = (
        args.expected_answer_loss_entropy_weight
        * max(
            0.0,
            1.0 - (args.steps / args.expected_answer_loss_entropy_decay_steps),
        )
        if args.expected_answer_loss_entropy_decay_steps > 0
        else args.expected_answer_loss_entropy_weight
    )
    metrics["expected_answer_loss_chunk_size"] = args.expected_answer_loss_chunk_size
    metrics["result_boundary_target_loss_weight"] = (
        args.result_boundary_target_loss_weight
    )
    metrics["final_result_boundary_target_loss_weight"] = (
        args.result_boundary_target_loss_weight
    )
    metrics["result_boundary_target_mode"] = args.result_boundary_target_mode
    metrics["result_boundary_target_temperature"] = (
        args.result_boundary_target_temperature
    )
    metrics["result_boundary_target_min_probability_floor"] = (
        args.result_boundary_target_min_probability_floor
    )
    metrics["result_boundary_target_chunk_size"] = (
        args.result_boundary_target_chunk_size
    )
    metrics["result_policy_entropy_weight"] = args.result_policy_entropy_weight
    metrics["result_policy_batch_diversity_weight"] = (
        args.result_policy_batch_diversity_weight
    )
    metrics["result_policy_improvement_assignment_weight"] = (
        args.result_policy_improvement_assignment_weight
    )
    metrics["result_policy_improvement_assignment_min_improvement"] = (
        args.result_policy_improvement_assignment_min_improvement
    )
    metrics["result_policy_improvement_assignment_quota_multiplier"] = (
        args.result_policy_improvement_assignment_quota_multiplier
    )
    metrics["result_policy_stabilization_temperature"] = (
        args.result_policy_stabilization_temperature
    )
    metrics["result_policy_stabilization_decay_steps"] = (
        args.result_policy_stabilization_decay_steps
    )
    metrics["final_result_policy_entropy_weight"] = (
        result_policy_stabilization_weight(
            initial_weight=args.result_policy_entropy_weight,
            decay_steps=args.result_policy_stabilization_decay_steps,
            step=args.steps,
        )
    )
    metrics["final_result_policy_batch_diversity_weight"] = (
        result_policy_stabilization_weight(
            initial_weight=args.result_policy_batch_diversity_weight,
            decay_steps=args.result_policy_stabilization_decay_steps,
            step=args.steps,
        )
    )
    metrics["final_result_policy_improvement_assignment_weight"] = (
        result_policy_stabilization_weight(
            initial_weight=args.result_policy_improvement_assignment_weight,
            decay_steps=args.result_policy_stabilization_decay_steps,
            step=args.steps,
        )
    )
    metrics["calculator_causal_gap_weight"] = args.calculator_causal_gap_weight
    metrics["calculator_causal_gap_margin"] = args.calculator_causal_gap_margin
    metrics["boundary_feedback_weight"] = args.boundary_feedback_weight
    metrics["final_boundary_feedback_weight"] = args.boundary_feedback_weight
    metrics["boundary_feedback_mode"] = args.boundary_feedback_mode
    metrics["boundary_feedback_seed"] = args.boundary_feedback_seed
    metrics["shadow_feedback_mode"] = args.shadow_feedback_mode
    metrics["shadow_feedback_ridge"] = args.shadow_feedback_ridge
    metrics["shadow_feedback_weight"] = args.shadow_feedback_weight
    metrics["shadow_feedback_heldout_fraction"] = (
        args.shadow_feedback_heldout_fraction
    )
    metrics["shadow_feedback_hidden_size"] = args.shadow_feedback_hidden_size
    metrics["shadow_feedback_dropout"] = args.shadow_feedback_dropout
    metrics["shadow_feedback_online_lr"] = args.shadow_feedback_online_lr
    metrics["shadow_feedback_weight_decay"] = args.shadow_feedback_weight_decay
    metrics["shadow_feedback_warmup_steps"] = args.shadow_feedback_warmup_steps
    metrics["shadow_feedback_updates_per_step"] = (
        args.shadow_feedback_updates_per_step
    )
    metrics["shadow_feedback_apply_max_norm"] = args.shadow_feedback_apply_max_norm
    metrics["shadow_feedback_refresh_every"] = args.shadow_feedback_refresh_every
    metrics["shadow_feedback_validation_fraction"] = (
        args.shadow_feedback_validation_fraction
    )
    metrics["shadow_feedback_validation_every"] = (
        args.shadow_feedback_validation_every
    )
    metrics["shadow_feedback_validation_loss_weight"] = (
        args.shadow_feedback_validation_loss_weight
    )
    metrics["shadow_feedback_validation_gradient_loss_weight"] = (
        args.shadow_feedback_validation_gradient_loss_weight
    )
    metrics["shadow_feedback_validation_gradient_norm_weight"] = (
        args.shadow_feedback_validation_gradient_norm_weight
    )
    metrics["shadow_feedback_target_normalization"] = (
        args.shadow_feedback_target_normalization
    )
    metrics["shadow_feedback_target_transform"] = (
        args.shadow_feedback_target_transform
    )
    metrics["shadow_feedback_feature_mode"] = args.shadow_feedback_feature_mode
    metrics["shadow_feedback_feature_normalization"] = (
        args.shadow_feedback_feature_normalization
    )
    metrics["shadow_feedback_loss_mode"] = args.shadow_feedback_loss_mode
    metrics["shadow_feedback_selection_score_mode"] = (
        args.shadow_feedback_selection_score_mode
    )
    metrics["shadow_feedback_selection_gap_penalty"] = (
        args.shadow_feedback_selection_gap_penalty
    )
    metrics["final_shadow_feedback_weight"] = args.shadow_feedback_weight
    if shadow_feedback_calibration_metrics:
        metrics["shadow_feedback_calibration"] = shadow_feedback_calibration_metrics
    metrics["calculator_result_head_hidden_size"] = (
        args.calculator_result_head_hidden_size
    )
    metrics["relaxed_calculator_temperature"] = args.relaxed_calculator_temperature
    metrics["relaxed_calculator_final_temperature"] = (
        args.relaxed_calculator_final_temperature
    )
    metrics["relaxed_calculator_temperature_decay_steps"] = (
        args.relaxed_calculator_temperature_decay_steps
    )
    metrics["final_relaxed_calculator_temperature"] = (
        relaxed_calculator_temperature(
            initial_temperature=args.relaxed_calculator_temperature,
            final_temperature=args.relaxed_calculator_final_temperature,
            decay_steps=args.relaxed_calculator_temperature_decay_steps,
            step=args.steps,
        )
    )
    metrics["relaxed_calculator_mode"] = args.relaxed_calculator_mode
    metrics["relaxed_calculator_hard_forward"] = (
        args.relaxed_calculator_hard_forward
    )
    metrics["relaxed_calculator_entropy_weight"] = (
        args.relaxed_calculator_entropy_weight
    )
    metrics["relaxed_calculator_entropy_decay_steps"] = (
        args.relaxed_calculator_entropy_decay_steps
    )
    metrics["final_relaxed_calculator_entropy_weight"] = (
        relaxed_calculator_entropy_weight(
            initial_weight=args.relaxed_calculator_entropy_weight,
            decay_steps=args.relaxed_calculator_entropy_decay_steps,
            step=args.steps,
        )
    )
    metrics["input_proj_anchor_checkpoint"] = (
        str(args.input_proj_anchor_checkpoint)
        if args.input_proj_anchor_checkpoint is not None
        else None
    )
    metrics["input_proj_anchor_weight"] = args.input_proj_anchor_weight
    metrics["input_proj_anchor_decay_steps"] = args.input_proj_anchor_decay_steps
    metrics["final_input_proj_anchor_weight"] = input_proj_anchor_weight(
        initial_weight=args.input_proj_anchor_weight,
        decay_steps=args.input_proj_anchor_decay_steps,
        step=args.steps,
    )
    if input_proj_anchor is not None:
        metrics["final_input_proj_anchor_loss"] = float(
            input_proj_anchor_loss(model, input_proj_anchor).item()
        )
        metrics["input_proj_anchor_delta"] = input_proj_anchor_delta_summary(
            model, input_proj_anchor
        )
    metrics["input_proj_lr"] = (
        args.lr if args.input_proj_lr is None else args.input_proj_lr
    )
    metrics["upstream_lr"] = args.lr if args.upstream_lr is None else args.upstream_lr
    metrics["optimizer_step_max_delta_norm"] = args.optimizer_step_max_delta_norm
    metrics["optimizer_step_acceptance_mode"] = args.optimizer_step_acceptance_mode
    metrics["optimizer_step_acceptance_tolerance"] = (
        args.optimizer_step_acceptance_tolerance
    )
    metrics["optimizer_step_line_search_scales"] = (
        args.optimizer_step_line_search_scales
    )
    if last_optimizer_step_metrics:
        metrics["final_optimizer_step"] = last_optimizer_step_metrics
    metrics["freeze_semantic_decoder"] = args.freeze_semantic_decoder
    metrics["freeze_upstream_encoder"] = args.freeze_upstream_encoder
    metrics["aux_operand_loss_grad_upstream"] = args.aux_operand_loss_grad_upstream
    metrics["trainable_parameter_groups"] = trainable_groups
    metrics["final_aux_operand_loss_weight"] = auxiliary_operand_weight(
        initial_weight=args.aux_operand_loss_weight,
        decay_steps=args.aux_operand_loss_decay_steps,
        floor=args.aux_operand_loss_floor,
        step=args.steps,
    )
    if args.variant == "model-c":
        aux_eval_rng = random.Random(seed + 40_000)
        aux_eval_batch = make_range_batch(
            batch_size=min(args.eval_samples, 128),
            num_digits=num_digits,
            operand_max=operand_max,
            rng=aux_eval_rng,
            fixed_width=True,
            device=device,
            answer_format=args.answer_format,
        )
        with torch.no_grad():
            metrics["final_aux_operand_loss"] = float(
                auxiliary_operand_loss(
                    model,
                    aux_eval_batch,
                    num_digits,
                    grad_upstream=args.aux_operand_loss_grad_upstream,
                ).item()
            )

    save_curve(run_dir / "training_curve.csv", curve)
    if snapshots:
        write_rows(run_dir / "diagnostic_snapshots.csv", snapshots)
    if args.variant == "model-c":
        trace_rows = calculator_trace_rows(
            model,
            num_digits=num_digits,
            operand_max=operand_max,
            samples=min(args.eval_samples, 128),
            seed=seed + 20_000,
            device=device,
            oracle_train=args.oracle_train,
            answer_format=args.answer_format,
        )
        trace_summary = summarize_trace_rows(trace_rows)
        metrics["diagnostic_summary"] = trace_summary
        counterfactual_samples = min(args.eval_samples, 128)
        metrics["counterfactuals"] = {
            "samples": counterfactual_samples,
            "injection_zero_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                answer_format=args.answer_format,
                device=device,
                oracle_train=False,
                injection_scale=0.0,
            )["exact_match"],
            "oracle_at_eval_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                answer_format=args.answer_format,
                device=device,
                oracle_train=True,
            )["exact_match"],
            "forced_zero_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                answer_format=args.answer_format,
                device=device,
                oracle_train=False,
                calculator_result_override="zero",
            )["exact_match"],
            "forced_random_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                answer_format=args.answer_format,
                device=device,
                oracle_train=False,
                calculator_result_override="random",
            )["exact_match"],
        }
        write_rows(run_dir / "calculator_trace_rows.csv", trace_rows)
        (run_dir / "diagnostic_summary.json").write_text(
            json.dumps(trace_summary, indent=2) + "\n"
        )
        if args.calculator_estimator in {
            "adaptive_interface",
            "action_loss_weighted_interface",
            "action_loss_replay_interface",
        }:
            adaptive_rows = adaptive_interface_trace_rows(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=min(args.eval_samples, 128),
                seed=seed + 22_000,
                device=device,
                target_mode=args.adaptive_interface_target_mode,
                answer_format=args.answer_format,
            )
            adaptive_summary = summarize_adaptive_interface_rows(adaptive_rows)
            metrics["adaptive_interface_diagnostic_summary"] = adaptive_summary
            write_rows(run_dir / "adaptive_interface_trace_rows.csv", adaptive_rows)
            (run_dir / "adaptive_interface_summary.json").write_text(
                json.dumps(adaptive_summary, indent=2) + "\n"
            )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(train_cfg),
            "metrics": metrics,
        },
        run_dir / "final_weights.pt",
    )

    print(
        f"variant={args.variant} digits={num_digits} eval exact-match "
        f"{metrics['correct']}/{metrics['samples']} "
        f"({metrics['exact_match']:.3f}); saved {run_dir}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny addition models with optional latent calculator hook."
    )
    parser.add_argument(
        "--variant",
        choices=["model-a", "model-b", "model-c"],
        default="model-a",
        help="model-a is the raw baseline, model-b wires the hook off, model-c turns addition on.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIGITS),
        help="Digit counts to train/evaluate separately.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument(
        "--answer-loss-weight",
        type=float,
        default=1.0,
        help="Weight on normal next-token answer loss; set to 0 for aux-only warm starts.",
    )
    parser.add_argument(
        "--answer-format",
        choices=ANSWER_FORMATS,
        default="sum",
        help=(
            "Answer target format. 'sum' preserves existing addition behavior; "
            "'sum_left_operand' emits zero-padded sum plus left operand."
        ),
    )
    parser.add_argument(
        "--operand-max",
        type=int,
        default=None,
        help="Restrict generated operands to 0..N while keeping fixed-width formatting.",
    )
    parser.add_argument(
        "--exhaustive-grid-batch",
        action="store_true",
        help=(
            "Reuse one fixed-width batch containing every ordered pair in "
            "0..operand_max exactly once."
        ),
    )
    parser.add_argument(
        "--calculator-operand-vocab-size",
        type=int,
        default=None,
        help=(
            "Override calculator operand classes. For true tiny-vocab runs, set this "
            "to operand_max + 1."
        ),
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--input-proj-lr",
        type=float,
        default=None,
        help="Adaptive-interface LR for calculator_hook.input_proj; defaults to --lr.",
    )
    parser.add_argument(
        "--upstream-lr",
        type=float,
        default=None,
        help="Adaptive-interface LR for trainable non-input-proj parameters; defaults to --lr.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--optimizer-step-max-delta-norm",
        type=float,
        default=0.0,
        help=(
            "Optional trust-region cap on the actual trainable-parameter "
            "update L2 norm after optimizer.step(). 0 disables."
        ),
    )
    parser.add_argument(
        "--optimizer-step-acceptance-mode",
        choices=["none", "answer_loss_decrease", "answer_loss_line_search"],
        default="none",
        help=(
            "Optional post-step acceptance test. answer_loss_decrease reverts "
            "optimizer steps that worsen hard-path answer loss on the current "
            "batch beyond the configured tolerance; answer_loss_line_search "
            "tries scaled versions of the proposed update."
        ),
    )
    parser.add_argument(
        "--optimizer-step-acceptance-tolerance",
        type=float,
        default=0.0,
        help=(
            "Allowed hard-path answer-loss increase for "
            "--optimizer-step-acceptance-mode answer_loss_decrease."
        ),
    )
    parser.add_argument(
        "--optimizer-step-line-search-scales",
        type=str,
        default="1,0.5,0.25,0.1,0",
        help=(
            "Comma-separated non-negative update scales for "
            "--optimizer-step-acceptance-mode answer_loss_line_search."
        ),
    )
    parser.add_argument(
        "--oracle-train",
        action="store_true",
        help="Feed true operands into Model C's calculator during training/eval.",
    )
    parser.add_argument(
        "--oracle-warmup-steps",
        type=int,
        default=0,
        help=(
            "For Model C, feed true operands for the first K training steps, then "
            "switch to learned operands. Final eval remains learned unless "
            "--oracle-train is also set."
        ),
    )
    parser.add_argument(
        "--aux-operand-loss-weight",
        type=float,
        default=0.0,
        help="Training-only diagnostic CE loss on learned calculator operand logits.",
    )
    parser.add_argument(
        "--aux-operand-loss-decay-steps",
        type=int,
        default=0,
        help="Linearly decay aux operand loss to zero over this many steps; 0 keeps it constant.",
    )
    parser.add_argument(
        "--aux-operand-loss-floor",
        type=float,
        default=0.0,
        help="Minimum aux operand loss weight after decay; only used with decay steps.",
    )
    parser.add_argument(
        "--aux-operand-loss-grad-upstream",
        action="store_true",
        help=(
            "Route aux operand loss through calculator_read_operand_logits so its "
            "gradient flows into the upstream encoder. Default uses the detached "
            "diagnostics path that only updates input_proj."
        ),
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="For Model C, save lightweight calculator-dependence diagnostics every N steps.",
    )
    parser.add_argument(
        "--snapshot-samples",
        type=int,
        default=64,
        help="Samples per periodic diagnostic snapshot.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "For Model C snapshots, also save model weights every N steps under "
            "checkpoint_snapshots/. Requires --snapshot-every to trigger the step."
        ),
    )
    parser.add_argument(
        "--injection-scale",
        type=float,
        default=1.0,
        help="Scale the calculator residual injection; default preserves existing behavior.",
    )
    parser.add_argument(
        "--calculator-estimator",
        choices=[
            "ste",
            "reinforce",
            "adaptive_interface",
            "action_loss_weighted_interface",
            "action_loss_replay_interface",
            "action_loss_full_enum_interface",
            "action_loss_full_enum_joint_interface",
            "identifiable_full_enum_local_target",
            "full_enum_expected_answer_loss",
            "gumbel_concrete_interface",
            "direct_feedback_alignment",
        ],
        default="ste",
        help="Estimator for the learned calculator input interface.",
    )
    parser.add_argument(
        "--calculator-action-head",
        choices=["independent_operands", "joint_pair", "result_space"],
        default="independent_operands",
        help="Calculator action parameterization used by the learned interface head.",
    )
    parser.add_argument(
        "--semantic-decoder-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint whose oracle-trained strict decoder/output interface should "
            "seed adaptive-interface training."
        ),
    )
    parser.add_argument(
        "--semantic-decoder-checkpoint-load-scope",
        choices=["full_model", "semantic_decoder_only"],
        default="full_model",
        help=(
            "Load the full checkpoint for backward compatibility, or only frozen "
            "answer-decoder/calculator-output semantic tensors."
        ),
    )
    parser.add_argument(
        "--adaptive-interface-loss-weight",
        type=float,
        default=1.0,
        help="Weight for counterfactual adaptive-interface operand target loss.",
    )
    parser.add_argument(
        "--adaptive-interface-loss-decay-steps",
        type=int,
        default=0,
        help=(
            "Linearly decay adaptive/action-loss interface objective weight to "
            "floor over this many steps; 0 keeps it constant."
        ),
    )
    parser.add_argument(
        "--adaptive-interface-loss-floor",
        type=float,
        default=0.0,
        help="Minimum adaptive/action-loss interface objective weight after decay.",
    )
    parser.add_argument(
        "--adaptive-interface-target-mode",
        choices=["hard_pair", "soft_result"],
        default="hard_pair",
        help="Hard best-pair CE or soft mass over all operand pairs producing the target result.",
    )
    parser.add_argument(
        "--adaptive-interface-entropy-weight",
        type=float,
        default=0.0,
        help="Entropy bonus weight for adaptive-interface operand distributions.",
    )
    parser.add_argument(
        "--action-loss-candidate-random",
        type=int,
        default=8,
        help="Random action pairs per prompt for action-loss weighted interface.",
    )
    parser.add_argument(
        "--action-loss-candidate-topk",
        type=int,
        default=2,
        help="Per-side top-k logits used as action-loss candidate pairs.",
    )
    parser.add_argument(
        "--action-loss-candidate-local-radius",
        type=int,
        default=1,
        help="Local +/- radius around the learned A/B actions for candidates.",
    )
    parser.add_argument(
        "--action-loss-candidate-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature over answer-NLL-ranked action candidates.",
    )
    parser.add_argument(
        "--action-loss-candidate-refresh-every",
        type=int,
        default=1,
        help=(
            "For action_loss_replay_interface, refresh cached per-prompt "
            "answer-NLL candidate targets every N steps."
        ),
    )
    parser.add_argument(
        "--action-loss-candidate-ema-beta",
        type=float,
        default=0.0,
        help=(
            "For action_loss_replay_interface, EMA coefficient for refreshed "
            "per-prompt soft targets."
        ),
    )
    parser.add_argument(
        "--action-loss-full-enum-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature over all answer-NLL-ranked action pairs.",
    )
    parser.add_argument(
        "--action-loss-full-enum-min-probability-floor",
        type=float,
        default=0.0,
        help="Optional per-action probability floor before renormalizing full-enum targets.",
    )
    parser.add_argument(
        "--action-loss-full-enum-chunk-size",
        type=int,
        default=64,
        help="Number of action pairs to score per forced-decoder chunk.",
    )
    parser.add_argument(
        "--action-loss-full-enum-target-mode",
        choices=["soft_pair", "hard_best_pair"],
        default="soft_pair",
        help=(
            "Target for independent full-enum local training: soft answer-NLL "
            "marginals or CE to the single best answer-NLL pair."
        ),
    )
    parser.add_argument(
        "--expected-answer-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the exact full-enum expected answer-loss interface "
            "objective. This objective uses model policy probabilities over "
            "all action pairs and detached answer NLL costs."
        ),
    )
    parser.add_argument(
        "--expected-answer-loss-policy-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for expected answer-loss operand policies.",
    )
    parser.add_argument(
        "--expected-answer-loss-cost-normalization",
        choices=["none", "center", "zscore"],
        default="none",
        help="Optional detached per-example cost normalization for expected answer loss.",
    )
    parser.add_argument(
        "--expected-answer-loss-entropy-weight",
        type=float,
        default=0.0,
        help="Entropy bonus weight for expected answer-loss action distributions.",
    )
    parser.add_argument(
        "--expected-answer-loss-entropy-decay-steps",
        type=int,
        default=0,
        help="Linearly decay expected answer-loss entropy weight to zero.",
    )
    parser.add_argument(
        "--expected-answer-loss-chunk-size",
        type=int,
        default=64,
        help="Number of action pairs to score per forced-decoder chunk.",
    )
    parser.add_argument(
        "--expected-answer-loss-gradient-diagnostic-only",
        action="store_true",
        help=(
            "Run one fixed-batch result-space exact expected answer-loss "
            "gradient diagnostic against sampled PG and boundary gradients, "
            "then exit."
        ),
    )
    parser.add_argument(
        "--result-boundary-target-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for answer-derived result-space boundary target CE/KL. "
            "The target is built from forced result-class answer NLLs."
        ),
    )
    parser.add_argument(
        "--result-boundary-target-mode",
        choices=["hard_best_result", "soft_result"],
        default="hard_best_result",
        help="CE to the lowest-NLL forced result or CE/KL to a soft result target.",
    )
    parser.add_argument(
        "--result-boundary-target-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for soft result boundary targets.",
    )
    parser.add_argument(
        "--result-boundary-target-min-probability-floor",
        type=float,
        default=0.0,
        help="Optional per-result probability floor before renormalizing soft targets.",
    )
    parser.add_argument(
        "--result-boundary-target-chunk-size",
        type=int,
        default=64,
        help="Number of forced result classes to score per decoder chunk.",
    )
    parser.add_argument(
        "--result-policy-entropy-weight",
        type=float,
        default=0.0,
        help=(
            "Non-prescriptive entropy bonus weight for result-space policy "
            "logits. Useful as a collapse-prevention stabilizer."
        ),
    )
    parser.add_argument(
        "--result-policy-batch-diversity-weight",
        type=float,
        default=0.0,
        help=(
            "Non-prescriptive batch-marginal diversity bonus weight for "
            "result-space policy logits."
        ),
    )
    parser.add_argument(
        "--result-policy-improvement-assignment-weight",
        type=float,
        default=0.0,
        help=(
            "Hard assignment-style result-policy CE weight. Targets are "
            "answer-loss-improving result classes selected with a per-result "
            "quota, tying diversity to per-example improvement."
        ),
    )
    parser.add_argument(
        "--result-policy-improvement-assignment-min-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum forced-answer-loss improvement required before an example "
            "can receive an improvement-assignment target."
        ),
    )
    parser.add_argument(
        "--result-policy-improvement-assignment-quota-multiplier",
        type=float,
        default=1.0,
        help=(
            "Per-result assignment quota multiplier relative to batch/result "
            "count for the hard improvement-assignment target."
        ),
    )
    parser.add_argument(
        "--result-policy-stabilization-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for result-policy stabilization metrics/loss.",
    )
    parser.add_argument(
        "--result-policy-stabilization-decay-steps",
        type=int,
        default=0,
        help=(
            "Linearly decay result-policy entropy/diversity weights to zero "
            "over this many steps; 0 keeps them constant."
        ),
    )
    parser.add_argument(
        "--calculator-causal-gap-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a non-prescriptive causal-use hinge. The objective "
            "encourages zero-injection answer loss to exceed normal answer "
            "loss by --calculator-causal-gap-margin."
        ),
    )
    parser.add_argument(
        "--calculator-causal-gap-margin",
        type=float,
        default=0.0,
        help=(
            "Required zero-injection minus normal answer-loss margin for the "
            "calculator causal-use hinge."
        ),
    )
    parser.add_argument(
        "--boundary-feedback-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for answer-gradient boundary feedback into result-space "
            "logits. Used with --calculator-estimator direct_feedback_alignment."
        ),
    )
    parser.add_argument(
        "--boundary-feedback-mode",
        choices=["output_proj_transpose", "direct_random"],
        default="output_proj_transpose",
        help=(
            "Map answer-loss gradients at the calculator injection back into "
            "result logits with either the output projection transpose or a "
            "fixed random direct-feedback matrix."
        ),
    )
    parser.add_argument(
        "--boundary-feedback-seed",
        type=int,
        default=0,
        help="Seed for the fixed random direct-feedback matrix.",
    )
    parser.add_argument(
        "--boundary-feedback-gradient-diagnostic-only",
        action="store_true",
        help=(
            "Run one fixed-batch boundary-feedback gradient diagnostic against "
            "the boundary-target ceiling, then exit."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-mode",
        choices=["fit_once_linear", "online_mlp"],
        default="fit_once_linear",
        help=(
            "Shadow-feedback estimator. fit_once_linear preserves the frozen "
            "linear map path; online_mlp runs the heldout warmup diagnostic."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-ridge",
        type=float,
        default=1e-3,
        help=(
            "Ridge penalty for the diagnostic linear shadow-feedback fit from "
            "answer-loss injection gradients to boundary result-logit gradients."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a frozen linear shadow-feedback map fitted once before "
            "training. Used with --calculator-estimator direct_feedback_alignment."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-heldout-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional deterministic heldout fraction for the shadow-feedback "
            "diagnostic. When positive, fit the shadow map on the remaining "
            "examples and report train/heldout gradient agreement separately."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-hidden-size",
        type=int,
        default=64,
        help="Hidden size for the online MLP shadow-feedback diagnostic.",
    )
    parser.add_argument(
        "--shadow-feedback-dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside the online MLP shadow-feedback module.",
    )
    parser.add_argument(
        "--shadow-feedback-online-lr",
        type=float,
        default=1e-3,
        help="Learning rate for the online MLP shadow-feedback warmup.",
    )
    parser.add_argument(
        "--shadow-feedback-weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay for the online MLP shadow-feedback warmup.",
    )
    parser.add_argument(
        "--shadow-feedback-warmup-steps",
        type=int,
        default=100,
        help="Number of frozen-main-model warmup steps for online shadow feedback.",
    )
    parser.add_argument(
        "--shadow-feedback-updates-per-step",
        type=int,
        default=1,
        help="Shadow module optimizer updates per warmup step.",
    )
    parser.add_argument(
        "--shadow-feedback-apply-max-norm",
        type=float,
        default=0.0,
        help=(
            "Optional maximum L2 norm for fixed online MLP shadow feedback "
            "during Stage 1 apply. 0 disables clamping."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-refresh-every",
        type=int,
        default=0,
        help=(
            "For online MLP shadow Stage 1, refit the shadow module every N "
            "training steps using the current model. 0 disables refresh."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-validation-fraction",
        type=float,
        default=0.0,
        help=(
            "Optional deterministic validation fraction for online MLP "
            "shadow-feedback checkpoint selection. This is separate from the "
            "heldout test fraction used for the final gate."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-validation-every",
        type=int,
        default=0,
        help=(
            "Validate online MLP shadow-feedback every N warmup steps and "
            "restore the best validation checkpoint before heldout reporting. "
            "0 disables validation selection."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-validation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional train-time validation split prediction-loss weight for "
            "online MLP shadow feedback. Heldout test split remains untouched."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-validation-gradient-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional train-time validation split model-gradient alignment "
            "loss weight for online MLP shadow feedback. Heldout test split "
            "remains untouched."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-validation-gradient-norm-weight",
        type=float,
        default=0.0,
        help=(
            "Optional relative-norm penalty inside the validation model-gradient "
            "alignment loss."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-target-normalization",
        choices=["none", "fit_zscore_per_result"],
        default="none",
        help=(
            "Normalize online MLP shadow target gradients using statistics "
            "fit on the fit split only; predictions are unnormalized before "
            "model-gradient diagnostics."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-target-transform",
        choices=["none", "unit_norm_per_example", "fit_result_prototype"],
        default="none",
        help=(
            "Optional target-gradient stabilization applied before target "
            "normalization in the online MLP shadow-feedback diagnostic."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-feature-mode",
        choices=[
            "injection_grad_logits",
            "injection_grad_logits_output_jacobian",
            "injection_grad_logits_result_input",
            "injection_grad_policy_state",
        ],
        default="injection_grad_logits",
        help=(
            "Feature state for the online MLP shadow module. Output-jacobian "
            "mode adds the local result-signal-to-injection J^T answer-loss "
            "scores; result-input mode adds the calculator result-projection "
            "input; policy-state mode adds result probabilities, "
            "log-probabilities, and entropy."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-feature-normalization",
        choices=["none", "fit_zscore_per_feature"],
        default="none",
        help=(
            "Normalize online MLP shadow input features using statistics fit "
            "on the fit split only."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-loss-mode",
        choices=["mse", "cosine", "mse_plus_cosine"],
        default="mse",
        help=(
            "Training loss for the online MLP shadow module. Cosine modes "
            "optimize normalized-target direction instead of plain MSE alone."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-selection-score-mode",
        choices=["min_result_upstream_cosine", "gap_penalized_min_cosine"],
        default="min_result_upstream_cosine",
        help=(
            "Validation checkpoint selection score for online MLP shadow "
            "feedback. Gap-penalized mode subtracts a train-validation "
            "cosine-gap penalty from the validation min-cosine score."
        ),
    )
    parser.add_argument(
        "--shadow-feedback-selection-gap-penalty",
        type=float,
        default=1.0,
        help="Penalty weight for gap-penalized online shadow checkpoint selection.",
    )
    parser.add_argument(
        "--shadow-feedback-gradient-diagnostic-only",
        action="store_true",
        help=(
            "Run one fixed-batch shadow-feedback diagnostic against the "
            "boundary-target ceiling, then exit."
        ),
    )
    parser.add_argument(
        "--calculator-result-head-hidden-size",
        type=int,
        default=0,
        help=(
            "Hidden size for a two-layer result-space request head. "
            "0 preserves the current linear result projection."
        ),
    )
    parser.add_argument(
        "--relaxed-calculator-temperature",
        type=float,
        default=1.0,
        help="Initial Concrete/softmax temperature for the relaxed calculator interface.",
    )
    parser.add_argument(
        "--relaxed-calculator-final-temperature",
        type=float,
        default=1.0,
        help="Final relaxed calculator temperature after linear decay.",
    )
    parser.add_argument(
        "--relaxed-calculator-temperature-decay-steps",
        type=int,
        default=0,
        help="Linearly decay relaxed calculator temperature over this many steps.",
    )
    parser.add_argument(
        "--relaxed-calculator-mode",
        choices=["deterministic", "gumbel"],
        default="deterministic",
        help="Use deterministic softmax or sampled Gumbel-Concrete relaxation.",
    )
    parser.add_argument(
        "--relaxed-calculator-hard-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a hard calculator signal in the forward pass with soft backward gradients.",
    )
    parser.add_argument(
        "--relaxed-calculator-entropy-weight",
        type=float,
        default=0.0,
        help="Entropy bonus weight for relaxed operand distributions.",
    )
    parser.add_argument(
        "--relaxed-calculator-entropy-decay-steps",
        type=int,
        default=0,
        help="Linearly decay relaxed calculator entropy bonus to zero.",
    )
    parser.add_argument(
        "--input-proj-anchor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint-relative L2 anchor target for "
            "calculator_hook.input_proj during adaptive-interface retention runs."
        ),
    )
    parser.add_argument(
        "--input-proj-anchor-weight",
        type=float,
        default=0.0,
        help="Initial weight for checkpoint-relative calculator_hook.input_proj L2 anchor.",
    )
    parser.add_argument(
        "--input-proj-anchor-decay-steps",
        type=int,
        default=0,
        help="Linearly decay input-proj anchor weight to zero over this many steps; 0 keeps it constant.",
    )
    parser.add_argument(
        "--freeze-semantic-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze calculator output projection and strict answer decoder.",
    )
    parser.add_argument(
        "--freeze-upstream-encoder",
        action="store_true",
        help="Diagnostic: freeze transformer encoder and train only the interface.",
    )
    parser.add_argument(
        "--calculator-read-position",
        choices=["eq", "operands", "operand_spans"],
        default="eq",
        help=(
            "Residual positions used for calculator operand logits. "
            "'eq' preserves existing behavior; 'operands' reads final A/B digits; "
            "'operand_spans' reads the full fixed-width A/B digit spans."
        ),
    )
    parser.add_argument(
        "--calculator-read-span-width",
        type=int,
        default=1,
        help="Digit-span width used by --calculator-read-position operand_spans.",
    )
    parser.add_argument(
        "--calculator-injection-mode",
        choices=["add", "replace"],
        default="add",
        help=(
            "How to apply the calculator injection. 'add' preserves the residual "
            "stream; 'replace' bottlenecks active '=' positions to the injection."
        ),
    )
    parser.add_argument(
        "--calculator-bottleneck-mode",
        choices=["none", "answer_decoder"],
        default="none",
        help=(
            "Optional stricter answer path. 'answer_decoder' predicts answer tokens "
            "only from calculator output plus answer-position metadata."
        ),
    )
    parser.add_argument(
        "--calculator-output-format",
        choices=["sum", "sum_left_operand"],
        default="sum",
        help=(
            "Calculator signal projected downstream. 'sum' preserves existing "
            "behavior; 'sum_left_operand' concatenates one-hot sum and left operand."
        ),
    )
    parser.add_argument(
        "--answer-decoder-interaction",
        choices=["none", "product"],
        default=None,
        help=(
            "Interaction used by the strict answer decoder. Default is 'none' for "
            "sum output and 'product' for new sum_left_operand configs."
        ),
    )
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument(
        "--mlp-expansion",
        type=int,
        default=4,
        help="MLP hidden-size multiplier relative to n_embd.",
    )
    parser.add_argument(
        "--calculator-hook-after-layer",
        type=int,
        default=None,
        help=(
            "Transformer layer after which to inject the calculator. "
            "Default is 2 for depth >=2, otherwise 1."
        ),
    )
    parser.add_argument(
        "--reinforce-baseline-beta",
        type=float,
        default=0.95,
        help="Exponential moving average coefficient for the answer-loss baseline.",
    )
    parser.add_argument(
        "--reinforce-baseline-mode",
        choices=["global_ema", "per_prompt_mean", "leave_one_out"],
        default="global_ema",
        help=(
            "Control variate for REINFORCE. global_ema preserves the legacy "
            "single-sample behavior; per_prompt_mean and leave_one_out use K "
            "samples per prompt."
        ),
    )
    parser.add_argument(
        "--reinforce-num-samples-per-prompt",
        type=int,
        default=1,
        help="Number of stochastic calculator-action samples per prompt.",
    )
    parser.add_argument(
        "--reinforce-entropy-weight",
        type=float,
        default=0.01,
        help="Entropy bonus weight for sampled operand distributions.",
    )
    parser.add_argument(
        "--reinforce-entropy-decay-steps",
        type=int,
        default=0,
        help="Linearly decay entropy weight to zero over this many steps; 0 keeps it constant.",
    )
    parser.add_argument(
        "--reinforce-gradient-diagnostic-only",
        action="store_true",
        help=(
            "Run one fixed-batch result-space policy-gradient diagnostic and "
            "PG-vs-boundary gradient agreement check, then exit."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oracle_train and args.variant != "model-c":
        raise ValueError("--oracle-train is only meaningful with --variant model-c")
    if args.oracle_warmup_steps < 0:
        raise ValueError("--oracle-warmup-steps must be non-negative")
    if args.oracle_warmup_steps > 0 and args.variant != "model-c":
        raise ValueError("--oracle-warmup-steps requires --variant model-c")
    if args.answer_loss_weight < 0:
        raise ValueError("--answer-loss-weight must be non-negative")
    if args.optimizer_step_max_delta_norm < 0:
        raise ValueError("--optimizer-step-max-delta-norm must be non-negative")
    if args.optimizer_step_acceptance_tolerance < 0:
        raise ValueError(
            "--optimizer-step-acceptance-tolerance must be non-negative"
        )
    parse_optimizer_step_line_search_scales(args.optimizer_step_line_search_scales)
    if args.calculator_read_span_width < 1:
        raise ValueError("--calculator-read-span-width must be positive")
    if (
        args.calculator_read_position == "operand_spans"
        and (
            len(args.digits) != 1
            or args.calculator_read_span_width != args.digits[0]
        )
    ):
        raise ValueError(
            "--calculator-read-position operand_spans requires "
            "--calculator-read-span-width to match the requested digit count"
        )
    if args.aux_operand_loss_weight < 0:
        raise ValueError("--aux-operand-loss-weight must be non-negative")
    if args.aux_operand_loss_decay_steps < 0:
        raise ValueError("--aux-operand-loss-decay-steps must be non-negative")
    if args.aux_operand_loss_floor < 0:
        raise ValueError("--aux-operand-loss-floor must be non-negative")
    if (
        args.aux_operand_loss_floor > 0
        and args.aux_operand_loss_weight <= 0
    ):
        raise ValueError("--aux-operand-loss-floor requires --aux-operand-loss-weight")
    if args.aux_operand_loss_floor > args.aux_operand_loss_weight:
        raise ValueError("--aux-operand-loss-floor cannot exceed aux weight")
    if args.snapshot_every < 0:
        raise ValueError("--snapshot-every must be non-negative")
    if args.snapshot_every > 0 and args.variant != "model-c":
        raise ValueError("--snapshot-every requires --variant model-c")
    if args.snapshot_samples < 1:
        raise ValueError("--snapshot-samples must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.checkpoint_every > 0 and args.snapshot_every <= 0:
        raise ValueError("--checkpoint-every requires --snapshot-every")
    if args.calculator_estimator == "reinforce" and args.variant != "model-c":
        raise ValueError("--calculator-estimator reinforce requires --variant model-c")
    result_space_reinforce_requested = (
        args.calculator_estimator == "reinforce"
        and args.calculator_action_head == "result_space"
    )
    if (
        (
            args.calculator_estimator
            in {
                "adaptive_interface",
                "action_loss_weighted_interface",
                "action_loss_replay_interface",
                "action_loss_full_enum_interface",
                "action_loss_full_enum_joint_interface",
                "identifiable_full_enum_local_target",
                "full_enum_expected_answer_loss",
                "gumbel_concrete_interface",
                "direct_feedback_alignment",
            }
            or result_space_reinforce_requested
        )
        and args.variant != "model-c"
    ):
        raise ValueError(
            "--calculator-estimator adaptive/action-loss interface requires --variant model-c"
        )
    if (
        args.calculator_estimator
        in {
            "adaptive_interface",
            "action_loss_weighted_interface",
            "action_loss_replay_interface",
            "action_loss_full_enum_interface",
            "action_loss_full_enum_joint_interface",
            "identifiable_full_enum_local_target",
            "full_enum_expected_answer_loss",
            "gumbel_concrete_interface",
            "direct_feedback_alignment",
        }
        or result_space_reinforce_requested
    ):
        if args.oracle_train:
            raise ValueError(
                "adaptive/action-loss interface is for learned operands, not --oracle-train"
            )
        if args.calculator_bottleneck_mode != "answer_decoder":
            raise ValueError(
                "adaptive/action-loss interface requires --calculator-bottleneck-mode answer_decoder"
            )
        if args.semantic_decoder_checkpoint is None:
            raise ValueError(
                "adaptive/action-loss interface requires --semantic-decoder-checkpoint"
            )
    if (
        args.semantic_decoder_checkpoint is not None
        and not args.semantic_decoder_checkpoint.exists()
    ):
        raise ValueError("--semantic-decoder-checkpoint does not exist")
    if args.adaptive_interface_loss_weight < 0:
        raise ValueError("--adaptive-interface-loss-weight must be non-negative")
    if args.adaptive_interface_loss_decay_steps < 0:
        raise ValueError("--adaptive-interface-loss-decay-steps must be non-negative")
    if args.adaptive_interface_loss_floor < 0:
        raise ValueError("--adaptive-interface-loss-floor must be non-negative")
    if (
        args.adaptive_interface_loss_floor > 0
        and args.adaptive_interface_loss_weight <= 0
    ):
        raise ValueError(
            "--adaptive-interface-loss-floor requires --adaptive-interface-loss-weight"
        )
    if args.adaptive_interface_loss_floor > args.adaptive_interface_loss_weight:
        raise ValueError(
            "--adaptive-interface-loss-floor cannot exceed adaptive interface weight"
        )
    if args.adaptive_interface_entropy_weight < 0:
        raise ValueError("--adaptive-interface-entropy-weight must be non-negative")
    if args.action_loss_candidate_random < 0:
        raise ValueError("--action-loss-candidate-random must be non-negative")
    if args.action_loss_candidate_topk < 0:
        raise ValueError("--action-loss-candidate-topk must be non-negative")
    if args.action_loss_candidate_local_radius < 0:
        raise ValueError("--action-loss-candidate-local-radius must be non-negative")
    if args.action_loss_candidate_temperature <= 0:
        raise ValueError("--action-loss-candidate-temperature must be positive")
    if args.action_loss_candidate_refresh_every < 1:
        raise ValueError("--action-loss-candidate-refresh-every must be positive")
    if not 0 <= args.action_loss_candidate_ema_beta < 1:
        raise ValueError("--action-loss-candidate-ema-beta must be in [0, 1)")
    if args.action_loss_full_enum_temperature <= 0:
        raise ValueError("--action-loss-full-enum-temperature must be positive")
    if args.action_loss_full_enum_min_probability_floor < 0:
        raise ValueError(
            "--action-loss-full-enum-min-probability-floor must be non-negative"
        )
    if args.action_loss_full_enum_chunk_size < 1:
        raise ValueError("--action-loss-full-enum-chunk-size must be positive")
    if args.expected_answer_loss_weight < 0:
        raise ValueError("--expected-answer-loss-weight must be non-negative")
    if args.expected_answer_loss_policy_temperature <= 0:
        raise ValueError(
            "--expected-answer-loss-policy-temperature must be positive"
        )
    if args.expected_answer_loss_entropy_weight < 0:
        raise ValueError(
            "--expected-answer-loss-entropy-weight must be non-negative"
        )
    if args.expected_answer_loss_entropy_decay_steps < 0:
        raise ValueError(
            "--expected-answer-loss-entropy-decay-steps must be non-negative"
        )
    if args.expected_answer_loss_chunk_size < 1:
        raise ValueError("--expected-answer-loss-chunk-size must be positive")
    if args.result_boundary_target_loss_weight < 0:
        raise ValueError("--result-boundary-target-loss-weight must be non-negative")
    if args.result_boundary_target_temperature <= 0:
        raise ValueError("--result-boundary-target-temperature must be positive")
    if args.result_boundary_target_min_probability_floor < 0:
        raise ValueError(
            "--result-boundary-target-min-probability-floor must be non-negative"
        )
    if args.result_boundary_target_chunk_size < 1:
        raise ValueError("--result-boundary-target-chunk-size must be positive")
    if args.result_policy_entropy_weight < 0:
        raise ValueError("--result-policy-entropy-weight must be non-negative")
    if args.result_policy_batch_diversity_weight < 0:
        raise ValueError(
            "--result-policy-batch-diversity-weight must be non-negative"
        )
    if args.result_policy_improvement_assignment_weight < 0:
        raise ValueError(
            "--result-policy-improvement-assignment-weight must be non-negative"
        )
    if args.result_policy_improvement_assignment_min_improvement < 0:
        raise ValueError(
            "--result-policy-improvement-assignment-min-improvement must be "
            "non-negative"
        )
    if args.result_policy_improvement_assignment_quota_multiplier <= 0:
        raise ValueError(
            "--result-policy-improvement-assignment-quota-multiplier must be "
            "positive"
        )
    if args.result_policy_stabilization_temperature <= 0:
        raise ValueError(
            "--result-policy-stabilization-temperature must be positive"
        )
    if args.result_policy_stabilization_decay_steps < 0:
        raise ValueError(
            "--result-policy-stabilization-decay-steps must be non-negative"
        )
    if args.calculator_causal_gap_weight < 0:
        raise ValueError("--calculator-causal-gap-weight must be non-negative")
    if args.calculator_causal_gap_margin < 0:
        raise ValueError("--calculator-causal-gap-margin must be non-negative")
    if args.boundary_feedback_weight < 0:
        raise ValueError("--boundary-feedback-weight must be non-negative")
    if args.shadow_feedback_ridge < 0:
        raise ValueError("--shadow-feedback-ridge must be non-negative")
    if args.shadow_feedback_weight < 0:
        raise ValueError("--shadow-feedback-weight must be non-negative")
    if not 0 <= args.shadow_feedback_heldout_fraction < 1:
        raise ValueError("--shadow-feedback-heldout-fraction must be in [0, 1)")
    if args.shadow_feedback_hidden_size < 1:
        raise ValueError("--shadow-feedback-hidden-size must be positive")
    if args.shadow_feedback_online_lr <= 0:
        raise ValueError("--shadow-feedback-online-lr must be positive")
    if args.shadow_feedback_warmup_steps < 1:
        raise ValueError("--shadow-feedback-warmup-steps must be positive")
    if args.shadow_feedback_updates_per_step < 1:
        raise ValueError("--shadow-feedback-updates-per-step must be positive")
    if args.shadow_feedback_apply_max_norm < 0:
        raise ValueError("--shadow-feedback-apply-max-norm must be non-negative")
    if args.shadow_feedback_refresh_every < 0:
        raise ValueError("--shadow-feedback-refresh-every must be non-negative")
    if not 0 <= args.shadow_feedback_validation_fraction < 1:
        raise ValueError("--shadow-feedback-validation-fraction must be in [0, 1)")
    if (
        args.shadow_feedback_validation_fraction
        + args.shadow_feedback_heldout_fraction
        >= 1
    ):
        raise ValueError(
            "--shadow-feedback-validation-fraction plus "
            "--shadow-feedback-heldout-fraction must leave fit examples"
        )
    if args.shadow_feedback_validation_every < 0:
        raise ValueError("--shadow-feedback-validation-every must be non-negative")
    if args.shadow_feedback_validation_loss_weight < 0:
        raise ValueError(
            "--shadow-feedback-validation-loss-weight must be non-negative"
        )
    if args.shadow_feedback_validation_gradient_loss_weight < 0:
        raise ValueError(
            "--shadow-feedback-validation-gradient-loss-weight must be "
            "non-negative"
        )
    if args.shadow_feedback_validation_gradient_norm_weight < 0:
        raise ValueError(
            "--shadow-feedback-validation-gradient-norm-weight must be "
            "non-negative"
        )
    if (
        args.shadow_feedback_validation_fraction > 0
        and args.shadow_feedback_validation_every < 1
    ):
        raise ValueError(
            "--shadow-feedback-validation-fraction requires "
            "--shadow-feedback-validation-every > 0"
        )
    if (
        args.shadow_feedback_validation_loss_weight > 0
        and args.shadow_feedback_validation_fraction <= 0
    ):
        raise ValueError(
            "--shadow-feedback-validation-loss-weight requires "
            "--shadow-feedback-validation-fraction > 0"
        )
    if (
        args.shadow_feedback_validation_gradient_loss_weight > 0
        and args.shadow_feedback_validation_fraction <= 0
    ):
        raise ValueError(
            "--shadow-feedback-validation-gradient-loss-weight requires "
            "--shadow-feedback-validation-fraction > 0"
        )
    if (
        args.shadow_feedback_mode == "online_mlp"
        and (
            args.shadow_feedback_gradient_diagnostic_only
            or args.shadow_feedback_weight > 0
        )
        and args.shadow_feedback_heldout_fraction <= 0
    ):
        raise ValueError(
            "--shadow-feedback-mode online_mlp warmup requires "
            "--shadow-feedback-heldout-fraction > 0"
        )
    if args.calculator_result_head_hidden_size < 0:
        raise ValueError("--calculator-result-head-hidden-size must be non-negative")
    if args.exhaustive_grid_batch and args.operand_max is None:
        raise ValueError("--exhaustive-grid-batch requires --operand-max")
    if (
        args.result_boundary_target_loss_weight > 0
        and args.calculator_action_head != "result_space"
    ):
        raise ValueError(
            "--result-boundary-target-loss-weight requires "
            "--calculator-action-head result_space"
        )
    if (
        args.result_boundary_target_loss_weight > 0
        and args.calculator_estimator != "gumbel_concrete_interface"
    ):
        raise ValueError(
            "--result-boundary-target-loss-weight requires "
            "--calculator-estimator gumbel_concrete_interface"
        )
    if (
        (
            args.result_policy_entropy_weight > 0
            or args.result_policy_batch_diversity_weight > 0
            or args.result_policy_improvement_assignment_weight > 0
        )
        and args.calculator_action_head != "result_space"
    ):
        raise ValueError(
            "result-policy stabilization requires "
            "--calculator-action-head result_space"
        )
    if (
        args.calculator_estimator == "full_enum_expected_answer_loss"
        and args.calculator_action_head
        not in {"independent_operands", "result_space"}
    ):
        raise ValueError(
            "full_enum_expected_answer_loss requires independent operand "
            "or result-space heads"
        )
    if (
        args.calculator_estimator == "direct_feedback_alignment"
        and args.calculator_action_head != "result_space"
    ):
        raise ValueError(
            "direct_feedback_alignment requires --calculator-action-head result_space"
        )
    if (
        args.calculator_estimator == "direct_feedback_alignment"
        and not args.boundary_feedback_gradient_diagnostic_only
        and not args.shadow_feedback_gradient_diagnostic_only
        and args.boundary_feedback_weight <= 0
        and args.shadow_feedback_weight <= 0
        and args.result_policy_entropy_weight <= 0
        and args.result_policy_batch_diversity_weight <= 0
        and args.result_policy_improvement_assignment_weight <= 0
    ):
        raise ValueError(
            "direct_feedback_alignment training requires "
            "--boundary-feedback-weight > 0, --shadow-feedback-weight > 0, "
            "or a result-policy stabilization weight > 0"
        )
    if args.relaxed_calculator_temperature <= 0:
        raise ValueError("--relaxed-calculator-temperature must be positive")
    if args.relaxed_calculator_final_temperature <= 0:
        raise ValueError("--relaxed-calculator-final-temperature must be positive")
    if args.relaxed_calculator_temperature_decay_steps < 0:
        raise ValueError(
            "--relaxed-calculator-temperature-decay-steps must be non-negative"
        )
    if args.relaxed_calculator_entropy_weight < 0:
        raise ValueError("--relaxed-calculator-entropy-weight must be non-negative")
    if args.relaxed_calculator_entropy_decay_steps < 0:
        raise ValueError(
            "--relaxed-calculator-entropy-decay-steps must be non-negative"
        )
    if (
        args.action_loss_full_enum_target_mode == "hard_best_pair"
        and args.calculator_action_head != "independent_operands"
    ):
        raise ValueError(
            "--action-loss-full-enum-target-mode hard_best_pair currently "
            "requires independent operand heads"
        )
    if (
        args.calculator_estimator == "action_loss_full_enum_joint_interface"
        and args.calculator_action_head != "joint_pair"
    ):
        raise ValueError(
            "action_loss_full_enum_joint_interface requires "
            "--calculator-action-head joint_pair"
        )
    if (
        args.calculator_action_head == "joint_pair"
        and args.calculator_estimator
        not in {"action_loss_full_enum_joint_interface", "gumbel_concrete_interface"}
    ):
        raise ValueError(
            "--calculator-action-head joint_pair is currently supported only with "
            "action_loss_full_enum_joint_interface or gumbel_concrete_interface"
        )
    if (
        args.calculator_action_head == "result_space"
        and args.calculator_estimator
        not in {
            "ste",
            "gumbel_concrete_interface",
            "reinforce",
            "full_enum_expected_answer_loss",
            "direct_feedback_alignment",
        }
    ):
        raise ValueError(
            "--calculator-action-head result_space is currently supported only with "
            "ste, gumbel_concrete_interface, reinforce, "
            "full_enum_expected_answer_loss, or direct_feedback_alignment"
        )
    if (
        args.calculator_action_head == "result_space"
        and args.calculator_output_format != "sum"
    ):
        raise ValueError(
            "--calculator-action-head result_space requires "
            "--calculator-output-format sum"
        )
    if args.input_proj_anchor_weight < 0:
        raise ValueError("--input-proj-anchor-weight must be non-negative")
    if args.input_proj_anchor_decay_steps < 0:
        raise ValueError("--input-proj-anchor-decay-steps must be non-negative")
    if args.input_proj_anchor_weight > 0 and args.input_proj_anchor_checkpoint is None:
        raise ValueError("--input-proj-anchor-weight requires --input-proj-anchor-checkpoint")
    if (
        args.input_proj_anchor_checkpoint is not None
        and not args.input_proj_anchor_checkpoint.exists()
    ):
        raise ValueError("--input-proj-anchor-checkpoint does not exist")
    if args.input_proj_lr is not None and args.input_proj_lr <= 0:
        raise ValueError("--input-proj-lr must be positive")
    if args.upstream_lr is not None and args.upstream_lr <= 0:
        raise ValueError("--upstream-lr must be positive")
    if not 0 <= args.reinforce_baseline_beta < 1:
        raise ValueError("--reinforce-baseline-beta must be in [0, 1)")
    if args.reinforce_num_samples_per_prompt < 1:
        raise ValueError("--reinforce-num-samples-per-prompt must be positive")
    if (
        args.reinforce_baseline_mode == "leave_one_out"
        and args.reinforce_num_samples_per_prompt < 2
    ):
        raise ValueError("--reinforce-baseline-mode leave_one_out requires K >= 2")
    if (
        args.reinforce_gradient_diagnostic_only
        and not result_space_reinforce_requested
    ):
        raise ValueError(
            "--reinforce-gradient-diagnostic-only requires result_space reinforce"
        )
    if args.expected_answer_loss_gradient_diagnostic_only and not (
        args.calculator_estimator == "full_enum_expected_answer_loss"
        and args.calculator_action_head == "result_space"
    ):
        raise ValueError(
            "--expected-answer-loss-gradient-diagnostic-only requires "
            "result_space full_enum_expected_answer_loss"
        )
    if args.boundary_feedback_gradient_diagnostic_only and not (
        args.calculator_estimator == "direct_feedback_alignment"
        and args.calculator_action_head == "result_space"
    ):
        raise ValueError(
            "--boundary-feedback-gradient-diagnostic-only requires "
            "result_space direct_feedback_alignment"
        )
    if args.shadow_feedback_gradient_diagnostic_only and not (
        args.calculator_estimator == "direct_feedback_alignment"
        and args.calculator_action_head == "result_space"
    ):
        raise ValueError(
            "--shadow-feedback-gradient-diagnostic-only requires "
            "result_space direct_feedback_alignment"
        )
    if args.reinforce_entropy_weight < 0:
        raise ValueError("--reinforce-entropy-weight must be non-negative")
    if args.reinforce_entropy_decay_steps < 0:
        raise ValueError("--reinforce-entropy-decay-steps must be non-negative")
    if args.n_layer < 1:
        raise ValueError("--n-layer must be positive")
    if args.n_head < 1:
        raise ValueError("--n-head must be positive")
    if args.n_embd < 1:
        raise ValueError("--n-embd must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    if args.mlp_expansion < 1:
        raise ValueError("--mlp-expansion must be positive")
    if (
        args.calculator_hook_after_layer is not None
        and not 0 <= args.calculator_hook_after_layer <= args.n_layer
    ):
        raise ValueError("--calculator-hook-after-layer must be within model depth")
    device = pick_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    suffix_parts = [args.variant]
    if args.oracle_train:
        suffix_parts.append("oracle")
    elif args.oracle_warmup_steps > 0:
        suffix_parts.append(f"oraclewarm{args.oracle_warmup_steps}")
    if args.operand_max is not None:
        suffix_parts.append(f"op0-{args.operand_max}")
    if args.exhaustive_grid_batch:
        suffix_parts.append("fullgrid")
    if args.calculator_estimator != "ste":
        suffix_parts.append(args.calculator_estimator)
    if args.calculator_estimator == "reinforce":
        if args.calculator_action_head != "independent_operands":
            suffix_parts.append(args.calculator_action_head)
        if args.reinforce_num_samples_per_prompt != 1:
            suffix_parts.append(f"K{args.reinforce_num_samples_per_prompt}")
        if args.reinforce_baseline_mode != "global_ema":
            suffix_parts.append(args.reinforce_baseline_mode)
        if args.input_proj_lr is not None:
            suffix_parts.append(f"inlr{args.input_proj_lr:g}")
        if args.upstream_lr is not None:
            suffix_parts.append(f"uplr{args.upstream_lr:g}")
        if args.reinforce_gradient_diagnostic_only:
            suffix_parts.append("graddiag")
    if args.calculator_estimator in {
        "adaptive_interface",
        "action_loss_weighted_interface",
        "action_loss_replay_interface",
        "action_loss_full_enum_interface",
        "action_loss_full_enum_joint_interface",
        "identifiable_full_enum_local_target",
        "full_enum_expected_answer_loss",
        "gumbel_concrete_interface",
    }:
        if args.calculator_action_head != "independent_operands":
            suffix_parts.append(args.calculator_action_head)
        if args.adaptive_interface_target_mode != "hard_pair":
            suffix_parts.append(args.adaptive_interface_target_mode)
        if args.input_proj_lr is not None:
            suffix_parts.append(f"inlr{args.input_proj_lr:g}")
        if args.upstream_lr is not None:
            suffix_parts.append(f"uplr{args.upstream_lr:g}")
        if args.adaptive_interface_entropy_weight > 0:
            suffix_parts.append(f"ient{args.adaptive_interface_entropy_weight:g}")
        if args.adaptive_interface_loss_decay_steps > 0:
            suffix_parts.append(
                f"ifacedecay{args.adaptive_interface_loss_decay_steps}"
            )
            if args.adaptive_interface_loss_floor > 0:
                suffix_parts.append(
                    f"ifacefloor{args.adaptive_interface_loss_floor:g}"
                )
        if args.calculator_estimator in {
            "action_loss_weighted_interface",
            "action_loss_replay_interface",
        }:
            suffix_parts.append(
                f"alrand{args.action_loss_candidate_random}"
                f"-altop{args.action_loss_candidate_topk}"
                f"-alloc{args.action_loss_candidate_local_radius}"
                f"-alt{args.action_loss_candidate_temperature:g}"
            )
        if args.calculator_estimator == "action_loss_replay_interface":
            suffix_parts.append(
                f"alrefresh{args.action_loss_candidate_refresh_every}"
                f"-alema{args.action_loss_candidate_ema_beta:g}"
            )
        if args.calculator_estimator in {
            "action_loss_full_enum_interface",
            "action_loss_full_enum_joint_interface",
            "identifiable_full_enum_local_target",
        }:
            suffix_parts.append(
                f"fullt{args.action_loss_full_enum_temperature:g}"
                f"-fullchunk{args.action_loss_full_enum_chunk_size}"
            )
            if args.action_loss_full_enum_target_mode != "soft_pair":
                suffix_parts.append(args.action_loss_full_enum_target_mode)
            if args.action_loss_full_enum_min_probability_floor > 0:
                suffix_parts.append(
                    "fullfloor"
                    f"{args.action_loss_full_enum_min_probability_floor:g}"
                )
        if args.calculator_estimator == "full_enum_expected_answer_loss":
            suffix_parts.append(
                f"expanspolt{args.expected_answer_loss_policy_temperature:g}"
                f"-expanschunk{args.expected_answer_loss_chunk_size}"
            )
            if args.expected_answer_loss_cost_normalization != "none":
                suffix_parts.append(args.expected_answer_loss_cost_normalization)
            if args.expected_answer_loss_entropy_weight > 0:
                suffix_parts.append(
                    f"expansent{args.expected_answer_loss_entropy_weight:g}"
                )
            if args.expected_answer_loss_gradient_diagnostic_only:
                suffix_parts.append("expansgraddiag")
        if args.calculator_estimator == "direct_feedback_alignment":
            suffix_parts.append(
                f"bf{args.boundary_feedback_weight:g}"
                f"-{args.boundary_feedback_mode}"
            )
            if args.boundary_feedback_mode == "direct_random":
                suffix_parts.append(f"bfseed{args.boundary_feedback_seed}")
            if args.boundary_feedback_gradient_diagnostic_only:
                suffix_parts.append("bfgraddiag")
            if args.shadow_feedback_gradient_diagnostic_only:
                suffix_parts.append(args.shadow_feedback_mode)
                if args.shadow_feedback_mode == "online_mlp":
                    suffix_parts.append(f"h{args.shadow_feedback_hidden_size}")
                    if args.shadow_feedback_dropout > 0:
                        suffix_parts.append(f"sdrop{args.shadow_feedback_dropout:g}")
                    suffix_parts.append(f"slr{args.shadow_feedback_online_lr:g}")
                    if args.shadow_feedback_weight_decay != 1e-2:
                        suffix_parts.append(
                            f"swd{args.shadow_feedback_weight_decay:g}"
                        )
                    suffix_parts.append(
                        f"warm{args.shadow_feedback_warmup_steps}"
                    )
                    if args.shadow_feedback_apply_max_norm > 0:
                        suffix_parts.append(
                            f"swapplymax{args.shadow_feedback_apply_max_norm:g}"
                        )
                    if args.shadow_feedback_refresh_every > 0:
                        suffix_parts.append(
                            f"swrefresh{args.shadow_feedback_refresh_every}"
                        )
                    if args.shadow_feedback_validation_fraction > 0:
                        suffix_parts.append(
                            f"val{args.shadow_feedback_validation_fraction:g}"
                        )
                        suffix_parts.append(
                            f"valevery{args.shadow_feedback_validation_every}"
                        )
                    if args.shadow_feedback_validation_loss_weight > 0:
                        suffix_parts.append(
                            f"valloss{args.shadow_feedback_validation_loss_weight:g}"
                        )
                    if args.shadow_feedback_validation_gradient_loss_weight > 0:
                        suffix_parts.append(
                            "valgrad"
                            f"{args.shadow_feedback_validation_gradient_loss_weight:g}"
                        )
                        if args.shadow_feedback_validation_gradient_norm_weight > 0:
                            suffix_parts.append(
                                "valgradnorm"
                                f"{args.shadow_feedback_validation_gradient_norm_weight:g}"
                            )
                    if args.shadow_feedback_target_normalization != "none":
                        suffix_parts.append(
                            f"tnorm{args.shadow_feedback_target_normalization}"
                        )
                    if args.shadow_feedback_target_transform != "none":
                        suffix_parts.append(
                            f"ttrans{args.shadow_feedback_target_transform}"
                        )
                    if args.shadow_feedback_feature_mode != "injection_grad_logits":
                        suffix_parts.append(
                            f"feat{args.shadow_feedback_feature_mode}"
                        )
                    if args.shadow_feedback_feature_normalization != "none":
                        suffix_parts.append(
                            f"fnorm{args.shadow_feedback_feature_normalization}"
                        )
                    if args.shadow_feedback_loss_mode != "mse":
                        suffix_parts.append(f"sloss{args.shadow_feedback_loss_mode}")
                    if (
                        args.shadow_feedback_selection_score_mode
                        != "min_result_upstream_cosine"
                    ):
                        suffix_parts.append(
                            f"sel{args.shadow_feedback_selection_score_mode}"
                        )
                        suffix_parts.append(
                            f"selgap{args.shadow_feedback_selection_gap_penalty:g}"
                        )
                else:
                    suffix_parts.append(
                        f"shadowridge{args.shadow_feedback_ridge:g}"
                    )
                if args.shadow_feedback_heldout_fraction > 0:
                    suffix_parts.append(
                        f"shadowheld{args.shadow_feedback_heldout_fraction:g}"
                    )
                suffix_parts.append("shadowgraddiag")
            if args.shadow_feedback_weight > 0:
                suffix_parts.append(
                    f"shadow{args.shadow_feedback_weight:g}"
                    f"-ridge{args.shadow_feedback_ridge:g}"
                )
        if args.result_boundary_target_loss_weight > 0:
            suffix_parts.append(
                f"rbt{args.result_boundary_target_loss_weight:g}"
                f"-{args.result_boundary_target_mode}"
                f"-rbtt{args.result_boundary_target_temperature:g}"
                f"-rbtchunk{args.result_boundary_target_chunk_size}"
            )
        if args.result_boundary_target_min_probability_floor > 0:
            suffix_parts.append(
                "rbtfloor"
                f"{args.result_boundary_target_min_probability_floor:g}"
            )
        if args.result_policy_entropy_weight > 0:
            suffix_parts.append(f"rpolent{args.result_policy_entropy_weight:g}")
        if args.result_policy_batch_diversity_weight > 0:
            suffix_parts.append(
                f"rpoldv{args.result_policy_batch_diversity_weight:g}"
            )
        if args.result_policy_improvement_assignment_weight > 0:
            suffix_parts.append(
                "rpolassign"
                f"{args.result_policy_improvement_assignment_weight:g}"
            )
            suffix_parts.append(
                "rpolassignmin"
                f"{args.result_policy_improvement_assignment_min_improvement:g}"
            )
            suffix_parts.append(
                "rpolassignq"
                f"{args.result_policy_improvement_assignment_quota_multiplier:g}"
            )
        if (
            args.result_policy_entropy_weight > 0
            or args.result_policy_batch_diversity_weight > 0
            or args.result_policy_improvement_assignment_weight > 0
        ):
            if args.result_policy_stabilization_temperature != 1.0:
                suffix_parts.append(
                    f"rpolt{args.result_policy_stabilization_temperature:g}"
                )
            if args.result_policy_stabilization_decay_steps > 0:
                suffix_parts.append(
                    f"rpoldcy{args.result_policy_stabilization_decay_steps}"
                )
        if args.calculator_causal_gap_weight > 0:
            suffix_parts.append(f"causalgapw{args.calculator_causal_gap_weight:g}")
            suffix_parts.append(f"causalgapm{args.calculator_causal_gap_margin:g}")
        if args.calculator_result_head_hidden_size > 0:
            suffix_parts.append(f"rhead{args.calculator_result_head_hidden_size}")
        if args.calculator_estimator == "gumbel_concrete_interface":
            suffix_parts.append(
                f"rtemp{args.relaxed_calculator_temperature:g}"
                f"-rfinal{args.relaxed_calculator_final_temperature:g}"
            )
            if args.relaxed_calculator_temperature_decay_steps > 0:
                suffix_parts.append(
                    f"rdecay{args.relaxed_calculator_temperature_decay_steps}"
                )
            if args.relaxed_calculator_mode != "deterministic":
                suffix_parts.append(args.relaxed_calculator_mode)
            if not args.relaxed_calculator_hard_forward:
                suffix_parts.append("softforward")
            if args.relaxed_calculator_entropy_weight > 0:
                suffix_parts.append(
                    f"rent{args.relaxed_calculator_entropy_weight:g}"
                )
        if args.input_proj_anchor_weight > 0:
            suffix_parts.append(f"inanchor{args.input_proj_anchor_weight:g}")
            if args.input_proj_anchor_decay_steps > 0:
                suffix_parts.append(f"inanchordecay{args.input_proj_anchor_decay_steps}")
        if args.optimizer_step_max_delta_norm > 0:
            suffix_parts.append(f"stepmax{args.optimizer_step_max_delta_norm:g}")
        if args.optimizer_step_acceptance_mode != "none":
            suffix_parts.append(f"stepaccept{args.optimizer_step_acceptance_mode}")
            if args.optimizer_step_acceptance_tolerance > 0:
                suffix_parts.append(
                    f"steptol{args.optimizer_step_acceptance_tolerance:g}"
                )
            if args.optimizer_step_acceptance_mode == "answer_loss_line_search":
                suffix_parts.append(
                    "stepscales"
                    + args.optimizer_step_line_search_scales.replace(",", "_")
                )
    if args.calculator_injection_mode != "add":
        suffix_parts.append(args.calculator_injection_mode)
    if args.calculator_bottleneck_mode != "none":
        suffix_parts.append(args.calculator_bottleneck_mode)
    if args.calculator_output_format != "sum":
        suffix_parts.append(args.calculator_output_format)
    effective_answer_decoder_interaction = (
        args.answer_decoder_interaction
        if args.answer_decoder_interaction is not None
        else (
            "product"
            if args.calculator_output_format == "sum_left_operand"
            else "none"
        )
    )
    if effective_answer_decoder_interaction != "none":
        suffix_parts.append(f"adec-{effective_answer_decoder_interaction}")
    if args.aux_operand_loss_weight > 0:
        suffix_parts.append(f"aux{args.aux_operand_loss_weight:g}")
        if args.aux_operand_loss_decay_steps > 0:
            suffix_parts.append(f"auxdecay{args.aux_operand_loss_decay_steps}")
        if args.aux_operand_loss_floor > 0:
            suffix_parts.append(f"auxfloor{args.aux_operand_loss_floor:g}")
    suffix = "-".join(suffix_parts)
    base_run_dir = create_unique_dir(args.run_root / f"{timestamp}_{suffix}")

    print(f"device: {device}")
    print(f"variant: {args.variant}")
    print(f"oracle train: {args.oracle_train}")
    print(f"oracle warmup steps: {args.oracle_warmup_steps}")
    print(f"answer loss weight: {args.answer_loss_weight}")
    print(f"injection scale: {args.injection_scale}")
    print(f"calculator injection mode: {args.calculator_injection_mode}")
    print(f"calculator bottleneck mode: {args.calculator_bottleneck_mode}")
    print(f"calculator output format: {args.calculator_output_format}")
    print(f"answer decoder interaction: {effective_answer_decoder_interaction}")
    print(f"exhaustive grid batch: {args.exhaustive_grid_batch}")
    print(f"calculator read span width: {args.calculator_read_span_width}")
    print(
        "aux operand loss: "
        f"weight={args.aux_operand_loss_weight} "
        f"decay_steps={args.aux_operand_loss_decay_steps} "
        f"floor={args.aux_operand_loss_floor}"
    )
    print(
        "diagnostic snapshots: "
        f"every={args.snapshot_every} samples={args.snapshot_samples} "
        f"checkpoint_every={args.checkpoint_every}"
    )
    print(f"calculator estimator: {args.calculator_estimator}")
    print(f"calculator action head: {args.calculator_action_head}")
    print(
        "reinforce: "
        f"baseline_mode={args.reinforce_baseline_mode} "
        f"samples_per_prompt={args.reinforce_num_samples_per_prompt} "
        f"baseline_beta={args.reinforce_baseline_beta} "
        f"entropy_weight={args.reinforce_entropy_weight} "
        f"entropy_decay_steps={args.reinforce_entropy_decay_steps} "
        f"gradient_diagnostic_only={args.reinforce_gradient_diagnostic_only}"
    )
    print(
        "adaptive interface: "
        f"target_mode={args.adaptive_interface_target_mode} "
        f"loss_weight={args.adaptive_interface_loss_weight} "
        f"loss_decay_steps={args.adaptive_interface_loss_decay_steps} "
        f"loss_floor={args.adaptive_interface_loss_floor} "
        f"entropy_weight={args.adaptive_interface_entropy_weight} "
        f"input_proj_lr={args.lr if args.input_proj_lr is None else args.input_proj_lr} "
        f"upstream_lr={args.lr if args.upstream_lr is None else args.upstream_lr} "
        f"input_proj_anchor_weight={args.input_proj_anchor_weight} "
        f"input_proj_anchor_decay_steps={args.input_proj_anchor_decay_steps}"
    )
    print(
        "optimizer trust region: "
        f"step_max_delta_norm={args.optimizer_step_max_delta_norm} "
        f"acceptance_mode={args.optimizer_step_acceptance_mode} "
        f"acceptance_tolerance={args.optimizer_step_acceptance_tolerance} "
        f"line_search_scales={args.optimizer_step_line_search_scales}"
    )
    print(
        "action-loss candidates: "
        f"random={args.action_loss_candidate_random} "
        f"topk={args.action_loss_candidate_topk} "
        f"local_radius={args.action_loss_candidate_local_radius} "
        f"temperature={args.action_loss_candidate_temperature} "
        f"refresh_every={args.action_loss_candidate_refresh_every} "
        f"ema_beta={args.action_loss_candidate_ema_beta}"
    )
    print(
        "expected answer loss: "
        f"weight={args.expected_answer_loss_weight} "
        f"policy_temperature={args.expected_answer_loss_policy_temperature} "
        f"cost_normalization={args.expected_answer_loss_cost_normalization} "
        f"entropy_weight={args.expected_answer_loss_entropy_weight} "
        f"entropy_decay_steps={args.expected_answer_loss_entropy_decay_steps} "
        f"chunk_size={args.expected_answer_loss_chunk_size} "
        f"gradient_diagnostic_only={args.expected_answer_loss_gradient_diagnostic_only}"
    )
    print(
        "result boundary target: "
        f"weight={args.result_boundary_target_loss_weight} "
        f"mode={args.result_boundary_target_mode} "
        f"temperature={args.result_boundary_target_temperature} "
        f"min_probability_floor={args.result_boundary_target_min_probability_floor} "
        f"chunk_size={args.result_boundary_target_chunk_size}"
    )
    print(
        "result policy stabilization: "
        f"entropy_weight={args.result_policy_entropy_weight} "
        f"batch_diversity_weight={args.result_policy_batch_diversity_weight} "
        "improvement_assignment_weight="
        f"{args.result_policy_improvement_assignment_weight} "
        "improvement_assignment_min="
        f"{args.result_policy_improvement_assignment_min_improvement} "
        "improvement_assignment_quota_multiplier="
        f"{args.result_policy_improvement_assignment_quota_multiplier} "
        f"temperature={args.result_policy_stabilization_temperature} "
        f"decay_steps={args.result_policy_stabilization_decay_steps}"
    )
    print(
        "calculator causal gap: "
        f"weight={args.calculator_causal_gap_weight} "
        f"margin={args.calculator_causal_gap_margin}"
    )
    print(
        "boundary feedback: "
        f"weight={args.boundary_feedback_weight} "
        f"mode={args.boundary_feedback_mode} "
        f"seed={args.boundary_feedback_seed} "
        f"gradient_diagnostic_only={args.boundary_feedback_gradient_diagnostic_only}"
    )
    print(
        "shadow feedback: "
        f"mode={args.shadow_feedback_mode} "
        f"weight={args.shadow_feedback_weight} "
        f"ridge={args.shadow_feedback_ridge} "
        f"heldout_fraction={args.shadow_feedback_heldout_fraction} "
        f"hidden_size={args.shadow_feedback_hidden_size} "
        f"dropout={args.shadow_feedback_dropout} "
        f"online_lr={args.shadow_feedback_online_lr} "
        f"weight_decay={args.shadow_feedback_weight_decay} "
        f"warmup_steps={args.shadow_feedback_warmup_steps} "
        f"updates_per_step={args.shadow_feedback_updates_per_step} "
        f"apply_max_norm={args.shadow_feedback_apply_max_norm} "
        f"refresh_every={args.shadow_feedback_refresh_every} "
        f"validation_fraction={args.shadow_feedback_validation_fraction} "
        f"validation_every={args.shadow_feedback_validation_every} "
        f"validation_loss_weight={args.shadow_feedback_validation_loss_weight} "
        f"validation_gradient_loss_weight={args.shadow_feedback_validation_gradient_loss_weight} "
        f"validation_gradient_norm_weight={args.shadow_feedback_validation_gradient_norm_weight} "
        f"target_normalization={args.shadow_feedback_target_normalization} "
        f"target_transform={args.shadow_feedback_target_transform} "
        f"feature_mode={args.shadow_feedback_feature_mode} "
        f"feature_normalization={args.shadow_feedback_feature_normalization} "
        f"loss_mode={args.shadow_feedback_loss_mode} "
        f"selection_score_mode={args.shadow_feedback_selection_score_mode} "
        f"selection_gap_penalty={args.shadow_feedback_selection_gap_penalty} "
        f"gradient_diagnostic_only={args.shadow_feedback_gradient_diagnostic_only}"
    )
    print(
        "calculator result head hidden size: "
        f"{args.calculator_result_head_hidden_size}"
    )
    print(
        "relaxed calculator: "
        f"temperature={args.relaxed_calculator_temperature} "
        f"final_temperature={args.relaxed_calculator_final_temperature} "
        f"temperature_decay_steps={args.relaxed_calculator_temperature_decay_steps} "
        f"mode={args.relaxed_calculator_mode} "
        f"hard_forward={args.relaxed_calculator_hard_forward} "
        f"entropy_weight={args.relaxed_calculator_entropy_weight} "
        f"entropy_decay_steps={args.relaxed_calculator_entropy_decay_steps}"
    )
    print(
        "architecture: "
        f"n_layer={args.n_layer} n_head={args.n_head} "
        f"n_embd={args.n_embd} mlp_expansion={args.mlp_expansion} "
        f"hook_after_layer={args.calculator_hook_after_layer}"
    )
    print(f"run root: {base_run_dir}")

    all_metrics = []
    for num_digits in args.digits:
        all_metrics.append(
            run_variant(
                num_digits=num_digits,
                args=args,
                base_run_dir=base_run_dir,
                device=device,
            )
        )

    summary = {"device": device, "variant": args.variant, "runs": all_metrics}
    (base_run_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("summary:")
    for metrics in all_metrics:
        print(
            f"  {args.variant} {metrics['num_digits']}-digit: "
            f"exact-match={metrics['exact_match']:.3f}, "
            f"final_loss={metrics['final_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
