import argparse
import csv
import json
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
    freeze_semantic_decoder: bool
    freeze_upstream_encoder: bool
    trainable_parameter_groups: list[dict[str, object]]
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
        "sampled_logp",
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
        "expected_answer_loss_best_nll",
        "expected_answer_loss_true_nll",
        "expected_answer_loss_learned_nll",
        "expected_answer_loss_expected_minus_best_gap",
        "expected_answer_loss_learned_minus_best_gap",
        "expected_answer_loss_learned_minus_true_gap",
        "expected_answer_loss_best_pair_probability",
        "expected_answer_loss_true_pair_probability",
        "expected_answer_loss_learned_pair_probability",
        "expected_answer_loss_hard_learned_best_fraction",
        "expected_answer_loss_hard_learned_pair_exact",
        "expected_answer_loss_hard_learned_calc_accuracy",
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
    return (
        model.calculator_hook.result_proj(result_input),
        positions["a"],
        positions["b"],
        positions["eq"],
    )


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
        }
        and args.freeze_semantic_decoder
    ):
        freeze_semantic_decoder_parameters(model)
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
        }
        and args.freeze_upstream_encoder
    ):
        freeze_upstream_encoder_parameters(model)
    trainable_groups = trainable_parameter_summary(model)
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
        freeze_semantic_decoder=args.freeze_semantic_decoder,
        freeze_upstream_encoder=args.freeze_upstream_encoder,
        trainable_parameter_groups=trainable_groups,
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

    curve: list[dict[str, float | int]] = []
    snapshots: list[dict[str, object]] = []
    final_loss = float("nan")
    policy_baseline: float | None = None
    action_loss_replay_cache = ActionLossReplayCache()
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
        use_relaxed_calculator = (
            args.variant == "model-c"
            and args.calculator_estimator == "gumbel_concrete_interface"
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
            logits, diagnostics = model(
                batch.x, oracle_operands=oracle_operands, return_diagnostics=True
            )
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
        sampled_logp_value = None
        operand_entropy_value = None
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
        relaxed_calculator_entropy_objective_value = None
        current_relaxed_entropy_weight = 0.0
        relaxed_calculator_metrics: dict[str, float] = {}
        anchor_loss_value = None
        anchor_weight = 0.0
        if use_reinforce:
            if policy_baseline is None:
                policy_baseline = float(answer_loss.detach().item())
            trace = diagnostics["calculator_trace"]
            eq_mask = trace["eq_mask"]
            eq_counts = eq_mask.long().sum(dim=-1)
            if not torch.all(eq_counts == 1):
                raise ValueError("REINFORCE training expects one '=' token per example")
            eq_pos = eq_mask.float().argmax(dim=-1).long()
            batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
            sampled_logp = trace["sampled_logp"][batch_idx, eq_pos]
            operand_entropy = (
                trace["a_entropy"][batch_idx, eq_pos]
                + trace["b_entropy"][batch_idx, eq_pos]
            )
            advantage = per_example_answer_loss.detach() - policy_baseline
            policy_loss = (advantage * sampled_logp).mean()
            if args.reinforce_entropy_decay_steps > 0:
                entropy_weight = args.reinforce_entropy_weight * max(
                    0.0, 1.0 - (step / args.reinforce_entropy_decay_steps)
                )
            else:
                entropy_weight = args.reinforce_entropy_weight
            entropy_loss = -entropy_weight * operand_entropy.mean()
            loss = loss + policy_loss + entropy_loss
            policy_loss_value = policy_loss.item()
            policy_advantage_mean = advantage.mean().item()
            sampled_logp_value = sampled_logp.mean().item()
            operand_entropy_value = operand_entropy.mean().item()
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
                curve_row["sampled_logp"] = sampled_logp_value
                curve_row["operand_entropy"] = operand_entropy_value
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
                    f" entropy={operand_entropy_value:.4f}"
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
        optim.step()
        if use_reinforce:
            answer_loss_value = float(answer_loss.detach().item())
            assert policy_baseline is not None
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
    metrics["input_proj_lr"] = args.lr if args.input_proj_lr is None else args.input_proj_lr
    metrics["upstream_lr"] = args.lr if args.upstream_lr is None else args.upstream_lr
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
        }
        and args.variant != "model-c"
    ):
        raise ValueError(
            "--calculator-estimator adaptive/action-loss interface requires --variant model-c"
        )
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
    if (
        args.calculator_estimator == "full_enum_expected_answer_loss"
        and args.calculator_action_head != "independent_operands"
    ):
        raise ValueError(
            "full_enum_expected_answer_loss requires independent operand heads"
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
        and args.calculator_estimator != "gumbel_concrete_interface"
    ):
        raise ValueError(
            "--calculator-action-head result_space is currently supported only with "
            "gumbel_concrete_interface"
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
    if args.calculator_estimator != "ste":
        suffix_parts.append(args.calculator_estimator)
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
        f"chunk_size={args.expected_answer_loss_chunk_size}"
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
