import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from scripts.overfit_one_batch import (  # noqa: E402
    action_loss_weights_from_losses,
    calculator_read_result_logits,
    fixed_width_operands_from_batch,
    flattened_group_gradients,
    freeze_semantic_decoder_parameters,
    full_enum_expected_answer_loss,
    gradient_cosine,
    gradient_l2,
    load_semantic_decoder_checkpoint,
    make_model_config,
    result_boundary_target_loss,
    score_forced_result_classes_chunked,
)
from src.data import (  # noqa: E402
    ANSWER_FORMATS,
    AnswerFormat,
    ArithmeticBatch,
    answer_target,
    make_loss_mask,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import TinyGPT  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEMANTIC_DECODER = (
    REPO_ROOT
    / "runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/"
    "stage0_candidates/tiny_operand_spans_dense/oracle_train/"
    "2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/"
    "model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt"
)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def exhaustive_batch(
    *,
    digits: int,
    operand_max: int,
    answer_format: AnswerFormat,
    device: str,
) -> ArithmeticBatch:
    seq_len = max_sequence_length(digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            prompt = f"{a:0{digits}d}+{b:0{digits}d}="
            ids = tokenize(
                prompt
                + answer_target(
                    a,
                    b,
                    digits,
                    answer_format=answer_format,
                    fixed_width=True,
                )
            )
            samples.append(pad_sequence(ids, seq_len))
            masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))
    return ArithmeticBatch(
        x=torch.tensor(samples, dtype=torch.long, device=device)[:, :-1],
        y=torch.tensor(samples, dtype=torch.long, device=device)[:, 1:],
        loss_mask=torch.tensor(masks, dtype=torch.bool, device=device)[:, 1:],
    )


def build_model(args: argparse.Namespace, *, device: str) -> TinyGPT:
    cfg = make_model_config(
        args.digits,
        "model-c",
        operand_vocab_size=args.calculator_operand_vocab_size,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operand_spans",
        calculator_read_span_width=args.calculator_read_span_width,
        calculator_injection_mode="add",
        calculator_bottleneck_mode="answer_decoder",
        calculator_output_format="sum",
        answer_decoder_interaction="product",
        answer_format=args.answer_format,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=args.calculator_hook_after_layer,
    )
    model = TinyGPT(cfg).to(device)
    checkpoint = resolve_path(args.semantic_decoder_checkpoint)
    load_semantic_decoder_checkpoint(
        model,
        checkpoint,
        load_scope="semantic_decoder_only",
    )
    freeze_semantic_decoder_parameters(model)
    return model


def target_weights_policy_reweighted(
    result_logits: torch.Tensor,
    losses: torch.Tensor,
    *,
    temperature: float,
    min_probability_floor: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("target temperature must be positive")
    logits = result_logits.detach().log_softmax(dim=-1) - (
        losses.detach() / temperature
    )
    weights = torch.softmax(logits, dim=-1)
    if min_probability_floor > 0:
        action_count = weights.shape[-1]
        if min_probability_floor * action_count >= 1.0:
            raise ValueError("min probability floor times action count must be < 1")
        weights = weights.clamp_min(min_probability_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.detach()


def target_weights_logit_descent(
    result_logits: torch.Tensor,
    losses: torch.Tensor,
    *,
    steps: int,
    lr: float,
    proximity_weight: float,
    temperature: float,
    min_probability_floor: float,
) -> torch.Tensor:
    if steps < 1:
        raise ValueError("logit descent steps must be positive")
    if lr <= 0:
        raise ValueError("logit descent lr must be positive")
    if proximity_weight < 0:
        raise ValueError("proximity weight must be non-negative")
    if temperature <= 0:
        raise ValueError("target temperature must be positive")
    start_logits = result_logits.detach()
    target_logits = start_logits.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([target_logits], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probs = torch.softmax(target_logits, dim=-1)
        # Each row is an independent local boundary variable. Sum over rows so
        # the per-example target step size does not shrink with grid size.
        expected_loss = (probs * losses.detach()).sum(dim=-1).sum()
        proximity = (
            0.5
            * proximity_weight
            * (target_logits - start_logits).pow(2).sum(dim=-1).sum()
        )
        objective = expected_loss + proximity
        objective.backward()
        optimizer.step()
    weights = torch.softmax(target_logits.detach() / temperature, dim=-1)
    if min_probability_floor > 0:
        action_count = weights.shape[-1]
        if min_probability_floor * action_count >= 1.0:
            raise ValueError("min probability floor times action count must be < 1")
        weights = weights.clamp_min(min_probability_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.detach()


def soft_target_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    if result_logits.shape != target_weights.shape:
        raise ValueError("target weights and result logits shape mismatch")
    return -(target_weights * result_logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def target_metrics(
    *,
    result_logits: torch.Tensor,
    losses: torch.Tensor,
    target_weights: torch.Tensor,
    true_sum: torch.Tensor,
    best_result: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    current_probs = torch.softmax(result_logits.detach(), dim=-1)
    current_expected = (current_probs * losses).sum(dim=-1)
    target_expected = (target_weights * losses).sum(dim=-1)
    target_entropy = -(
        target_weights * target_weights.clamp_min(1e-12).log()
    ).sum(dim=-1)
    target_argmax = target_weights.argmax(dim=-1)
    true_prob = target_weights.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    best_prob = target_weights.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    current_argmax = result_logits.detach().argmax(dim=-1)
    return {
        f"{prefix}_target_expected_loss": float(target_expected.mean().item()),
        f"{prefix}_current_expected_loss": float(current_expected.mean().item()),
        f"{prefix}_target_expected_improvement": float(
            (current_expected - target_expected).mean().item()
        ),
        f"{prefix}_target_entropy": float(target_entropy.mean().item()),
        f"{prefix}_target_effective_results": float(
            target_entropy.exp().mean().item()
        ),
        f"{prefix}_target_true_probability": float(true_prob.mean().item()),
        f"{prefix}_target_best_probability": float(best_prob.mean().item()),
        f"{prefix}_target_argmax_accuracy": float(
            (target_argmax == true_sum).float().mean().item()
        ),
        f"{prefix}_target_argmax_matches_best": float(
            (target_argmax == best_result).float().mean().item()
        ),
        f"{prefix}_target_argmax_matches_current": float(
            (target_argmax == current_argmax).float().mean().item()
        ),
    }


def gradient_summary(
    *,
    model: TinyGPT,
    batch: ArithmeticBatch,
    num_digits: int,
    local_loss: torch.Tensor,
    local_prefix: str,
    boundary_target_temperature: float,
    boundary_chunk_size: int,
    expected_policy_temperature: float,
    expected_chunk_size: int,
) -> dict[str, float | str]:
    model.zero_grad(set_to_none=True)
    local_loss.backward()
    local_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    boundary_loss, boundary_metrics = result_boundary_target_loss(
        model,
        batch,
        num_digits=num_digits,
        target_mode="hard_best_result",
        temperature=boundary_target_temperature,
        min_probability_floor=0.0,
        chunk_size=boundary_chunk_size,
    )
    boundary_loss.backward()
    boundary_grads = flattened_group_gradients(model)

    model.zero_grad(set_to_none=True)
    expected_loss, expected_metrics = full_enum_expected_answer_loss(
        model,
        batch,
        num_digits=num_digits,
        policy_temperature=expected_policy_temperature,
        cost_normalization="none",
        entropy_weight=0.0,
        chunk_size=expected_chunk_size,
    )
    expected_loss.backward()
    expected_grads = flattened_group_gradients(model)
    model.zero_grad(set_to_none=True)

    result_group = "calculator_hook.result_proj"
    upstream_group = "upstream"
    summary: dict[str, float | str] = {
        "local_target_mode": local_prefix,
        f"{local_prefix}_loss": float(local_loss.detach().item()),
        "boundary_loss": float(boundary_loss.detach().item()),
        "expected_answer_loss": float(expected_loss.detach().item()),
        f"{local_prefix}_result_proj_grad_l2": gradient_l2(
            local_grads, result_group
        ),
        f"{local_prefix}_upstream_grad_l2": gradient_l2(local_grads, upstream_group),
        "boundary_result_proj_grad_l2": gradient_l2(boundary_grads, result_group),
        "boundary_upstream_grad_l2": gradient_l2(boundary_grads, upstream_group),
        "expected_result_proj_grad_l2": gradient_l2(expected_grads, result_group),
        "expected_upstream_grad_l2": gradient_l2(expected_grads, upstream_group),
        f"{local_prefix}_vs_boundary_result_proj_cosine": gradient_cosine(
            local_grads, boundary_grads, result_group
        ),
        f"{local_prefix}_vs_boundary_upstream_cosine": gradient_cosine(
            local_grads, boundary_grads, upstream_group
        ),
        f"{local_prefix}_vs_expected_result_proj_cosine": gradient_cosine(
            local_grads, expected_grads, result_group
        ),
        f"{local_prefix}_vs_expected_upstream_cosine": gradient_cosine(
            local_grads, expected_grads, upstream_group
        ),
        "expected_vs_boundary_result_proj_cosine": gradient_cosine(
            expected_grads, boundary_grads, result_group
        ),
        "expected_vs_boundary_upstream_cosine": gradient_cosine(
            expected_grads, boundary_grads, upstream_group
        ),
    }
    for key, value in boundary_metrics.items():
        summary[f"boundary_{key}"] = value
    for key, value in expected_metrics.items():
        summary[f"expected_{key}"] = value
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    batch = exhaustive_batch(
        digits=args.digits,
        operand_max=args.operand_max,
        answer_format=args.answer_format,
        device=device,
    )
    model = build_model(args, device=device)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    full_losses = score_forced_result_classes_chunked(
        model,
        batch,
        chunk_size=args.result_chunk_size,
    ).detach()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    best_result = full_losses.argmin(dim=-1)
    soft_ceiling = action_loss_weights_from_losses(
        full_losses,
        temperature=args.boundary_target_temperature,
        min_probability_floor=0.0,
    )

    rows: list[dict[str, Any]] = []
    for temperature in args.policy_reweighted_temperatures:
        weights = target_weights_policy_reweighted(
            result_logits,
            full_losses,
            temperature=temperature,
            min_probability_floor=args.min_probability_floor,
        )
        local_loss = soft_target_loss(model, batch, weights)
        prefix = f"policy_reweighted_t{temperature:g}".replace(".", "p")
        row = {
            "diagnostic": "local_target_propagation_stage0_gate",
            "seed": int(args.seed),
            "batch_size": int(batch.x.shape[0]),
            "operand_max": int(args.operand_max),
            "target_family": "policy_reweighted",
            "target_temperature": float(temperature),
            "target_descent_steps": 0,
            "target_descent_lr": 0.0,
            "target_proximity_weight": 0.0,
        }
        row.update(
            target_metrics(
                result_logits=result_logits,
                losses=full_losses,
                target_weights=weights,
                true_sum=true_sum,
                best_result=best_result,
                prefix=prefix,
            )
        )
        row.update(
            gradient_summary(
                model=model,
                batch=batch,
                num_digits=args.digits,
                local_loss=local_loss,
                local_prefix=prefix,
                boundary_target_temperature=args.boundary_target_temperature,
                boundary_chunk_size=args.result_chunk_size,
                expected_policy_temperature=args.expected_policy_temperature,
                expected_chunk_size=args.result_chunk_size,
            )
        )
        rows.append(row)

    for proximity_weight in args.logit_descent_proximity_weights:
        weights = target_weights_logit_descent(
            result_logits,
            full_losses,
            steps=args.logit_descent_steps,
            lr=args.logit_descent_lr,
            proximity_weight=proximity_weight,
            temperature=args.logit_descent_target_temperature,
            min_probability_floor=args.min_probability_floor,
        )
        local_loss = soft_target_loss(model, batch, weights)
        prefix = f"logit_descent_p{proximity_weight:g}".replace(".", "p")
        row = {
            "diagnostic": "local_target_propagation_stage0_gate",
            "seed": int(args.seed),
            "batch_size": int(batch.x.shape[0]),
            "operand_max": int(args.operand_max),
            "target_family": "logit_descent",
            "target_temperature": float(args.logit_descent_target_temperature),
            "target_descent_steps": int(args.logit_descent_steps),
            "target_descent_lr": float(args.logit_descent_lr),
            "target_proximity_weight": float(proximity_weight),
        }
        row.update(
            target_metrics(
                result_logits=result_logits,
                losses=full_losses,
                target_weights=weights,
                true_sum=true_sum,
                best_result=best_result,
                prefix=prefix,
            )
        )
        row.update(
            gradient_summary(
                model=model,
                batch=batch,
                num_digits=args.digits,
                local_loss=local_loss,
                local_prefix=prefix,
                boundary_target_temperature=args.boundary_target_temperature,
                boundary_chunk_size=args.result_chunk_size,
                expected_policy_temperature=args.expected_policy_temperature,
                expected_chunk_size=args.result_chunk_size,
            )
        )
        rows.append(row)

    hard_best_accuracy = float((best_result == true_sum).float().mean().item())
    soft_true_probability = float(
        soft_ceiling.gather(1, true_sum.unsqueeze(-1)).squeeze(-1).mean().item()
    )
    summary: dict[str, Any] = {
        "diagnostic": "local_target_propagation_stage0_gate",
        "seed": int(args.seed),
        "device": device,
        "batch_size": int(batch.x.shape[0]),
        "operand_max": int(args.operand_max),
        "semantic_decoder_checkpoint": str(resolve_path(args.semantic_decoder_checkpoint)),
        "hard_best_equals_true_sum": hard_best_accuracy,
        "soft_ceiling_true_probability": soft_true_probability,
        "rows": rows,
    }
    return summary


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one float")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 0 gate for local-target/target-propagation style result "
            "boundary targets in Phase 7."
        )
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="sum")
    parser.add_argument(
        "--semantic-decoder-checkpoint",
        type=Path,
        default=DEFAULT_SEMANTIC_DECODER,
    )
    parser.add_argument("--calculator-operand-vocab-size", type=int, default=20)
    parser.add_argument("--calculator-read-span-width", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--mlp-expansion", type=int, default=1)
    parser.add_argument("--calculator-hook-after-layer", type=int, default=1)
    parser.add_argument("--result-chunk-size", type=int, default=16)
    parser.add_argument("--boundary-target-temperature", type=float, default=1.0)
    parser.add_argument("--expected-policy-temperature", type=float, default=1.0)
    parser.add_argument("--min-probability-floor", type=float, default=0.0)
    parser.add_argument(
        "--policy-reweighted-temperatures",
        type=parse_float_list,
        default=parse_float_list("0.25,0.5,1,2"),
    )
    parser.add_argument(
        "--logit-descent-proximity-weights",
        type=parse_float_list,
        default=parse_float_list("0.01,0.1,1"),
    )
    parser.add_argument("--logit-descent-steps", type=int, default=25)
    parser.add_argument("--logit-descent-lr", type=float, default=1.0)
    parser.add_argument("--logit-descent-target-temperature", type=float, default=1.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "runs/2026-05-29_phase7_local_target_propagation_gate",
    )
    args = parser.parse_args()

    if args.result_chunk_size < 1:
        raise ValueError("--result-chunk-size must be positive")
    if args.logit_descent_steps < 1:
        raise ValueError("--logit-descent-steps must be positive")
    if args.min_probability_floor < 0:
        raise ValueError("--min-probability-floor must be non-negative")
    if args.boundary_target_temperature <= 0:
        raise ValueError("--boundary-target-temperature must be positive")
    if args.expected_policy_temperature <= 0:
        raise ValueError("--expected-policy-temperature must be positive")

    summary = run_gate(args)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "local_target_propagation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    write_rows(output_root / "local_target_propagation_rows.csv", summary["rows"])
    best_result = max(
        summary["rows"],
        key=lambda row: row.get(
            f"{row['local_target_mode']}_vs_boundary_upstream_cosine",
            float("-inf"),
        ),
    )
    mode = best_result["local_target_mode"]
    print(
        f"wrote {output_root}; best_upstream={best_result[mode + '_vs_boundary_upstream_cosine']:.4f} "
        f"best_result={best_result[mode + '_vs_boundary_result_proj_cosine']:.4f} "
        f"mode={mode}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
