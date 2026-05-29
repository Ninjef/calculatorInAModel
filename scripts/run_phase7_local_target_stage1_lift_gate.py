import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.overfit_one_batch import (  # noqa: E402
    adaptive_optimizer_param_groups,
    calculator_read_result_logits,
    fixed_width_operands_from_batch,
    full_enum_expected_answer_loss,
    result_boundary_target_loss,
    score_forced_result_classes_chunked,
    snapshot_row_from_model,
    trainable_parameter_summary,
)
from scripts.run_phase7_local_target_propagation_gate import (  # noqa: E402
    DEFAULT_SEMANTIC_DECODER,
    build_model,
    exhaustive_batch,
    pick_device,
    resolve_path,
    soft_target_loss,
    target_weights_logit_descent,
    target_weights_policy_reweighted,
    write_rows,
)
from src.data import ANSWER_FORMATS  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_branch_specs(text: str) -> list[str]:
    branches = [part.strip() for part in text.split(",") if part.strip()]
    if not branches:
        raise argparse.ArgumentTypeError("expected at least one branch")
    allowed_prefixes = (
        "hard_boundary",
        "expected_loss",
        "policy_reweighted_t",
        "logit_descent_p",
    )
    for branch in branches:
        if not branch.startswith(allowed_prefixes):
            raise argparse.ArgumentTypeError(f"unknown branch {branch!r}")
    return branches


def branch_loss(
    model,
    batch,
    *,
    branch: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    if branch == "hard_boundary":
        loss, metrics = result_boundary_target_loss(
            model,
            batch,
            num_digits=args.digits,
            target_mode="hard_best_result",
            temperature=args.boundary_target_temperature,
            min_probability_floor=0.0,
            chunk_size=args.result_chunk_size,
        )
        return loss, {
            "branch_loss_mode": branch,
            "branch_loss": float(loss.detach().item()),
            **{f"boundary_{key}": value for key, value in metrics.items()},
        }
    if branch == "expected_loss":
        loss, metrics = full_enum_expected_answer_loss(
            model,
            batch,
            num_digits=args.digits,
            policy_temperature=args.expected_policy_temperature,
            cost_normalization="none",
            entropy_weight=0.0,
            chunk_size=args.result_chunk_size,
        )
        return loss, {
            "branch_loss_mode": branch,
            "branch_loss": float(loss.detach().item()),
            **{f"expected_{key}": value for key, value in metrics.items()},
        }

    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    full_losses = score_forced_result_classes_chunked(
        model,
        batch,
        chunk_size=args.result_chunk_size,
    ).detach()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    best_result = full_losses.argmin(dim=-1)
    if branch.startswith("policy_reweighted_t"):
        temperature = float(branch.removeprefix("policy_reweighted_t").replace("p", "."))
        weights = target_weights_policy_reweighted(
            result_logits,
            full_losses,
            temperature=temperature,
            min_probability_floor=args.min_probability_floor,
        )
        family_metrics: dict[str, float | str] = {
            "target_family": "policy_reweighted",
            "target_temperature": float(temperature),
        }
    elif branch.startswith("logit_descent_p"):
        proximity = float(branch.removeprefix("logit_descent_p").replace("p", "."))
        weights = target_weights_logit_descent(
            result_logits,
            full_losses,
            steps=args.logit_descent_steps,
            lr=args.logit_descent_lr,
            proximity_weight=proximity,
            temperature=args.logit_descent_target_temperature,
            min_probability_floor=args.min_probability_floor,
        )
        family_metrics = {
            "target_family": "logit_descent",
            "target_temperature": float(args.logit_descent_target_temperature),
            "target_descent_steps": int(args.logit_descent_steps),
            "target_descent_lr": float(args.logit_descent_lr),
            "target_proximity_weight": float(proximity),
        }
    else:
        raise ValueError(f"unknown branch {branch!r}")

    loss = soft_target_loss(model, batch, weights)
    target_argmax = weights.argmax(dim=-1)
    target_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    result_pred = result_logits.detach().argmax(dim=-1)
    current_probs = result_logits.detach().softmax(dim=-1)
    target_expected_loss = (weights * full_losses).sum(dim=-1)
    current_expected_loss = (current_probs * full_losses).sum(dim=-1)
    true_prob = weights.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    best_prob = weights.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    return loss, {
        "branch_loss_mode": branch,
        "branch_loss": float(loss.detach().item()),
        "target_entropy": float(target_entropy.mean().item()),
        "target_effective_results": float(target_entropy.exp().mean().item()),
        "target_true_probability": float(true_prob.mean().item()),
        "target_best_probability": float(best_prob.mean().item()),
        "target_argmax_accuracy": float((target_argmax == true_sum).float().mean().item()),
        "target_argmax_matches_best": float(
            (target_argmax == best_result).float().mean().item()
        ),
        "target_argmax_matches_current": float(
            (target_argmax == result_pred).float().mean().item()
        ),
        "target_expected_loss": float(target_expected_loss.mean().item()),
        "current_expected_loss": float(current_expected_loss.mean().item()),
        "target_expected_improvement": float(
            (current_expected_loss - target_expected_loss).mean().item()
        ),
        **family_metrics,
    }


@torch.no_grad()
def exact_grid_policy_metrics(model, batch, *, digits: int) -> dict[str, float | str]:
    logits, diagnostics = model(batch.x, return_diagnostics=True)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
    true_sum = true_a + true_b
    eq_mask = diagnostics["calculator_trace"]["eq_mask"]
    eq_pos = eq_mask.float().argmax(dim=1).long()
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    result_pred = diagnostics["calculator_trace"]["result_pred"][batch_idx, eq_pos]
    probs = result_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    true_prob = probs.gather(1, true_sum.unsqueeze(1)).squeeze(1)
    token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.y.reshape(-1),
        reduction="none",
    ).reshape_as(batch.y)
    loss_mask = batch.loss_mask.to(token_loss.dtype)
    answer_nll = (token_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)
    return {
        "exact_grid_answer_nll": float(answer_nll.mean().item()),
        "exact_grid_calculator_result_accuracy": float(
            (result_pred == true_sum).float().mean().item()
        ),
        "exact_grid_result_true_probability": float(true_prob.mean().item()),
        "exact_grid_result_entropy": float(entropy.mean().item()),
        "exact_grid_result_effective_results": float(entropy.exp().mean().item()),
    }


def train_branch(
    *,
    branch: str,
    args: argparse.Namespace,
    device: str,
    batch,
) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args, device=device)
    initial_state = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(
        adaptive_optimizer_param_groups(
            model,
            lr=args.lr,
            input_proj_lr=args.input_proj_lr,
            upstream_lr=args.upstream_lr,
            weight_decay=args.weight_decay,
        )
    )
    rows: list[dict[str, Any]] = []

    def append_snapshot(step: int, train_metrics: dict[str, float | str]) -> None:
        row = {
            "branch": branch,
            "step": int(step),
            **train_metrics,
            **exact_grid_policy_metrics(model, batch, digits=args.digits),
        }
        run_sampled_controls = (
            step == 0
            or step == args.steps
            or (
                args.control_eval_every > 0
                and step % args.control_eval_every == 0
            )
        )
        if run_sampled_controls:
            row.update(
                {
                    f"sampled_{key}": value
                    for key, value in snapshot_row_from_model(
                        model,
                        step=step,
                        num_digits=args.digits,
                        operand_max=args.operand_max,
                        samples=args.eval_samples,
                        seed=args.seed + 1000,
                        device=device,
                        answer_format=args.answer_format,
                    ).items()
                }
            )
        rows.append(row)

    initial_loss, initial_metrics = branch_loss(model, batch, branch=branch, args=args)
    append_snapshot(0, initial_metrics)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = branch_loss(model, batch, branch=branch, args=args)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            append_snapshot(step, metrics)

    final = rows[-1]
    sampled_rows = [row for row in rows if "sampled_normal_exact_match" in row]
    best = max(sampled_rows, key=lambda row: float(row["sampled_normal_exact_match"]))
    calc_best = max(rows, key=lambda row: float(row["exact_grid_calculator_result_accuracy"]))
    return {
        "branch": branch,
        "steps": int(args.steps),
        "trainable_parameter_groups": trainable_parameter_summary(model),
        "initial_state_tensors": int(len(initial_state)),
        "final": final,
        "best_sampled_normal": best,
        "best_exact_grid_calc": calc_best,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device()
    batch = exhaustive_batch(
        digits=args.digits,
        operand_max=args.operand_max,
        answer_format=args.answer_format,
        device=device,
    )
    branches = parse_branch_specs(args.branches)
    results = [
        train_branch(branch=branch, args=args, device=device, batch=batch)
        for branch in branches
    ]
    return {
        "diagnostic": "local_target_stage1_lift_gate",
        "device": device,
        "seed": int(args.seed),
        "batch_size": int(batch.x.shape[0]),
        "operand_max": int(args.operand_max),
        "steps": int(args.steps),
        "eval_every": int(args.eval_every),
        "eval_samples": int(args.eval_samples),
        "semantic_decoder_checkpoint": str(resolve_path(args.semantic_decoder_checkpoint)),
        "branches": branches,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short Stage 1 lift gate for Phase 7 local-target branches."
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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--control-eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--input-proj-lr", type=float, default=1e-2)
    parser.add_argument("--upstream-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--result-chunk-size", type=int, default=16)
    parser.add_argument("--boundary-target-temperature", type=float, default=1.0)
    parser.add_argument("--expected-policy-temperature", type=float, default=1.0)
    parser.add_argument("--min-probability-floor", type=float, default=0.0)
    parser.add_argument("--logit-descent-steps", type=int, default=25)
    parser.add_argument("--logit-descent-lr", type=float, default=1.0)
    parser.add_argument("--logit-descent-target-temperature", type=float, default=1.0)
    parser.add_argument(
        "--branches",
        default="hard_boundary,expected_loss,policy_reweighted_t1,logit_descent_p0p1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs/2026-05-29_phase7_local_target_stage1_lift_gate",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    if args.control_eval_every < 0:
        raise ValueError("--control-eval-every must be non-negative")
    if args.result_chunk_size < 1:
        raise ValueError("--result-chunk-size must be positive")

    summary = run(args)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "local_target_stage1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    all_rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        branch_dir = output_root / result["branch"]
        branch_dir.mkdir(parents=True, exist_ok=True)
        write_rows(branch_dir / "training_curve.csv", result["rows"])
        all_rows.extend(result["rows"])
    write_rows(output_root / "local_target_stage1_rows.csv", all_rows)
    for result in summary["results"]:
        final = result["final"]
        best = result["best_sampled_normal"]
        print(
            f"{result['branch']}: final_normal={final['sampled_normal_exact_match']:.4f} "
            f"final_calc={final['exact_grid_calculator_result_accuracy']:.4f} "
            f"best_normal={best['sampled_normal_exact_match']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
