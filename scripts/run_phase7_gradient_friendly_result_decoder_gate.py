from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.diagnose_calculator_protocol import pick_device
from scripts.overfit_one_batch import (
    fixed_width_operands_from_batch,
    freeze_semantic_decoder_parameters,
    load_semantic_decoder_checkpoint,
    make_exhaustive_range_batch,
    make_model_config,
    make_range_batch,
    masked_cross_entropy_per_example,
    run_expected_answer_loss_gradient_diagnostic,
)
from src.data import ANSWER_FORMATS, EQ_ID, AnswerFormat, ArithmeticBatch
from src.model import TinyGPT, masked_cross_entropy


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_CHECKPOINT = (
    REPO_ROOT
    / "runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/"
    "stage0_candidates/tiny_operand_spans_dense/oracle_train/"
    "2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/"
    "model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt"
)
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate"
)
SEMANTIC_PREFIXES = (
    "answer_offset_emb.",
    "answer_decoder.",
    "calculator_hook.output_proj.",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")


def natural_model(
    *,
    estimator: str,
    action_head: str,
    device: str | torch.device,
    answer_format: AnswerFormat,
) -> TinyGPT:
    cfg = make_model_config(
        2,
        "model-c",
        operand_vocab_size=20,
        calculator_estimator=estimator,
        calculator_action_head=action_head,
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        calculator_output_format="sum",
        answer_decoder_interaction="product",
        answer_format=answer_format,
        n_layer=2,
        n_head=1,
        n_embd=16,
        mlp_expansion=1,
        calculator_hook_after_layer=1,
    )
    return TinyGPT(cfg).to(device)


def semantic_parameters(model: TinyGPT) -> list[torch.nn.Parameter]:
    return [
        param
        for name, param in model.named_parameters()
        if name.startswith(SEMANTIC_PREFIXES)
    ]


def freeze_except_semantic_decoder(model: TinyGPT) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in semantic_parameters(model):
        param.requires_grad = True


def make_oracle_operands_from_batch(
    batch: ArithmeticBatch, *, num_digits: int
) -> torch.Tensor:
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    oracle = torch.zeros((*batch.x.shape, 2), dtype=torch.long, device=batch.x.device)
    oracle[..., 0] = true_a.unsqueeze(-1)
    oracle[..., 1] = true_b.unsqueeze(-1)
    return oracle


def true_sum_from_batch(batch: ArithmeticBatch, *, num_digits: int) -> torch.Tensor:
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    return true_a + true_b


def answer_token_exact(logits: torch.Tensor, batch: ArithmeticBatch) -> float:
    pred = logits.argmax(dim=-1)
    correct_or_masked = (pred == batch.y) | (~batch.loss_mask)
    return float(correct_or_masked.all(dim=-1).float().mean().item())


@torch.no_grad()
def forced_true_and_oracle_metrics(
    model: TinyGPT, batch: ArithmeticBatch, *, num_digits: int
) -> dict[str, float]:
    model.eval()
    true_sum = true_sum_from_batch(batch, num_digits=num_digits)
    forced_logits = model(batch.x, forced_calculator_result_class=true_sum)
    forced_loss = masked_cross_entropy(forced_logits, batch.y, batch.loss_mask)
    oracle_logits = model(
        batch.x,
        oracle_operands=make_oracle_operands_from_batch(batch, num_digits=num_digits),
    )
    oracle_loss = masked_cross_entropy(oracle_logits, batch.y, batch.loss_mask)
    return {
        "forced_true_exact_accuracy": answer_token_exact(forced_logits, batch),
        "forced_true_nll": float(forced_loss.item()),
        "oracle_exact_accuracy": answer_token_exact(oracle_logits, batch),
        "oracle_nll": float(oracle_loss.item()),
    }


def soft_result_distribution(
    true_sum: torch.Tensor,
    *,
    result_vocab_size: int,
    sigma: float,
    uniform_mix: float,
) -> torch.Tensor:
    classes = torch.arange(result_vocab_size, device=true_sum.device).float()
    distances = classes.unsqueeze(0) - true_sum.float().unsqueeze(1)
    logits = -0.5 * distances.pow(2) / max(sigma, 1e-6) ** 2
    probs = torch.softmax(logits, dim=-1)
    if uniform_mix > 0:
        uniform = torch.full_like(probs, 1.0 / result_vocab_size)
        probs = (1.0 - uniform_mix) * probs + uniform_mix * uniform
    return probs


def answer_decoder_logits_from_result_distribution(
    model: TinyGPT, batch: ArithmeticBatch, result_probs: torch.Tensor
) -> torch.Tensor:
    if model.calculator_hook is None:
        raise ValueError("soft result decoder loss requires a calculator hook")
    if model.answer_offset_emb is None or model.answer_decoder is None:
        raise ValueError("soft result decoder loss requires answer_decoder bottleneck")
    B, T = batch.x.shape
    eq_mask = batch.x == EQ_ID
    has_eq = eq_mask.any(dim=-1)
    eq_pos = eq_mask.float().argmax(dim=-1).long()
    answer_mask = (
        torch.arange(T, device=batch.x.device).unsqueeze(0) >= eq_pos.unsqueeze(-1)
    ) & has_eq.unsqueeze(-1)
    offsets = (
        torch.arange(T, device=batch.x.device).unsqueeze(0) - eq_pos.unsqueeze(-1)
    ).clamp(min=0, max=model.cfg.block_size - 1)
    selected_signal = model.calculator_hook.output_proj(result_probs)
    selected_signal = selected_signal * model.calculator_hook.injection_scale
    offset_h = model.answer_offset_emb(offsets)
    selected_signal = selected_signal.unsqueeze(1)
    decoder_h = selected_signal + offset_h + (selected_signal * offset_h)
    decoder_logits = model.answer_decoder(decoder_h)
    filler = torch.zeros_like(decoder_logits)
    return torch.where(answer_mask.unsqueeze(-1), decoder_logits, filler)


def soft_result_calibration_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    sigma: float,
    uniform_mix: float,
) -> torch.Tensor:
    true_sum = true_sum_from_batch(batch, num_digits=num_digits)
    result_probs = soft_result_distribution(
        true_sum,
        result_vocab_size=model.cfg.calculator_result_vocab_size,
        sigma=sigma,
        uniform_mix=uniform_mix,
    )
    logits = answer_decoder_logits_from_result_distribution(model, batch, result_probs)
    return masked_cross_entropy(logits, batch.y, batch.loss_mask)


def trainable_forced_result_losses(
    model: TinyGPT, batch: ArithmeticBatch, forced_result: torch.Tensor
) -> torch.Tensor:
    expanded_x = batch.x.repeat_interleave(forced_result.shape[1], dim=0)
    expanded_y = batch.y.repeat_interleave(forced_result.shape[1], dim=0)
    expanded_mask = batch.loss_mask.repeat_interleave(forced_result.shape[1], dim=0)
    forced = forced_result.reshape(-1)
    logits = model(expanded_x, forced_calculator_result_class=forced)
    losses = masked_cross_entropy_per_example(
        logits, expanded_y, expanded_mask
    ).reshape(batch.x.shape[0], forced_result.shape[1])
    return losses


def contrastive_margin_loss(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    num_digits: int,
    wrong_classes_per_prompt: int,
    margin: float,
    generator: torch.Generator,
) -> torch.Tensor:
    true_sum = true_sum_from_batch(batch, num_digits=num_digits)
    result_vocab_size = model.cfg.calculator_result_vocab_size
    random_offsets = torch.randint(
        1,
        result_vocab_size,
        (batch.x.shape[0], wrong_classes_per_prompt),
        generator=generator,
        device=batch.x.device,
    )
    wrong = (true_sum.unsqueeze(1) + random_offsets) % result_vocab_size
    forced = torch.cat([true_sum.unsqueeze(1), wrong], dim=1)
    losses = trainable_forced_result_losses(model, batch, forced)
    true_loss = losses[:, :1]
    wrong_loss = losses[:, 1:]
    return torch.relu(margin + true_loss - wrong_loss).mean()


def semantic_delta_summary(before_path: Path, after_path: Path) -> dict[str, float]:
    before_payload = torch.load(before_path, map_location="cpu")
    after_payload = torch.load(after_path, map_location="cpu")
    before_state = before_payload.get("model_state_dict", before_payload)
    after_state = after_payload.get("model_state_dict", after_payload)
    squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, before in before_state.items():
        if not name.startswith(SEMANTIC_PREFIXES) or name not in after_state:
            continue
        after = after_state[name]
        if before.shape != after.shape:
            continue
        delta = after.detach().float() - before.detach().float()
        squared += float(delta.pow(2).sum().item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.abs().max().item() > 0)
        total += 1
    return {
        "semantic_decoder_delta_l2": math.sqrt(squared),
        "semantic_decoder_delta_max_abs": max_abs,
        "semantic_decoder_changed_tensors": changed,
        "semantic_decoder_tensor_count": total,
    }


def save_checkpoint(
    model: TinyGPT,
    path: Path,
    *,
    branch: str,
    step: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "branch": branch,
                "step": step,
                "model": asdict(model.cfg),
                "metrics": metrics,
            },
        },
        path,
    )


def train_decoder_candidate(
    *,
    branch: str,
    baseline_checkpoint: Path,
    run_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    model = natural_model(
        estimator="ste",
        action_head="independent_operands",
        device=device,
        answer_format=args.answer_format,
    )
    load_semantic_decoder_checkpoint(
        model, baseline_checkpoint, load_scope="semantic_decoder_only"
    )
    freeze_except_semantic_decoder(model)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.decoder_lr,
        betas=(0.9, 0.95),
        weight_decay=args.decoder_weight_decay,
    )
    rng = random.Random(args.seed + (17 if branch == "soft_calibration" else 29))
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + (1700 if branch == "soft_calibration" else 2900))
    exhaustive_batch = make_exhaustive_range_batch(
        num_digits=2,
        operand_max=19,
        fixed_width=True,
        device=device,
        answer_format=args.answer_format,
    )
    curve: list[dict[str, float | int | str]] = []
    best_checkpoint = run_dir / f"{branch}_best_weights.pt"
    best_loss = float("inf")
    best_metrics: dict[str, float] = {}
    for step in range(args.decoder_steps + 1):
        if args.decoder_exhaustive_grid_batch:
            batch = exhaustive_batch
        else:
            batch = make_range_batch(
                batch_size=args.decoder_batch_size,
                num_digits=2,
                operand_max=19,
                rng=rng,
                fixed_width=True,
                device=device,
                answer_format=args.answer_format,
            )
        true_sum = true_sum_from_batch(batch, num_digits=2)
        logits = model(batch.x, forced_calculator_result_class=true_sum)
        hard_loss = masked_cross_entropy(logits, batch.y, batch.loss_mask)
        extra_loss = batch.x.new_tensor(0.0, dtype=torch.float)
        if branch == "soft_calibration":
            extra_loss = soft_result_calibration_loss(
                model,
                batch,
                num_digits=2,
                sigma=args.soft_result_sigma,
                uniform_mix=args.soft_result_uniform_mix,
            )
        elif branch == "contrastive_margin":
            extra_loss = contrastive_margin_loss(
                model,
                batch,
                num_digits=2,
                wrong_classes_per_prompt=args.contrastive_wrong_classes,
                margin=args.contrastive_margin,
                generator=generator,
            )
        else:
            raise ValueError(f"unknown branch: {branch}")
        loss = hard_loss + args.decoder_extra_loss_weight * extra_loss
        if step < args.decoder_steps:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
        if step % args.decoder_log_every == 0 or step == args.decoder_steps:
            eval_metrics = forced_true_and_oracle_metrics(
                model, exhaustive_batch, num_digits=2
            )
            row: dict[str, float | int | str] = {
                "step": step,
                "branch": branch,
                "loss": float(loss.detach().item()),
                "hard_loss": float(hard_loss.detach().item()),
                "extra_loss": float(extra_loss.detach().item()),
                **eval_metrics,
            }
            curve.append(row)
            print(
                f"{branch} step={step:04d} loss={row['loss']:.4f} "
                f"forced_exact={row['forced_true_exact_accuracy']:.4f} "
                f"oracle_exact={row['oracle_exact_accuracy']:.4f}"
            )
            if float(row["forced_true_nll"]) < best_loss:
                best_loss = float(row["forced_true_nll"])
                best_metrics = {
                    key: float(value)
                    for key, value in row.items()
                    if isinstance(value, (int, float)) and key != "step"
                }
                save_checkpoint(
                    model,
                    best_checkpoint,
                    branch=branch,
                    step=step,
                    metrics=best_metrics,
                )
    write_json(run_dir / f"{branch}_curve.json", curve)
    final_checkpoint = run_dir / f"{branch}_final_weights.pt"
    save_checkpoint(
        model,
        final_checkpoint,
        branch=branch,
        step=args.decoder_steps,
        metrics=forced_true_and_oracle_metrics(model, exhaustive_batch, num_digits=2),
    )
    summary = {
        "branch": branch,
        "best_checkpoint": str(best_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "best_forced_true_nll": best_loss,
        "best_metrics": best_metrics,
        **semantic_delta_summary(baseline_checkpoint, best_checkpoint),
    }
    write_json(run_dir / f"{branch}_decoder_summary.json", summary)
    return best_checkpoint, summary


def downstream_alignment_diagnostic(
    *,
    label: str,
    checkpoint: Path,
    run_dir: Path,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(args.seed + 2)
    model = natural_model(
        estimator="full_enum_expected_answer_loss",
        action_head="result_space",
        device=device,
        answer_format=args.answer_format,
    )
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope="semantic_decoder_only")
    freeze_semantic_decoder_parameters(model)
    batch = make_exhaustive_range_batch(
        num_digits=2,
        operand_max=19,
        fixed_width=True,
        device=device,
        answer_format=args.answer_format,
    )
    forced_metrics = forced_true_and_oracle_metrics(model, batch, num_digits=2)
    diagnostic = run_expected_answer_loss_gradient_diagnostic(
        model,
        batch,
        num_digits=2,
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
    summary = {
        "label": label,
        "checkpoint": str(checkpoint),
        **forced_metrics,
        **diagnostic,
    }
    write_json(run_dir / f"{label}_alignment_summary.json", summary)
    return summary


def stage0_passed(summary: dict[str, Any]) -> bool:
    return (
        summary["forced_true_exact_accuracy"] >= 0.99
        and summary["boundary_result_boundary_target_hard_best_equals_true_sum"] >= 0.99
        and summary["exact_result_proj_grad_l2"] > 0.0
        and summary["exact_upstream_grad_l2"] > 0.0
        and summary["exact_semantic_decoder_grad_l2"] == 0.0
        and summary["exact_vs_boundary_result_proj_cosine"] > 0.0
        and summary["exact_vs_boundary_upstream_cosine"] > 0.0
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 gradient-friendly result decoder alignment gate."
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT,
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="sum")
    parser.add_argument("--decoder-steps", type=int, default=300)
    parser.add_argument("--decoder-batch-size", type=int, default=400)
    parser.add_argument("--decoder-exhaustive-grid-batch", action="store_true")
    parser.add_argument("--decoder-lr", type=float, default=0.003)
    parser.add_argument("--decoder-weight-decay", type=float, default=0.0)
    parser.add_argument("--decoder-extra-loss-weight", type=float, default=1.0)
    parser.add_argument("--decoder-log-every", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--branches", nargs="+", default=["soft_calibration", "contrastive_margin"])
    parser.add_argument("--soft-result-sigma", type=float, default=2.5)
    parser.add_argument("--soft-result-uniform-mix", type=float, default=0.05)
    parser.add_argument("--contrastive-wrong-classes", type=int, default=8)
    parser.add_argument("--contrastive-margin", type=float, default=1.0)
    parser.add_argument("--expected-answer-loss-policy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--expected-answer-loss-cost-normalization",
        choices=["none", "center", "zscore"],
        default="none",
    )
    parser.add_argument("--expected-answer-loss-entropy-weight", type=float, default=0.0)
    parser.add_argument("--expected-answer-loss-chunk-size", type=int, default=64)
    parser.add_argument(
        "--reinforce-baseline-mode",
        choices=["global_ema", "per_prompt_mean", "leave_one_out"],
        default="leave_one_out",
    )
    parser.add_argument("--reinforce-num-samples-per-prompt", type=int, default=16)
    parser.add_argument("--reinforce-entropy-weight", type=float, default=0.0)
    parser.add_argument(
        "--result-boundary-target-mode",
        choices=["hard_best_result", "soft_result"],
        default="hard_best_result",
    )
    parser.add_argument("--result-boundary-target-temperature", type=float, default=1.0)
    parser.add_argument(
        "--result-boundary-target-min-probability-floor", type=float, default=0.0
    )
    parser.add_argument("--result-boundary-target-chunk-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_checkpoint.exists():
        raise FileNotFoundError(args.baseline_checkpoint)
    if args.decoder_steps < 0:
        raise ValueError("--decoder-steps must be non-negative")
    if args.decoder_log_every < 1:
        raise ValueError("--decoder-log-every must be positive")
    unknown = sorted(set(args.branches) - {"soft_calibration", "contrastive_margin"})
    if unknown:
        raise ValueError(f"unknown branches: {unknown}")
    device = pick_device() if args.device == "auto" else args.device
    run_dir = args.run_root / timestamp()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        | {"device": device, "run_dir": str(run_dir)},
    )

    summaries: list[dict[str, Any]] = []
    baseline_summary = downstream_alignment_diagnostic(
        label="baseline",
        checkpoint=args.baseline_checkpoint,
        run_dir=run_dir,
        device=device,
        args=args,
    )
    summaries.append(baseline_summary)
    print(
        "baseline "
        f"forced_exact={baseline_summary['forced_true_exact_accuracy']:.4f} "
        f"best_true={baseline_summary['boundary_result_boundary_target_hard_best_equals_true_sum']:.4f} "
        f"exact_cos={baseline_summary['exact_vs_boundary_result_proj_cosine']:.4f}/"
        f"{baseline_summary['exact_vs_boundary_upstream_cosine']:.4f}"
    )

    decoder_summaries: list[dict[str, Any]] = []
    for branch in args.branches:
        checkpoint, decoder_summary = train_decoder_candidate(
            branch=branch,
            baseline_checkpoint=args.baseline_checkpoint,
            run_dir=run_dir,
            device=device,
            args=args,
        )
        decoder_summaries.append(decoder_summary)
        summary = downstream_alignment_diagnostic(
            label=branch,
            checkpoint=checkpoint,
            run_dir=run_dir,
            device=device,
            args=args,
        )
        summaries.append(summary)
        print(
            f"{branch} "
            f"forced_exact={summary['forced_true_exact_accuracy']:.4f} "
            f"best_true={summary['boundary_result_boundary_target_hard_best_equals_true_sum']:.4f} "
            f"exact_cos={summary['exact_vs_boundary_result_proj_cosine']:.4f}/"
            f"{summary['exact_vs_boundary_upstream_cosine']:.4f} "
            f"pg_exact={summary['pg_vs_exact_result_proj_cosine']:.4f}/"
            f"{summary['pg_vs_exact_upstream_cosine']:.4f}"
        )

    passing = [summary for summary in summaries if stage0_passed(summary)]
    decision = (
        "gradient_friendly_decoder_alignment_pass"
        if passing
        else "gradient_friendly_decoder_alignment_negative"
    )
    result = {
        "decision": decision,
        "run_dir": str(run_dir),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "decoder_summaries": decoder_summaries,
        "alignment_summaries": summaries,
        "passing_labels": [summary["label"] for summary in passing],
    }
    write_json(run_dir / "stage0_gradient_friendly_decoder_gate_summary.json", result)
    print(f"decision={decision} run_dir={run_dir}")


if __name__ == "__main__":
    main()
