#!/usr/bin/env python3
"""Diagnose sparse amortized prediction of result-boundary targets.

This is a static gate for the Phase 7 result-boundary direction: given a model
checkpoint, score all forced result classes once for evaluation, then ask
whether a shared critic trained on only a few forced-result scores per training
prompt can recover the full-enumeration best result on heldout prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch

from overfit_one_batch import (
    TinyGPT,
    calculator_read_result_logits_and_input,
    fixed_width_operands_from_batch,
    load_semantic_decoder_checkpoint,
    make_exhaustive_range_batch,
    make_model_config,
    score_forced_result_classes_chunked,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparse amortized critic diagnostic for result-boundary targets."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-label", action="append", default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--heldout-prompts", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--uncertainty-candidates", type=int, default=8)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument(
        "--critic-loss-mode",
        choices=["pointwise", "pairwise", "hybrid"],
        default="pointwise",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_model(config: dict[str, Any], *, device: str) -> TinyGPT:
    model_cfg = make_model_config(
        int(config["num_digits"]),
        str(config["variant"]),
        injection_scale=1.0,
        operand_vocab_size=int(config["calculator_operand_vocab_size"]),
        calculator_estimator=str(config["calculator_estimator"]),
        calculator_action_head=str(config["calculator_action_head"]),
        calculator_read_position=str(config["calculator_read_position"]),
        calculator_read_span_width=int(config["calculator_read_span_width"]),
        calculator_injection_mode=str(config["calculator_injection_mode"]),
        calculator_bottleneck_mode=str(config["calculator_bottleneck_mode"]),
        calculator_output_format=str(config["calculator_output_format"]),
        answer_decoder_interaction=str(config["answer_decoder_interaction"]),
        calculator_result_head_hidden_size=int(
            config.get("calculator_result_head_hidden_size", 0)
        ),
        relaxed_calculator_temperature=float(
            config.get("relaxed_calculator_temperature", 1.0)
        ),
        relaxed_calculator_mode=str(config.get("relaxed_calculator_mode", "deterministic")),
        relaxed_calculator_hard_forward=bool(
            config.get("relaxed_calculator_hard_forward", True)
        ),
        answer_format=str(config.get("answer_format", "sum")),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        mlp_expansion=int(config["mlp_expansion"]),
        calculator_hook_after_layer=int(config["calculator_hook_after_layer"]),
    )
    return TinyGPT(model_cfg).to(device)


@torch.no_grad()
def candidate_features(model: TinyGPT, batch, *, device: str) -> torch.Tensor:
    _logits, result_input, _positions = calculator_read_result_logits_and_input(
        model, batch
    )
    if model.calculator_hook is None:
        raise ValueError("calculator hook is required")
    output_weight = model.calculator_hook.output_proj.weight.detach().T
    result_count = output_weight.shape[0]
    prompt_features = torch.nn.functional.normalize(result_input.detach(), dim=-1)
    result_features = torch.nn.functional.normalize(output_weight, dim=-1)
    prompt_expanded = prompt_features.unsqueeze(1).expand(-1, result_count, -1)
    result_expanded = result_features.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    prompt_norm = result_input.detach().norm(dim=-1, keepdim=True)
    result_norm = output_weight.norm(dim=-1, keepdim=True)
    prompt_norm = prompt_norm.unsqueeze(1).expand(-1, result_count, -1)
    result_norm = result_norm.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    result_id = torch.linspace(-1.0, 1.0, result_count, device=device)
    result_id = result_id.view(1, result_count, 1).expand(batch.x.shape[0], -1, -1)
    return torch.cat(
        [prompt_expanded, result_expanded, prompt_norm, result_norm, result_id],
        dim=-1,
    )


def sample_training_pairs(
    train_indices: torch.Tensor,
    *,
    result_count: int,
    samples_per_prompt: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_indices = train_indices.repeat_interleave(samples_per_prompt)
    result_indices = torch.randint(
        low=0,
        high=result_count,
        size=(int(prompt_indices.shape[0]),),
        generator=generator,
        device=device,
    )
    return prompt_indices, result_indices


def train_critic(
    train_features: torch.Tensor,
    train_losses: torch.Tensor,
    *,
    prompts: int,
    samples_per_prompt: int,
    hidden_size: int,
    epochs: int,
    critic_loss_mode: str,
    lr: float,
    weight_decay: float,
) -> tuple[torch.nn.Module, dict[str, float]]:
    if train_features.shape[0] != prompts * samples_per_prompt:
        raise ValueError("train feature count must match prompts * samples_per_prompt")
    if critic_loss_mode not in {"pointwise", "pairwise", "hybrid"}:
        raise ValueError("unknown critic loss mode")
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    target_mean = train_losses.mean()
    target_std = train_losses.std().clamp_min(1.0e-6)
    normalized_features = (train_features - feature_mean) / feature_std
    normalized_targets = ((train_losses - target_mean) / target_std).unsqueeze(-1)

    critic = torch.nn.Sequential(
        torch.nn.Linear(normalized_features.shape[-1], hidden_size),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_size, 1),
    ).to(train_features.device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    final_loss = float("nan")
    critic.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = critic(normalized_features)
        pointwise_loss = torch.nn.functional.smooth_l1_loss(
            prediction, normalized_targets
        )
        grouped_prediction = prediction.reshape(prompts, samples_per_prompt)
        grouped_targets = normalized_targets.reshape(prompts, samples_per_prompt)
        prediction_diff = grouped_prediction.unsqueeze(2) - grouped_prediction.unsqueeze(1)
        target_diff = grouped_targets.unsqueeze(2) - grouped_targets.unsqueeze(1)
        pair_mask = target_diff.abs() > 1.0e-6
        pair_targets = (target_diff > 0).to(prediction_diff.dtype)
        if bool(pair_mask.any().item()):
            pairwise_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction_diff[pair_mask],
                pair_targets[pair_mask],
            )
        else:
            pairwise_loss = pointwise_loss.new_tensor(0.0)
        if critic_loss_mode == "pointwise":
            loss = pointwise_loss
        elif critic_loss_mode == "pairwise":
            loss = pairwise_loss
        else:
            loss = pointwise_loss + pairwise_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().item())

    critic.eval()
    critic.feature_mean = feature_mean  # type: ignore[attr-defined]
    critic.feature_std = feature_std  # type: ignore[attr-defined]
    critic.target_mean = target_mean  # type: ignore[attr-defined]
    critic.target_std = target_std  # type: ignore[attr-defined]
    return critic, {"critic_train_loss": final_loss}


@torch.no_grad()
def predict_losses(critic: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1])
    normalized = (flat - critic.feature_mean) / critic.feature_std  # type: ignore[attr-defined]
    prediction = critic(normalized).reshape(features.shape[:-1])
    return prediction * critic.target_std + critic.target_mean  # type: ignore[attr-defined]


def evaluate_predictions(
    predicted_losses: torch.Tensor,
    full_losses: torch.Tensor,
    heldout_indices: torch.Tensor,
    true_sum: torch.Tensor,
) -> dict[str, float]:
    heldout_pred = predicted_losses[heldout_indices]
    heldout_full = full_losses[heldout_indices]
    heldout_true_sum = true_sum[heldout_indices]
    full_best = heldout_full.argmin(dim=-1)
    pred_best = heldout_pred.argmin(dim=-1)
    pred_top3 = heldout_pred.topk(k=min(3, heldout_pred.shape[-1]), largest=False).indices
    pred_top5 = heldout_pred.topk(k=min(5, heldout_pred.shape[-1]), largest=False).indices
    best_losses = heldout_full.gather(1, full_best.unsqueeze(-1)).squeeze(-1)
    pred_losses = heldout_full.gather(1, pred_best.unsqueeze(-1)).squeeze(-1)
    true_losses = heldout_full.gather(1, heldout_true_sum.unsqueeze(-1)).squeeze(-1)
    return {
        "heldout_full_best_equals_true_sum": float(
            (full_best == heldout_true_sum).float().mean().item()
        ),
        "heldout_pred_best_equals_full_best": float(
            (pred_best == full_best).float().mean().item()
        ),
        "heldout_pred_best_equals_true_sum": float(
            (pred_best == heldout_true_sum).float().mean().item()
        ),
        "heldout_pred_top3_contains_full_best": float(
            (pred_top3 == full_best.unsqueeze(-1)).any(dim=-1).float().mean().item()
        ),
        "heldout_pred_top5_contains_full_best": float(
            (pred_top5 == full_best.unsqueeze(-1)).any(dim=-1).float().mean().item()
        ),
        "heldout_mean_regret": float((pred_losses - best_losses).mean().item()),
        "heldout_median_regret": float((pred_losses - best_losses).median().item()),
        "heldout_true_minus_best_gap": float((true_losses - best_losses).mean().item()),
    }


def evaluate_candidate_proposals(
    mean_losses: torch.Tensor,
    std_losses: torch.Tensor,
    full_losses: torch.Tensor,
    heldout_indices: torch.Tensor,
    true_sum: torch.Tensor,
    *,
    candidate_count: int,
    uncertainty_beta: float,
) -> dict[str, float]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if uncertainty_beta < 0.0:
        raise ValueError("uncertainty_beta must be non-negative")
    heldout_mean = mean_losses[heldout_indices]
    heldout_std = std_losses[heldout_indices]
    heldout_full = full_losses[heldout_indices]
    heldout_true_sum = true_sum[heldout_indices]
    full_best = heldout_full.argmin(dim=-1)
    best_losses = heldout_full.gather(1, full_best.unsqueeze(-1)).squeeze(-1)

    k = min(candidate_count, heldout_mean.shape[-1])
    mean_candidates = heldout_mean.topk(k=k, largest=False).indices
    lcb = heldout_mean - uncertainty_beta * heldout_std
    lcb_candidates = lcb.topk(k=k, largest=False).indices

    def score_selected(prefix: str, candidates: torch.Tensor) -> dict[str, float]:
        selected_full_losses = heldout_full.gather(1, candidates)
        selected_best_position = selected_full_losses.argmin(dim=-1)
        selected_best = candidates.gather(
            1,
            selected_best_position.unsqueeze(-1),
        ).squeeze(-1)
        selected_best_losses = heldout_full.gather(
            1,
            selected_best.unsqueeze(-1),
        ).squeeze(-1)
        return {
            f"{prefix}_topk_contains_full_best": float(
                (candidates == full_best.unsqueeze(-1)).any(dim=-1).float().mean().item()
            ),
            f"{prefix}_scored_best_equals_full_best": float(
                (selected_best == full_best).float().mean().item()
            ),
            f"{prefix}_scored_best_equals_true_sum": float(
                (selected_best == heldout_true_sum).float().mean().item()
            ),
            f"{prefix}_mean_regret": float(
                (selected_best_losses - best_losses).mean().item()
            ),
            f"{prefix}_median_regret": float(
                (selected_best_losses - best_losses).median().item()
            ),
        }

    return {
        **score_selected("heldout_mean_proposal", mean_candidates),
        **score_selected("heldout_lcb_proposal", lcb_candidates),
    }


def diagnose_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    model = build_model(config, device=args.device)
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope="full_model")
    model.eval()
    batch = make_exhaustive_range_batch(
        num_digits=int(config["num_digits"]),
        operand_max=int(config["operand_max"]),
        fixed_width=bool(config.get("fixed_width", True)),
        device=args.device,
        answer_format=str(config.get("answer_format", "sum")),
    )
    full_losses = score_forced_result_classes_chunked(
        model,
        batch,
        chunk_size=args.chunk_size,
    )
    features = candidate_features(model, batch, device=args.device)
    true_a, true_b = fixed_width_operands_from_batch(
        batch.x,
        num_digits=int(config["num_digits"]),
    )
    true_sum = true_a + true_b

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    prompt_count = int(batch.x.shape[0])
    permutation = torch.randperm(prompt_count, generator=generator, device=args.device)
    heldout_count = min(args.heldout_prompts, prompt_count - 1)
    heldout_indices = permutation[:heldout_count]
    train_indices = permutation[heldout_count:]
    result_count = int(full_losses.shape[-1])

    ensemble_predictions = []
    train_losses_used = 0
    train_loss_values = []
    for member_idx in range(int(args.ensemble_size)):
        torch.manual_seed(int(args.seed) + member_idx)
        prompt_indices, result_indices = sample_training_pairs(
            train_indices,
            result_count=result_count,
            samples_per_prompt=args.samples_per_prompt,
            generator=generator,
            device=args.device,
        )
        train_features = features[prompt_indices, result_indices]
        train_losses = full_losses[prompt_indices, result_indices]
        critic, member_train_metrics = train_critic(
            train_features,
            train_losses,
            prompts=int(train_indices.shape[0]),
            samples_per_prompt=args.samples_per_prompt,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            critic_loss_mode=args.critic_loss_mode,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        ensemble_predictions.append(predict_losses(critic, features))
        train_losses_used += int(train_losses.numel())
        train_loss_values.append(float(member_train_metrics["critic_train_loss"]))

    stacked_predictions = torch.stack(ensemble_predictions, dim=0)
    predicted_losses = stacked_predictions.mean(dim=0)
    prediction_std = stacked_predictions.std(dim=0, unbiased=False)
    train_metrics = {
        "critic_train_loss": float(
            torch.tensor(train_loss_values, device=args.device).mean().item()
        ),
        "critic_train_loss_mean": float(
            torch.tensor(train_loss_values, device=args.device).mean().item()
        ),
        "critic_train_loss_min": float(min(train_loss_values)),
        "critic_train_loss_max": float(max(train_loss_values)),
    }
    metrics = evaluate_predictions(
        predicted_losses,
        full_losses,
        heldout_indices,
        true_sum,
    )
    proposal_metrics = evaluate_candidate_proposals(
        predicted_losses,
        prediction_std,
        full_losses,
        heldout_indices,
        true_sum,
        candidate_count=int(args.uncertainty_candidates),
        uncertainty_beta=float(args.uncertainty_beta),
    )
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "prompt_count": prompt_count,
        "train_prompts": int(train_indices.shape[0]),
        "heldout_prompts": int(heldout_indices.shape[0]),
        "result_count": result_count,
        "samples_per_prompt": int(args.samples_per_prompt),
        "ensemble_size": int(args.ensemble_size),
        "uncertainty_candidates": int(args.uncertainty_candidates),
        "uncertainty_beta": float(args.uncertainty_beta),
        "critic_loss_mode": str(args.critic_loss_mode),
        "forced_scores_used": train_losses_used,
        "heldout_proposal_scores_if_used": int(
            heldout_indices.shape[0] * min(args.uncertainty_candidates, result_count)
        ),
        "proposal_score_fraction_of_full_heldout_enum": float(
            min(args.uncertainty_candidates, result_count) / result_count
        ),
        "full_enum_scores_for_eval": int(full_losses.numel()),
        **train_metrics,
        **metrics,
        **proposal_metrics,
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be positive")
    if args.uncertainty_candidates < 1:
        raise ValueError("--uncertainty-candidates must be positive")
    if args.uncertainty_beta < 0.0:
        raise ValueError("--uncertainty-beta must be non-negative")
    config = load_config(args.config)
    labels = args.checkpoint_label
    if labels is None:
        labels = [path.stem for path in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise ValueError("--checkpoint-label count must match --checkpoint count")
    rows = [
        diagnose_checkpoint(checkpoint, label=label, config=config, args=args)
        for checkpoint, label in zip(args.checkpoint, labels, strict=True)
    ]
    output = {"rows": rows}
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
