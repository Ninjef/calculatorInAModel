#!/usr/bin/env python3
"""Test whether result-boundary proposal critics survive checkpoint drift."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from diagnose_result_boundary_amortized_critic import (
    build_model,
    candidate_features,
    evaluate_candidate_proposals,
    evaluate_predictions,
    load_config,
    sample_training_pairs,
    train_critic,
)
from overfit_one_batch import (
    fixed_width_operands_from_batch,
    load_semantic_decoder_checkpoint,
    make_exhaustive_range_batch,
    score_forced_result_classes_chunked,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a sparse result-boundary critic on one checkpoint and evaluate "
            "candidate proposals on one or more other checkpoints."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-checkpoint", type=Path, required=True)
    parser.add_argument("--train-label", default="train")
    parser.add_argument("--eval-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--eval-label", action="append", default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--heldout-prompts", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--proposal-candidates", type=int, default=8)
    parser.add_argument("--adapt-samples-per-prompt", type=int, default=0)
    parser.add_argument("--adapt-epochs", type=int, default=100)
    parser.add_argument("--adapt-lr", type=float, default=3.0e-4)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument(
        "--critic-loss-mode",
        choices=["pointwise", "pairwise", "hybrid"],
        default="pairwise",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def checkpoint_tensors(
    checkpoint: Path,
    *,
    config: dict[str, Any],
    device: str,
    chunk_size: int,
) -> dict[str, torch.Tensor]:
    model = build_model(config, device=device)
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope="full_model")
    model.eval()
    batch = make_exhaustive_range_batch(
        num_digits=int(config["num_digits"]),
        operand_max=int(config["operand_max"]),
        fixed_width=bool(config.get("fixed_width", True)),
        device=device,
        answer_format=str(config.get("answer_format", "sum")),
    )
    full_losses = score_forced_result_classes_chunked(
        model,
        batch,
        chunk_size=chunk_size,
    )
    features = candidate_features(model, batch, device=device)
    true_a, true_b = fixed_width_operands_from_batch(
        batch.x,
        num_digits=int(config["num_digits"]),
    )
    return {
        "features": features,
        "full_losses": full_losses,
        "true_sum": true_a + true_b,
    }


def feature_drift_metrics(
    critic: torch.nn.Module,
    features: torch.Tensor,
    heldout_indices: torch.Tensor,
) -> dict[str, float]:
    heldout = features[heldout_indices].reshape(-1, features.shape[-1])
    normalized = (heldout - critic.feature_mean) / critic.feature_std  # type: ignore[attr-defined]
    return {
        "feature_standardized_abs_mean": float(normalized.abs().mean().item()),
        "feature_standardized_abs_p95": float(
            normalized.abs().quantile(0.95).item()
        ),
    }


def retarget_critic_normalization(
    critic: torch.nn.Module,
    *,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> None:
    """Change feature/target normalization while preserving raw predictions."""
    if not isinstance(critic, torch.nn.Sequential):
        raise TypeError("normalization retargeting expects a Sequential critic")
    first = critic[0]
    last = critic[-1]
    if not isinstance(first, torch.nn.Linear) or not isinstance(last, torch.nn.Linear):
        raise TypeError("normalization retargeting expects Linear endpoints")

    old_feature_mean = critic.feature_mean.to(  # type: ignore[attr-defined]
        feature_mean.device
    )
    old_feature_std = critic.feature_std.to(  # type: ignore[attr-defined]
        feature_std.device
    )
    old_target_mean = critic.target_mean.to(  # type: ignore[attr-defined]
        target_mean.device
    )
    old_target_std = critic.target_std.to(  # type: ignore[attr-defined]
        target_std.device
    )

    feature_scale = (feature_std / old_feature_std).squeeze(0)
    feature_offset = ((feature_mean - old_feature_mean) / old_feature_std).squeeze(0)
    with torch.no_grad():
        old_first_weight = first.weight.detach().clone()
        first.weight.copy_(old_first_weight * feature_scale.unsqueeze(0))
        first.bias.add_(old_first_weight @ feature_offset)

        target_scale = old_target_std / target_std
        last.weight.mul_(target_scale)
        last.bias.copy_(
            (last.bias * old_target_std + old_target_mean - target_mean) / target_std
        )

    critic.feature_mean = feature_mean  # type: ignore[attr-defined]
    critic.feature_std = feature_std  # type: ignore[attr-defined]
    critic.target_mean = target_mean  # type: ignore[attr-defined]
    critic.target_std = target_std  # type: ignore[attr-defined]


def fit_existing_critic(
    critic: torch.nn.Module,
    train_features: torch.Tensor,
    train_losses: torch.Tensor,
    *,
    prompts: int,
    samples_per_prompt: int,
    epochs: int,
    critic_loss_mode: str,
    lr: float,
    weight_decay: float,
) -> dict[str, float]:
    if train_features.shape[0] != prompts * samples_per_prompt:
        raise ValueError("train feature count must match prompts * samples_per_prompt")
    feature_mean = train_features.mean(dim=0, keepdim=True)
    feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1.0e-6)
    target_mean = train_losses.mean()
    target_std = train_losses.std().clamp_min(1.0e-6)
    retarget_critic_normalization(
        critic,
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    normalized_features = (train_features - feature_mean) / feature_std
    normalized_targets = ((train_losses - target_mean) / target_std).unsqueeze(-1)
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
        prediction_diff = grouped_prediction.unsqueeze(2) - grouped_prediction.unsqueeze(
            1
        )
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
    return {"adapt_critic_loss": final_loss}


@torch.no_grad()
def predict_losses(critic: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    flat = features.reshape(-1, features.shape[-1])
    normalized = (flat - critic.feature_mean) / critic.feature_std  # type: ignore[attr-defined]
    prediction = critic(normalized).reshape(features.shape[:-1])
    return prediction * critic.target_std + critic.target_mean  # type: ignore[attr-defined]


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be positive")
    if args.heldout_prompts < 1:
        raise ValueError("--heldout-prompts must be positive")
    if args.ensemble_size < 1:
        raise ValueError("--ensemble-size must be positive")
    if args.proposal_candidates < 1:
        raise ValueError("--proposal-candidates must be positive")
    if args.adapt_samples_per_prompt < 0:
        raise ValueError("--adapt-samples-per-prompt must be non-negative")
    if args.adapt_samples_per_prompt > 0 and args.adapt_epochs < 1:
        raise ValueError("--adapt-epochs must be positive when adaptation is enabled")
    if args.adapt_lr <= 0:
        raise ValueError("--adapt-lr must be positive")
    if args.uncertainty_beta < 0.0:
        raise ValueError("--uncertainty-beta must be non-negative")

    eval_labels = args.eval_label
    if eval_labels is None:
        eval_labels = [path.stem for path in args.eval_checkpoint]
    if len(eval_labels) != len(args.eval_checkpoint):
        raise ValueError("--eval-label count must match --eval-checkpoint count")

    config = load_config(args.config)
    train_data = checkpoint_tensors(
        args.train_checkpoint,
        config=config,
        device=args.device,
        chunk_size=int(args.chunk_size),
    )
    prompt_count = int(train_data["full_losses"].shape[0])
    result_count = int(train_data["full_losses"].shape[-1])
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(args.seed))
    permutation = torch.randperm(prompt_count, generator=generator, device=args.device)
    heldout_count = min(int(args.heldout_prompts), prompt_count - 1)
    heldout_indices = permutation[:heldout_count]
    train_indices = permutation[heldout_count:]

    critics = []
    train_losses_used = 0
    train_loss_values = []
    for member_idx in range(int(args.ensemble_size)):
        torch.manual_seed(int(args.seed) + member_idx)
        prompt_indices, result_indices = sample_training_pairs(
            train_indices,
            result_count=result_count,
            samples_per_prompt=int(args.samples_per_prompt),
            generator=generator,
            device=args.device,
        )
        train_features = train_data["features"][prompt_indices, result_indices]
        train_losses = train_data["full_losses"][prompt_indices, result_indices]
        critic, train_metrics = train_critic(
            train_features,
            train_losses,
            prompts=int(train_indices.shape[0]),
            samples_per_prompt=int(args.samples_per_prompt),
            hidden_size=int(args.hidden_size),
            epochs=int(args.epochs),
            critic_loss_mode=str(args.critic_loss_mode),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        critics.append(critic)
        train_losses_used += int(train_losses.numel())
        train_loss_values.append(float(train_metrics["critic_train_loss"]))

    rows: list[dict[str, float | int | str]] = []
    eval_items = [
        (args.train_checkpoint, args.train_label),
        *zip(args.eval_checkpoint, eval_labels),
    ]
    for checkpoint, label in eval_items:
        if Path(checkpoint) == args.train_checkpoint:
            eval_data = train_data
        else:
            eval_data = checkpoint_tensors(
                Path(checkpoint),
                config=config,
                device=args.device,
                chunk_size=int(args.chunk_size),
            )
        def append_row(
            *,
            mode: str,
            active_critics: list[torch.nn.Module],
            adaptation_scores_used: int,
            adaptation_loss: float | None,
        ) -> None:
            predictions = [
                predict_losses(critic, eval_data["features"])
                for critic in active_critics
            ]
            stacked_predictions = torch.stack(predictions, dim=0)
            predicted_losses = stacked_predictions.mean(dim=0)
            prediction_std = stacked_predictions.std(dim=0, unbiased=False)
            prediction_metrics = evaluate_predictions(
                predicted_losses,
                eval_data["full_losses"],
                heldout_indices,
                eval_data["true_sum"],
            )
            proposal_metrics = evaluate_candidate_proposals(
                predicted_losses,
                prediction_std,
                eval_data["full_losses"],
                heldout_indices,
                eval_data["true_sum"],
                candidate_count=int(args.proposal_candidates),
                uncertainty_beta=float(args.uncertainty_beta),
            )
            drift_metrics = feature_drift_metrics(
                active_critics[0],
                eval_data["features"],
                heldout_indices,
            )
            rows.append(
                {
                    "proposal_mode": mode,
                    "train_label": str(args.train_label),
                    "train_checkpoint": str(args.train_checkpoint),
                    "eval_label": str(label),
                    "eval_checkpoint": str(checkpoint),
                    "prompt_count": prompt_count,
                    "train_prompts": int(train_indices.shape[0]),
                    "heldout_prompts": int(heldout_indices.shape[0]),
                    "result_count": result_count,
                    "samples_per_prompt": int(args.samples_per_prompt),
                    "adapt_samples_per_prompt": int(args.adapt_samples_per_prompt),
                    "adapt_epochs": int(args.adapt_epochs),
                    "adaptation_scores_used": adaptation_scores_used,
                    "adapt_critic_loss": adaptation_loss,
                    "ensemble_size": int(args.ensemble_size),
                    "proposal_candidates": int(args.proposal_candidates),
                    "uncertainty_beta": float(args.uncertainty_beta),
                    "critic_loss_mode": str(args.critic_loss_mode),
                    "forced_scores_used": train_losses_used,
                    "critic_train_loss": float(
                        torch.tensor(train_loss_values, device=args.device).mean().item()
                    ),
                    **prediction_metrics,
                    **proposal_metrics,
                    **drift_metrics,
                }
            )

        append_row(
            mode="frozen",
            active_critics=critics,
            adaptation_scores_used=0,
            adaptation_loss=None,
        )
        if int(args.adapt_samples_per_prompt) <= 0:
            continue

        adapted_critics = [copy.deepcopy(critic) for critic in critics]
        adaptation_losses = []
        adaptation_scores_used = 0
        for member_idx, critic in enumerate(adapted_critics):
            adapt_generator = torch.Generator(device=args.device)
            adapt_generator.manual_seed(int(args.seed) + 10_000 + member_idx)
            prompt_indices, result_indices = sample_training_pairs(
                train_indices,
                result_count=result_count,
                samples_per_prompt=int(args.adapt_samples_per_prompt),
                generator=adapt_generator,
                device=args.device,
            )
            train_features = eval_data["features"][prompt_indices, result_indices]
            train_losses = eval_data["full_losses"][prompt_indices, result_indices]
            metrics = fit_existing_critic(
                critic,
                train_features,
                train_losses,
                prompts=int(train_indices.shape[0]),
                samples_per_prompt=int(args.adapt_samples_per_prompt),
                epochs=int(args.adapt_epochs),
                critic_loss_mode=str(args.critic_loss_mode),
                lr=float(args.adapt_lr),
                weight_decay=float(args.weight_decay),
            )
            adaptation_losses.append(float(metrics["adapt_critic_loss"]))
            adaptation_scores_used += int(train_losses.numel())

        append_row(
            mode="adapted",
            active_critics=adapted_critics,
            adaptation_scores_used=adaptation_scores_used,
            adaptation_loss=float(
                torch.tensor(adaptation_losses, device=args.device).mean().item()
            ),
        )

    output = {"rows": rows}
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
