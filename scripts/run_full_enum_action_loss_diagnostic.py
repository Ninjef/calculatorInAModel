import argparse
import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.diagnose_calculator_protocol import (  # noqa: E402
    decode_tokens,
    load_checkpoint,
    pick_device,
)
from scripts.overfit_one_batch import (  # noqa: E402
    action_loss_soft_targets,
    action_loss_weights_from_losses,
    calculator_read_operand_logits,
    calculator_read_pair_logits,
    fixed_width_operands_from_batch,
    full_enum_action_pairs,
    make_range_batch,
    score_action_loss_candidates_chunked,
)
from src.data import ANSWER_FORMATS, AnswerFormat, EQ_ID  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_prompt(row: torch.Tensor) -> str:
    ids = row.detach().cpu().tolist()
    if EQ_ID in ids:
        return decode_tokens(ids[: ids.index(EQ_ID) + 1])
    return decode_tokens(ids)


@torch.no_grad()
def full_enum_diagnostic(
    *,
    checkpoint: Path,
    samples: int,
    batch_size: int,
    digits: int,
    operand_max: int,
    temperature: float,
    min_probability_floor: float,
    near_best_tolerance: float,
    chunk_size: int,
    seed: int,
    device: str,
    answer_format: AnswerFormat,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, train_config = load_checkpoint(checkpoint, device=device, injection_scale=None)
    model.eval()
    rng = random.Random(seed + 92_000)
    rows: list[dict[str, Any]] = []
    seen = 0
    while seen < samples:
        current_batch = min(batch_size, samples - seen)
        batch = make_range_batch(
            batch_size=current_batch,
            num_digits=digits,
            operand_max=operand_max,
            rng=rng,
            fixed_width=True,
            device=device,
            answer_format=answer_format,
        )
        if model.cfg.calculator_action_head == "joint_pair":
            pair_logits, _, _, _ = calculator_read_pair_logits(model, batch)
            classes = model.cfg.calculator_operand_vocab_size
            a_logits = torch.logsumexp(
                pair_logits.log_softmax(dim=-1).reshape(
                    current_batch, classes, classes
                ),
                dim=-1,
            )
            b_logits = torch.logsumexp(
                pair_logits.log_softmax(dim=-1).reshape(
                    current_batch, classes, classes
                ),
                dim=-2,
            )
            learned_idx_from_head = pair_logits.argmax(dim=-1)
        else:
            a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
            classes = a_logits.shape[-1]
            pair_logits = None
            learned_idx_from_head = None
        pairs = full_enum_action_pairs(classes=classes, device=device)
        candidates = pairs.unsqueeze(0).expand(current_batch, -1, -1)
        losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
        weights = action_loss_weights_from_losses(
            losses,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
        )
        target_a, target_b = action_loss_soft_targets(
            a_logits, b_logits, candidates, weights
        )
        true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
        learned_a = a_logits.argmax(dim=-1)
        learned_b = b_logits.argmax(dim=-1)
        learned_idx = (
            learned_idx_from_head
            if learned_idx_from_head is not None
            else learned_a * classes + learned_b
        )
        learned_a = learned_idx // classes
        learned_b = learned_idx % classes
        true_idx = true_a * classes + true_b
        best_idx = losses.argmin(dim=-1)
        batch_idx = torch.arange(current_batch, device=device)
        learned_losses = losses.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
        true_losses = losses.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
        best_losses = losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        best_pairs = pairs.index_select(0, best_idx)
        pair_sums = pairs[:, 0] + pairs[:, 1]
        result_count = int(2 * classes - 1)
        result_losses = losses.new_full((current_batch, result_count), float("inf"))
        for result_idx in range(result_count):
            mask = pair_sums == result_idx
            if bool(mask.any()):
                result_losses[:, result_idx] = losses[:, mask].min(dim=1).values
        true_sums = true_a + true_b
        learned_sums = learned_a + learned_b
        best_result_idx = result_losses.argmin(dim=-1)
        best_result_losses = result_losses.gather(
            1, best_result_idx.unsqueeze(-1)
        ).squeeze(-1)
        learned_result_losses = result_losses.gather(
            1, learned_sums.unsqueeze(-1)
        ).squeeze(-1)
        true_result_losses = result_losses.gather(
            1, true_sums.unsqueeze(-1)
        ).squeeze(-1)
        result_weights = action_loss_weights_from_losses(
            result_losses,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
        )
        result_entropy = -(
            result_weights * result_weights.clamp_min(1e-12).log()
        ).sum(dim=-1)
        same_true_sum_near_best = (
            (pair_sums.unsqueeze(0) == true_sums.unsqueeze(-1))
            & (losses <= best_losses.unsqueeze(-1) + near_best_tolerance)
        ).sum(dim=-1)
        same_best_sum_near_best = (
            (pair_sums.unsqueeze(0) == best_result_idx.unsqueeze(-1))
            & (losses <= best_losses.unsqueeze(-1) + near_best_tolerance)
        ).sum(dim=-1)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
        true_pair_probs = weights.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
        true_pair_ranks = (losses < true_losses.unsqueeze(-1)).sum(dim=-1) + 1
        best_tie_tolerance = 1e-6
        true_within_best_tie = true_losses <= best_losses + best_tie_tolerance
        sorted_weights = weights.sort(dim=-1, descending=True).values
        top1_mass = sorted_weights[:, :1].sum(dim=-1)
        top3_mass = sorted_weights[:, :3].sum(dim=-1)
        top5_mass = sorted_weights[:, :5].sum(dim=-1)
        learned_tie_1e3 = learned_losses <= best_losses + 1e-3
        learned_tie_1e2 = learned_losses <= best_losses + 1e-2
        if pair_logits is not None:
            pair_probs = pair_logits.softmax(dim=-1)
            pair_logit_entropy = -(
                pair_probs * pair_probs.clamp_min(1e-12).log()
            ).sum(dim=-1)
            learned_pair_probability = pair_probs.gather(
                1, learned_idx.unsqueeze(-1)
            ).squeeze(-1)
            true_pair_head_probability = pair_probs.gather(
                1, true_idx.unsqueeze(-1)
            ).squeeze(-1)
        else:
            pair_logit_entropy = torch.full_like(entropy, float("nan"))
            learned_pair_probability = torch.full_like(entropy, float("nan"))
            true_pair_head_probability = torch.full_like(entropy, float("nan"))
        for i in range(current_batch):
            best_a = int(best_pairs[i, 0].item())
            best_b = int(best_pairs[i, 1].item())
            rows.append(
                {
                    "sample": seen + i,
                    "prompt": format_prompt(batch.x[i]),
                    "true_a": int(true_a[i].item()),
                    "true_b": int(true_b[i].item()),
                    "true_sum": int((true_a[i] + true_b[i]).item()),
                    "learned_a": int(learned_a[i].item()),
                    "learned_b": int(learned_b[i].item()),
                    "learned_sum": int((learned_a[i] + learned_b[i]).item()),
                    "best_a": best_a,
                    "best_b": best_b,
                    "best_sum": best_a + best_b,
                    "best_result": int(best_result_idx[i].item()),
                    "best_full_enum_mean_nll": float(best_losses[i].item()),
                    "best_result_mean_nll": float(best_result_losses[i].item()),
                    "learned_mean_nll": float(learned_losses[i].item()),
                    "learned_result_mean_nll": float(
                        learned_result_losses[i].item()
                    ),
                    "true_mean_nll": float(true_losses[i].item()),
                    "true_result_mean_nll": float(true_result_losses[i].item()),
                    "learned_minus_true_gap": float(
                        (learned_losses[i] - true_losses[i]).item()
                    ),
                    "learned_minus_best_gap": float(
                        (learned_losses[i] - best_losses[i]).item()
                    ),
                    "learned_result_minus_best_result_gap": float(
                        (learned_result_losses[i] - best_result_losses[i]).item()
                    ),
                    "true_result_minus_best_result_gap": float(
                        (true_result_losses[i] - best_result_losses[i]).item()
                    ),
                    "true_is_best": bool(best_idx[i].item() == true_idx[i].item()),
                    "true_result_is_best_result": bool(
                        best_result_idx[i].item() == true_sums[i].item()
                    ),
                    "learned_result_is_best_result": bool(
                        best_result_idx[i].item() == learned_sums[i].item()
                    ),
                    "true_within_best_tie": bool(true_within_best_tie[i].item()),
                    "learned_is_best": bool(
                        best_idx[i].item() == learned_idx[i].item()
                    ),
                    "learned_within_1e-3_best": bool(learned_tie_1e3[i].item()),
                    "learned_within_1e-2_best": bool(learned_tie_1e2[i].item()),
                    "best_matches_true_operands": bool(
                        best_a == true_a[i].item() and best_b == true_b[i].item()
                    ),
                    "best_result_matches_true_sum": bool(
                        best_a + best_b == (true_a[i] + true_b[i]).item()
                    ),
                    "best_result_group_matches_true_sum": bool(
                        best_result_idx[i].item() == true_sums[i].item()
                    ),
                    "best_left_operand_matches_true": bool(
                        best_a == true_a[i].item()
                    ),
                    "learned_result_matches_true_sum": bool(
                        (learned_a[i] + learned_b[i]).item()
                        == (true_a[i] + true_b[i]).item()
                    ),
                    "soft_target_true_a_mass": float(
                        target_a[i, true_a[i]].item()
                    ),
                    "soft_target_true_b_mass": float(
                        target_b[i, true_b[i]].item()
                    ),
                    "soft_target_true_pair_probability": float(
                        true_pair_probs[i].item()
                    ),
                    "true_pair_rank": int(true_pair_ranks[i].item()),
                    "top1_target_mass": float(top1_mass[i].item()),
                    "top3_target_mass": float(top3_mass[i].item()),
                    "top5_target_mass": float(top5_mass[i].item()),
                    "same_true_sum_near_best_pair_count": int(
                        same_true_sum_near_best[i].item()
                    ),
                    "same_best_sum_near_best_pair_count": int(
                        same_best_sum_near_best[i].item()
                    ),
                    "learned_pair_head_probability": float(
                        learned_pair_probability[i].item()
                    ),
                    "true_pair_head_probability": float(
                        true_pair_head_probability[i].item()
                    ),
                    "pair_entropy": float(entropy[i].item()),
                    "effective_pair_count": float(entropy[i].exp().item()),
                    "result_entropy": float(result_entropy[i].item()),
                    "effective_result_count": float(result_entropy[i].exp().item()),
                    "pair_logit_entropy": float(pair_logit_entropy[i].item()),
                    "pair_logit_effective_pair_count": float(
                        pair_logit_entropy[i].exp().item()
                    ),
                    "action_pair_count": int(candidates.shape[1]),
                }
            )
        seen += current_batch

    summary = {
        "checkpoint": str(checkpoint),
        "samples": len(rows),
        "digits": digits,
        "answer_format": answer_format,
        "calculator_output_format": model.cfg.calculator_output_format,
        "operand_max": operand_max,
        "temperature": temperature,
        "min_probability_floor": min_probability_floor,
        "near_best_tolerance": near_best_tolerance,
        "chunk_size": chunk_size,
        "action_pair_count": rows[0]["action_pair_count"] if rows else 0,
        "mean_best_full_enum_nll": mean(
            float(row["best_full_enum_mean_nll"]) for row in rows
        ),
        "mean_learned_nll": mean(float(row["learned_mean_nll"]) for row in rows),
        "mean_true_nll": mean(float(row["true_mean_nll"]) for row in rows),
        "mean_learned_minus_true_gap": mean(
            float(row["learned_minus_true_gap"]) for row in rows
        ),
        "mean_learned_minus_best_gap": mean(
            float(row["learned_minus_best_gap"]) for row in rows
        ),
        "mean_learned_result_minus_best_result_gap": mean(
            float(row["learned_result_minus_best_result_gap"]) for row in rows
        ),
        "mean_true_result_minus_best_result_gap": mean(
            float(row["true_result_minus_best_result_gap"]) for row in rows
        ),
        "true_best_fraction": mean(int(row["true_is_best"]) for row in rows),
        "true_result_best_fraction": mean(
            int(row["true_result_is_best_result"]) for row in rows
        ),
        "learned_result_best_fraction": mean(
            int(row["learned_result_is_best_result"]) for row in rows
        ),
        "tie_aware_true_best_fraction": mean(
            int(row["true_within_best_tie"]) for row in rows
        ),
        "learned_best_fraction": mean(int(row["learned_is_best"]) for row in rows),
        "learned_within_1e-3_best_fraction": mean(
            int(row["learned_within_1e-3_best"]) for row in rows
        ),
        "learned_within_1e-2_best_fraction": mean(
            int(row["learned_within_1e-2_best"]) for row in rows
        ),
        "best_matches_true_operands_fraction": mean(
            int(row["best_matches_true_operands"]) for row in rows
        ),
        "best_result_matches_true_sum_fraction": mean(
            int(row["best_result_matches_true_sum"]) for row in rows
        ),
        "best_result_group_matches_true_sum_fraction": mean(
            int(row["best_result_group_matches_true_sum"]) for row in rows
        ),
        "best_left_operand_matches_true_fraction": mean(
            int(row["best_left_operand_matches_true"]) for row in rows
        ),
        "learned_result_matches_true_sum_fraction": mean(
            int(row["learned_result_matches_true_sum"]) for row in rows
        ),
        "mean_soft_target_true_a_mass": mean(
            float(row["soft_target_true_a_mass"]) for row in rows
        ),
        "mean_soft_target_true_b_mass": mean(
            float(row["soft_target_true_b_mass"]) for row in rows
        ),
        "mean_soft_target_true_pair_probability": mean(
            float(row["soft_target_true_pair_probability"]) for row in rows
        ),
        "mean_true_pair_rank": mean(float(row["true_pair_rank"]) for row in rows),
        "mean_top1_target_mass": mean(float(row["top1_target_mass"]) for row in rows),
        "mean_top3_target_mass": mean(float(row["top3_target_mass"]) for row in rows),
        "mean_top5_target_mass": mean(float(row["top5_target_mass"]) for row in rows),
        "mean_same_true_sum_near_best_pair_count": mean(
            float(row["same_true_sum_near_best_pair_count"]) for row in rows
        ),
        "mean_same_best_sum_near_best_pair_count": mean(
            float(row["same_best_sum_near_best_pair_count"]) for row in rows
        ),
        "mean_pair_entropy": mean(float(row["pair_entropy"]) for row in rows),
        "mean_effective_pair_count": mean(
            float(row["effective_pair_count"]) for row in rows
        ),
        "mean_result_entropy": mean(float(row["result_entropy"]) for row in rows),
        "mean_effective_result_count": mean(
            float(row["effective_result_count"]) for row in rows
        ),
        "mean_pair_logit_entropy": mean(
            float(row["pair_logit_entropy"])
            for row in rows
            if float(row["pair_logit_entropy"]) == float(row["pair_logit_entropy"])
        )
        if any(
            float(row["pair_logit_entropy"]) == float(row["pair_logit_entropy"])
            for row in rows
        )
        else float("nan"),
        "mean_pair_logit_effective_pair_count": mean(
            float(row["pair_logit_effective_pair_count"])
            for row in rows
            if float(row["pair_logit_effective_pair_count"])
            == float(row["pair_logit_effective_pair_count"])
        )
        if any(
            float(row["pair_logit_effective_pair_count"])
            == float(row["pair_logit_effective_pair_count"])
            for row in rows
        )
        else float("nan"),
        "calculator_action_head": model.cfg.calculator_action_head,
        "train_config": train_config,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score every calculator action pair by answer NLL."
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--digits", type=int, default=2)
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
        "--calculator-output-format",
        choices=["sum", "sum_left_operand"],
        default="sum",
        help=(
            "Recorded expectation for the checkpoint calculator signal. Checkpoint "
            "configuration remains authoritative."
        ),
    )
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-probability-floor", type=float, default=0.0)
    parser.add_argument("--near-best-tolerance", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.min_probability_floor < 0:
        raise ValueError("--min-probability-floor must be non-negative")
    if args.near_best_tolerance < 0:
        raise ValueError("--near-best-tolerance must be non-negative")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    device = pick_device()
    summaries: list[dict[str, Any]] = []
    for checkpoint_arg in args.checkpoint:
        checkpoint = (
            checkpoint_arg if checkpoint_arg.is_absolute() else REPO_ROOT / checkpoint_arg
        )
        output_dir = (
            args.output_root / checkpoint.parent.name
            if args.output_root is not None
            else checkpoint.parent / "full_enum_action_loss"
        )
        rows, summary = full_enum_diagnostic(
            checkpoint=checkpoint,
            samples=args.samples,
            batch_size=args.batch_size,
            digits=args.digits,
            operand_max=args.operand_max,
            temperature=args.temperature,
            min_probability_floor=args.min_probability_floor,
            near_best_tolerance=args.near_best_tolerance,
            chunk_size=args.chunk_size,
            seed=args.seed,
            device=device,
            answer_format=args.answer_format,
        )
        summary["requested_calculator_output_format"] = args.calculator_output_format
        summary["output_dir"] = str(output_dir)
        summary["device"] = device
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "full_enum_rows.csv", rows)
        (output_dir / "full_enum_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries.append(summary)
        print(
            f"{checkpoint}: learned_true_gap="
            f"{summary['mean_learned_minus_true_gap']:.4f} "
            f"learned_best={summary['learned_best_fraction']:.3f} "
            f"true_best={summary['true_best_fraction']:.3f} "
            f"effective_pairs={summary['mean_effective_pair_count']:.1f}"
        )
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "full_enum_summary_all.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
