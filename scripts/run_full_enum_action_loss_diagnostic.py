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
    fixed_width_operands_from_batch,
    full_enum_action_pairs,
    make_range_batch,
    score_action_loss_candidates_chunked,
)
from src.data import EQ_ID  # noqa: E402


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
    chunk_size: int,
    seed: int,
    device: str,
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
        )
        a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
        classes = a_logits.shape[-1]
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
        learned_idx = learned_a * classes + learned_b
        true_idx = true_a * classes + true_b
        best_idx = losses.argmin(dim=-1)
        batch_idx = torch.arange(current_batch, device=device)
        learned_losses = losses.gather(1, learned_idx.unsqueeze(-1)).squeeze(-1)
        true_losses = losses.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
        best_losses = losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
        best_pairs = pairs.index_select(0, best_idx)
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
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
                    "best_full_enum_mean_nll": float(best_losses[i].item()),
                    "learned_mean_nll": float(learned_losses[i].item()),
                    "true_mean_nll": float(true_losses[i].item()),
                    "learned_minus_true_gap": float(
                        (learned_losses[i] - true_losses[i]).item()
                    ),
                    "learned_minus_best_gap": float(
                        (learned_losses[i] - best_losses[i]).item()
                    ),
                    "true_is_best": bool(best_idx[i].item() == true_idx[i].item()),
                    "learned_is_best": bool(
                        best_idx[i].item() == learned_idx[i].item()
                    ),
                    "best_matches_true_operands": bool(
                        best_a == true_a[i].item() and best_b == true_b[i].item()
                    ),
                    "best_result_matches_true_sum": bool(
                        best_a + best_b == (true_a[i] + true_b[i]).item()
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
                    "pair_entropy": float(entropy[i].item()),
                    "effective_pair_count": float(entropy[i].exp().item()),
                    "action_pair_count": int(candidates.shape[1]),
                }
            )
        seen += current_batch

    summary = {
        "checkpoint": str(checkpoint),
        "samples": len(rows),
        "digits": digits,
        "operand_max": operand_max,
        "temperature": temperature,
        "min_probability_floor": min_probability_floor,
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
        "true_best_fraction": mean(int(row["true_is_best"]) for row in rows),
        "learned_best_fraction": mean(int(row["learned_is_best"]) for row in rows),
        "best_matches_true_operands_fraction": mean(
            int(row["best_matches_true_operands"]) for row in rows
        ),
        "best_result_matches_true_sum_fraction": mean(
            int(row["best_result_matches_true_sum"]) for row in rows
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
        "mean_pair_entropy": mean(float(row["pair_entropy"]) for row in rows),
        "mean_effective_pair_count": mean(
            float(row["effective_pair_count"]) for row in rows
        ),
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
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-probability-floor", type=float, default=0.0)
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
            chunk_size=args.chunk_size,
            seed=args.seed,
            device=device,
        )
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
