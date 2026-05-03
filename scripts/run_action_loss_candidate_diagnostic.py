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
    action_loss_candidate_pairs,
    calculator_read_operand_logits,
    make_range_batch,
    score_action_loss_candidates,
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
def candidate_diagnostic(
    *,
    checkpoint: Path,
    samples: int,
    batch_size: int,
    digits: int,
    operand_max: int,
    random_actions: int,
    topk: int,
    local_radius: int,
    seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, train_config = load_checkpoint(checkpoint, device=device, injection_scale=None)
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 91_000)
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
        candidates = action_loss_candidate_pairs(
            a_logits,
            b_logits,
            random_actions=random_actions,
            topk=topk,
            local_radius=local_radius,
            generator=generator,
        )
        losses = score_action_loss_candidates(model, batch, candidates)
        learned_losses = losses[:, 0]
        best_idx = losses.argmin(dim=-1)
        batch_idx = torch.arange(current_batch, device=device)
        best_losses = losses[batch_idx, best_idx]
        best_pairs = candidates[batch_idx, best_idx]
        learned_pairs = candidates[:, 0]
        for i in range(current_batch):
            best_i = int(best_idx[i].item())
            learned_a = int(learned_pairs[i, 0].item())
            learned_b = int(learned_pairs[i, 1].item())
            best_a = int(best_pairs[i, 0].item())
            best_b = int(best_pairs[i, 1].item())
            prompt = format_prompt(batch.x[i])
            true_a = int(prompt.split("+")[0])
            true_b = int(prompt.split("+")[1].split("=")[0])
            rows.append(
                {
                    "sample": seen + i,
                    "prompt": prompt,
                    "true_a": true_a,
                    "true_b": true_b,
                    "true_sum": true_a + true_b,
                    "learned_a": learned_a,
                    "learned_b": learned_b,
                    "learned_sum": learned_a + learned_b,
                    "best_candidate_index": best_i,
                    "best_a": best_a,
                    "best_b": best_b,
                    "best_sum": best_a + best_b,
                    "learned_mean_nll": float(learned_losses[i].item()),
                    "best_mean_nll": float(best_losses[i].item()),
                    "best_improvement": float(
                        (learned_losses[i] - best_losses[i]).item()
                    ),
                    "candidate_count": int(candidates.shape[1]),
                    "best_beats_learned": bool(
                        best_losses[i].item() < learned_losses[i].item() - 1e-8
                    ),
                    "best_matches_true_operands": bool(
                        best_a == true_a and best_b == true_b
                    ),
                    "best_result_matches_true_sum": bool(
                        best_a + best_b == true_a + true_b
                    ),
                    "learned_result_matches_true_sum": bool(
                        learned_a + learned_b == true_a + true_b
                    ),
                }
            )
        seen += current_batch

    summary = {
        "checkpoint": str(checkpoint),
        "samples": len(rows),
        "digits": digits,
        "operand_max": operand_max,
        "random_actions": random_actions,
        "topk": topk,
        "local_radius": local_radius,
        "candidate_count": rows[0]["candidate_count"] if rows else 0,
        "mean_learned_nll": mean(float(row["learned_mean_nll"]) for row in rows),
        "mean_best_nll": mean(float(row["best_mean_nll"]) for row in rows),
        "mean_best_improvement": mean(float(row["best_improvement"]) for row in rows),
        "better_fraction": mean(int(row["best_beats_learned"]) for row in rows),
        "best_matches_true_operands_fraction": mean(
            int(row["best_matches_true_operands"]) for row in rows
        ),
        "best_result_matches_true_sum_fraction": mean(
            int(row["best_result_matches_true_sum"]) for row in rows
        ),
        "learned_result_matches_true_sum_fraction": mean(
            int(row["learned_result_matches_true_sum"]) for row in rows
        ),
        "train_config": train_config,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate whether answer-NLL-ranked learned/random/top-k/local action "
            "candidates contain actions that beat the current learned action."
        )
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--random-actions", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--local-radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.random_actions < 0:
        raise ValueError("--random-actions must be non-negative")
    if args.topk < 0:
        raise ValueError("--topk must be non-negative")
    if args.local_radius < 0:
        raise ValueError("--local-radius must be non-negative")
    device = pick_device()
    summaries: list[dict[str, Any]] = []
    for checkpoint_arg in args.checkpoint:
        checkpoint = checkpoint_arg if checkpoint_arg.is_absolute() else REPO_ROOT / checkpoint_arg
        output_dir = (
            args.output_root / checkpoint.parent.name
            if args.output_root is not None
            else checkpoint.parent / "action_loss_candidate_diagnostic"
        )
        rows, summary = candidate_diagnostic(
            checkpoint=checkpoint,
            samples=args.samples,
            batch_size=args.batch_size,
            digits=args.digits,
            operand_max=args.operand_max,
            random_actions=args.random_actions,
            topk=args.topk,
            local_radius=args.local_radius,
            seed=args.seed,
            device=device,
        )
        summary["output_dir"] = str(output_dir)
        summary["device"] = device
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "candidate_rows.csv", rows)
        (output_dir / "candidate_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries.append(summary)
        print(
            f"{checkpoint}: better_fraction={summary['better_fraction']:.3f} "
            f"mean_best_improvement={summary['mean_best_improvement']:.4f} "
            f"best_result_acc={summary['best_result_matches_true_sum_fraction']:.3f}"
        )
    if args.output_root is not None:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "candidate_summary_all.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
