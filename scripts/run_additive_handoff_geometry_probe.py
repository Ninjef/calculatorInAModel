import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from scripts.diagnose_calculator_protocol import pick_device  # noqa: E402
from scripts.overfit_one_batch import (  # noqa: E402
    adaptive_optimizer_param_groups,
    fixed_width_operands_from_batch,
    freeze_calculator_policy_parameters,
    load_semantic_decoder_checkpoint,
    make_model_config,
    make_oracle_operands_from_batch,
    trainable_parameter_summary,
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


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def checkpoint_label(checkpoint: Path) -> str:
    try:
        stable_name = checkpoint.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        stable_name = checkpoint.as_posix()
    digest = hashlib.sha1(stable_name.encode("utf-8")).hexdigest()[:8]
    return f"{checkpoint.parent.name}__{checkpoint.stem}__{digest}"


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def loss_per_example(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    batch, seq_len, vocab = logits.shape
    losses = F.cross_entropy(
        logits.reshape(batch * seq_len, vocab),
        targets.reshape(batch * seq_len),
        reduction="none",
    ).reshape(batch, seq_len)
    mask_f = mask.to(losses.dtype)
    return (losses * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)


def masked_loss(batch: ArithmeticBatch, logits: torch.Tensor) -> torch.Tensor:
    return loss_per_example(logits, batch.y, batch.loss_mask).mean()


def load_additive_model(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[TinyGPT, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu")
    train_config = payload["config"]
    cfg = make_model_config(
        args.digits,
        "model-c",
        operand_vocab_size=args.calculator_operand_vocab_size,
        calculator_estimator="ste",
        calculator_action_head="result_space",
        calculator_read_position="operand_spans",
        calculator_read_span_width=args.calculator_read_span_width,
        calculator_injection_mode="add",
        calculator_bottleneck_mode="none",
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
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope="compatible_model")
    return model, train_config


@torch.no_grad()
def trace_summary(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    digits: int,
) -> dict[str, float]:
    logits, diagnostics = model(batch.x, return_diagnostics=True)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
    true_sum = true_a + true_b
    trace = diagnostics["calculator_trace"]
    eq_mask = trace["eq_mask"]
    eq_pos = eq_mask.float().argmax(dim=1).long()
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    result = trace["result_pred"][batch_idx, eq_pos]
    return {
        "normal_loss": float(masked_loss(batch, logits).item()),
        "normal_calculator_result_accuracy": float((result == true_sum).float().mean().item()),
    }


@torch.no_grad()
def basic_losses(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    digits: int,
) -> dict[str, float]:
    oracle = make_oracle_operands_from_batch(batch.x, num_digits=digits)
    normal = trace_summary(model, batch, digits=digits)
    zero_logits = model(batch.x, calculator_result_override="zero")
    oracle_logits = model(batch.x, oracle_operands=oracle)
    return {
        **normal,
        "zero_forced_loss": float(masked_loss(batch, zero_logits).item()),
        "oracle_true_loss": float(masked_loss(batch, oracle_logits).item()),
    }


@torch.no_grad()
def forced_result_geometry(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    digits: int,
    chunk_size: int,
) -> dict[str, float]:
    if model.calculator_hook is None:
        raise ValueError("calculator hook required")
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
    true_sum = true_a + true_b
    _, diagnostics = model(batch.x, return_diagnostics=True)
    trace = diagnostics["calculator_trace"]
    eq_pos = trace["eq_mask"].float().argmax(dim=1).long()
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    learned_result = trace["result_pred"][batch_idx, eq_pos]

    all_losses: list[torch.Tensor] = []
    result_vocab = model.cfg.calculator_result_vocab_size
    for start in range(0, result_vocab, chunk_size):
        classes = torch.arange(
            start,
            min(start + chunk_size, result_vocab),
            device=batch.x.device,
        )
        expanded_x = batch.x.repeat_interleave(classes.numel(), dim=0)
        expanded_y = batch.y.repeat_interleave(classes.numel(), dim=0)
        expanded_mask = batch.loss_mask.repeat_interleave(classes.numel(), dim=0)
        forced = classes.repeat(batch.x.shape[0])
        logits = model(expanded_x, forced_calculator_result_class=forced)
        losses = loss_per_example(logits, expanded_y, expanded_mask)
        all_losses.append(losses.reshape(batch.x.shape[0], classes.numel()))

    loss_matrix = torch.cat(all_losses, dim=1)
    best = loss_matrix.argmin(dim=1)
    sorted_idx = loss_matrix.argsort(dim=1)
    top3 = sorted_idx[:, :3]
    true_loss = loss_matrix.gather(1, true_sum.unsqueeze(1)).squeeze(1)
    learned_loss = loss_matrix.gather(1, learned_result.unsqueeze(1)).squeeze(1)
    best_loss = loss_matrix.gather(1, best.unsqueeze(1)).squeeze(1)
    return {
        "forced_best_true_fraction": float((best == true_sum).float().mean().item()),
        "forced_top3_true_fraction": float(
            (top3 == true_sum.unsqueeze(1)).any(dim=1).float().mean().item()
        ),
        "forced_best_learned_fraction": float(
            (best == learned_result).float().mean().item()
        ),
        "forced_true_loss": float(true_loss.mean().item()),
        "forced_learned_loss": float(learned_loss.mean().item()),
        "forced_best_loss": float(best_loss.mean().item()),
        "forced_true_minus_best": float((true_loss - best_loss).mean().item()),
        "forced_learned_minus_best": float((learned_loss - best_loss).mean().item()),
        "forced_learned_matches_true_fraction": float(
            (learned_result == true_sum).float().mean().item()
        ),
    }


def train_slope(
    model: TinyGPT,
    batch: ArithmeticBatch,
    *,
    args: argparse.Namespace,
    checkpoint_label_value: str,
) -> list[dict[str, Any]]:
    freeze_calculator_policy_parameters(model)
    optimizer = torch.optim.AdamW(
        adaptive_optimizer_param_groups(
            model,
            lr=args.lr,
            input_proj_lr=args.lr,
            upstream_lr=args.lr,
            weight_decay=args.weight_decay,
        ),
        betas=(0.9, 0.95),
    )
    wanted_steps = set(args.slope_steps)
    max_step = max(wanted_steps)
    rows: list[dict[str, Any]] = []

    def append(step: int) -> None:
        losses = basic_losses(model, batch, digits=args.digits)
        rows.append(
            {
                "checkpoint_label": checkpoint_label_value,
                "step": step,
                **losses,
            }
        )

    if 0 in wanted_steps:
        append(0)
    for step in range(1, max_step + 1):
        logits = model(batch.x)
        loss = masked_loss(batch, logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        if step in wanted_steps:
            append(step)
    return rows


def parse_steps(raw: str) -> list[int]:
    steps = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not steps or steps[0] < 0:
        raise ValueError("--slope-steps must contain non-negative integers")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether a learned calculator source has additive handoff "
            "geometry by measuring forced-result losses and short downstream "
            "learning slope."
        )
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="sum")
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--calculator-operand-vocab-size", type=int, default=20)
    parser.add_argument("--calculator-read-span-width", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--mlp-expansion", type=int, default=1)
    parser.add_argument("--calculator-hook-after-layer", type=int, default=1)
    parser.add_argument("--forced-result-chunk-size", type=int, default=16)
    parser.add_argument("--slope-steps", type=parse_steps, default=parse_steps("0,50,100"))
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operand_max >= args.calculator_operand_vocab_size:
        raise ValueError("--calculator-operand-vocab-size must exceed --operand-max")
    if args.forced_result_chunk_size < 1:
        raise ValueError("--forced-result-chunk-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    batch = exhaustive_batch(
        digits=args.digits,
        operand_max=args.operand_max,
        answer_format=args.answer_format,
        device=device,
    )
    output_root = resolve_path(args.output_root) if args.output_root else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_slope_rows: list[dict[str, Any]] = []
    for checkpoint_arg in args.checkpoint:
        checkpoint = resolve_path(checkpoint_arg)
        label = checkpoint_label(checkpoint)
        model, train_config = load_additive_model(checkpoint, args=args, device=device)
        base = basic_losses(model, batch, digits=args.digits)
        geometry = forced_result_geometry(
            model,
            batch,
            digits=args.digits,
            chunk_size=args.forced_result_chunk_size,
        )
        trainable = trainable_parameter_summary(model)
        slope_model, _ = load_additive_model(checkpoint, args=args, device=device)
        slope_rows = train_slope(
            slope_model,
            batch,
            args=args,
            checkpoint_label_value=label,
        )
        all_slope_rows.extend(slope_rows)
        final_slope = slope_rows[-1]
        summary = {
            "checkpoint": str(checkpoint),
            "checkpoint_label": label,
            "device": device,
            "digits": args.digits,
            "operand_max": args.operand_max,
            "slope_steps": args.slope_steps,
            "trainable_parameter_groups": trainable,
            "source_train_config": train_config,
            **base,
            **geometry,
            "slope_final_step": final_slope["step"],
            "slope_final_normal_loss": final_slope["normal_loss"],
            "slope_loss_delta": base["normal_loss"] - final_slope["normal_loss"],
            "slope_final_calc_accuracy": final_slope[
                "normal_calculator_result_accuracy"
            ],
        }
        summaries.append(summary)
        print(
            f"{label}: calc={summary['normal_calculator_result_accuracy']:.4f} "
            f"true_best={summary['forced_best_true_fraction']:.4f} "
            f"true_minus_best={summary['forced_true_minus_best']:.4f} "
            f"slope_delta={summary['slope_loss_delta']:.4f} "
            f"slope_final_loss={summary['slope_final_normal_loss']:.4f}"
        )

    if output_root is not None:
        (output_root / "additive_handoff_geometry_summary.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )
        write_rows(output_root / "additive_handoff_slope_rows.csv", all_slope_rows)


if __name__ == "__main__":
    main()
