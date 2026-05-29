import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.diagnose_calculator_protocol import pick_device  # noqa: E402
from src.data import (  # noqa: E402
    ANSWER_FORMATS,
    EQ_ID,
    AnswerFormat,
    answer_target,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import GPTConfig, TinyGPT  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def output_dir_for_checkpoint(output_root: Path, checkpoint: Path) -> Path:
    try:
        stable_name = checkpoint.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        stable_name = checkpoint.as_posix()
    digest = hashlib.sha1(stable_name.encode("utf-8")).hexdigest()[:8]
    return output_root / f"{checkpoint.parent.name}__{checkpoint.stem}__{digest}"


def load_probe_model(
    checkpoint: Path,
    *,
    device: str,
    additive_compatible: bool,
) -> tuple[TinyGPT, dict[str, Any], int, list[str]]:
    payload = torch.load(checkpoint, map_location=device)
    train_config = payload["config"]
    model_config = dict(train_config["model"])
    if additive_compatible:
        model_config["calculator_bottleneck_mode"] = "none"
        model_config["answer_decoder_interaction"] = "none"
        model_config["calculator_estimator"] = "ste"
    cfg = GPTConfig(**model_config)
    model = TinyGPT(cfg).to(device)
    checkpoint_state = payload["model_state_dict"]
    model_state = model.state_dict()
    if additive_compatible:
        compatible_state = {
            key: value
            for key, value in checkpoint_state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        loaded_tensors = len(compatible_state)
        skipped = sorted(set(checkpoint_state) - set(compatible_state))
        skipped.extend(f"missing::{name}" for name in sorted(missing))
        skipped.extend(f"unexpected::{name}" for name in sorted(unexpected))
    else:
        model.load_state_dict(checkpoint_state)
        loaded_tensors = len(checkpoint_state)
        skipped = []
    model.eval()
    return model, train_config, loaded_tensors, skipped


def exact_grid_inputs(
    *,
    digits: int,
    operand_max: int,
    answer_format: AnswerFormat,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = max_sequence_length(digits, answer_format=answer_format)
    inputs: list[list[int]] = []
    targets: list[int] = []
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
            inputs.append(pad_sequence(ids, seq_len)[:-1])
            targets.append(a + b)
    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def collect_features(model: TinyGPT, x: torch.Tensor) -> dict[str, torch.Tensor]:
    _, diagnostics = model(x, return_diagnostics=True)
    eq_mask = x == EQ_ID
    if not eq_mask.any(dim=1).all():
        raise ValueError("every prompt must contain an '=' token")
    eq_pos = eq_mask.float().argmax(dim=1)
    batch_idx = torch.arange(x.shape[0], device=x.device)
    read_positions = model._calculator_read_positions(x)
    a_pos = read_positions["a"]
    b_pos = read_positions["b"]
    features: dict[str, torch.Tensor] = {}
    read_residual = diagnostics["calculator_read_residual"]
    read_a = read_residual[batch_idx, a_pos]
    read_b = read_residual[batch_idx, b_pos]
    features["read_eq"] = read_residual[batch_idx, eq_pos]
    features["read_a"] = read_a
    features["read_b"] = read_b
    features["read_pair"] = torch.cat([read_a, read_b], dim=-1)
    for layer, residual in diagnostics.get("layer_residuals", {}).items():
        features[f"layer{layer}_eq"] = residual[batch_idx, eq_pos]
        layer_a = residual[batch_idx, a_pos]
        layer_b = residual[batch_idx, b_pos]
        features[f"layer{layer}_a"] = layer_a
        features[f"layer{layer}_b"] = layer_b
        features[f"layer{layer}_pair"] = torch.cat([layer_a, layer_b], dim=-1)
    return features


def split_indices(samples: int, *, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices = list(range(samples))
    random.Random(seed).shuffle(indices)
    split = max(1, min(samples - 1, int(train_fraction * samples)))
    return torch.tensor(indices[:split]), torch.tensor(indices[split:])


def train_linear_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    train_idx: torch.Tensor,
    eval_idx: torch.Tensor,
    *,
    classes: int,
    steps: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    train_idx = train_idx.to(features.device)
    eval_idx = eval_idx.to(features.device)
    head = nn.Linear(features.shape[-1], classes).to(features.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        logits = head(features[train_idx])
        loss = F.cross_entropy(logits, targets[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_logits = head(features[train_idx])
        eval_logits = head(features[eval_idx])
        train_loss = F.cross_entropy(train_logits, targets[train_idx])
        eval_loss = F.cross_entropy(eval_logits, targets[eval_idx])
        train_acc = (train_logits.argmax(dim=-1) == targets[train_idx]).float().mean()
        eval_acc = (eval_logits.argmax(dim=-1) == targets[eval_idx]).float().mean()
    return {
        "train_loss": float(train_loss.item()),
        "eval_loss": float(eval_loss.item()),
        "train_accuracy": float(train_acc.item()),
        "eval_accuracy": float(eval_acc.item()),
    }


def probe_checkpoint(
    checkpoint: Path,
    *,
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, train_config, loaded_tensors, skipped = load_probe_model(
        checkpoint,
        device=device,
        additive_compatible=args.additive_compatible,
    )
    x, targets = exact_grid_inputs(
        digits=args.digits,
        operand_max=args.operand_max,
        answer_format=args.answer_format,
        device=device,
    )
    if targets.max().item() >= model.cfg.calculator_result_vocab_size:
        raise ValueError("sum targets exceed checkpoint result vocabulary")
    train_idx, eval_idx = split_indices(
        targets.shape[0],
        train_fraction=args.train_fraction,
        seed=args.seed + 123,
    )
    feature_map = collect_features(model, x)
    requested_features = args.features or sorted(feature_map)
    rows: list[dict[str, Any]] = []
    for feature_name in requested_features:
        if feature_name not in feature_map:
            raise ValueError(
                f"feature {feature_name!r} not available; "
                f"available features: {', '.join(sorted(feature_map))}"
            )
        metrics = train_linear_probe(
            feature_map[feature_name],
            targets,
            train_idx,
            eval_idx,
            classes=model.cfg.calculator_result_vocab_size,
            steps=args.probe_steps,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            seed=args.seed + 456,
        )
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "feature": feature_name,
                "feature_dim": int(feature_map[feature_name].shape[-1]),
                "train_samples": int(train_idx.numel()),
                "eval_samples": int(eval_idx.numel()),
                **metrics,
            }
        )
    best_row = max(rows, key=lambda row: float(row["eval_accuracy"]))
    summary = {
        "checkpoint": str(checkpoint),
        "device": device,
        "digits": args.digits,
        "operand_max": args.operand_max,
        "answer_format": args.answer_format,
        "additive_compatible": args.additive_compatible,
        "loaded_tensors": loaded_tensors,
        "skipped_tensors": skipped,
        "train_fraction": args.train_fraction,
        "probe_steps": args.probe_steps,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "seed": args.seed,
        "best_feature": best_row["feature"],
        "best_eval_accuracy": best_row["eval_accuracy"],
        "mean_eval_accuracy": mean(float(row["eval_accuracy"]) for row in rows),
        "features": rows,
        "train_config": train_config,
    }
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train cheap linear probes on frozen calculator-source states to "
            "estimate additive handoff geometry."
        )
    )
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument(
        "--answer-format",
        choices=ANSWER_FORMATS,
        default="sum",
    )
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument(
        "--additive-compatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load compatible tensors into an additive non-bottleneck model before "
            "probing, matching bottleneck-to-additive handoff setup."
        ),
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Probe only named features such as read_eq or layer2_eq.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--probe-steps", type=int, default=800)
    parser.add_argument("--probe-lr", type=float, default=0.05)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")
    if args.probe_steps < 1:
        raise ValueError("--probe-steps must be positive")
    if args.probe_lr <= 0:
        raise ValueError("--probe-lr must be positive")
    if args.probe_weight_decay < 0:
        raise ValueError("--probe-weight-decay must be non-negative")
    if args.operand_max >= 10**args.digits:
        raise ValueError("--operand-max must fit inside --digits")

    device = pick_device()
    output_root = args.output_root
    if output_root is not None:
        output_root = resolve_path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for checkpoint_arg in args.checkpoint:
        checkpoint = resolve_path(checkpoint_arg)
        rows, summary = probe_checkpoint(checkpoint, args=args, device=device)
        summaries.append(summary)
        if output_root is not None:
            output_dir = output_dir_for_checkpoint(output_root, checkpoint)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_rows(output_dir / "frozen_state_probe_rows.csv", rows)
            (output_dir / "frozen_state_probe_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
        print(
            f"{checkpoint}: best_feature={summary['best_feature']} "
            f"best_eval_accuracy={summary['best_eval_accuracy']:.4f} "
            f"mean_eval_accuracy={summary['mean_eval_accuracy']:.4f}"
        )

    if output_root is not None:
        (output_root / "frozen_state_probe_summary_all.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
