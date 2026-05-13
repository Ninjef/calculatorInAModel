import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.overfit_one_batch import (  # noqa: E402
    fixed_width_operands_from_batch,
    load_semantic_decoder_checkpoint,
    make_model_config,
    score_forced_result_classes_chunked,
)
from src.data import (  # noqa: E402
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
DEFAULT_CHECKPOINT = REPO_ROOT / (
    "runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/"
    "stage0_candidates/tiny_operand_spans_dense/oracle_train/"
    "2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/"
    "model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt"
)
DEFAULT_RUN_ROOT = REPO_ROOT / (
    "runs/2026-05-13_phase7_result_feature_separability_and_upstream_open"
)


@dataclass(frozen=True)
class ProbeResult:
    head_kind: str
    hidden_size: int
    seed: int
    train_accuracy: float
    eval_accuracy: float
    steps: int


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_int_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return [int(value) for value in values]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def exhaustive_natural_batch(
    *,
    operand_max: int,
    num_digits: int,
    fixed_width: bool,
    answer_format: AnswerFormat,
    device: str | torch.device,
) -> ArithmeticBatch:
    seq_len = max_sequence_length(num_digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for a in range(operand_max + 1):
        for b in range(operand_max + 1):
            prompt = (
                f"{a:0{num_digits}d}+{b:0{num_digits}d}="
                if fixed_width
                else f"{a}+{b}="
            )
            ids = tokenize(
                prompt
                + answer_target(
                    a,
                    b,
                    num_digits,
                    answer_format=answer_format,
                    fixed_width=fixed_width,
                )
            )
            samples.append(pad_sequence(ids, seq_len))
            masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))
    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(x=tokens[:, :-1], y=tokens[:, 1:], loss_mask=loss_mask[:, 1:])


def build_phase7_model(
    *,
    checkpoint: Path,
    device: str,
    num_digits: int,
    operand_max: int,
    operand_vocab_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    mlp_expansion: int,
    calculator_hook_after_layer: int,
    read_span_width: int,
) -> TinyGPT:
    if operand_max >= operand_vocab_size:
        raise ValueError("--calculator-operand-vocab-size must exceed --operand-max")
    cfg = make_model_config(
        num_digits,
        "model-c",
        operand_vocab_size=operand_vocab_size,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="result_space",
        calculator_read_position="operand_spans",
        calculator_read_span_width=read_span_width,
        calculator_bottleneck_mode="answer_decoder",
        calculator_output_format="sum",
        answer_decoder_interaction="product",
        answer_format="sum",
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_expansion=mlp_expansion,
        calculator_hook_after_layer=calculator_hook_after_layer,
    )
    model = TinyGPT(cfg).to(device)
    load_semantic_decoder_checkpoint(
        model,
        checkpoint,
        load_scope="semantic_decoder_only",
    )
    model.eval()
    return model


@torch.no_grad()
def collect_probe_features(
    model: TinyGPT, batch: ArithmeticBatch
) -> dict[str, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("feature extraction requires a calculator hook")
    if model.cfg.calculator_action_head != "result_space":
        raise ValueError("feature extraction requires result_space action head")
    B, T = batch.x.shape
    if T > model.cfg.block_size:
        raise ValueError(f"sequence length {T} > block_size {model.cfg.block_size}")
    pos = torch.arange(T, device=batch.x.device)
    residual = model.tok_emb(batch.x) + model.pos_emb(pos)
    read_residual = residual
    for i, block in enumerate(model.blocks, start=1):
        read_residual = block(read_residual)
        if i == model.cfg.calculator_hook_after_layer:
            break
    positions = model._calculator_read_positions(batch.x)
    a_read, b_read = model.calculator_hook._operand_span_inputs(
        read_residual, positions
    )
    exact_result_proj_input = torch.cat([a_read, b_read], dim=-1)

    final_residual = residual
    for block in model.blocks:
        final_residual = block(final_residual)
    final_residual = model.ln_f(final_residual)
    a_final, b_final = model.calculator_hook._operand_span_inputs(
        final_residual, positions
    )
    return {
        "exact_result_proj_input": exact_result_proj_input.detach().cpu(),
        "operand_a_span": a_read.detach().cpu(),
        "operand_b_span": b_read.detach().cpu(),
        "final_layer_operand_spans": torch.cat([a_final, b_final], dim=-1)
        .detach()
        .cpu(),
        "calculator_read_residual_paired": exact_result_proj_input.detach().cpu(),
    }


def normalize_from_train(
    train_x: torch.Tensor, eval_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


class ProbeHead(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, *, head_kind: str, hidden_size: int
    ) -> None:
        super().__init__()
        if head_kind == "linear":
            self.net = nn.Linear(in_features, num_classes)
        elif head_kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            )
        else:
            raise ValueError("probe head kind must be 'linear' or 'mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    train_indices: torch.Tensor,
    eval_indices: torch.Tensor,
    head_kind: str,
    hidden_size: int,
    seed: int,
    steps: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[ProbeResult, torch.Tensor]:
    if steps < 1:
        raise ValueError("probe steps must be positive")
    if head_kind == "mlp" and hidden_size < 1:
        raise ValueError("MLP hidden size must be positive")
    if head_kind not in {"linear", "mlp"}:
        raise ValueError("probe head kind must be 'linear' or 'mlp'")
    torch.manual_seed(seed)
    train_x_raw = features.index_select(0, train_indices).to(device=device)
    eval_x_raw = features.index_select(0, eval_indices).to(device=device)
    train_y = labels.index_select(0, train_indices).to(device=device)
    eval_y = labels.index_select(0, eval_indices).to(device=device)
    train_x, eval_x = normalize_from_train(train_x_raw, eval_x_raw)
    head = ProbeHead(
        train_x.shape[-1],
        int(labels.max().item()) + 1,
        head_kind=head_kind,
        hidden_size=hidden_size,
    ).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(train_x), train_y)
        loss.backward()
        optim.step()
    with torch.no_grad():
        train_pred = head(train_x).argmax(dim=-1)
        eval_pred = head(eval_x).argmax(dim=-1)
        result = ProbeResult(
            head_kind=head_kind,
            hidden_size=hidden_size,
            seed=seed,
            train_accuracy=float((train_pred == train_y).float().mean().item()),
            eval_accuracy=float((eval_pred == eval_y).float().mean().item()),
            steps=steps,
        )
        _, all_x = normalize_from_train(train_x_raw, features.to(device=device))
        all_pred = head(all_x).argmax(dim=-1).detach().cpu()
    return result, all_pred


def kfold_indices(n: int, folds: int, *, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    if folds > n:
        raise ValueError("--folds cannot exceed dataset size")
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    fold_sizes = [n // folds for _ in range(folds)]
    for i in range(n % folds):
        fold_sizes[i] += 1
    splits = []
    start = 0
    for size in fold_sizes:
        eval_list = order[start : start + size]
        eval_set = set(eval_list)
        train_list = [idx for idx in order if idx not in eval_set]
        splits.append(
            (
                torch.tensor(train_list, dtype=torch.long),
                torch.tensor(eval_list, dtype=torch.long),
            )
        )
        start += size
    return splits


def summarize_feature_norms(features: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, tensor in features.items():
        norms = tensor.norm(dim=-1)
        summary[name] = {
            "feature_dim": int(tensor.shape[-1]),
            "mean_l2": float(norms.mean().item()),
            "std_l2": float(norms.std(unbiased=False).item()),
            "min_l2": float(norms.min().item()),
            "max_l2": float(norms.max().item()),
            "feature_mean_abs": float(tensor.abs().mean().item()),
            "feature_std": float(tensor.std(unbiased=False).item()),
        }
    return summary


def run_probe_suite(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    seeds: list[int],
    folds: int,
    head_kind: str,
    hidden_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, torch.Tensor]]:
    n = features.shape[0]
    all_indices = torch.arange(n, dtype=torch.long)
    all_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_predictions: dict[int, torch.Tensor] = {}
    for seed in seeds:
        all_result, all_pred = train_probe(
            features,
            labels,
            train_indices=all_indices,
            eval_indices=all_indices,
            head_kind=head_kind,
            hidden_size=hidden_size,
            seed=seed,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        all_predictions[seed] = all_pred
        all_rows.append(asdict(all_result))
        for fold_idx, (train_idx, eval_idx) in enumerate(
            kfold_indices(n, folds, seed=seed)
        ):
            fold_result, _ = train_probe(
                features,
                labels,
                train_indices=train_idx,
                eval_indices=eval_idx,
                head_kind=head_kind,
                hidden_size=hidden_size,
                seed=seed,
                steps=steps,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
            )
            row = asdict(fold_result)
            row["fold"] = fold_idx
            row["train_size"] = int(train_idx.numel())
            row["eval_size"] = int(eval_idx.numel())
            fold_rows.append(row)
    return all_rows, fold_rows, all_predictions


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def build_confusion_by_result_class(
    labels: torch.Tensor, predictions: torch.Tensor
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_class in sorted(set(labels.tolist())):
        mask = labels == result_class
        class_preds = predictions[mask]
        counts: dict[int, int] = {}
        for value in class_preds.tolist():
            counts[value] = counts.get(value, 0) + 1
        rows.append(
            {
                "result_class": int(result_class),
                "count": int(mask.sum().item()),
                "accuracy": float((class_preds == result_class).float().mean().item()),
                "predicted_counts": counts,
            }
        )
    return rows


def validate_args(args: argparse.Namespace) -> None:
    for head in parse_int_list(args.mlp_hidden_sizes):
        if head < 1:
            raise ValueError("--mlp-hidden-sizes entries must be positive")
    probe_heads = [head.strip() for head in args.probe_heads.split(",") if head.strip()]
    if not probe_heads:
        raise ValueError("--probe-heads must include at least one head kind")
    unsupported = sorted(set(probe_heads) - {"linear", "mlp"})
    if unsupported:
        raise ValueError(f"unsupported probe head kind(s): {unsupported}")
    if args.linear_steps < 1:
        raise ValueError("--linear-steps must be positive")
    if args.mlp_steps < 1:
        raise ValueError("--mlp-steps must be positive")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.result_boundary_target_chunk_size < 1:
        raise ValueError("--result-boundary-target-chunk-size must be positive")
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise ValueError("--checkpoint does not exist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7 frozen feature result separability probe."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--probe-seeds", default="2,4,5")
    parser.add_argument("--probe-heads", default="linear,mlp")
    parser.add_argument("--mlp-hidden-sizes", default="64,128")
    parser.add_argument("--linear-steps", type=int, default=2500)
    parser.add_argument("--mlp-steps", type=int, default=3500)
    parser.add_argument("--linear-lr", type=float, default=0.05)
    parser.add_argument("--mlp-lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--calculator-operand-vocab-size", type=int, default=20)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--mlp-expansion", type=int, default=1)
    parser.add_argument("--calculator-hook-after-layer", type=int, default=1)
    parser.add_argument("--calculator-read-span-width", type=int, default=2)
    parser.add_argument("--result-boundary-target-chunk-size", type=int, default=64)
    parser.add_argument("--write-prediction-rows", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    device = args.device or pick_device()
    seeds = parse_int_list(args.probe_seeds)
    hidden_sizes = parse_int_list(args.mlp_hidden_sizes)
    probe_heads = [head.strip() for head in args.probe_heads.split(",") if head.strip()]
    args.run_root.mkdir(parents=True, exist_ok=True)

    model = build_phase7_model(
        checkpoint=args.checkpoint,
        device=device,
        num_digits=args.digits,
        operand_max=args.operand_max,
        operand_vocab_size=args.calculator_operand_vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=args.calculator_hook_after_layer,
        read_span_width=args.calculator_read_span_width,
    )
    batch = exhaustive_natural_batch(
        operand_max=args.operand_max,
        num_digits=args.digits,
        fixed_width=True,
        answer_format="sum",
        device=device,
    )
    features = collect_probe_features(model, batch)
    expected_width = 2 * args.calculator_read_span_width * args.n_embd
    actual_width = features["exact_result_proj_input"].shape[-1]
    if actual_width != expected_width:
        raise ValueError(
            "exact result_proj input width mismatch: "
            f"expected {expected_width}, got {actual_width}"
        )

    with torch.no_grad():
        forced_losses = score_forced_result_classes_chunked(
            model,
            batch,
            chunk_size=args.result_boundary_target_chunk_size,
        ).detach().cpu()
        target_labels = forced_losses.argmin(dim=-1).long()
        true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
        true_a = true_a.detach().cpu()
        true_b = true_b.detach().cpu()
        true_sum = (true_a + true_b).long()
    parity = float((target_labels == true_sum).float().mean().item())

    all_probe_rows: list[dict[str, Any]] = []
    fold_probe_rows: list[dict[str, Any]] = []
    predictions_by_name: dict[str, torch.Tensor] = {}

    if "linear" in probe_heads:
        all_rows, fold_rows, preds = run_probe_suite(
            features=features["exact_result_proj_input"],
            labels=target_labels,
            seeds=seeds,
            folds=args.folds,
            head_kind="linear",
            hidden_size=0,
            steps=args.linear_steps,
            lr=args.linear_lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        all_probe_rows.extend(all_rows)
        fold_probe_rows.extend(fold_rows)
        predictions_by_name["linear"] = preds[seeds[0]]

    for hidden_size in hidden_sizes:
        if "mlp" not in probe_heads:
            continue
        all_rows, fold_rows, preds = run_probe_suite(
            features=features["exact_result_proj_input"],
            labels=target_labels,
            seeds=seeds,
            folds=args.folds,
            head_kind="mlp",
            hidden_size=hidden_size,
            steps=args.mlp_steps,
            lr=args.mlp_lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        all_probe_rows.extend(all_rows)
        fold_probe_rows.extend(fold_rows)
        predictions_by_name[f"mlp{hidden_size}"] = preds[seeds[0]]

    operand_probe_rows: list[dict[str, Any]] = []
    for name, operand_labels in [
        ("operand_a_span", true_a),
        ("operand_b_span", true_b),
    ]:
        rows, _, _ = run_probe_suite(
            features=features[name],
            labels=operand_labels,
            seeds=seeds,
            folds=args.folds,
            head_kind="linear",
            hidden_size=0,
            steps=args.linear_steps,
            lr=args.linear_lr,
            weight_decay=args.weight_decay,
            device=device,
        )
        for row in rows:
            row["feature_name"] = name
        operand_probe_rows.extend(rows)

    def all_acc(head_kind: str, hidden_size: int) -> float:
        rows = [
            row
            for row in all_probe_rows
            if row["head_kind"] == head_kind and row["hidden_size"] == hidden_size
        ]
        return mean([float(row["eval_accuracy"]) for row in rows])

    def fold_accs(head_kind: str, hidden_size: int) -> list[float]:
        return [
            float(row["eval_accuracy"])
            for row in fold_probe_rows
            if row["head_kind"] == head_kind and row["hidden_size"] == hidden_size
        ]

    linear_folds = fold_accs("linear", 0)
    mlp64_folds = fold_accs("mlp", 64)
    mlp128_folds = fold_accs("mlp", 128)
    operand_a_rows = [
        row for row in operand_probe_rows if row["feature_name"] == "operand_a_span"
    ]
    operand_b_rows = [
        row for row in operand_probe_rows if row["feature_name"] == "operand_b_span"
    ]

    confusion_source = predictions_by_name.get(
        "linear",
        next(iter(predictions_by_name.values())) if predictions_by_name else target_labels,
    )
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "run_root": str(args.run_root),
        "config": {
            "digits": args.digits,
            "operand_max": args.operand_max,
            "calculator_operand_vocab_size": args.calculator_operand_vocab_size,
            "calculator_result_vocab_size": model.cfg.calculator_result_vocab_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "mlp_expansion": args.mlp_expansion,
            "calculator_hook_after_layer": args.calculator_hook_after_layer,
            "answer_format": "sum",
            "calculator_output_format": "sum",
            "calculator_bottleneck_mode": "answer_decoder",
            "answer_decoder_interaction": "product",
            "calculator_action_head": "result_space",
            "calculator_read_position": "operand_spans",
            "calculator_read_span_width": args.calculator_read_span_width,
            "semantic_decoder_checkpoint_load_scope": "semantic_decoder_only",
            "freeze_semantic_decoder": True,
            "oracle_train": False,
            "aux_operand_loss_weight": 0.0,
            "adaptive_interface_loss_weight": 0.0,
            "expected_answer_loss_weight": 0.0,
            "relaxed_calculator_entropy_weight": 0.0,
            "input_proj_anchor_weight": 0.0,
        },
        "probe_seeds": seeds,
        "folds": args.folds,
        "result_target_parity_with_true_sum": parity,
        "linear_all400_accuracy": all_acc("linear", 0),
        "linear_5fold_mean_accuracy": mean(linear_folds),
        "linear_5fold_min_accuracy": min(linear_folds) if linear_folds else float("nan"),
        "mlp64_all400_accuracy": all_acc("mlp", 64),
        "mlp64_5fold_mean_accuracy": mean(mlp64_folds),
        "mlp128_all400_accuracy": all_acc("mlp", 128),
        "mlp128_5fold_mean_accuracy": mean(mlp128_folds),
        "operand_a_linear_accuracy": mean(
            [float(row["eval_accuracy"]) for row in operand_a_rows]
        ),
        "operand_b_linear_accuracy": mean(
            [float(row["eval_accuracy"]) for row in operand_b_rows]
        ),
        "confusion_by_result_class": build_confusion_by_result_class(
            target_labels, confusion_source
        ),
        "feature_norm_summary": summarize_feature_norms(features),
        "all400_probe_rows": all_probe_rows,
        "fold_probe_rows": fold_probe_rows,
        "operand_probe_rows": operand_probe_rows,
    }

    (args.run_root / "result_feature_separability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_rows(args.run_root / "result_feature_probe_all400.csv", all_probe_rows)
    write_rows(args.run_root / "result_feature_probe_5fold.csv", fold_probe_rows)
    write_rows(args.run_root / "result_feature_operand_probes.csv", operand_probe_rows)

    example_rows: list[dict[str, Any]] = []
    linear_preds = predictions_by_name.get("linear")
    mlp64_preds = predictions_by_name.get("mlp64")
    mlp128_preds = predictions_by_name.get("mlp128")
    for idx in range(target_labels.shape[0]):
        row: dict[str, Any] = {
            "sample": idx,
            "true_a": int(true_a[idx].item()),
            "true_b": int(true_b[idx].item()),
            "true_sum": int(true_sum[idx].item()),
            "target_result": int(target_labels[idx].item()),
        }
        if linear_preds is not None:
            row["linear_prediction"] = int(linear_preds[idx].item())
        if mlp64_preds is not None:
            row["mlp64_prediction"] = int(mlp64_preds[idx].item())
        if mlp128_preds is not None:
            row["mlp128_prediction"] = int(mlp128_preds[idx].item())
        example_rows.append(row)
    write_rows(args.run_root / "result_feature_predictions.csv", example_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
