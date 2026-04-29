import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from src.data import (
    EOS_ID,
    EQ_ID,
    ID_TO_TOKEN,
    ArithmeticBatch,
    make_loss_mask,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import GPTConfig, TinyGPT, masked_cross_entropy


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def make_model_config(
    *,
    num_digits: int,
    variant: str,
    injection_scale: float = 1.0,
    operand_vocab_size: int | None = None,
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    mlp_expansion: int = 4,
    calculator_hook_after_layer: int | None = None,
) -> GPTConfig:
    operand_vocab_size = operand_vocab_size or 10**num_digits
    if calculator_hook_after_layer is None:
        calculator_hook_after_layer = min(2, n_layer)
    return GPTConfig(
        block_size=max_sequence_length(num_digits) - 1,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_expansion=mlp_expansion,
        calculator_enabled=variant in {"model-b", "model-c"},
        calculator_mode="add" if variant == "model-c" else "off",
        calculator_hook_after_layer=calculator_hook_after_layer,
        calculator_operand_vocab_size=operand_vocab_size,
        calculator_result_vocab_size=(2 * operand_vocab_size) - 1,
        calculator_injection_scale=injection_scale,
    )


def make_problem(a: int, b: int, num_digits: int) -> tuple[list[int], str]:
    prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
    return tokenize(prompt), f"{a + b}<eos>"


def trim_after_eos(ids: list[int]) -> list[int]:
    if EOS_ID in ids:
        return ids[: ids.index(EOS_ID) + 1]
    return ids


def make_oracle_operands(
    *, a: int, b: int, shape: tuple[int, int], device: str | torch.device
) -> torch.Tensor:
    oracle = torch.zeros((*shape, 2), dtype=torch.long, device=device)
    oracle[..., 0] = a
    oracle[..., 1] = b
    return oracle


def make_range_batch(
    *,
    batch_size: int,
    num_digits: int,
    operand_max: int,
    rng: random.Random,
    device: str | torch.device,
) -> ArithmeticBatch:
    seq_len = max_sequence_length(num_digits)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for _ in range(batch_size):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        ids = tokenize(f"{a:0{num_digits}d}+{b:0{num_digits}d}={a + b}<eos>")
        samples.append(pad_sequence(ids, seq_len))
        masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))

    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(x=tokens[:, :-1], y=tokens[:, 1:], loss_mask=loss_mask[:, 1:])


def train_fresh_model(args: argparse.Namespace, device: str) -> TinyGPT:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    cfg = make_model_config(
        num_digits=args.digits,
        variant=args.variant,
        injection_scale=args.injection_scale,
        operand_vocab_size=args.calculator_operand_vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=args.calculator_hook_after_layer,
    )
    model = TinyGPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    model.train()
    for step in range(args.steps):
        batch = make_range_batch(
            batch_size=args.batch_size,
            num_digits=args.digits,
            operand_max=args.operand_max,
            rng=rng,
            device=device,
        )
        logits = model(batch.x)
        loss = masked_cross_entropy(logits, batch.y, batch.loss_mask)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if args.log_every > 0 and step % args.log_every == 0:
            print(f"fresh step={step:5d} loss={loss.item():.4f}")
    return model


def load_checkpoint(
    checkpoint: Path, device: str, injection_scale: float | None
) -> tuple[TinyGPT, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    train_config = payload["config"]
    model_config = dict(train_config["model"])
    if injection_scale is not None:
        model_config["calculator_injection_scale"] = injection_scale
    cfg = GPTConfig(**model_config)
    model = TinyGPT(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, train_config


@torch.no_grad()
def generate_answer(
    model: TinyGPT,
    *,
    prompt_ids: list[int],
    a: int,
    b: int,
    max_new_tokens: int,
    device: str | torch.device,
    oracle: bool,
    calculator_result_override: str,
) -> tuple[list[int], float]:
    model.eval()
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    confidences: list[float] = []
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.cfg.block_size :]
        oracle_operands = None
        if oracle:
            oracle_operands = make_oracle_operands(
                a=a, b=b, shape=ids_cond.shape, device=device
            )
        logits = model(
            ids_cond,
            oracle_operands=oracle_operands,
            calculator_result_override=calculator_result_override,
        )
        probs = logits[:, -1, :].softmax(dim=-1)
        next_id = probs.argmax(dim=-1, keepdim=True)
        confidences.append(float(probs.max().item()))
        ids = torch.cat([ids, next_id], dim=1)
    generated = ids[0, len(prompt_ids) :].tolist()
    confidence = sum(confidences) / max(len(confidences), 1)
    return generated, confidence


@torch.no_grad()
def diagnostic_rows(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
    oracle: bool,
    calculator_result_override: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    max_answer_tokens = num_digits + 2
    model.eval()
    for i in range(samples):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        prompt_ids, target = make_problem(a, b, num_digits)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        oracle_operands = None
        if oracle:
            oracle_operands = make_oracle_operands(a=a, b=b, shape=x.shape, device=device)
        logits, diagnostics = model(
            x,
            return_diagnostics=True,
            oracle_operands=oracle_operands,
            calculator_result_override=calculator_result_override,
        )
        probs = logits[:, -1, :].softmax(dim=-1)
        first_token_confidence = float(probs.max().item())
        pred_ids, prediction_confidence = generate_answer(
            model,
            prompt_ids=prompt_ids,
            a=a,
            b=b,
            max_new_tokens=max_answer_tokens,
            device=device,
            oracle=oracle,
            calculator_result_override=calculator_result_override,
        )
        pred = decode_tokens(trim_after_eos(pred_ids))
        eq_pos = prompt_ids.index(EQ_ID)
        trace = diagnostics.get("calculator_trace", {})

        def trace_value(name: str, default: float | int | bool) -> float | int | bool:
            if name not in trace:
                return default
            value = trace[name][0, eq_pos]
            if value.dtype == torch.bool:
                return bool(value.item())
            if value.dtype.is_floating_point:
                return float(value.item())
            return int(value.item())

        rows.append(
            {
                "sample": i,
                "prompt": decode_tokens(prompt_ids),
                "true_a": a,
                "true_b": b,
                "true_sum": a + b,
                "target_answer": target,
                "prediction": pred,
                "correct": pred == target,
                "prediction_confidence": prediction_confidence,
                "first_token_confidence": first_token_confidence,
                "eq_position": eq_pos,
                "a_pred": trace_value("a_pred", -1),
                "b_pred": trace_value("b_pred", -1),
                "calculator_result": trace_value("result_pred", -1),
                "a_confidence": trace_value("a_confidence", float("nan")),
                "b_confidence": trace_value("b_confidence", float("nan")),
                "a_entropy": trace_value("a_entropy", float("nan")),
                "b_entropy": trace_value("b_entropy", float("nan")),
                "injection_norm": trace_value("injection_norm", float("nan")),
                "unscaled_injection_norm": trace_value(
                    "unscaled_injection_norm", float("nan")
                ),
                "oracle_used": trace_value("oracle_used", False),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(int(row["correct"]) for row in rows)
    operand_rows = [row for row in rows if row["a_pred"] >= 0 and row["b_pred"] >= 0]
    operand_correct = sum(
        int(row["a_pred"] == row["true_a"] and row["b_pred"] == row["true_b"])
        for row in operand_rows
    )
    result_correct = sum(
        int(row["calculator_result"] == row["true_sum"]) for row in operand_rows
    )
    finite_operand_rows = [
        row
        for row in operand_rows
        if row["a_confidence"] == row["a_confidence"]
        and row["b_confidence"] == row["b_confidence"]
        and row["a_entropy"] == row["a_entropy"]
        and row["b_entropy"] == row["b_entropy"]
    ]

    def mean_field(name: str) -> float:
        if not finite_operand_rows:
            return float("nan")
        return sum(float(row[name]) for row in finite_operand_rows) / len(
            finite_operand_rows
        )

    return {
        "samples": len(rows),
        "exact_match": correct / max(len(rows), 1),
        "correct": correct,
        "operand_exact_match": operand_correct / max(len(operand_rows), 1),
        "calculator_result_accuracy": result_correct / max(len(operand_rows), 1),
        "mean_a_confidence": mean_field("a_confidence"),
        "mean_b_confidence": mean_field("b_confidence"),
        "mean_a_entropy": mean_field("a_entropy"),
        "mean_b_entropy": mean_field("b_entropy"),
    }


@torch.no_grad()
def collect_probe_data(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    layer: int | None,
    position: str,
    samples: int,
    seed: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    residuals: list[torch.Tensor] = []
    targets_a: list[int] = []
    targets_b: list[int] = []
    model.eval()
    for _ in range(samples):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        prompt_ids, _ = make_problem(a, b, num_digits)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        _, diagnostics = model(x, return_diagnostics=True)
        eq_pos = prompt_ids.index(EQ_ID)
        if position == "eq":
            read_pos = eq_pos
        elif position == "a":
            read_pos = num_digits - 1
        elif position == "b":
            read_pos = (num_digits + 1) + (num_digits - 1)
        else:
            raise ValueError(f"unknown probe position: {position}")

        if layer is None:
            residual = diagnostics["calculator_read_residual"]
        else:
            residual = diagnostics["layer_residuals"][layer]
        residuals.append(residual[0, read_pos].detach())
        targets_a.append(a)
        targets_b.append(b)
    x_probe = torch.stack(residuals)
    y_a = torch.tensor(targets_a, dtype=torch.long, device=device)
    y_b = torch.tensor(targets_b, dtype=torch.long, device=device)
    return x_probe, y_a, y_b


def train_probe_head(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    classes: int,
    steps: int,
    lr: float,
) -> dict[str, float]:
    head = nn.Linear(x_train.shape[-1], classes).to(x_train.device)
    optim = torch.optim.AdamW(head.parameters(), lr=lr)
    for _ in range(steps):
        logits = head(x_train)
        loss = nn.functional.cross_entropy(logits, y_train)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    with torch.no_grad():
        eval_logits = head(x_eval)
        eval_loss = nn.functional.cross_entropy(eval_logits, y_eval)
        eval_acc = (eval_logits.argmax(dim=-1) == y_eval).float().mean()
    return {"loss": float(eval_loss.item()), "accuracy": float(eval_acc.item())}


def run_probe(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    layer: int | None,
    position: str,
    samples: int,
    seed: int,
    device: str | torch.device,
    steps: int,
    lr: float,
) -> dict[str, Any]:
    x_probe, y_a, y_b = collect_probe_data(
        model=model,
        num_digits=num_digits,
        operand_max=operand_max,
        layer=layer,
        position=position,
        samples=samples,
        seed=seed,
        device=device,
    )
    split = max(1, int(0.8 * samples))
    classes = model.cfg.calculator_operand_vocab_size
    return {
        "samples": samples,
        "train_samples": split,
        "eval_samples": samples - split,
        "layer": layer if layer is not None else model.cfg.calculator_hook_after_layer,
        "position": position,
        "operand_a": train_probe_head(
            x_probe[:split],
            y_a[:split],
            x_probe[split:],
            y_a[split:],
            classes=classes,
            steps=steps,
            lr=lr,
        ),
        "operand_b": train_probe_head(
            x_probe[:split],
            y_b[:split],
            x_probe[split:],
            y_b[split:],
            classes=classes,
            steps=steps,
            lr=lr,
        ),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the latent protocol used by the hard calculator hook."
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--variant",
        choices=["model-a", "model-b", "model-c"],
        default="model-c",
        help="Used only when training a fresh diagnostic model.",
    )
    parser.add_argument("--digits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--operand-max", type=int, default=None)
    parser.add_argument(
        "--calculator-operand-vocab-size",
        type=int,
        default=None,
        help="Override calculator operand classes when training a fresh diagnostic model.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--injection-scale", type=float, default=1.0)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument(
        "--mlp-expansion",
        type=int,
        default=4,
        help="MLP hidden-size multiplier relative to n_embd.",
    )
    parser.add_argument(
        "--calculator-hook-after-layer",
        type=int,
        default=None,
        help=(
            "Transformer layer after which to inject the calculator. "
            "Default is 2 for depth >=2, otherwise 1."
        ),
    )
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument(
        "--calculator-result-override",
        choices=["add", "zero", "plus_one", "random"],
        default="add",
        help=(
            "Eval-only counterfactual for the calculator result class. "
            "'add' preserves normal behavior."
        ),
    )
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--probe-layers", type=int, nargs="+", default=None)
    parser.add_argument(
        "--probe-positions",
        choices=["a", "b", "eq"],
        nargs="+",
        default=["eq"],
    )
    parser.add_argument("--probe-steps", type=int, default=200)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    operand_max = args.operand_max
    if operand_max is None:
        operand_max = 10**args.digits - 1
    if operand_max >= 10**args.digits:
        raise ValueError("--operand-max must fit inside --digits")
    if args.calculator_operand_vocab_size is not None:
        if operand_max >= args.calculator_operand_vocab_size:
            raise ValueError(
                "--calculator-operand-vocab-size must be greater than --operand-max"
            )
    if args.n_layer < 1:
        raise ValueError("--n-layer must be positive")
    if args.n_head < 1:
        raise ValueError("--n-head must be positive")
    if args.n_embd < 1:
        raise ValueError("--n-embd must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    if args.mlp_expansion < 1:
        raise ValueError("--mlp-expansion must be positive")
    if (
        args.calculator_hook_after_layer is not None
        and not 0 <= args.calculator_hook_after_layer <= args.n_layer
    ):
        raise ValueError("--calculator-hook-after-layer must be within model depth")

    train_config: dict[str, Any] | None = None
    if args.checkpoint is not None:
        model, train_config = load_checkpoint(
            args.checkpoint, device=device, injection_scale=args.injection_scale
        )
    else:
        model = train_fresh_model(args, device=device)

    if args.output_dir is None:
        if args.checkpoint is not None:
            output_dir = args.checkpoint.resolve().parent / "diagnostics"
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            output_dir = (
                Path(__file__).resolve().parent.parent
                / "runs"
                / f"{timestamp}_diagnostics"
            )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = diagnostic_rows(
        model,
        num_digits=args.digits,
        operand_max=operand_max,
        samples=args.samples,
        seed=args.seed + 10_000,
        device=device,
        oracle=args.oracle,
        calculator_result_override=args.calculator_result_override,
    )
    summary = summarize_rows(rows)
    summary.update(
        {
            "device": device,
            "digits": args.digits,
            "operand_max": operand_max,
            "oracle": args.oracle,
            "calculator_result_override": args.calculator_result_override,
            "injection_scale": args.injection_scale,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "train_config": train_config,
            "fresh_config": None if args.checkpoint else asdict(model.cfg),
        }
    )
    if args.probe:
        probe_layers = args.probe_layers
        if probe_layers is None:
            probe_layers = [model.cfg.calculator_hook_after_layer]
        summary["probe"] = {}
        for layer in probe_layers:
            if not 1 <= layer <= model.cfg.n_layer:
                raise ValueError(f"probe layer {layer} outside model depth")
            for position in args.probe_positions:
                key = f"layer{layer}_{position}"
                summary["probe"][key] = run_probe(
                    model,
                    num_digits=args.digits,
                    operand_max=operand_max,
                    layer=layer,
                    position=position,
                    samples=max(args.samples, 8),
                    seed=args.seed + 20_000,
                    device=device,
                    steps=args.probe_steps,
                    lr=args.probe_lr,
                )

    write_rows(output_dir / "calculator_trace_rows.csv", rows)
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"saved diagnostics: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
