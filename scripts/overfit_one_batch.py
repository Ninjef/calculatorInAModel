import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import (
    EOS_ID,
    ID_TO_TOKEN,
    detokenize,
    make_batch,
    max_sequence_length,
    tokenize,
)
from src.model import GPTConfig, TinyGPT, masked_cross_entropy

DEFAULT_DIGITS = (1, 2, 3)
DEFAULT_STEPS = 1000
DEFAULT_EVAL_SAMPLES = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-3
DEFAULT_SEED = 0
LOG_EVERY = 50


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    run_name: str
    seed: int
    num_digits: int
    steps: int
    batch_size: int
    eval_samples: int
    lr: float
    weight_decay: float
    grad_clip: float
    fixed_width: bool
    oracle_train: bool
    model: dict[str, object]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def make_problem(
    a: int, b: int, num_digits: int, fixed_width: bool
) -> tuple[list[int], str]:
    if fixed_width:
        prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
    else:
        prompt = f"{a}+{b}="
    return tokenize(prompt), f"{a + b}<eos>"


def generate_answer(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str | torch.device,
    oracle_operands: tuple[int, int] | None = None,
) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.cfg.block_size :]
        oracle_tensor = None
        if oracle_operands is not None:
            oracle_tensor = make_oracle_operands_from_values(
                a=oracle_operands[0],
                b=oracle_operands[1],
                shape=ids_cond.shape,
                device=device,
            )
        logits = model(ids_cond, oracle_operands=oracle_tensor)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt_ids) :].tolist()


def trim_after_eos(ids: list[int]) -> list[int]:
    if EOS_ID in ids:
        return ids[: ids.index(EOS_ID) + 1]
    return ids


def evaluate(
    model: TinyGPT,
    *,
    num_digits: int,
    samples: int,
    seed: int,
    fixed_width: bool,
    device: str | torch.device,
    oracle_train: bool,
) -> dict[str, object]:
    rng = random.Random(seed)
    high = 10**num_digits - 1
    max_answer_tokens = num_digits + 2
    exact = 0
    examples: list[dict[str, str | bool]] = []

    model.eval()
    for i in range(samples):
        a = rng.randint(0, high)
        b = rng.randint(0, high)
        prompt_ids, target = make_problem(a, b, num_digits, fixed_width=fixed_width)
        oracle_operands = (a, b) if oracle_train else None
        pred_ids = trim_after_eos(
            generate_answer(
                model,
                prompt_ids,
                max_answer_tokens,
                device,
                oracle_operands=oracle_operands,
            )
        )
        pred = decode_tokens(pred_ids)
        ok = pred == target
        exact += int(ok)
        if i < 8:
            examples.append(
                {
                    "prompt": detokenize(prompt_ids),
                    "target": target,
                    "prediction": pred,
                    "correct": ok,
                }
            )

    return {
        "num_digits": num_digits,
        "samples": samples,
        "exact_match": exact / samples,
        "correct": exact,
        "examples": examples,
    }


def save_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(curve)


def make_oracle_operands_from_values(
    *,
    a: int,
    b: int,
    shape: tuple[int, int],
    device: str | torch.device,
) -> torch.Tensor:
    oracle = torch.zeros((*shape, 2), dtype=torch.long, device=device)
    oracle[..., 0] = a
    oracle[..., 1] = b
    return oracle


def make_oracle_operands_from_batch(
    x: torch.Tensor, *, num_digits: int
) -> torch.Tensor:
    powers = torch.tensor(
        [10**i for i in range(num_digits - 1, -1, -1)],
        dtype=torch.long,
        device=x.device,
    )
    a = (x[:, :num_digits].long() * powers).sum(dim=-1)
    b_start = num_digits + 1
    b_end = b_start + num_digits
    b = (x[:, b_start:b_end].long() * powers).sum(dim=-1)
    oracle = torch.zeros((*x.shape, 2), dtype=torch.long, device=x.device)
    oracle[..., 0] = a.unsqueeze(-1)
    oracle[..., 1] = b.unsqueeze(-1)
    return oracle


def make_model_config(
    num_digits: int, variant: str, *, injection_scale: float = 1.0
) -> GPTConfig:
    operand_vocab_size = 10**num_digits
    calculator_enabled = variant in {"model-b", "model-c"}
    calculator_mode = "add" if variant == "model-c" else "off"
    return GPTConfig(
        block_size=max_sequence_length(num_digits) - 1,
        calculator_enabled=calculator_enabled,
        calculator_mode=calculator_mode,
        calculator_hook_after_layer=2,
        calculator_operand_vocab_size=operand_vocab_size,
        calculator_result_vocab_size=(2 * operand_vocab_size) - 1,
        calculator_injection_scale=injection_scale,
    )


def run_variant(
    *,
    num_digits: int,
    args: argparse.Namespace,
    base_run_dir: Path,
    device: str,
) -> dict[str, object]:
    seed = args.seed + num_digits
    torch.manual_seed(seed)
    rng = random.Random(seed)

    cfg = make_model_config(
        num_digits, args.variant, injection_scale=args.injection_scale
    )
    model = TinyGPT(cfg).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.variant}-{num_digits}digit-seed{seed}"
    run_dir = base_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_cfg = TrainConfig(
        variant=args.variant,
        run_name=run_name,
        seed=seed,
        num_digits=num_digits,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_samples=args.eval_samples,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        fixed_width=True,
        oracle_train=args.oracle_train,
        model=asdict(cfg),
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(train_cfg), indent=2) + "\n"
    )

    curve: list[dict[str, float | int]] = []
    final_loss = float("nan")
    model.train()
    for step in range(args.steps + 1):
        batch = make_batch(
            batch_size=args.batch_size,
            num_digits=num_digits,
            rng=rng,
            fixed_width=True,
            device=device,
        )
        oracle_operands = None
        if args.oracle_train:
            oracle_operands = make_oracle_operands_from_batch(
                batch.x, num_digits=num_digits
            )
        logits = model(batch.x, oracle_operands=oracle_operands)
        loss = masked_cross_entropy(logits, batch.y, batch.loss_mask)

        if step % args.log_every == 0:
            loss_value = loss.item()
            curve.append({"step": step, "loss": loss_value})
            print(
                f"variant={args.variant} digits={num_digits} "
                f"step={step:5d} loss={loss_value:.4f}"
            )

        if step == args.steps:
            final_loss = loss.item()
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()

    metrics = evaluate(
        model,
        num_digits=num_digits,
        samples=args.eval_samples,
        seed=seed + 10_000,
        fixed_width=True,
        device=device,
        oracle_train=args.oracle_train,
    )
    metrics["final_loss"] = final_loss
    metrics["parameter_count"] = model.num_params()
    metrics["run_dir"] = str(run_dir)
    metrics["variant"] = args.variant
    metrics["oracle_train"] = args.oracle_train

    save_curve(run_dir / "training_curve.csv", curve)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(train_cfg),
            "metrics": metrics,
        },
        run_dir / "final_weights.pt",
    )

    print(
        f"variant={args.variant} digits={num_digits} eval exact-match "
        f"{metrics['correct']}/{metrics['samples']} "
        f"({metrics['exact_match']:.3f}); saved {run_dir}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny addition models with optional latent calculator hook."
    )
    parser.add_argument(
        "--variant",
        choices=["model-a", "model-b", "model-c"],
        default="model-a",
        help="model-a is the raw baseline, model-b wires the hook off, model-c turns addition on.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIGITS),
        help="Digit counts to train/evaluate separately.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--oracle-train",
        action="store_true",
        help="Feed true operands into Model C's calculator during training/eval.",
    )
    parser.add_argument(
        "--injection-scale",
        type=float,
        default=1.0,
        help="Scale the calculator residual injection; default preserves existing behavior.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oracle_train and args.variant != "model-c":
        raise ValueError("--oracle-train is only meaningful with --variant model-c")
    device = pick_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = f"{args.variant}-oracle" if args.oracle_train else args.variant
    base_run_dir = args.run_root / f"{timestamp}_{suffix}"
    base_run_dir.mkdir(parents=True, exist_ok=False)

    print(f"device: {device}")
    print(f"variant: {args.variant}")
    print(f"oracle train: {args.oracle_train}")
    print(f"injection scale: {args.injection_scale}")
    print(f"run root: {base_run_dir}")

    all_metrics = []
    for num_digits in args.digits:
        all_metrics.append(
            run_variant(
                num_digits=num_digits,
                args=args,
                base_run_dir=base_run_dir,
                device=device,
            )
        )

    summary = {"device": device, "variant": args.variant, "runs": all_metrics}
    (base_run_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("summary:")
    for metrics in all_metrics:
        print(
            f"  {args.variant} {metrics['num_digits']}-digit: "
            f"exact-match={metrics['exact_match']:.3f}, "
            f"final_loss={metrics['final_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
