"""Train heldout result logits from an amortized prior fit to prompt memory traces."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import ArithmeticBatch
from src.model import GPTConfig, TinyGPT


def load_overfit_module():
    script_path = Path(__file__).resolve().parent / "overfit_one_batch.py"
    spec = importlib.util.spec_from_file_location("overfit_replay_gate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_batch_from_pairs(
    overfit,
    pairs: list[tuple[int, int]],
    *,
    num_digits: int,
    answer_format: str,
    device: str | torch.device,
) -> ArithmeticBatch:
    seq_len = overfit.max_sequence_length(num_digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for a, b in pairs:
        prompt_ids, answer = overfit.make_problem(
            a,
            b,
            num_digits,
            fixed_width=True,
            answer_format=answer_format,
        )
        ids = prompt_ids + overfit.tokenize(answer)
        samples.append(overfit.pad_sequence(ids, seq_len))
        masks.append(overfit.pad_sequence(overfit.make_loss_mask(ids), seq_len, pad_id=0))
    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def set_train_scope(model: TinyGPT, scope: str) -> None:
    for param in model.parameters():
        param.requires_grad = scope == "all"
    if scope == "result_head":
        for hook in model.calculator_hook_modules():
            assert hook.result_proj is not None
            for param in hook.result_proj.parameters():
                param.requires_grad = True
    elif scope == "calculator_policy":
        for name, param in model.named_parameters():
            if ".result_proj." in name or ".input_proj." in name:
                param.requires_grad = True
            elif name.startswith("calculator_hook.result_proj.") or name.startswith(
                "calculator_hook.input_proj."
            ):
                param.requires_grad = True
    elif scope != "all":
        raise ValueError(f"unknown train scope: {scope}")


def fit_prior_from_trace(
    overfit,
    train_rows: list[dict[str, str]],
    *,
    num_digits: int,
    operand_vocab_size: int,
    result_vocab_size: int,
    hidden_size: int,
    feature_mode: str,
    lr: float,
    steps: int,
    seed: int,
    device: str | torch.device,
):
    memory = overfit.ResultBoundaryPromptHardMemory(
        entries={},
        target_mode="zero_improvement",
        scoring_bottleneck_mode="answer_decoder",
        expected_prompt_count=len(train_rows),
    )
    for row in train_rows:
        memory.entries[tuple(overfit.tokenize(row["prompt"]))] = {
            "best_result": int(row["calculator_result"]),
            "best_loss": 0.0,
        }
    prior = overfit.init_result_boundary_amortized_prior(
        operand_vocab_size=operand_vocab_size,
        result_vocab_size=result_vocab_size,
        hidden_size=hidden_size,
        feature_mode=feature_mode,
        lr=lr,
        min_entries=1,
        replay_batch_size=0,
        device=device,
    )
    rng = random.Random(seed)
    train_metrics = {}
    for _ in range(steps):
        train_metrics = overfit.train_result_boundary_amortized_prior(
            prior,
            memory,
            num_digits=num_digits,
            device=device,
            rng=rng,
        )
    return prior, train_metrics


def row_pairs(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    return [(int(row["true_a"]), int(row["true_b"])) for row in rows]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prior-steps", type=int, default=2000)
    parser.add_argument("--prior-lr", type=float, default=0.01)
    parser.add_argument("--prior-hidden-size", type=int, default=64)
    parser.add_argument(
        "--train-replay-weight",
        type=float,
        default=0.0,
        help=(
            "Optional CE weight for replaying train prompts through the same "
            "prior while training heldout prompts."
        ),
    )
    parser.add_argument(
        "--prior-feature-mode",
        choices=["embedding", "numeric"],
        default="numeric",
    )
    parser.add_argument(
        "--train-scope",
        choices=["result_head", "calculator_policy", "all"],
        default="result_head",
    )
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overfit = load_overfit_module()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = GPTConfig(**config["model"])
    device = "cpu"
    model = TinyGPT(cfg).to(device)
    checkpoint = torch.load(args.run_dir / "final_weights.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    train_rows = read_rows(args.run_dir / "train_prompt_trace_rows.csv")
    heldout_rows = read_rows(args.run_dir / "heldout_prompt_trace_rows.csv")
    train_pairs = row_pairs(train_rows)
    heldout_pairs = row_pairs(heldout_rows)
    answer_format = config.get("answer_format", "sum")
    num_digits = int(config["num_digits"])

    prior, prior_train_metrics = fit_prior_from_trace(
        overfit,
        train_rows,
        num_digits=num_digits,
        operand_vocab_size=int(config["calculator_operand_vocab_size"]),
        result_vocab_size=cfg.calculator_result_vocab_size,
        hidden_size=args.prior_hidden_size,
        feature_mode=args.prior_feature_mode,
        lr=args.prior_lr,
        steps=args.prior_steps,
        seed=args.seed,
        device=device,
    )
    prior_train_eval = overfit.evaluate_result_boundary_amortized_prior(
        prior,
        train_pairs,
        device=device,
    )
    prior_heldout_eval = overfit.evaluate_result_boundary_amortized_prior(
        prior,
        heldout_pairs,
        device=device,
    )

    heldout_batch = make_batch_from_pairs(
        overfit,
        heldout_pairs,
        num_digits=num_digits,
        answer_format=answer_format,
        device=device,
    )
    train_batch = make_batch_from_pairs(
        overfit,
        train_pairs,
        num_digits=num_digits,
        answer_format=answer_format,
        device=device,
    )
    set_train_scope(model, args.train_scope)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    def evaluate(prefix: str, step: int) -> dict[str, object]:
        train_traces = overfit.calculator_trace_rows(
            model,
            num_digits=num_digits,
            operand_max=int(config["operand_max"]),
            samples=len(train_pairs),
            seed=args.seed + 10_000 + step,
            device=device,
            oracle_train=False,
            answer_format=answer_format,
            pairs=train_pairs,
        )
        heldout_traces = overfit.calculator_trace_rows(
            model,
            num_digits=num_digits,
            operand_max=int(config["operand_max"]),
            samples=len(heldout_pairs),
            seed=args.seed + 20_000 + step,
            device=device,
            oracle_train=False,
            answer_format=answer_format,
            pairs=heldout_pairs,
        )
        train_summary = overfit.summarize_trace_rows(train_traces)
        heldout_summary = overfit.summarize_trace_rows(heldout_traces)
        return {
            "prefix": prefix,
            "step": step,
            "loss": float("nan"),
            "train_exact_match": train_summary["exact_match"],
            "train_calculator_result_accuracy": train_summary[
                "calculator_result_accuracy"
            ],
            "heldout_exact_match": heldout_summary["exact_match"],
            "heldout_calculator_result_accuracy": heldout_summary[
                "calculator_result_accuracy"
            ],
        }

    rows: list[dict[str, object]] = [evaluate("initial", 0)]
    model.train()
    for step in range(1, args.steps + 1):
        loss, _metrics = overfit.result_boundary_amortized_prior_model_loss(
            model,
            prior,
            heldout_batch,
            num_digits=num_digits,
        )
        if args.train_replay_weight > 0:
            train_loss, _train_metrics = overfit.result_boundary_amortized_prior_model_loss(
                model,
                prior,
                train_batch,
                num_digits=num_digits,
            )
            loss = loss + args.train_replay_weight * train_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if args.eval_every > 0 and step % args.eval_every == 0:
            row = evaluate("replay", step)
            row["loss"] = float(loss.detach().item())
            rows.append(row)

    final = rows[-1]
    output_dir = args.output_dir or (
        args.run_dir / f"amortized_prior_replay_{args.prior_feature_mode}_{args.train_scope}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    overfit.write_rows(output_dir / "replay_curve.csv", rows)
    summary = {
        "run_dir": str(args.run_dir),
        "steps": args.steps,
        "lr": args.lr,
        "prior_steps": args.prior_steps,
        "prior_lr": args.prior_lr,
        "prior_hidden_size": args.prior_hidden_size,
        "prior_feature_mode": args.prior_feature_mode,
        "train_replay_weight": args.train_replay_weight,
        "train_scope": args.train_scope,
        "prior_train_fit_accuracy": prior_train_metrics[
            "result_boundary_target_amortized_prior_train_accuracy"
        ],
        "prior_train_accuracy_vs_true": prior_train_eval["accuracy"],
        "prior_heldout_accuracy_vs_true": prior_heldout_eval["accuracy"],
        "initial": rows[0],
        "final": final,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
