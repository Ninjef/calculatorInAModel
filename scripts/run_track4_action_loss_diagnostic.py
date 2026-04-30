import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnose_calculator_protocol import (  # noqa: E402
    decode_tokens,
    load_checkpoint,
    make_oracle_operands,
    make_problem,
    pick_device,
)
from src.data import EQ_ID, tokenize  # noqa: E402
from src.model import TinyGPT  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_HISTORY = (
    REPO_ROOT
    / "aiAgentWorkHistory/2026-04-30-track-4-optimization-estimators.md"
)
TASK_DOC = (
    REPO_ROOT
    / "aiAgentProjectTasks/2026-04-30-1202-Track-4-optimization-estimators.md"
)
COMPLETED_TASK_DOC = (
    REPO_ROOT
    / "aiAgentProjectTasks/completed/2026-04-30-1202-Track-4-optimization-estimators.md"
)


@dataclass(frozen=True)
class Track4Checkpoint:
    name: str
    purpose: str
    checkpoint: str
    digits: int = 2
    operand_max: int = 19
    oracle: bool = False


MANIFEST = [
    Track4Checkpoint(
        name="additive_answer_only_model_c",
        purpose="Additive answer-only learned Model C",
        checkpoint="runs/2026-04-30_124622_062528_model-c-op0-19/model-c-2digit-seed2/final_weights.pt",
    ),
    Track4Checkpoint(
        name="additive_aux001_model_c",
        purpose="Additive high-answer aux 0.01 Model C",
        checkpoint="runs/2026-04-30_124941_086322_model-c-op0-19-aux0.01/model-c-2digit-seed2/final_weights.pt",
    ),
    Track4Checkpoint(
        name="replace_model_b_off_leakage_control",
        purpose="Replacement Model B/off leakage control",
        checkpoint="runs/2026-04-30_131732_611676_model-b-op0-19-replace/model-b-2digit-seed2/final_weights.pt",
    ),
    Track4Checkpoint(
        name="replace_oracle_model_c",
        purpose="Replacement oracle Model C control",
        checkpoint="runs/2026-04-30_131816_053136_model-c-oracle-op0-19-replace/model-c-2digit-seed2/final_weights.pt",
        oracle=True,
    ),
    Track4Checkpoint(
        name="replace_answer_only_model_c",
        purpose="Replacement answer-only learned Model C",
        checkpoint="runs/2026-04-30_131959_337278_model-c-op0-19-replace/model-c-2digit-seed2/final_weights.pt",
    ),
    Track4Checkpoint(
        name="replace_aux_decay_model_c",
        purpose="Replacement aux decay transient candidate",
        checkpoint="runs/2026-04-30_132810_628910_model-c-op0-19-replace-aux0.03-auxdecay1000/model-c-2digit-seed2/final_weights.pt",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def make_sample_specs(
    *, samples: int, operand_max: int, seed: int
) -> list[dict[str, int]]:
    rng = random.Random(seed)
    return [
        {
            "sample": i,
            "true_a": rng.randint(0, operand_max),
            "true_b": rng.randint(0, operand_max),
        }
        for i in range(samples)
    ]


def answer_loss_for_action(
    model: TinyGPT,
    *,
    prompt_ids: list[int],
    target_answer: str,
    true_a: int,
    true_b: int,
    forced_a: int | None,
    forced_b: int | None,
    device: str | torch.device,
    oracle_base: bool,
) -> tuple[float, float]:
    target_ids = tokenize(target_answer)
    full_ids = prompt_ids + target_ids
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    oracle_operands = None
    if forced_a is not None and forced_b is not None:
        oracle_operands = make_oracle_operands(
            a=forced_a, b=forced_b, shape=x.shape, device=device
        )
    elif oracle_base:
        oracle_operands = make_oracle_operands(
            a=true_a, b=true_b, shape=x.shape, device=device
        )
    logits = model(x, oracle_operands=oracle_operands)
    log_probs = logits.log_softmax(dim=-1)
    start = len(prompt_ids) - 1
    token_log_probs = log_probs[0, start : start + len(target_ids)].gather(
        -1, y[0, start : start + len(target_ids)].unsqueeze(-1)
    )
    total_nll = -float(token_log_probs.sum().item())
    mean_nll = total_nll / max(len(target_ids), 1)
    return total_nll, mean_nll


def learned_action_for_prompt(
    model: TinyGPT,
    *,
    prompt_ids: list[int],
    true_a: int,
    true_b: int,
    device: str | torch.device,
    oracle_base: bool,
) -> dict[str, int | float | bool]:
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    oracle_operands = None
    if oracle_base:
        oracle_operands = make_oracle_operands(
            a=true_a, b=true_b, shape=x.shape, device=device
        )
    _, diagnostics = model(x, return_diagnostics=True, oracle_operands=oracle_operands)
    trace = diagnostics.get("calculator_trace", {})
    eq_pos = prompt_ids.index(EQ_ID)

    def scalar(name: str, default: int | float | bool) -> int | float | bool:
        if name not in trace:
            return default
        value = trace[name][0, eq_pos]
        if value.dtype == torch.bool:
            return bool(value.item())
        if value.dtype.is_floating_point:
            return float(value.item())
        return int(value.item())

    return {
        "learned_a": int(scalar("a_pred", -1)),
        "learned_b": int(scalar("b_pred", -1)),
        "learned_result": int(scalar("result_pred", -1)),
        "a_entropy": float(scalar("a_entropy", float("nan"))),
        "b_entropy": float(scalar("b_entropy", float("nan"))),
        "a_confidence": float(scalar("a_confidence", float("nan"))),
        "b_confidence": float(scalar("b_confidence", float("nan"))),
        "oracle_used": bool(scalar("oracle_used", False)),
    }


@torch.no_grad()
def action_loss_diagnostic(
    model: TinyGPT,
    *,
    sample_specs: list[dict[str, int]],
    num_digits: int,
    operand_max: int,
    random_actions: int,
    seed: int,
    device: str | torch.device,
    oracle_base: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    model.eval()
    shuffled_pairs = [
        (
            sample_specs[(i + 1) % len(sample_specs)]["true_a"],
            sample_specs[(i + 1) % len(sample_specs)]["true_b"],
        )
        for i in range(len(sample_specs))
    ]
    action_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []

    for spec, shuffled_pair in zip(sample_specs, shuffled_pairs):
        sample = int(spec["sample"])
        true_a = int(spec["true_a"])
        true_b = int(spec["true_b"])
        prompt_ids, target_answer = make_problem(true_a, true_b, num_digits)
        learned = learned_action_for_prompt(
            model,
            prompt_ids=prompt_ids,
            true_a=true_a,
            true_b=true_b,
            device=device,
            oracle_base=oracle_base,
        )
        learned_a = int(learned["learned_a"])
        learned_b = int(learned["learned_b"])
        candidates: list[tuple[str, int | None, int | None]] = [
            ("normal", None, None),
            ("learned_forced", learned_a, learned_b),
            ("true", true_a, true_b),
            ("shuffled_true", int(shuffled_pair[0]), int(shuffled_pair[1])),
        ]
        for i in range(random_actions):
            candidates.append(
                (
                    f"random_{i}",
                    rng.randint(0, operand_max),
                    rng.randint(0, operand_max),
                )
            )

        losses: list[float] = []
        named_losses: dict[str, float] = {}
        for action_kind, forced_a, forced_b in candidates:
            total_nll, mean_nll = answer_loss_for_action(
                model,
                prompt_ids=prompt_ids,
                target_answer=target_answer,
                true_a=true_a,
                true_b=true_b,
                forced_a=forced_a,
                forced_b=forced_b,
                device=device,
                oracle_base=oracle_base,
            )
            action_rows.append(
                {
                    "sample": sample,
                    "prompt": decode_tokens(prompt_ids),
                    "target_answer": target_answer,
                    "true_a": true_a,
                    "true_b": true_b,
                    "true_sum": true_a + true_b,
                    "action_kind": action_kind,
                    "forced_a": "" if forced_a is None else forced_a,
                    "forced_b": "" if forced_b is None else forced_b,
                    "forced_sum": ""
                    if forced_a is None or forced_b is None
                    else forced_a + forced_b,
                    "target_total_nll": total_nll,
                    "target_mean_nll": mean_nll,
                    "learned_a": learned_a,
                    "learned_b": learned_b,
                    "learned_result": learned["learned_result"],
                    "a_entropy": learned["a_entropy"],
                    "b_entropy": learned["b_entropy"],
                    "a_confidence": learned["a_confidence"],
                    "b_confidence": learned["b_confidence"],
                    "oracle_used": learned["oracle_used"],
                }
            )
            losses.append(mean_nll)
            named_losses[action_kind] = mean_nll

        random_losses = [
            loss
            for name, loss in named_losses.items()
            if name.startswith("random_")
        ]
        best_kind, best_loss = min(named_losses.items(), key=lambda item: item[1])
        prompt_rows.append(
            {
                "sample": sample,
                "prompt": decode_tokens(prompt_ids),
                "target_answer": target_answer,
                "true_a": true_a,
                "true_b": true_b,
                "true_sum": true_a + true_b,
                "learned_a": learned_a,
                "learned_b": learned_b,
                "learned_result": learned["learned_result"],
                "normal_mean_nll": named_losses["normal"],
                "learned_forced_mean_nll": named_losses["learned_forced"],
                "true_mean_nll": named_losses["true"],
                "shuffled_true_mean_nll": named_losses["shuffled_true"],
                "random_mean_nll": mean(random_losses),
                "random_best_mean_nll": min(random_losses),
                "random_std_mean_nll": pstdev(random_losses)
                if len(random_losses) > 1
                else 0.0,
                "action_loss_std": pstdev(losses) if len(losses) > 1 else 0.0,
                "action_loss_range": max(losses) - min(losses),
                "random_minus_true_gap": mean(random_losses) - named_losses["true"],
                "shuffled_minus_true_gap": named_losses["shuffled_true"]
                - named_losses["true"],
                "learned_minus_true_gap": named_losses["learned_forced"]
                - named_losses["true"],
                "true_is_best": best_kind == "true",
                "learned_is_best": best_kind == "learned_forced",
                "best_action_kind": best_kind,
                "best_mean_nll": best_loss,
                "a_entropy": learned["a_entropy"],
                "b_entropy": learned["b_entropy"],
            }
        )

    def mean_field(name: str) -> float:
        return mean(float(row[name]) for row in prompt_rows)

    summary = {
        "samples": len(prompt_rows),
        "random_actions_per_prompt": random_actions,
        "mean_normal_nll": mean_field("normal_mean_nll"),
        "mean_learned_forced_nll": mean_field("learned_forced_mean_nll"),
        "mean_true_nll": mean_field("true_mean_nll"),
        "mean_random_nll": mean_field("random_mean_nll"),
        "mean_shuffled_true_nll": mean_field("shuffled_true_mean_nll"),
        "mean_action_loss_std": mean_field("action_loss_std"),
        "mean_action_loss_range": mean_field("action_loss_range"),
        "mean_random_minus_true_gap": mean_field("random_minus_true_gap"),
        "mean_shuffled_minus_true_gap": mean_field("shuffled_minus_true_gap"),
        "mean_learned_minus_true_gap": mean_field("learned_minus_true_gap"),
        "true_best_fraction": sum(bool(row["true_is_best"]) for row in prompt_rows)
        / max(len(prompt_rows), 1),
        "learned_best_fraction": sum(
            bool(row["learned_is_best"]) for row in prompt_rows
        )
        / max(len(prompt_rows), 1),
        "operand_exact_match": sum(
            int(row["learned_a"] == row["true_a"] and row["learned_b"] == row["true_b"])
            for row in prompt_rows
        )
        / max(len(prompt_rows), 1),
        "calculator_result_accuracy": sum(
            int(row["learned_result"] == row["true_sum"]) for row in prompt_rows
        )
        / max(len(prompt_rows), 1),
        "mean_a_entropy": mean_field("a_entropy"),
        "mean_b_entropy": mean_field("b_entropy"),
        "digits": num_digits,
        "operand_max": operand_max,
    }
    return action_rows, prompt_rows, summary


def track3_classification(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.parent / "track3_diagnostics" / "diagnostic_summary.json"
    if not path.exists():
        return {
            "category": "track3_summary_missing",
            "bottleneck_classification": "track3_summary_missing",
        }
    summary = read_json(path)
    return summary.get("classification", {})


def run_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.limit < 1 or args.limit > len(MANIFEST):
        raise ValueError(f"--limit must be in [1, {len(MANIFEST)}]")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.random_actions < 1:
        raise ValueError("--random-actions must be positive")
    device = pick_device()
    results: list[dict[str, Any]] = []
    for item in MANIFEST[: args.limit]:
        checkpoint = REPO_ROOT / item.checkpoint
        output_dir = (
            args.output_root / item.name
            if args.output_root is not None
            else checkpoint.parent / "track4_action_loss"
        )
        cmd_info = {
            "checkpoint": str(checkpoint),
            "output_dir": str(output_dir),
            "samples": args.samples,
            "random_actions": args.random_actions,
        }
        if args.dry_run:
            print(json.dumps(cmd_info, indent=2))
            results.append(
                {
                    "name": item.name,
                    "purpose": item.purpose,
                    "checkpoint": item.checkpoint,
                    "output_dir": str(output_dir.relative_to(REPO_ROOT)),
                    "summary": {},
                    "classification": {},
                }
            )
            continue
        model, train_config = load_checkpoint(
            checkpoint, device=device, injection_scale=None
        )
        sample_specs = make_sample_specs(
            samples=args.samples,
            operand_max=item.operand_max,
            seed=args.seed + 30_000,
        )
        action_rows, prompt_rows, summary = action_loss_diagnostic(
            model,
            sample_specs=sample_specs,
            num_digits=item.digits,
            operand_max=item.operand_max,
            random_actions=args.random_actions,
            seed=args.seed + 40_000,
            device=device,
            oracle_base=item.oracle,
        )
        classification = track3_classification(checkpoint)
        summary.update(
            {
                "name": item.name,
                "purpose": item.purpose,
                "checkpoint": item.checkpoint,
                "device": device,
                "oracle_base": item.oracle,
                "calculator_injection_mode": model.cfg.calculator_injection_mode,
                "calculator_read_position": model.cfg.calculator_read_position,
                "calculator_estimator": model.cfg.calculator_estimator,
                "train_config": train_config,
                "track3_classification": classification,
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "action_loss_rows.csv", action_rows)
        write_rows(output_dir / "prompt_action_loss_summary.csv", prompt_rows)
        (output_dir / "action_loss_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        results.append(
            {
                "name": item.name,
                "purpose": item.purpose,
                "checkpoint": item.checkpoint,
                "output_dir": str(output_dir.relative_to(REPO_ROOT)),
                "summary": summary,
                "classification": classification,
            }
        )
    return results


def write_work_history(results: list[dict[str, Any]], *, args: argparse.Namespace) -> None:
    lines = [
        "# 2026-04-30 - Track 4 optimization estimators and action-loss signal",
        "",
        "Task: start Track 4 by measuring whether calculator operand actions create enough downstream answer-loss signal to justify estimator sweeps.",
        "",
        "## Implementation",
        "",
        "- Added `scripts/run_track4_action_loss_diagnostic.py`.",
        "- Reused the Track 3 six-checkpoint manifest and classification labels.",
        "- For each prompt, the diagnostic measures target answer NLL under normal learned actions, forced learned operands, true operands, shuffled true operands, and random operand actions.",
        "- The output includes per-action CSV rows, per-prompt summaries, and JSON aggregates for action-loss variance and true/random/shuffled/learned gaps.",
        "",
        "## Diagnostic Run",
        "",
        "```bash",
        f"PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples {args.samples} --random-actions {args.random_actions}",
        "```",
        "",
        "| Checkpoint | Track 3 class | Bottleneck label | True NLL | Random NLL | Random-true gap | Action-loss std | True best | Learned best | Operand exact | Output |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        summary = result["summary"]
        classification = result["classification"]
        lines.append(
            "| "
            + " | ".join(
                [
                    result["purpose"],
                    classification.get("category", "n/a"),
                    classification.get("bottleneck_classification", "n/a"),
                    format_float(summary.get("mean_true_nll")),
                    format_float(summary.get("mean_random_nll")),
                    format_float(summary.get("mean_random_minus_true_gap")),
                    format_float(summary.get("mean_action_loss_std")),
                    format_float(summary.get("true_best_fraction")),
                    format_float(summary.get("learned_best_fraction")),
                    format_float(summary.get("operand_exact_match")),
                    f"`{result['output_dir']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The diagnostic is intentionally checkpoint-first: it measures the action-loss landscape before changing estimators.",
            "- Existing additive checkpoints remain bypass baselines, and existing replacement checkpoints keep Track 3's invalid/leaky bottleneck label.",
            "- Positive random-true gaps mean true calculator actions lower target loss relative to sampled random actions; near-zero gaps mean the downstream answer loss has little operand-action signal to exploit.",
            "- Estimator comparisons should still wait for a stricter calculator-required bottleneck, because Track 2/3 showed the current replacement mode leaks through autoregressive answer-token context.",
            "",
            "## Validation",
            "",
            "```bash",
            "PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_track4_action_loss_diagnostic.py",
            "PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q",
            "PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples 4 --random-actions 2 --limit 1",
            f"PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_track4_action_loss_diagnostic.py --samples {args.samples} --random-actions {args.random_actions}",
            "```",
            "",
            "Conclusion: Track 4 now has a direct operand-action loss diagnostic. The next necessary step is a stricter bottleneck that passes Model B/off and oracle controls before estimator sweeps.",
        ]
    )
    WORK_HISTORY.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Track 4 operand-action loss sensitivity diagnostics."
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--random-actions", type=int, default=16)
    parser.add_argument("--limit", type=int, default=len(MANIFEST))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-work-history", action="store_true")
    parser.add_argument("--move-task-doc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is not None:
        if args.output_root == Path("auto"):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            args.output_root = REPO_ROOT / "runs" / f"{timestamp}_track4_action_loss"
        elif not args.output_root.is_absolute():
            args.output_root = REPO_ROOT / args.output_root
    results = run_manifest(args)
    if not args.dry_run and not args.no_work_history:
        write_work_history(results, args=args)
        if args.move_task_doc and TASK_DOC.exists():
            COMPLETED_TASK_DOC.parent.mkdir(parents=True, exist_ok=True)
            TASK_DOC.rename(COMPLETED_TASK_DOC)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
