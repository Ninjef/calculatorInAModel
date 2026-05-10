from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "runs" / "2026-05-10_phase6_identifiable_full_enum_local_target"
LOCAL_STAGE0B_CHECKPOINT = (
    REPO_ROOT
    / "runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-"
    "sum_left_operand/model-c-2digit-seed2/final_weights.pt"
)
RECORDED_STAGE0B_CHECKPOINT = Path(
    "/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/"
    "runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-"
    "sum_left_operand/model-c-2digit-seed2/final_weights.pt"
)
STAGE0B_CHECKPOINT = (
    LOCAL_STAGE0B_CHECKPOINT
    if LOCAL_STAGE0B_CHECKPOINT.exists()
    else RECORDED_STAGE0B_CHECKPOINT
)
PHASE4_RETAINED_POSITIVE = (
    REPO_ROOT
    / "runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/"
    "2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-inlr0.0003-"
    "uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/codex_pycache")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    with (RUN_ROOT / "commands.jsonl").open("a") as handle:
        handle.write(json.dumps({"command": command, "cwd": str(cwd)}) + "\n")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def phase6_train_command(
    *,
    checkpoint: Path,
    run_root: Path,
    estimator: str,
    steps: int,
    seed: int,
    freeze_upstream: bool,
    input_proj_lr: float,
    upstream_lr: float,
    local_target_weight: float,
    target_mode: str,
    snapshot_every: int,
    checkpoint_every: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/overfit_one_batch.py",
        "--variant",
        "model-c",
        "--digits",
        "2",
        "--operand-max",
        "19",
        "--calculator-operand-vocab-size",
        "20",
        "--n-layer",
        "2",
        "--n-head",
        "1",
        "--n-embd",
        "16",
        "--mlp-expansion",
        "1",
        "--calculator-hook-after-layer",
        "1",
        "--answer-format",
        "sum_left_operand",
        "--calculator-output-format",
        "sum_left_operand",
        "--calculator-read-position",
        "operand_spans",
        "--calculator-read-span-width",
        "2",
        "--calculator-bottleneck-mode",
        "answer_decoder",
        "--calculator-estimator",
        estimator,
        "--semantic-decoder-checkpoint",
        str(checkpoint),
        "--semantic-decoder-checkpoint-load-scope",
        "full_model",
        "--answer-loss-weight",
        "1.0",
        "--aux-operand-loss-weight",
        "0.0",
        "--adaptive-interface-loss-weight",
        str(local_target_weight),
        "--adaptive-interface-loss-decay-steps",
        "0",
        "--adaptive-interface-loss-floor",
        "0.0",
        "--input-proj-anchor-weight",
        "0.0",
        "--input-proj-lr",
        str(input_proj_lr),
        "--upstream-lr",
        str(upstream_lr),
        "--steps",
        str(steps),
        "--batch-size",
        "64",
        "--eval-samples",
        "512",
        "--snapshot-every",
        str(snapshot_every),
        "--snapshot-samples",
        "128",
        "--checkpoint-every",
        str(checkpoint_every),
        "--seed",
        str(seed),
        "--run-root",
        str(run_root),
    ]
    if estimator == "identifiable_full_enum_local_target":
        command.extend(
            [
                "--action-loss-full-enum-target-mode",
                target_mode,
                "--action-loss-full-enum-temperature",
                "0.25",
                "--action-loss-full-enum-chunk-size",
                "64",
            ]
        )
    if freeze_upstream:
        command.append("--freeze-upstream-encoder")
    return command


def summarize_target_pass(summary: dict[str, Any]) -> bool:
    return (
        summary["best_matches_true_operands_fraction"] >= 0.90
        and summary["mean_true_pair_rank"] <= 1.10
        and summary["mean_effective_pair_count"] < 10.0
    )


def cmd_summarize_target(args: argparse.Namespace) -> None:
    checkpoints = [STAGE0B_CHECKPOINT]
    if PHASE4_RETAINED_POSITIVE.exists():
        checkpoints.append(PHASE4_RETAINED_POSITIVE)
    output_root = RUN_ROOT / "target_sharpness"
    command = [
        sys.executable,
        "scripts/run_full_enum_action_loss_diagnostic.py",
        "--checkpoint",
        *[str(path) for path in checkpoints],
        "--samples",
        str(args.samples),
        "--batch-size",
        "64",
        "--digits",
        "2",
        "--operand-max",
        "19",
        "--answer-format",
        "sum_left_operand",
        "--calculator-output-format",
        "sum_left_operand",
        "--temperature",
        str(args.temperature),
        "--chunk-size",
        "64",
        "--seed",
        "0",
        "--output-root",
        str(output_root),
    ]
    run_command(command)
    summaries = load_json(output_root / "full_enum_summary_all.json")
    gate = {
        f"checkpoint_{index}_{Path(row['checkpoint']).parent.parent.name}": {
            "best_matches_true_operands_fraction": row[
                "best_matches_true_operands_fraction"
            ],
            "tie_aware_true_best_fraction": row["tie_aware_true_best_fraction"],
            "mean_true_pair_rank": row["mean_true_pair_rank"],
            "mean_effective_pair_count": row["mean_effective_pair_count"],
            "mean_soft_target_true_pair_probability": row[
                "mean_soft_target_true_pair_probability"
            ],
            "mean_top5_target_mass": row["mean_top5_target_mass"],
            "passes_gate": summarize_target_pass(row),
        }
        for index, row in enumerate(summaries)
    }
    write_json(RUN_ROOT / "target_sharpness_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


def latest_model_run(stage_root: Path) -> Path:
    matches = sorted(stage_root.glob("*/model-c-2digit-seed*/metrics.json"))
    if not matches:
        raise FileNotFoundError(f"no metrics.json found under {stage_root}")
    return matches[-1].parent


def best_snapshot_checkpoint(run_dir: Path, *, threshold: float) -> Path | None:
    snapshot_path = run_dir / "diagnostic_snapshots.csv"
    if not snapshot_path.exists():
        return None
    best: tuple[float, int] | None = None
    for row in read_rows(snapshot_path):
        score = min(
            float(row["operand_exact_match"]),
            float(row["pair_exact_match"]),
            float(row["calculator_result_accuracy"]),
        )
        step = int(row["step"])
        if best is None or score > best[0]:
            best = (score, step)
    if best is None or best[0] < threshold:
        return None
    checkpoint = run_dir / "checkpoint_snapshots" / f"step_{best[1]:05d}_weights.pt"
    return checkpoint if checkpoint.exists() else None


def target_gate_passed() -> bool:
    gate_path = RUN_ROOT / "target_sharpness_gate.json"
    if not gate_path.exists():
        return False
    gate = load_json(gate_path)
    return any(row.get("passes_gate") for row in gate.values())


def cmd_run_smoke(args: argparse.Namespace) -> None:
    if not target_gate_passed() and not args.force:
        raise SystemExit(
            "target sharpness gate has not passed; run summarize-target first "
            "or pass --force for development only"
        )
    stage1_root = RUN_ROOT / "stage1" / "frozen_upstream"
    if not args.skip_frozen:
        run_command(
            phase6_train_command(
                checkpoint=STAGE0B_CHECKPOINT,
                run_root=stage1_root,
                estimator="identifiable_full_enum_local_target",
                steps=args.steps,
                seed=0,
                freeze_upstream=True,
                input_proj_lr=args.input_proj_lr,
                upstream_lr=0.0003,
                local_target_weight=1.0,
                target_mode=args.target_mode,
                snapshot_every=args.snapshot_every,
                checkpoint_every=args.snapshot_every,
            )
        )
    elif not stage1_root.exists():
        raise SystemExit("--skip-frozen requested, but no frozen branch exists")
    stage1_run = latest_model_run(stage1_root)
    retention_start = best_snapshot_checkpoint(stage1_run, threshold=0.90)
    if retention_start is not None:
        stage2_root = RUN_ROOT / "stage2" / "frozen_upstream_retention"
        run_command(
            phase6_train_command(
                checkpoint=retention_start,
                run_root=stage2_root,
                estimator="adaptive_interface",
                steps=args.retention_steps,
                seed=0,
                freeze_upstream=True,
                input_proj_lr=0.0003,
                upstream_lr=0.0003,
                local_target_weight=0.0,
                target_mode=args.target_mode,
                snapshot_every=args.snapshot_every,
                checkpoint_every=args.snapshot_every,
            )
        )
    elif args.include_upstream:
        stage1_open_root = RUN_ROOT / "stage1" / "upstream_open"
        run_command(
            phase6_train_command(
                checkpoint=STAGE0B_CHECKPOINT,
                run_root=stage1_open_root,
                estimator="identifiable_full_enum_local_target",
                steps=1000,
                seed=0,
                freeze_upstream=False,
                input_proj_lr=0.0003,
                upstream_lr=0.00003,
                local_target_weight=1.0,
                target_mode=args.target_mode,
                snapshot_every=50,
                checkpoint_every=50,
            )
        )


def diagnostic_commands(checkpoint: Path, output_root: Path) -> list[list[str]]:
    return [
        [
            sys.executable,
            "scripts/run_causal_calculator_protocol_diagnostics.py",
            "--checkpoint",
            str(checkpoint),
            "--digits",
            "2",
            "--operand-max",
            "19",
            "--samples",
            "256",
            "--answer-format",
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
            "--output-dir",
            str(output_root / "canonical"),
        ],
        [
            sys.executable,
            "scripts/diagnose_private_protocol.py",
            "--checkpoint",
            str(checkpoint),
            "--digits",
            "2",
            "--operand-max",
            "19",
            "--answer-format",
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
            "--output-dir",
            str(output_root / "private"),
        ],
        [
            sys.executable,
            "scripts/run_full_enum_action_loss_diagnostic.py",
            "--checkpoint",
            str(checkpoint),
            "--samples",
            "128",
            "--batch-size",
            "64",
            "--digits",
            "2",
            "--operand-max",
            "19",
            "--answer-format",
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
            "--temperature",
            "0.25",
            "--chunk-size",
            "64",
            "--output-root",
            str(output_root / "full_enum"),
        ],
    ]


def selected_checkpoints() -> dict[str, Path]:
    checkpoints = {"stage0b": STAGE0B_CHECKPOINT}
    for label, root in [
        ("stage1_frozen_final", RUN_ROOT / "stage1" / "frozen_upstream"),
        ("stage2_retention_final", RUN_ROOT / "stage2" / "frozen_upstream_retention"),
        ("stage1_upstream_open_final", RUN_ROOT / "stage1" / "upstream_open"),
    ]:
        try:
            checkpoints[label] = latest_model_run(root) / "final_weights.pt"
        except FileNotFoundError:
            pass
    for label, root in [
        ("stage1_frozen_best_snapshot", RUN_ROOT / "stage1" / "frozen_upstream"),
        ("stage1_upstream_open_best_snapshot", RUN_ROOT / "stage1" / "upstream_open"),
    ]:
        try:
            run_dir = latest_model_run(root)
            best = best_snapshot_checkpoint(run_dir, threshold=0.35)
            if best is not None:
                checkpoints[label] = best
        except FileNotFoundError:
            pass
    return checkpoints


def cmd_diagnostics(args: argparse.Namespace) -> None:
    for label, checkpoint in selected_checkpoints().items():
        output_root = RUN_ROOT / "diagnostics" / label
        for command in diagnostic_commands(checkpoint, output_root):
            run_command(command)


def compact_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    snapshots = read_rows(run_dir / "diagnostic_snapshots.csv")
    best = max(
        snapshots,
        key=lambda row: min(
            float(row["operand_exact_match"]),
            float(row["pair_exact_match"]),
            float(row["calculator_result_accuracy"]),
        ),
    )
    return {
        "run_dir": str(run_dir),
        "final_exact_match": metrics.get("exact_match"),
        "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
        "final_local_target_loss_weight": metrics.get(
            "final_local_target_loss_weight"
        ),
        "final_adaptive_interface_loss_weight": metrics.get(
            "final_adaptive_interface_loss_weight"
        ),
        "final_input_proj_anchor_weight": metrics.get(
            "final_input_proj_anchor_weight"
        ),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        "best_snapshot": best,
    }


def cmd_summarize(args: argparse.Namespace) -> None:
    summary: dict[str, Any] = {
        "run_root": str(RUN_ROOT),
        "stage0b_checkpoint": str(STAGE0B_CHECKPOINT),
        "target_sharpness_gate": load_json(RUN_ROOT / "target_sharpness_gate.json")
        if (RUN_ROOT / "target_sharpness_gate.json").exists()
        else {},
        "runs": {},
    }
    for label, root in [
        ("stage1_frozen", RUN_ROOT / "stage1" / "frozen_upstream"),
        ("stage2_retention", RUN_ROOT / "stage2" / "frozen_upstream_retention"),
        ("stage1_upstream_open", RUN_ROOT / "stage1" / "upstream_open"),
    ]:
        try:
            summary["runs"][label] = compact_run(latest_model_run(root))
        except FileNotFoundError:
            pass
    write_json(RUN_ROOT / "summary.json", summary)
    lines = [
        "# Phase 6 Identifiable Full-Enum Local Target Summary",
        "",
        f"Run root: `{RUN_ROOT}`",
        f"Stage 0B checkpoint: `{STAGE0B_CHECKPOINT}`",
        "",
        "## Target Sharpness Gate",
    ]
    for label, row in summary["target_sharpness_gate"].items():
        lines.append(
            "- "
            f"{label}: best=true {row['best_matches_true_operands_fraction']:.3f}, "
            f"rank {row['mean_true_pair_rank']:.3f}, "
            f"effective pairs {row['mean_effective_pair_count']:.3f}, "
            f"true-pair prob {row['mean_soft_target_true_pair_probability']:.3f}, "
            f"pass={row['passes_gate']}"
        )
    lines.extend(["", "## Runs"])
    for label, row in summary["runs"].items():
        best = row["best_snapshot"]
        lines.append(
            "- "
            f"{label}: final exact {row['final_exact_match']}, "
            f"best step {best['step']} normal/operand/pair/calc "
            f"{float(best['normal_exact_match']):.3f}/"
            f"{float(best['operand_exact_match']):.3f}/"
            f"{float(best['pair_exact_match']):.3f}/"
            f"{float(best['calculator_result_accuracy']):.3f}; "
            f"aux={row['final_aux_operand_loss_weight']}, "
            f"local={row['final_local_target_loss_weight']}, "
            f"adaptive={row['final_adaptive_interface_loss_weight']}, "
            f"anchor={row['final_input_proj_anchor_weight']}"
        )
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 identifiable full-enum local-target runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    target = subparsers.add_parser("summarize-target")
    target.add_argument("--samples", type=int, default=128)
    target.add_argument("--temperature", type=float, default=0.25)

    smoke = subparsers.add_parser("run-smoke")
    smoke.add_argument("--steps", type=int, default=500)
    smoke.add_argument("--retention-steps", type=int, default=500)
    smoke.add_argument("--snapshot-every", type=int, default=50)
    smoke.add_argument("--input-proj-lr", type=float, default=0.001)
    smoke.add_argument(
        "--target-mode",
        choices=["hard_best_pair", "soft_pair"],
        default="hard_best_pair",
    )
    smoke.add_argument("--include-upstream", action="store_true")
    smoke.add_argument("--skip-frozen", action="store_true")
    smoke.add_argument("--force", action="store_true")

    subparsers.add_parser("diagnostics")
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "summarize-target":
        cmd_summarize_target(args)
    elif args.command == "run-smoke":
        cmd_run_smoke(args)
    elif args.command == "diagnostics":
        cmd_diagnostics(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
