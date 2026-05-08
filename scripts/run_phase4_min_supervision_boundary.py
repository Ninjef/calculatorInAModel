import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "runs" / "2026-05-07_phase4_min_supervision_boundary"
STAGE0B_CHECKPOINT = Path(
    "/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/"
    "runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-"
    "sum_left_operand/model-c-2digit-seed2/final_weights.pt"
)
SEEDS = {
    2: 0,
    4: 2,
    5: 3,
}
THRESHOLDS = [0.25, 0.50, 0.75, 0.90, 0.95]
PRIMARY_STAGE1_STEPS = [10, 25, 50, 75, 100, 125, 150]
RETENTION_THRESHOLD = 0.95


def common_args(*, steps: int, cli_seed: int, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/overfit_one_batch.py",
        "--variant",
        "model-c",
        "--digits",
        "2",
        "--steps",
        str(steps),
        "--batch-size",
        "64",
        "--eval-samples",
        "512",
        "--operand-max",
        "19",
        "--calculator-operand-vocab-size",
        "20",
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
        "adaptive_interface",
        "--semantic-decoder-checkpoint",
        str(checkpoint),
        "--freeze-semantic-decoder",
        "--freeze-upstream-encoder",
        "--input-proj-lr",
        "0.03",
        "--upstream-lr",
        "0.003",
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
        "--seed",
        str(cli_seed),
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(args: list[str], run_root: Path, log_name: str) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / log_name
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    full_args = args + ["--run-root", str(run_root)]
    with log_path.open("w") as log:
        subprocess.run(
            full_args,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    summary_paths = sorted(
        run_root.glob("*/summary_metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not summary_paths:
        raise RuntimeError(f"no summary_metrics.json under {run_root}")
    run_dir = Path(load_json(summary_paths[-1])["runs"][0]["run_dir"])
    return run_dir


def run_many(tasks: list[tuple[str, list[str], Path]], *, jobs: int) -> dict[str, str]:
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_command, command, root, f"{label}.log"): label
            for label, command, root in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            run_dir = future.result()
            print(f"{label}: {run_dir}", flush=True)
            results[label] = str(run_dir)
    return results


def stage1a(jobs: int) -> None:
    tasks: list[tuple[str, list[str], Path]] = []
    for effective_seed, cli_seed in SEEDS.items():
        args = common_args(steps=150, cli_seed=cli_seed, checkpoint=STAGE0B_CHECKPOINT)
        args += [
            "--answer-loss-weight",
            "0.0",
            "--adaptive-interface-loss-weight",
            "0.0",
            "--aux-operand-loss-weight",
            "1.0",
            "--aux-operand-loss-decay-steps",
            "0",
            "--snapshot-every",
            "5",
            "--checkpoint-every",
            "5",
            "--snapshot-samples",
            "256",
            "--log-every",
            "5",
        ]
        tasks.append(
            (
                f"stage1a_seed{effective_seed}",
                args,
                RUN_ROOT / "stage1a" / f"seed{effective_seed}",
            )
        )
    results = run_many(tasks, jobs=jobs)
    manifest = load_manifest()
    manifest.setdefault("stage1a", {}).update(results)
    write_manifest(manifest)


def checkpoint_for_row(run_dir: Path, row: dict[str, str]) -> str:
    step = int(row["step"])
    return str(run_dir / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt")


def summarize_stage1a(manifest: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label, run_dir_text in manifest.get("stage1a", {}).items():
        run_dir = Path(run_dir_text)
        rows = read_rows(run_dir / "diagnostic_snapshots.csv")
        by_step = {int(row["step"]): row for row in rows}
        thresholds: dict[str, Any] = {}
        for threshold in THRESHOLDS + [1.0]:
            selected = next(
                (
                    row
                    for row in rows
                    if float(row["operand_exact_match"]) >= threshold
                ),
                None,
            )
            key = f"{threshold:.2f}"
            thresholds[key] = (
                None
                if selected is None
                else {
                    "step": int(selected["step"]),
                    "checkpoint": checkpoint_for_row(run_dir, selected),
                    "operand_exact_match": float(selected["operand_exact_match"]),
                    "normal_exact_match": float(selected["normal_exact_match"]),
                    "calculator_result_accuracy": float(
                        selected["calculator_result_accuracy"]
                    ),
                }
            )
        primary = {}
        for step in PRIMARY_STAGE1_STEPS:
            row = by_step.get(step)
            if row is not None:
                primary[str(step)] = row_to_metrics(row)
        summary[label] = {
            "run_dir": str(run_dir),
            "metrics": fast_metrics(run_dir),
            "primary_steps": primary,
            "thresholds": thresholds,
        }
    return summary


def row_to_metrics(row: dict[str, str]) -> dict[str, float | int]:
    keys = [
        "normal_exact_match",
        "injection_zero_exact_match",
        "forced_random_exact_match",
        "oracle_exact_match",
        "operand_exact_match",
        "pair_exact_match",
        "calculator_result_accuracy",
        "mean_a_entropy",
        "mean_b_entropy",
        "mean_a_confidence",
        "mean_b_confidence",
    ]
    metrics: dict[str, float | int] = {"step": int(row["step"])}
    for key in keys:
        metrics[key] = float(row[key])
    return metrics


def fast_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    snapshots = read_rows(run_dir / "diagnostic_snapshots.csv")
    final_snapshot = row_to_metrics(snapshots[-1]) if snapshots else {}
    return {
        "run_dir": str(run_dir),
        "final_weights": str(run_dir / "final_weights.pt"),
        "final_snapshot": final_snapshot,
        "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
        "final_adaptive_interface_loss_weight": metrics.get(
            "final_adaptive_interface_loss_weight"
        ),
        "final_input_proj_anchor_weight": metrics.get("final_input_proj_anchor_weight"),
        "freeze_semantic_decoder": metrics.get("freeze_semantic_decoder"),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
    }


def selected_stage2_handoffs(stage1_summary: dict[str, Any]) -> list[dict[str, Any]]:
    handoffs: list[dict[str, Any]] = []
    for label, summary in stage1_summary.items():
        effective_seed = int(label.rsplit("seed", 1)[1])
        seen_checkpoints: dict[str, dict[str, Any]] = {}
        for threshold in THRESHOLDS:
            selected = summary["thresholds"][f"{threshold:.2f}"]
            if selected is None:
                continue
            checkpoint = selected["checkpoint"]
            entry = seen_checkpoints.setdefault(
                checkpoint,
                {
                    "effective_seed": effective_seed,
                    "cli_seed": SEEDS[effective_seed],
                    "checkpoint": checkpoint,
                    "stage1_step": selected["step"],
                    "stage1_operand_exact_match": selected["operand_exact_match"],
                    "thresholds": [],
                },
            )
            entry["thresholds"].append(threshold)
        handoffs.extend(seen_checkpoints.values())
    return handoffs


def stage2(jobs: int) -> None:
    manifest = load_manifest()
    stage1_summary = summarize_stage1a(manifest)
    handoffs = selected_stage2_handoffs(stage1_summary)
    tasks: list[tuple[str, list[str], Path]] = []
    for handoff in handoffs:
        effective_seed = handoff["effective_seed"]
        cli_seed = handoff["cli_seed"]
        step = handoff["stage1_step"]
        threshold_label = "_".join(
            f"{threshold:.2f}".replace(".", "p") for threshold in handoff["thresholds"]
        )
        label = f"stage2_seed{effective_seed}_step{step}_{threshold_label}"
        args = common_args(
            steps=1000,
            cli_seed=cli_seed,
            checkpoint=Path(handoff["checkpoint"]),
        )
        args += [
            "--input-proj-lr",
            "0.0003",
            "--upstream-lr",
            "0.0003",
            "--answer-loss-weight",
            "1.0",
            "--adaptive-interface-loss-weight",
            "0.0",
            "--aux-operand-loss-weight",
            "0.0",
            "--input-proj-anchor-weight",
            "0.0",
            "--snapshot-every",
            "50",
            "--checkpoint-every",
            "50",
            "--snapshot-samples",
            "256",
            "--log-every",
            "50",
        ]
        tasks.append(
            (
                label,
                args,
                RUN_ROOT / "stage2" / f"seed{effective_seed}" / f"step{step}",
            )
        )
    results = run_many(tasks, jobs=jobs)
    manifest.setdefault("stage2", {}).update(results)
    manifest["stage2_handoffs"] = handoffs
    write_manifest(manifest)


def stage2_lower(jobs: int) -> None:
    manifest = load_manifest()
    stage1_summary = summarize_stage1a(manifest)
    stage2_summary = summarize_stage2(manifest)
    tasks: list[tuple[str, list[str], Path]] = []
    for seed_label, rows in group_stage2_by_seed(stage2_summary).items():
        recovered_lowest = any(
            row["stage1_step"] == min(item["stage1_step"] for item in rows)
            and row["final_snapshot"]["operand_exact_match"] >= 0.95
            for row in rows
        )
        if not recovered_lowest:
            continue
        effective_seed = int(seed_label.rsplit("seed", 1)[1])
        stage1_run = Path(stage1_summary[f"stage1a_seed{effective_seed}"]["run_dir"])
        rows1 = read_rows(stage1_run / "diagnostic_snapshots.csv")
        lower = [
            row
            for row in rows1
            if float(row["operand_exact_match"]) < 0.25 and int(row["step"]) > 0
        ]
        if not lower:
            continue
        selected = lower[-1]
        step = int(selected["step"])
        label = f"stage2_lower_seed{effective_seed}_step{step}"
        args = common_args(
            steps=1000,
            cli_seed=SEEDS[effective_seed],
            checkpoint=Path(checkpoint_for_row(stage1_run, selected)),
        )
        args += [
            "--input-proj-lr",
            "0.0003",
            "--upstream-lr",
            "0.0003",
            "--answer-loss-weight",
            "1.0",
            "--adaptive-interface-loss-weight",
            "0.0",
            "--aux-operand-loss-weight",
            "0.0",
            "--input-proj-anchor-weight",
            "0.0",
            "--snapshot-every",
            "50",
            "--checkpoint-every",
            "50",
            "--snapshot-samples",
            "256",
            "--log-every",
            "50",
        ]
        tasks.append(
            (
                label,
                args,
                RUN_ROOT / "stage2_lower" / f"seed{effective_seed}" / f"step{step}",
            )
        )
    if not tasks:
        print("no lower Stage 2 handoffs selected", flush=True)
        return
    results = run_many(tasks, jobs=jobs)
    manifest.setdefault("stage2_lower", {}).update(results)
    write_manifest(manifest)


def group_stage2_by_seed(stage2_summary: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in stage2_summary:
        grouped.setdefault(f"seed{row['effective_seed']}", []).append(row)
    return grouped


def stage1b(jobs: int) -> None:
    tasks: list[tuple[str, list[str], Path]] = []
    for effective_seed, cli_seed in SEEDS.items():
        for decay in [25, 50, 100]:
            args = common_args(steps=300, cli_seed=cli_seed, checkpoint=STAGE0B_CHECKPOINT)
            args += [
                "--answer-loss-weight",
                "1.0",
                "--adaptive-interface-loss-weight",
                "0.0",
                "--aux-operand-loss-weight",
                "1.0",
                "--aux-operand-loss-decay-steps",
                str(decay),
                "--snapshot-every",
                "25",
                "--checkpoint-every",
                "25",
                "--snapshot-samples",
                "256",
                "--log-every",
                "25",
            ]
            tasks.append(
                (
                    f"stage1b_seed{effective_seed}_decay{decay}",
                    args,
                    RUN_ROOT / "stage1b" / f"seed{effective_seed}" / f"decay{decay}",
                )
            )
    results = run_many(tasks, jobs=jobs)
    manifest = load_manifest()
    manifest.setdefault("stage1b", {}).update(results)
    write_manifest(manifest)


def summarize_stage2(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_run_dir = {
        str(handoff["checkpoint"]): handoff for handoff in manifest.get("stage2_handoffs", [])
    }
    rows = []
    for label, run_dir_text in {
        **manifest.get("stage2", {}),
        **manifest.get("stage2_lower", {}),
    }.items():
        run_dir = Path(run_dir_text)
        metrics = fast_metrics(run_dir)
        config = load_json(run_dir / "config.json")
        source = by_run_dir.get(config["semantic_decoder_checkpoint"], {})
        effective_seed = source.get(
            "effective_seed", int(label.split("seed", 1)[1].split("_", 1)[0])
        )
        row = {
            "label": label,
            "run_dir": str(run_dir),
            "selected_checkpoint": str(run_dir / "final_weights.pt"),
            "effective_seed": effective_seed,
            "cli_seed": source.get("cli_seed", SEEDS[effective_seed]),
            "stage1_checkpoint": config["semantic_decoder_checkpoint"],
            "stage1_step": source.get("stage1_step", step_from_checkpoint(config["semantic_decoder_checkpoint"])),
            "stage1_operand_exact_match": source.get(
                "stage1_operand_exact_match",
                operand_exact_for_checkpoint(config["semantic_decoder_checkpoint"]),
            ),
            "thresholds": source.get("thresholds", []),
            **metrics,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["effective_seed"], row["stage1_step"]))


def step_from_checkpoint(path: str) -> int:
    name = Path(path).stem
    if name.startswith("step_") and name.endswith("_weights"):
        return int(name.removeprefix("step_").removesuffix("_weights"))
    return -1


def operand_exact_for_checkpoint(path: str) -> float | None:
    checkpoint = Path(path)
    step = step_from_checkpoint(path)
    if step < 0:
        return None
    snapshot_path = checkpoint.parent.parent / "diagnostic_snapshots.csv"
    if not snapshot_path.exists():
        return None
    for row in read_rows(snapshot_path):
        if int(row["step"]) == step:
            return float(row["operand_exact_match"])
    return None


def summarize_stage1b(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, run_dir_text in manifest.get("stage1b", {}).items():
        run_dir = Path(run_dir_text)
        config = load_json(run_dir / "config.json")
        effective_seed = int(label.split("seed", 1)[1].split("_", 1)[0])
        rows.append(
            {
                "label": label,
                "effective_seed": effective_seed,
                "cli_seed": SEEDS[effective_seed],
                "decay_steps": config["aux_operand_loss_decay_steps"],
                **fast_metrics(run_dir),
            }
        )
    return sorted(rows, key=lambda row: (row["effective_seed"], row["decay_steps"]))


def write_summary() -> None:
    manifest = load_manifest()
    summary = {
        "stage1a": summarize_stage1a(manifest),
        "stage2": summarize_stage2(manifest),
        "stage1b": summarize_stage1b(manifest),
    }
    write_json(RUN_ROOT / "summary.json", summary)
    print(RUN_ROOT / "summary.json")


def diagnostic_selections(summary: dict[str, Any]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    stage2_rows = summary.get("stage2", [])
    for effective_seed in sorted({row["effective_seed"] for row in stage2_rows}):
        rows = [
            row for row in stage2_rows if row["effective_seed"] == effective_seed
        ]
        retained = [
            row
            for row in rows
            if row["final_snapshot"]["operand_exact_match"] >= RETENTION_THRESHOLD
        ]
        if retained:
            retained_row = min(retained, key=lambda row: row["stage1_step"])
            selections.append(
                {
                    "kind": f"seed{effective_seed}_lowest_retained",
                    "label": retained_row["label"],
                    "checkpoint": retained_row["selected_checkpoint"],
                    "run_dir": retained_row["run_dir"],
                }
            )
            failures_below = [
                row
                for row in rows
                if row["stage1_step"] < retained_row["stage1_step"]
                and row["final_snapshot"]["operand_exact_match"] < RETENTION_THRESHOLD
            ]
            if failures_below:
                failure_row = max(failures_below, key=lambda row: row["stage1_step"])
                selections.append(
                    {
                        "kind": f"seed{effective_seed}_nearest_failed_below",
                        "label": failure_row["label"],
                        "checkpoint": failure_row["selected_checkpoint"],
                        "run_dir": failure_row["run_dir"],
                    }
                )

    stage1b_rows = summary.get("stage1b", [])
    if stage1b_rows:
        best_stage1b = max(
            stage1b_rows,
            key=lambda row: row["final_snapshot"]["operand_exact_match"],
        )
        selections.append(
            {
                "kind": "stage1b_best_decayed",
                "label": best_stage1b["label"],
                "checkpoint": best_stage1b["final_weights"],
                "run_dir": best_stage1b["run_dir"],
            }
        )
    return selections


def run_diagnostic_command(args: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w") as log:
        subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def diagnostics() -> None:
    write_summary()
    summary = load_json(RUN_ROOT / "summary.json")
    selections = diagnostic_selections(summary)
    write_json(RUN_ROOT / "diagnostic_selections.json", selections)
    for selection in selections:
        run_dir = Path(selection["run_dir"])
        checkpoint = Path(selection["checkpoint"])
        prefix = selection["kind"]
        print(f"{prefix}: {checkpoint}", flush=True)
        run_diagnostic_command(
            [
                sys.executable,
                "-m",
                "scripts.run_causal_calculator_protocol_diagnostics",
                "--checkpoint",
                str(checkpoint),
                "--samples",
                "256",
                "--digits",
                "2",
                "--operand-max",
                "19",
                "--answer-format",
                "sum_left_operand",
                "--calculator-output-format",
                "sum_left_operand",
                "--forced-result-sweep",
                "--forced-result-batch-size",
                "64",
                "--output-dir",
                str(run_dir / f"{prefix}_canonical_causal_diagnostics"),
            ],
            run_dir / f"{prefix}_canonical_causal_diagnostics.log",
        )
        run_diagnostic_command(
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
                str(run_dir / f"{prefix}_private_protocol_diagnostics"),
            ],
            run_dir / f"{prefix}_private_protocol_diagnostics.log",
        )
        run_diagnostic_command(
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
                "1.0",
                "--chunk-size",
                "64",
                "--output-root",
                str(run_dir / f"{prefix}_full_enum_action_loss"),
            ],
            run_dir / f"{prefix}_full_enum_action_loss.log",
        )


def load_manifest() -> dict[str, Any]:
    path = RUN_ROOT / "manifest.json"
    if path.exists():
        return load_json(path)
    return {}


def write_manifest(manifest: dict[str, Any]) -> None:
    write_json(RUN_ROOT / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "stage1a",
            "stage2",
            "stage2-lower",
            "stage1b",
            "summarize",
            "diagnostics",
        ],
    )
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "stage1a":
        stage1a(args.jobs)
    elif args.stage == "stage2":
        stage2(args.jobs)
    elif args.stage == "stage2-lower":
        stage2_lower(args.jobs)
    elif args.stage == "stage1b":
        stage1b(args.jobs)
    elif args.stage == "summarize":
        write_summary()
    elif args.stage == "diagnostics":
        diagnostics()


if __name__ == "__main__":
    main()
