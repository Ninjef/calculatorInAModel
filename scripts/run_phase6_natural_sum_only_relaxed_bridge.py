from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "runs/2026-05-12_phase6_natural_sum_only_relaxed_bridge"
STAGE0_SUM_CHECKPOINT = (
    REPO_ROOT
    / "runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/"
    "model-c-2digit-seed2/final_weights.pt"
)
CLI_SEEDS = [0, 2, 3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run_command(args: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    with log_path.open("w") as log:
        log.write("$ " + " ".join(args) + "\n\n")
        subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def latest_run_dir(run_root: Path) -> Path:
    summaries = sorted(
        run_root.glob("*/summary_metrics.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not summaries:
        raise RuntimeError(f"no summary_metrics.json under {run_root}")
    return Path(load_json(summaries[-1])["runs"][0]["run_dir"])


def row_to_metrics(row: dict[str, str]) -> dict[str, float | int]:
    keys = [
        "normal_exact_match",
        "injection_zero_exact_match",
        "forced_zero_exact_match",
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
        if key in row and row[key] != "":
            metrics[key] = float(row[key])
    return metrics


def common_overfit_args(
    *,
    seed: int,
    steps: int,
    checkpoint: Path,
    load_scope: str,
    estimator: str,
    freeze_upstream: bool,
    input_proj_lr: float,
    upstream_lr: float,
    snapshot_every: int,
    checkpoint_every: int,
    snapshot_samples: int,
    run_root: Path,
) -> list[str]:
    args = [
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
        "--calculator-estimator",
        estimator,
        "--calculator-action-head",
        "independent_operands",
        "--semantic-decoder-checkpoint",
        str(checkpoint),
        "--semantic-decoder-checkpoint-load-scope",
        load_scope,
        "--freeze-semantic-decoder",
        "--answer-loss-weight",
        "1.0",
        "--adaptive-interface-loss-weight",
        "0.0",
        "--aux-operand-loss-weight",
        "0.0",
        "--expected-answer-loss-weight",
        "0.0",
        "--input-proj-anchor-weight",
        "0.0",
        "--input-proj-lr",
        f"{input_proj_lr:g}",
        "--upstream-lr",
        f"{upstream_lr:g}",
        "--calculator-read-position",
        "operand_spans",
        "--calculator-read-span-width",
        "2",
        "--calculator-bottleneck-mode",
        "answer_decoder",
        "--calculator-output-format",
        "sum",
        "--answer-format",
        "sum",
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
        str(seed),
        "--snapshot-every",
        str(snapshot_every),
        "--checkpoint-every",
        str(checkpoint_every),
        "--snapshot-samples",
        str(snapshot_samples),
        "--run-root",
        str(run_root),
        "--log-every",
        str(max(snapshot_every, 1)),
    ]
    if freeze_upstream:
        args.append("--freeze-upstream-encoder")
    return args


def relaxed_args(
    *,
    seed: int,
    run_root: Path,
    freeze_upstream: bool = True,
    upstream_lr: float = 0.0003,
) -> list[str]:
    args = common_overfit_args(
        seed=seed,
        steps=300,
        checkpoint=STAGE0_SUM_CHECKPOINT,
        load_scope="semantic_decoder_only",
        estimator="gumbel_concrete_interface",
        freeze_upstream=freeze_upstream,
        input_proj_lr=0.03,
        upstream_lr=upstream_lr,
        snapshot_every=25,
        checkpoint_every=25,
        snapshot_samples=128,
        run_root=run_root,
    )
    args.extend(
        [
            "--relaxed-calculator-mode",
            "deterministic",
            "--relaxed-calculator-hard-forward",
            "--relaxed-calculator-temperature",
            "2.0",
            "--relaxed-calculator-final-temperature",
            "0.5",
            "--relaxed-calculator-temperature-decay-steps",
            "300",
            "--relaxed-calculator-entropy-weight",
            "0.0",
        ]
    )
    return args


def retention_args(
    *,
    seed: int,
    checkpoint: Path,
    run_root: Path,
    freeze_upstream: bool = True,
    upstream_lr: float = 0.0003,
) -> list[str]:
    return common_overfit_args(
        seed=seed,
        steps=1000,
        checkpoint=checkpoint,
        load_scope="full_model",
        estimator="adaptive_interface",
        freeze_upstream=freeze_upstream,
        input_proj_lr=0.0003,
        upstream_lr=upstream_lr,
        snapshot_every=50,
        checkpoint_every=50,
        snapshot_samples=128,
        run_root=run_root,
    )


def run_parallel_tasks(
    tasks: list[tuple[str, list[str], Path]], *, jobs: int
) -> dict[str, str]:
    results: dict[str, str] = {}
    remaining: list[tuple[str, list[str], Path]] = []
    for label, command, root in tasks:
        existing = sorted(root.glob("*/summary_metrics.json"))
        if existing:
            run_dir = Path(load_json(existing[-1])["runs"][0]["run_dir"])
            print(f"{label}: already complete at {run_dir}", flush=True)
            results[label] = str(run_dir)
        else:
            remaining.append((label, command, root))
    if not remaining:
        return results
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_command, command, log_path=root / f"{label}.log"): (
                label,
                root,
            )
            for label, command, root in remaining
        }
        for future in as_completed(futures):
            label, root = futures[future]
            future.result()
            run_dir = latest_run_dir(root)
            print(f"{label}: {run_dir}", flush=True)
            results[label] = str(run_dir)
    return results


def checkpoint_for_step(run_dir: Path, step: int) -> Path:
    cfg = load_json(run_dir / "config.json")
    if step == int(cfg.get("steps", -1)):
        return run_dir / "final_weights.pt"
    return run_dir / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"


def final_objective_weights(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_loss_weight": metrics.get("answer_loss_weight"),
        "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
        "final_adaptive_interface_loss_weight": metrics.get(
            "final_adaptive_interface_loss_weight"
        ),
        "final_local_target_loss_weight": metrics.get("final_local_target_loss_weight"),
        "final_expected_answer_loss_weight": metrics.get(
            "final_expected_answer_loss_weight"
        ),
        "final_relaxed_calculator_entropy_weight": metrics.get(
            "final_relaxed_calculator_entropy_weight"
        ),
        "final_input_proj_anchor_weight": metrics.get(
            "final_input_proj_anchor_weight"
        ),
    }


def analyze_stage1_run(run_dir: Path, threshold: float = 0.95) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    cfg = load_json(run_dir / "config.json")
    effective_seed = int(cfg["seed"])
    snapshots = [
        row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")
    ]
    qualifying = [
        row
        for row in snapshots
        if min(
            float(row.get("normal_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
        )
        >= threshold
    ]
    best = max(
        snapshots,
        key=lambda row: (
            float(row.get("calculator_result_accuracy", 0.0)),
            float(row.get("normal_exact_match", 0.0)),
            float(row.get("pair_exact_match", 0.0)),
            -int(row["step"]),
        ),
    )
    first = qualifying[0] if qualifying else None
    return {
        "run_dir": str(run_dir),
        "effective_seed": effective_seed,
        "cli_seed": effective_seed - 2,
        "final_eval_exact_match": metrics.get("exact_match"),
        "final_snapshot": snapshots[-1],
        "first_gate": first,
        "best_snapshot": best,
        "first_gate_checkpoint": (
            str(checkpoint_for_step(run_dir, int(first["step"]))) if first else None
        ),
        "best_checkpoint": str(checkpoint_for_step(run_dir, int(best["step"]))),
        "passes_fast_gate": first is not None,
        "final_objective_weights": final_objective_weights(metrics),
        "freeze_semantic_decoder": metrics.get("freeze_semantic_decoder"),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
    }


def analyze_retention_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    cfg = load_json(run_dir / "config.json")
    snapshots = [
        row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")
    ]
    qualifying = [
        row
        for row in snapshots
        if min(
            float(row.get("normal_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
        )
        >= 0.99
    ]
    best = max(
        snapshots,
        key=lambda row: (
            float(row.get("calculator_result_accuracy", 0.0)),
            float(row.get("normal_exact_match", 0.0)),
            float(row.get("pair_exact_match", 0.0)),
            int(row["step"]),
        ),
    )
    selected = snapshots[-1] if snapshots[-1] in qualifying else (
        qualifying[-1] if qualifying else best
    )
    checkpoint = checkpoint_for_step(run_dir, int(selected["step"]))
    return {
        "run_dir": str(run_dir),
        "source_checkpoint": metrics.get("semantic_decoder_checkpoint"),
        "effective_seed": int(cfg["seed"]),
        "selected_snapshot": selected,
        "selected_checkpoint": str(checkpoint),
        "best_snapshot": best,
        "final_snapshot": snapshots[-1],
        "final_eval_exact_match": metrics.get("exact_match"),
        "passes_fast_retention": bool(qualifying),
        "final_objective_weights": final_objective_weights(metrics),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
    }


def latest_summaries_by_parent(paths: Any) -> list[Path]:
    latest: dict[Path, Path] = {}
    for path in sorted(paths):
        parent = path.parent.parent
        previous = latest.get(parent)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest[parent] = path
    return sorted(latest.values())


def stage1_runs() -> list[dict[str, Any]]:
    rows = []
    for summary in sorted((RUN_ROOT / "stage1").glob("cli_seed*/*/summary_metrics.json")):
        rows.append(analyze_stage1_run(Path(load_json(summary)["runs"][0]["run_dir"])))
    rows.sort(key=lambda row: int(row["effective_seed"]))
    return rows


def retention_runs(root_name: str = "stage2_retention") -> list[dict[str, Any]]:
    rows = []
    for summary in latest_summaries_by_parent(
        (RUN_ROOT / root_name).glob("*/*/summary_metrics.json")
    ):
        rows.append(analyze_retention_run(Path(load_json(summary)["runs"][0]["run_dir"])))
    rows.sort(key=lambda row: row["run_dir"])
    return rows


def stage4_stage1_runs() -> list[dict[str, Any]]:
    rows = []
    for summary in latest_summaries_by_parent(
        (RUN_ROOT / "stage4_upstream_open/stage1_cli_seed0").glob(
            "*/summary_metrics.json"
        )
    ):
        rows.append(analyze_stage1_run(Path(load_json(summary)["runs"][0]["run_dir"])))
    return rows


def stage4_retention_runs() -> list[dict[str, Any]]:
    rows = []
    for summary in latest_summaries_by_parent(
        (RUN_ROOT / "stage4_upstream_open").glob(
            "retention_*/*/summary_metrics.json"
        )
    ):
        row = analyze_retention_run(Path(load_json(summary)["runs"][0]["run_dir"]))
        row["branch"] = summary.parents[1].name
        rows.append(row)
    rows.sort(key=lambda row: row["branch"])
    return rows


def checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }


def parameter_group_for_name(name: str) -> str:
    if name.startswith("calculator_hook.input_proj."):
        return "calculator_hook.input_proj"
    if name.startswith(("calculator_hook.output_proj.", "answer_offset_emb.", "answer_decoder.")):
        return "semantic_decoder"
    if name.startswith(("tok_emb.", "pos_emb.", "blocks.", "ln_f.", "lm_head.")):
        return "upstream_encoder"
    return "other"


def checkpoint_delta_summary(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = checkpoint_state_dict(before_path)
    after = checkpoint_state_dict(after_path)
    groups: dict[str, dict[str, Any]] = {}
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None or before_tensor.shape != after_tensor.shape:
            continue
        group = parameter_group_for_name(name)
        row = groups.setdefault(
            group,
            {"l2": 0.0, "max_abs": 0.0, "changed_tensor_count": 0, "tensor_count": 0},
        )
        delta = (after_tensor.float() - before_tensor.float()).reshape(-1)
        row["l2"] += float(torch.dot(delta, delta).item())
        row["max_abs"] = max(float(row["max_abs"]), float(delta.abs().max().item()))
        row["changed_tensor_count"] += int(delta.abs().max().item() > 0.0)
        row["tensor_count"] += 1
    for row in groups.values():
        row["l2"] = math.sqrt(float(row["l2"]))
    return {
        "before_checkpoint": str(before_path),
        "after_checkpoint": str(after_path),
        "groups": groups,
    }


def add_parameter_deltas(summary: dict[str, Any]) -> None:
    for section in ["stage1", "stage4_upstream_open_stage1"]:
        for row in summary[section]:
            step0 = Path(row["run_dir"]) / "checkpoint_snapshots/step_00000_weights.pt"
            if step0.exists():
                row["parameter_delta_step0_to_best"] = checkpoint_delta_summary(
                    step0, Path(row["best_checkpoint"])
                )


def stage0() -> None:
    if not STAGE0_SUM_CHECKPOINT.exists():
        raise FileNotFoundError(STAGE0_SUM_CHECKPOINT)
    root = RUN_ROOT / "stage0" / "semantic_only_seed0"
    if not sorted(root.glob("*/summary_metrics.json")):
        run_command(
            common_overfit_args(
                seed=0,
                steps=0,
                checkpoint=STAGE0_SUM_CHECKPOINT,
                load_scope="semantic_decoder_only",
                estimator="adaptive_interface",
                freeze_upstream=True,
                input_proj_lr=0.03,
                upstream_lr=0.0003,
                snapshot_every=1,
                checkpoint_every=1,
                snapshot_samples=128,
                run_root=root,
            ),
            log_path=root / "stage0.log",
        )
    run_dir = latest_run_dir(root)
    checkpoint = run_dir / "final_weights.pt"
    full_enum_dir = RUN_ROOT / "stage0" / "full_enum"
    run_command(
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
            "sum",
            "--calculator-output-format",
            "sum",
            "--temperature",
            "1.0",
            "--chunk-size",
            "64",
            "--output-root",
            str(full_enum_dir),
        ],
        log_path=RUN_ROOT / "stage0" / "full_enum.log",
    )
    snapshot = row_to_metrics(read_rows(run_dir / "diagnostic_snapshots.csv")[0])
    metrics = load_json(run_dir / "metrics.json")
    full_enum_summary = next(full_enum_dir.glob("*/full_enum_summary.json"))
    full_enum = load_json(full_enum_summary)
    semantic_delta = checkpoint_delta_summary(STAGE0_SUM_CHECKPOINT, checkpoint)[
        "groups"
    ].get("semantic_decoder", {"l2": 0.0, "max_abs": 0.0})
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "source_semantic_checkpoint": str(STAGE0_SUM_CHECKPOINT),
        "snapshot": snapshot,
        "final_objective_weights": final_objective_weights(metrics),
        "full_enum": {
            "best_result_matches_true_sum_fraction": full_enum.get(
                "best_result_matches_true_sum_fraction"
            ),
            "best_result_group_matches_true_sum_fraction": full_enum.get(
                "best_result_group_matches_true_sum_fraction"
            ),
            "true_best_fraction": full_enum.get("true_best_fraction"),
            "best_matches_true_operands_fraction": full_enum.get(
                "best_matches_true_operands_fraction"
            ),
            "true_result_best_fraction": full_enum.get("true_result_best_fraction"),
            "learned_result_best_fraction": full_enum.get(
                "learned_result_best_fraction"
            ),
            "mean_learned_result_minus_best_result_gap": full_enum.get(
                "mean_learned_result_minus_best_result_gap"
            ),
            "learned_result_matches_true_sum_fraction": full_enum.get(
                "learned_result_matches_true_sum_fraction"
            ),
            "mean_same_true_sum_near_best_pair_count": full_enum.get(
                "mean_same_true_sum_near_best_pair_count"
            ),
            "mean_same_best_sum_near_best_pair_count": full_enum.get(
                "mean_same_best_sum_near_best_pair_count"
            ),
            "mean_effective_pair_count": full_enum.get("mean_effective_pair_count"),
            "mean_effective_result_count": full_enum.get(
                "mean_effective_result_count"
            ),
        },
        "semantic_decoder_delta": semantic_delta,
        "passes_wiring_gate": (
            float(snapshot.get("oracle_exact_match", 0.0)) >= 0.98
            and float(snapshot.get("injection_zero_exact_match", 1.0)) <= 0.10
            and float(snapshot.get("forced_random_exact_match", 1.0)) <= 0.10
            and float(
                full_enum.get("best_result_group_matches_true_sum_fraction", 0.0)
            )
            >= 0.98
            and float(semantic_delta.get("l2", 1.0)) == 0.0
        ),
    }
    write_json(RUN_ROOT / "stage0_summary.json", summary)
    write_summary()


def stage1(jobs: int) -> None:
    if not STAGE0_SUM_CHECKPOINT.exists():
        raise FileNotFoundError(STAGE0_SUM_CHECKPOINT)
    tasks = []
    for seed in CLI_SEEDS:
        root = RUN_ROOT / "stage1" / f"cli_seed{seed}"
        tasks.append(
            (
                f"cli_seed{seed}",
                relaxed_args(seed=seed, run_root=root, freeze_upstream=True),
                root,
            )
        )
    run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


def stage2(jobs: int) -> None:
    tasks = []
    for row in stage1_runs():
        if not row["passes_fast_gate"]:
            continue
        seed = int(row["cli_seed"])
        first_checkpoint = Path(row["first_gate_checkpoint"])
        best_checkpoint = Path(row["best_checkpoint"])
        sources = [("first", first_checkpoint)]
        if best_checkpoint != first_checkpoint:
            sources.append(("best", best_checkpoint))
        for source_label, checkpoint in sources:
            root = (
                RUN_ROOT
                / "stage2_retention"
                / f"seed{row['effective_seed']}_{source_label}_step{checkpoint.stem.split('_')[1]}"
            )
            tasks.append(
                (
                    f"stage2_seed{row['effective_seed']}_{source_label}",
                    retention_args(seed=seed, checkpoint=checkpoint, run_root=root),
                    root,
                )
            )
    if not tasks:
        print("No Stage 1 passing seeds; skipping Stage 2.", flush=True)
        write_summary()
        return
    run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


def stage3(jobs: int) -> None:
    retained = {
        int(row["effective_seed"])
        for row in retention_runs()
        if row["passes_fast_retention"]
    }
    if 2 not in retained:
        print("Seed 2 did not pass retention; skipping replication.", flush=True)
        write_summary()
        return
    missing = [seed for seed in CLI_SEEDS if seed + 2 not in {row["effective_seed"] for row in stage1_runs()}]
    if not missing:
        print("Stage 1 already ran all configured replication seeds.", flush=True)
    else:
        tasks = []
        for seed in missing:
            root = RUN_ROOT / "stage1" / f"cli_seed{seed}"
            tasks.append(
                (
                    f"cli_seed{seed}",
                    relaxed_args(seed=seed, run_root=root, freeze_upstream=True),
                    root,
                )
            )
        run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


def stage4(jobs: int) -> None:
    retained = {
        int(row["effective_seed"])
        for row in retention_runs()
        if row["passes_fast_retention"]
    }
    if len(retained) < 2:
        print("Need at least two retained seeds before upstream-open stress.", flush=True)
        write_summary()
        return
    root = RUN_ROOT / "stage4_upstream_open" / "stage1_cli_seed0"
    run_parallel_tasks(
        [
            (
                "upstream_open_cli_seed0",
                relaxed_args(
                    seed=0,
                    run_root=root,
                    freeze_upstream=False,
                    upstream_lr=0.00003,
                ),
                root,
            )
        ],
        jobs=1,
    )
    row = analyze_stage1_run(latest_run_dir(root))
    if row["passes_fast_gate"]:
        first = Path(row["first_gate_checkpoint"])
        best = Path(row["best_checkpoint"])
        tasks = []
        sources = [("first", first)]
        if best != first:
            sources.append(("best", best))
        for source_label, checkpoint in sources:
            for frozen in [True, False]:
                label = "frozen_retention" if frozen else "open_retention"
                ret_root = (
                    RUN_ROOT
                    / "stage4_upstream_open"
                    / f"retention_{source_label}_{label}"
                )
                tasks.append(
                    (
                        f"{source_label}_{label}",
                        retention_args(
                            seed=0,
                            checkpoint=checkpoint,
                            run_root=ret_root,
                            freeze_upstream=frozen,
                            upstream_lr=0.00003,
                        ),
                        ret_root,
                    )
                )
        run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


def selection(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    checkpoint = row.get("selected_checkpoint") or row.get("best_checkpoint")
    run_dir = Path(row["run_dir"])
    safe = prefix + "_" + run_dir.parent.name
    return {
        "kind": safe,
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "canonical_dir_name": safe + "_canonical",
        "private_dir_name": safe + "_private",
        "full_enum_dir_name": safe + "_full_enum",
    }


def diagnostic_selections(summary: dict[str, Any]) -> list[dict[str, Any]]:
    selections = []
    for row in summary["stage1"]:
        if row["passes_fast_gate"]:
            selections.append(selection(row, "stage1_selected"))
        else:
            selections.append(selection(row, "stage1_failure_best"))
    for row in summary["stage2_retention"]:
        selections.append(selection(row, "stage2_selected"))
    for row in summary["stage4_upstream_open_stage1"]:
        selections.append(selection(row, "stage4_upstream_selected"))
    for row in summary["stage4_upstream_open_retention"]:
        selections.append(selection(row, f"stage4_{row['branch']}_selected"))
    return selections


def run_diagnostic_commands(item: dict[str, Any], checkpoint: Path, run_dir: Path) -> None:
    canonical_dir = run_dir / item["canonical_dir_name"]
    private_dir = run_dir / item["private_dir_name"]
    full_enum_dir = run_dir / item["full_enum_dir_name"]
    run_command(
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
            "sum",
            "--calculator-output-format",
            "sum",
            "--forced-result-sweep",
            "--forced-result-batch-size",
            "64",
            "--output-dir",
            str(canonical_dir),
        ],
        log_path=run_dir / f"{item['canonical_dir_name']}.log",
    )
    run_command(
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
            "sum",
            "--calculator-output-format",
            "sum",
            "--output-dir",
            str(private_dir),
        ],
        log_path=run_dir / f"{item['private_dir_name']}.log",
    )
    run_command(
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
            "sum",
            "--calculator-output-format",
            "sum",
            "--temperature",
            "1.0",
            "--chunk-size",
            "64",
            "--output-root",
            str(full_enum_dir),
        ],
        log_path=run_dir / f"{item['full_enum_dir_name']}.log",
    )


def diagnostics() -> None:
    write_summary()
    summary = load_json(RUN_ROOT / "summary.json")
    selections = diagnostic_selections(summary)
    write_json(RUN_ROOT / "diagnostic_selections.json", selections)
    for item in selections:
        checkpoint = Path(item["checkpoint"])
        run_dir = Path(item["run_dir"])
        print(f"{item['kind']}: {checkpoint}", flush=True)
        run_diagnostic_commands(item, checkpoint, run_dir)
    write_summary()


def counterfactual(summary: dict[str, Any], condition: str) -> float | None:
    for row in summary.get("counterfactual_exact_match", []):
        if row.get("condition") == condition:
            return row.get("exact_match")
    return None


def compact_canonical(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "normal_exact_match": summary.get("exact_match"),
        "operand_exact_match": summary.get("operand_exact_match"),
        "pair_exact_match": summary.get("pair_exact_match"),
        "calculator_result_accuracy": summary.get("calculator_result_accuracy"),
        "injection_zero_exact_match": counterfactual(summary, "injection_zero"),
        "forced_random_exact_match": counterfactual(summary, "forced_random"),
        "oracle_at_eval_exact_match": counterfactual(summary, "oracle_at_eval"),
    }


def compact_private(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_exact_match": summary.get("answer_exact_match"),
        "operand_exact_match": summary.get("operand_exact_match"),
        "pair_exact_match": summary.get("pair_exact_match"),
        "calculator_result_accuracy": summary.get("calculator_result_accuracy"),
        "mapped_operand_exact_match": summary.get("mapped_operand_exact_match"),
    }


def compact_full_enum(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_learned_minus_best_gap": summary.get("mean_learned_minus_best_gap"),
        "learned_best_fraction": summary.get("learned_best_fraction"),
        "best_matches_true_operands_fraction": summary.get(
            "best_matches_true_operands_fraction"
        ),
        "best_result_matches_true_sum_fraction": summary.get(
            "best_result_matches_true_sum_fraction"
        ),
        "best_result_group_matches_true_sum_fraction": summary.get(
            "best_result_group_matches_true_sum_fraction"
        ),
        "learned_result_matches_true_sum_fraction": summary.get(
            "learned_result_matches_true_sum_fraction"
        ),
        "learned_result_best_fraction": summary.get("learned_result_best_fraction"),
        "mean_learned_result_minus_best_result_gap": summary.get(
            "mean_learned_result_minus_best_result_gap"
        ),
        "mean_same_true_sum_near_best_pair_count": summary.get(
            "mean_same_true_sum_near_best_pair_count"
        ),
        "mean_effective_result_count": summary.get("mean_effective_result_count"),
    }


def find_summary(root: Path, filename: str) -> Path | None:
    matches = sorted(root.glob(f"**/{filename}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple {filename} under {root}: {matches}")
    return matches[0]


def collect_diagnostics() -> list[dict[str, Any]]:
    selections_path = RUN_ROOT / "diagnostic_selections.json"
    if not selections_path.exists():
        return []
    rows = []
    for item in load_json(selections_path):
        run_dir = Path(item["run_dir"])
        canonical = run_dir / item["canonical_dir_name"] / "diagnostic_summary.json"
        private = run_dir / item["private_dir_name"] / "private_protocol_summary.json"
        full_enum = find_summary(run_dir / item["full_enum_dir_name"], "full_enum_summary.json")
        row = {
            **item,
            "complete": canonical.exists() and private.exists() and full_enum is not None,
        }
        if row["complete"]:
            row["canonical"] = compact_canonical(load_json(canonical))
            row["private"] = compact_private(load_json(private))
            row["full_enum"] = compact_full_enum(load_json(full_enum))
        rows.append(row)
    return rows


def retained_seed_count(summary: dict[str, Any]) -> int:
    return len(
        {
            int(row["effective_seed"])
            for row in summary["stage2_retention"]
            if row["passes_fast_retention"]
        }
    )


def interpretation_labels(summary: dict[str, Any]) -> list[str]:
    labels = []
    stage1_passes = [row for row in summary["stage1"] if row["passes_fast_gate"]]
    retained = retained_seed_count(summary)
    if retained >= 2:
        labels.append("natural_sum_only_positive")
        labels.append("natural_sum_only_seed_robust_positive")
    elif retained == 1:
        labels.append("natural_sum_only_fragile")
    elif stage1_passes:
        labels.append("natural_sum_only_retention_failure")
    elif summary["stage1"]:
        labels.append("natural_sum_only_negative")
    if any(row["passes_fast_gate"] for row in summary["stage4_upstream_open_stage1"]):
        if any(
            row["passes_fast_retention"] and not row["freeze_upstream_encoder"]
            for row in summary["stage4_upstream_open_retention"]
        ):
            labels.append("natural_sum_only_upstream_open_positive")
        else:
            labels.append("natural_sum_only_upstream_open_instability")
    return labels


def metric_triple(row: dict[str, Any]) -> str:
    return "/".join(
        fmt(row.get(key))
        for key in [
            "normal_exact_match",
            "pair_exact_match",
            "calculator_result_accuracy",
        ]
    )


def write_summary() -> None:
    summary = {
        "run_root": str(RUN_ROOT),
        "stage0": load_json(RUN_ROOT / "stage0_summary.json")
        if (RUN_ROOT / "stage0_summary.json").exists()
        else {},
        "stage1": stage1_runs(),
        "stage2_retention": retention_runs(),
        "stage4_upstream_open_stage1": stage4_stage1_runs(),
        "stage4_upstream_open_retention": stage4_retention_runs(),
        "diagnostics": collect_diagnostics(),
    }
    add_parameter_deltas(summary)
    summary["interpretation_labels"] = interpretation_labels(summary)
    write_json(RUN_ROOT / "summary.json", summary)
    write_summary_md(summary)


def write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Natural Sum-Only Relaxed Bridge",
        "",
        f"Run root: `{RUN_ROOT}`",
        "",
        "## Stage 0 Wiring And Landscape",
        "",
    ]
    stage0_summary = summary.get("stage0", {})
    if stage0_summary:
        snap = stage0_summary["snapshot"]
        full_enum = stage0_summary["full_enum"]
        sem = stage0_summary["semantic_decoder_delta"]
        lines.extend(
            [
                "| oracle | injection-zero | forced-random | initial normal | initial calc | best result=true | true-pair best | same-sum near-best | semantic delta |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        fmt(snap.get("oracle_exact_match")),
                        fmt(snap.get("injection_zero_exact_match")),
                        fmt(snap.get("forced_random_exact_match")),
                        fmt(snap.get("normal_exact_match")),
                        fmt(snap.get("calculator_result_accuracy")),
                        fmt(
                            full_enum.get(
                                "best_result_group_matches_true_sum_fraction"
                            )
                        ),
                        fmt(full_enum.get("true_best_fraction")),
                        fmt(full_enum.get("mean_same_true_sum_near_best_pair_count")),
                        fmt(sem.get("l2")),
                    ]
                )
                + " |",
                "",
            ]
        )
    lines.extend(
        [
            "## Stage 1 Frozen-Upstream Deterministic Concrete",
            "",
            "| effective seed | first gate | best step | best normal/pair/calc | final normal/pair/calc | selected checkpoint |",
            "| ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in summary["stage1"]:
        first = row.get("first_gate")
        best = row.get("best_snapshot")
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["effective_seed"]),
                    fmt(first["step"] if first else None),
                    fmt(best["step"] if best else None),
                    metric_triple(best or {}),
                    metric_triple(row.get("final_snapshot", {})),
                    f"`{row.get('best_checkpoint')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Stage 2 Relaxation-Off Retention",
            "",
            "| effective seed | selected step | selected normal/pair/calc | final normal/pair/calc | passed | source checkpoint |",
            "| ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in summary["stage2_retention"]:
        selected = row["selected_snapshot"]
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["effective_seed"]),
                    fmt(selected["step"]),
                    metric_triple(selected),
                    metric_triple(row["final_snapshot"]),
                    str(row["passes_fast_retention"]),
                    f"`{row['source_checkpoint']}`",
                ]
            )
            + " |"
        )
    if summary["stage4_upstream_open_stage1"]:
        lines.extend(
            [
                "",
                "## Stage 4 Upstream-Open Stress",
                "",
                "| branch | effective seed | first gate | best normal/pair/calc | final normal/pair/calc |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in summary["stage4_upstream_open_stage1"]:
            first = row.get("first_gate")
            lines.append(
                "| stage1 | "
                + " | ".join(
                    [
                        fmt(row["effective_seed"]),
                        fmt(first["step"] if first else None),
                        metric_triple(row["best_snapshot"]),
                        metric_triple(row["final_snapshot"]),
                    ]
                )
                + " |"
            )
        for row in summary["stage4_upstream_open_retention"]:
            selected = row["selected_snapshot"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["branch"],
                        fmt(row["effective_seed"]),
                        fmt(selected["step"]),
                        metric_triple(selected),
                        metric_triple(row["final_snapshot"]),
                    ]
                )
                + " |"
            )
    if summary["diagnostics"]:
        lines.extend(
            [
                "",
                "## Selected Diagnostics",
                "",
                "| kind | canonical normal/calc | private answer/calc | learned result=true | learned result best | result gap |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in summary["diagnostics"]:
            if not row.get("complete"):
                continue
            can = row["canonical"]
            priv = row["private"]
            full = row["full_enum"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["kind"],
                        f"{fmt(can.get('normal_exact_match'))}/{fmt(can.get('calculator_result_accuracy'))}",
                        f"{fmt(priv.get('answer_exact_match'))}/{fmt(priv.get('calculator_result_accuracy'))}",
                        fmt(full.get("learned_result_matches_true_sum_fraction")),
                        fmt(full.get("learned_result_best_fraction")),
                        fmt(full.get("mean_learned_result_minus_best_result_gap")),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Labels",
            "",
            ", ".join(f"`{label}`" for label in summary["interpretation_labels"])
            or "_none yet_",
            "",
        ]
    )
    (RUN_ROOT / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 natural sum-only deterministic Concrete bridge runner."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["stage0", "stage1", "stage2", "stage3", "stage4"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--jobs", type=int, default=1)
    sub.add_parser("diagnostics")
    sub.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "stage0":
        stage0()
    elif args.command == "stage1":
        stage1(jobs=args.jobs)
    elif args.command == "stage2":
        stage2(jobs=args.jobs)
    elif args.command == "stage3":
        stage3(jobs=args.jobs)
    elif args.command == "stage4":
        stage4(jobs=args.jobs)
    elif args.command == "diagnostics":
        diagnostics()
    elif args.command == "summarize":
        write_summary()
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
