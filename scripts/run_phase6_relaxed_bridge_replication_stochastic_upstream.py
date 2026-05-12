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
RUN_ROOT = (
    REPO_ROOT
    / "runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream"
)
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
DETERMINISTIC_CLI_SEEDS = [0, 2, 3]
GUMBEL_GATE_SEEDS = [7201, 7202, 7203]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
        if key in row and row[key] != "":
            metrics[key] = float(row[key])
    return metrics


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
    summary = load_json(summaries[-1])
    return Path(summary["runs"][0]["run_dir"])


def stage0() -> None:
    if not STAGE0B_CHECKPOINT.exists():
        raise FileNotFoundError(STAGE0B_CHECKPOINT)
    outputs = []
    commands = [
        (
            "deterministic_seed6201",
            [
                sys.executable,
                "scripts/run_phase6_gumbel_concrete_interface_bridge.py",
                "stage0-gradient-gate",
                "--samples",
                "128",
                "--temperature",
                "2.0",
                "--mode",
                "deterministic",
                "--seed",
                "6201",
                "--chunk-size",
                "64",
                "--output",
                str(RUN_ROOT / "stage0/deterministic_seed6201.json"),
            ],
        )
    ]
    for seed in GUMBEL_GATE_SEEDS:
        commands.append(
            (
                f"gumbel_seed{seed}",
                [
                    sys.executable,
                    "scripts/run_phase6_gumbel_concrete_interface_bridge.py",
                    "stage0-gradient-gate",
                    "--samples",
                    "128",
                    "--temperature",
                    "2.0",
                    "--mode",
                    "gumbel",
                    "--seed",
                    str(seed),
                    "--chunk-size",
                    "64",
                    "--output",
                    str(RUN_ROOT / f"stage0/gumbel_seed{seed}.json"),
                ],
            )
        )
    for label, command in commands:
        run_command(command, log_path=RUN_ROOT / "stage0" / f"{label}.log")
        payload = load_json(Path(command[-1]))
        payload["gate_label"] = label
        payload["gate_seed"] = int(label.rsplit("seed", 1)[1])
        outputs.append(payload)
        print(f"{label}: {command[-1]}", flush=True)

    gumbel_rows = [row for row in outputs if row["mode"] == "gumbel"]
    summary = {
        "outputs": outputs,
        "deterministic_passes_required_gate": deterministic_gate_passes(outputs[0]),
        "gumbel_mean_best_pair_probability_delta": mean(
            [row["best_pair_probability_delta"] for row in gumbel_rows]
        ),
        "gumbel_min_best_pair_probability_delta": min(
            row["best_pair_probability_delta"] for row in gumbel_rows
        ),
        "gumbel_max_best_pair_probability_delta": max(
            row["best_pair_probability_delta"] for row in gumbel_rows
        ),
        "gumbel_positive_gradient_cosine_count": sum(
            row["gradient_cosine_relaxed_answer_vs_hard_best_ce"] > 0.0
            for row in gumbel_rows
        ),
        "gumbel_passes_required_gate": (
            mean([row["best_pair_probability_delta"] for row in gumbel_rows]) > 0.0
            and sum(
                row["gradient_cosine_relaxed_answer_vs_hard_best_ce"] > 0.0
                for row in gumbel_rows
            )
            >= 2
            and all(
                row["semantic_decoder_grad_l2"] == 0.0
                and row["parameter_delta_after_one_step"]
                .get("semantic_decoder", {})
                .get("l2", 0.0)
                == 0.0
                for row in gumbel_rows
            )
        ),
    }
    write_json(RUN_ROOT / "stage0_summary.json", summary)
    write_summary()


def deterministic_gate_passes(row: dict[str, Any]) -> bool:
    deltas = row["parameter_delta_after_one_step"]
    return (
        row["best_pair_probability_delta"] > 0.0
        and row["gradient_cosine_relaxed_answer_vs_hard_best_ce"] > 0.0
        and deltas.get("calculator_hook.input_proj", {}).get("l2", 0.0) > 0.0
        and deltas.get("upstream_encoder", {}).get("l2", 0.0) == 0.0
        and deltas.get("semantic_decoder", {}).get("l2", 0.0) == 0.0
        and row["semantic_decoder_grad_l2"] == 0.0
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


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
        "sum_left_operand",
        "--answer-format",
        "sum_left_operand",
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
        str(snapshot_every),
    ]
    if freeze_upstream:
        args.append("--freeze-upstream-encoder")
    return args


def relaxed_args(
    *,
    seed: int,
    run_root: Path,
    mode: str,
    freeze_upstream: bool = True,
    upstream_lr: float = 0.0003,
    temperature: float = 2.0,
    final_temperature: float = 0.5,
    entropy_weight: float = 0.0,
    entropy_decay_steps: int = 0,
) -> list[str]:
    args = common_overfit_args(
        seed=seed,
        steps=300,
        checkpoint=STAGE0B_CHECKPOINT,
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
            mode,
            "--relaxed-calculator-hard-forward",
            "--relaxed-calculator-temperature",
            f"{temperature:g}",
            "--relaxed-calculator-final-temperature",
            f"{final_temperature:g}",
            "--relaxed-calculator-temperature-decay-steps",
            "300",
            "--relaxed-calculator-entropy-weight",
            f"{entropy_weight:g}",
        ]
    )
    if entropy_decay_steps > 0:
        args.extend(
            ["--relaxed-calculator-entropy-decay-steps", str(entropy_decay_steps)]
        )
    return args


def stage1_deterministic(jobs: int) -> None:
    if not STAGE0B_CHECKPOINT.exists():
        raise FileNotFoundError(STAGE0B_CHECKPOINT)
    tasks = []
    for seed in DETERMINISTIC_CLI_SEEDS:
        root = RUN_ROOT / "stage1_deterministic" / f"cli_seed{seed}"
        tasks.append(
            (
                f"cli_seed{seed}",
                relaxed_args(
                    seed=seed,
                    run_root=root,
                    mode="deterministic",
                    freeze_upstream=True,
                ),
                root,
            )
        )
    run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


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
            pool.submit(
                run_command,
                command,
                log_path=root / f"{label}.log",
            ): (label, root)
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


def analyze_run(run_dir: Path, threshold: float = 0.95) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    cfg = load_json(run_dir / "config.json")
    effective_seed = int(cfg["seed"])
    snapshots = [row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")]
    qualifying = [
        row
        for row in snapshots
        if min(
            float(row.get("normal_exact_match", 0.0)),
            float(row.get("operand_exact_match", 0.0)),
            float(row.get("pair_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
        )
        >= threshold
    ]
    best = max(
        snapshots,
        key=lambda row: (
            float(row.get("pair_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
            float(row.get("operand_exact_match", 0.0)),
            float(row.get("normal_exact_match", 0.0)),
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
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        "freeze_semantic_decoder": metrics.get("freeze_semantic_decoder"),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
    }


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


def deterministic_runs() -> list[dict[str, Any]]:
    rows = []
    for summary in sorted((RUN_ROOT / "stage1_deterministic").glob("cli_seed*/*/summary_metrics.json")):
        run_dir = Path(load_json(summary)["runs"][0]["run_dir"])
        rows.append(analyze_run(run_dir))
    rows.sort(key=lambda row: int(row["effective_seed"]))
    return rows


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


def stage2_retention(jobs: int) -> None:
    tasks = []
    for row in deterministic_runs():
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
        print("No deterministic Stage 1 passing seeds; skipping Stage 2.", flush=True)
        write_summary()
        return
    run_parallel_tasks(tasks, jobs=jobs)
    write_summary()


def retention_runs(root_name: str = "stage2_retention") -> list[dict[str, Any]]:
    rows = []
    root = RUN_ROOT / root_name
    for summary in latest_summaries_by_parent(root.glob("*/*/summary_metrics.json")):
        run_dir = Path(load_json(summary)["runs"][0]["run_dir"])
        rows.append(analyze_retention_run(run_dir))
    rows.sort(key=lambda row: row["run_dir"])
    return rows


def analyze_retention_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    cfg = load_json(run_dir / "config.json")
    snapshots = [row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")]
    qualifying = [
        row
        for row in snapshots
        if min(
            float(row.get("operand_exact_match", 0.0)),
            float(row.get("pair_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
        )
        >= 0.99
    ]
    best = max(
        snapshots,
        key=lambda row: (
            float(row.get("pair_exact_match", 0.0)),
            float(row.get("calculator_result_accuracy", 0.0)),
            float(row.get("operand_exact_match", 0.0)),
            int(row["step"]),
        ),
    )
    selected = snapshots[-1] if snapshots[-1] in qualifying else (qualifying[-1] if qualifying else best)
    selected_step = int(selected["step"])
    checkpoint = (
        run_dir / "final_weights.pt"
        if selected_step == int(cfg.get("steps", -1))
        else run_dir / "checkpoint_snapshots" / f"step_{selected_step:05d}_weights.pt"
    )
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


def stage3_stochastic(jobs: int) -> None:
    stage0_summary = load_json(RUN_ROOT / "stage0_summary.json")
    det_passing = [row for row in deterministic_runs() if row["passes_fast_gate"]]
    if not stage0_summary.get("gumbel_passes_required_gate") and det_passing:
        print("Gate B failed; running primary stochastic seed 2 as negative documentation.", flush=True)
    tasks = [
        (
            "primary_cli_seed0",
            relaxed_args(
                seed=0,
                run_root=RUN_ROOT / "stage3_stochastic" / "primary_cli_seed0",
                mode="gumbel",
            ),
            RUN_ROOT / "stage3_stochastic" / "primary_cli_seed0",
        )
    ]
    run_parallel_tasks(tasks, jobs=1)
    primary = analyze_run(latest_run_dir(RUN_ROOT / "stage3_stochastic" / "primary_cli_seed0"))
    extra_tasks = []
    if primary["passes_fast_gate"]:
        for row in det_passing:
            if int(row["cli_seed"]) == 0:
                continue
            root = RUN_ROOT / "stage3_stochastic" / f"primary_cli_seed{row['cli_seed']}"
            extra_tasks.append(
                (
                    f"primary_cli_seed{row['cli_seed']}",
                    relaxed_args(seed=int(row["cli_seed"]), run_root=root, mode="gumbel"),
                    root,
                )
            )
    elif len(det_passing) >= 2:
        root = RUN_ROOT / "stage3_stochastic" / "stabilized_cli_seed0"
        extra_tasks.append(
            (
                "stabilized_cli_seed0",
                relaxed_args(
                    seed=0,
                    run_root=root,
                    mode="gumbel",
                    temperature=1.5,
                    final_temperature=0.5,
                    entropy_weight=0.01,
                    entropy_decay_steps=200,
                ),
                root,
            )
        )
    if extra_tasks:
        run_parallel_tasks(extra_tasks, jobs=jobs)
    write_summary()


def stage4_upstream_open(jobs: int) -> None:
    if len([row for row in deterministic_runs() if row["passes_fast_gate"]]) < 2:
        print("Deterministic replication did not pass on at least two seeds; skipping Stage 4.", flush=True)
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
                    mode="deterministic",
                    freeze_upstream=False,
                    upstream_lr=0.00003,
                ),
                root,
            )
        ],
        jobs=1,
    )
    upstream_row = analyze_run(latest_run_dir(root))
    if upstream_row["passes_fast_gate"]:
        first = Path(upstream_row["first_gate_checkpoint"])
        best = Path(upstream_row["best_checkpoint"])
        sources = [("first", first)]
        if best != first:
            sources.append(("best", best))
        tasks = []
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


def stage3_runs() -> list[dict[str, Any]]:
    rows = []
    root = RUN_ROOT / "stage3_stochastic"
    for summary in latest_summaries_by_parent(root.glob("*/*/summary_metrics.json")):
        run_dir = Path(load_json(summary)["runs"][0]["run_dir"])
        row = analyze_run(run_dir)
        row["branch"] = summary.parents[1].name
        rows.append(row)
    rows.sort(key=lambda row: row["branch"])
    return rows


def stage4_stage1_runs() -> list[dict[str, Any]]:
    rows = []
    root = RUN_ROOT / "stage4_upstream_open" / "stage1_cli_seed0"
    for summary in latest_summaries_by_parent(root.glob("*/summary_metrics.json")):
        rows.append(analyze_run(Path(load_json(summary)["runs"][0]["run_dir"])))
    return rows


def stage4_retention_runs() -> list[dict[str, Any]]:
    rows = []
    root = RUN_ROOT / "stage4_upstream_open"
    for summary in latest_summaries_by_parent(root.glob("retention_*/*/summary_metrics.json")):
        run_dir = Path(load_json(summary)["runs"][0]["run_dir"])
        row = analyze_retention_run(run_dir)
        row["branch"] = summary.parents[1].name
        rows.append(row)
    rows.sort(key=lambda row: row["branch"])
    return rows


def latest_summaries_by_parent(paths: Any) -> list[Path]:
    latest: dict[Path, Path] = {}
    for path in sorted(paths):
        parent = path.parent.parent
        previous = latest.get(parent)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            latest[parent] = path
    return sorted(latest.values())


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
    return {"before_checkpoint": str(before_path), "after_checkpoint": str(after_path), "groups": groups}


def add_stage1_deltas(summary: dict[str, Any]) -> None:
    for row in summary["stage1_deterministic"]:
        run_dir = Path(row["run_dir"])
        step0 = run_dir / "checkpoint_snapshots/step_00000_weights.pt"
        if step0.exists():
            row["parameter_delta_step0_to_best"] = checkpoint_delta_summary(
                step0, Path(row["best_checkpoint"])
            )
    for row in summary["stage3_stochastic"]:
        run_dir = Path(row["run_dir"])
        step0 = run_dir / "checkpoint_snapshots/step_00000_weights.pt"
        if step0.exists():
            row["parameter_delta_step0_to_best"] = checkpoint_delta_summary(
                step0, Path(row["best_checkpoint"])
            )
    for row in summary["stage4_upstream_open_stage1"]:
        run_dir = Path(row["run_dir"])
        step0 = run_dir / "checkpoint_snapshots/step_00000_weights.pt"
        if step0.exists():
            row["parameter_delta_step0_to_best"] = checkpoint_delta_summary(
                step0, Path(row["best_checkpoint"])
            )


def diagnostic_selections(summary: dict[str, Any]) -> list[dict[str, Any]]:
    selections = []
    for row in summary["stage1_deterministic"]:
        if row["passes_fast_gate"]:
            selections.append(selection(row, "stage1_deterministic_selected"))
        else:
            selections.append(selection(row, "stage1_deterministic_failure_best"))
    for row in summary["stage2_retention"]:
        selections.append(selection(row, "stage2_retention_selected"))
    for row in summary["stage3_stochastic"]:
        selections.append(selection(row, f"stage3_{row['branch']}_best"))
    for row in summary["stage4_upstream_open_stage1"]:
        if row["passes_fast_gate"]:
            selections.append(selection(row, "stage4_upstream_open_selected"))
        else:
            selections.append(selection(row, "stage4_upstream_open_failure_best"))
    for row in summary["stage4_upstream_open_retention"]:
        selections.append(selection(row, f"stage4_{row['branch']}_selected"))
    return selections


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


def diagnostics() -> None:
    write_summary()
    summary = load_json(RUN_ROOT / "summary.json")
    selections = diagnostic_selections(summary)
    write_json(RUN_ROOT / "diagnostic_selections.json", selections)
    for item in selections:
        run_dir = Path(item["run_dir"])
        checkpoint = Path(item["checkpoint"])
        print(f"{item['kind']}: {checkpoint}", flush=True)
        run_diagnostic_commands(item, checkpoint, run_dir)
    write_summary()


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
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
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
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
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
            "sum_left_operand",
            "--calculator-output-format",
            "sum_left_operand",
            "--temperature",
            "1.0",
            "--chunk-size",
            "64",
            "--output-root",
            str(full_enum_dir),
        ],
        log_path=run_dir / f"{item['full_enum_dir_name']}.log",
    )


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


def counterfactual(summary: dict[str, Any], condition: str) -> float | None:
    for row in summary.get("counterfactual_exact_match", []):
        if row.get("condition") == condition:
            return row.get("exact_match")
    return None


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
        "mean_learned_minus_true_gap": summary.get("mean_learned_minus_true_gap"),
        "mean_learned_minus_best_gap": summary.get("mean_learned_minus_best_gap"),
        "learned_best_fraction": summary.get("learned_best_fraction"),
        "true_best_fraction": summary.get("true_best_fraction"),
        "best_matches_true_operands_fraction": summary.get(
            "best_matches_true_operands_fraction"
        ),
    }


def find_summary(root: Path, filename: str) -> Path | None:
    matches = sorted(root.glob(f"**/{filename}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple {filename} under {root}: {matches}")
    return matches[0]


def collect_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    selections_path = RUN_ROOT / "diagnostic_selections.json"
    if not selections_path.exists():
        return rows
    for item in load_json(selections_path):
        run_dir = Path(item["run_dir"])
        canonical = run_dir / item["canonical_dir_name"] / "diagnostic_summary.json"
        private = run_dir / item["private_dir_name"] / "private_protocol_summary.json"
        full_enum = find_summary(run_dir / item["full_enum_dir_name"], "full_enum_summary.json")
        row = {**item, "complete": canonical.exists() and private.exists() and full_enum is not None}
        if row["complete"]:
            row["canonical"] = compact_canonical(load_json(canonical))
            row["private"] = compact_private(load_json(private))
            row["full_enum"] = compact_full_enum(load_json(full_enum))
        rows.append(row)
    return rows


def write_summary() -> None:
    stage0_summary = (
        load_json(RUN_ROOT / "stage0_summary.json")
        if (RUN_ROOT / "stage0_summary.json").exists()
        else {}
    )
    for idx, row in enumerate(stage0_summary.get("outputs", [])):
        if "gate_seed" not in row:
            row["gate_seed"] = [6201, *GUMBEL_GATE_SEEDS][idx]
    summary = {
        "run_root": str(RUN_ROOT),
        "stage0": stage0_summary,
        "stage1_deterministic": deterministic_runs(),
        "stage2_retention": retention_runs(),
        "stage3_stochastic": stage3_runs(),
        "stage4_upstream_open_stage1": stage4_stage1_runs(),
        "stage4_upstream_open_retention": stage4_retention_runs(),
    }
    add_stage1_deltas(summary)
    summary["diagnostics"] = collect_diagnostics(summary)
    summary["interpretation_labels"] = interpretation_labels(summary)
    write_json(RUN_ROOT / "summary.json", summary)
    write_summary_md(summary)


def interpretation_labels(summary: dict[str, Any]) -> list[str]:
    labels = []
    stage1_pass = [row for row in summary["stage1_deterministic"] if row["passes_fast_gate"]]
    stage2_pass = retained_seed_count(summary) >= 2
    if len(stage1_pass) >= 2 and stage2_pass:
        labels.append("deterministic_concrete_positive")
    elif summary["stage1_deterministic"]:
        labels.append("relaxed_bridge_replication_failure")
    if any(row["passes_fast_gate"] for row in summary["stage3_stochastic"]):
        labels.append("stochastic_gumbel_positive_candidate")
    elif summary["stage3_stochastic"]:
        labels.append("stochastic_gumbel_negative")
    if any(row["passes_fast_gate"] for row in summary["stage4_upstream_open_stage1"]):
        if any(row["passes_fast_retention"] and not row["freeze_upstream_encoder"] for row in summary["stage4_upstream_open_retention"]):
            labels.append("upstream_open_positive")
        else:
            labels.append("upstream_open_instability")
    return labels


def retained_seed_count(summary: dict[str, Any]) -> int:
    seeds = {
        int(row["effective_seed"])
        for row in summary["stage2_retention"]
        if row["passes_fast_retention"]
    }
    return len(seeds)


def metric_quad(row: dict[str, Any]) -> str:
    return "/".join(
        fmt(row.get(key))
        for key in [
            "normal_exact_match",
            "operand_exact_match",
            "pair_exact_match",
            "calculator_result_accuracy",
        ]
    )


def write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Relaxed Bridge Replication, Stochastic, Upstream",
        "",
        f"Run root: `{RUN_ROOT}`",
        "",
        "## Stage 0 Gates",
        "",
        "| mode | seed | best-pair delta | grad cosine | input delta | upstream delta | semantic grad/delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("stage0", {}).get("outputs", []):
        deltas = row["parameter_delta_after_one_step"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["mode"],
                    fmt(row.get("gate_seed", row.get("seed", ""))),
                    fmt(row["best_pair_probability_delta"]),
                    fmt(row["gradient_cosine_relaxed_answer_vs_hard_best_ce"]),
                    fmt(deltas.get("calculator_hook.input_proj", {}).get("l2", 0.0)),
                    fmt(deltas.get("upstream_encoder", {}).get("l2", 0.0)),
                    fmt(row["semantic_decoder_grad_l2"]) + "/" + fmt(deltas.get("semantic_decoder", {}).get("l2", 0.0)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Stage 1 Deterministic",
            "",
            "| eff seed | first gate | best step | best fast normal/operand/pair/calc | final fast normal/operand/pair/calc | final eval | selected checkpoint |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["stage1_deterministic"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["effective_seed"]),
                    fmt(row["first_gate"]["step"] if row["first_gate"] else None),
                    fmt(row["best_snapshot"]["step"]),
                    metric_quad(row["best_snapshot"]),
                    metric_quad(row["final_snapshot"]),
                    fmt(row["final_eval_exact_match"]),
                    f"`{row['best_checkpoint']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Stage 2 Retention",
            "",
            "| eff seed | source checkpoint | selected step | selected fast normal/operand/pair/calc | final fast normal/operand/pair/calc | final eval |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["stage2_retention"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["effective_seed"]),
                    f"`{row['source_checkpoint']}`",
                    fmt(row["selected_snapshot"]["step"]),
                    metric_quad(row["selected_snapshot"]),
                    metric_quad(row["final_snapshot"]),
                    fmt(row["final_eval_exact_match"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Stage 3 Stochastic",
            "",
            "| branch | eff seed | gate | best fast normal/operand/pair/calc | final fast normal/operand/pair/calc | final eval |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["stage3_stochastic"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["branch"],
                    fmt(row["effective_seed"]),
                    fmt(row["passes_fast_gate"]),
                    metric_quad(row["best_snapshot"]),
                    metric_quad(row["final_snapshot"]),
                    fmt(row["final_eval_exact_match"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Stage 4 Upstream Open",
            "",
            "| eff seed | gate | best fast normal/operand/pair/calc | final fast normal/operand/pair/calc | final eval | upstream delta | input delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["stage4_upstream_open_stage1"]:
        deltas = row.get("parameter_delta_step0_to_best", {}).get("groups", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["effective_seed"]),
                    fmt(row["passes_fast_gate"]),
                    metric_quad(row["best_snapshot"]),
                    metric_quad(row["final_snapshot"]),
                    fmt(row["final_eval_exact_match"]),
                    fmt(deltas.get("upstream_encoder", {}).get("l2", 0.0)),
                    fmt(deltas.get("calculator_hook.input_proj", {}).get("l2", 0.0)),
                ]
            )
            + " |"
        )
    if summary["diagnostics"]:
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "| selection | canonical operand/pair/calc | private answer/operand/pair/calc | full-enum gaps | learned-best |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["diagnostics"]:
            if not row["complete"]:
                lines.append(f"| {row['kind']} | pending | pending | pending | pending |")
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["kind"],
                        "/".join(
                            fmt(row["canonical"].get(key))
                            for key in [
                                "operand_exact_match",
                                "pair_exact_match",
                                "calculator_result_accuracy",
                            ]
                        ),
                        "/".join(
                            fmt(row["private"].get(key))
                            for key in [
                                "answer_exact_match",
                                "operand_exact_match",
                                "pair_exact_match",
                                "calculator_result_accuracy",
                            ]
                        ),
                        "/".join(
                            fmt(row["full_enum"].get(key))
                            for key in [
                                "mean_learned_minus_true_gap",
                                "mean_learned_minus_best_gap",
                            ]
                        ),
                        fmt(row["full_enum"].get("learned_best_fraction")),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Labels", "", ", ".join(summary["interpretation_labels"]) or "pending", ""])
    (RUN_ROOT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "stage0",
            "stage1",
            "stage2",
            "stage3",
            "stage4",
            "diagnostics",
            "summarize",
        ],
    )
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "stage0":
        stage0()
    elif args.stage == "stage1":
        stage1_deterministic(args.jobs)
    elif args.stage == "stage2":
        stage2_retention(args.jobs)
    elif args.stage == "stage3":
        stage3_stochastic(args.jobs)
    elif args.stage == "stage4":
        stage4_upstream_open(args.jobs)
    elif args.stage == "diagnostics":
        diagnostics()
    elif args.stage == "summarize":
        write_summary()


if __name__ == "__main__":
    main()
