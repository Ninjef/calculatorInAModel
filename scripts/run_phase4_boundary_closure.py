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
RUN_ROOT = REPO_ROOT / "runs" / "2026-05-08_phase4_boundary_closure"
PREVIOUS_RUN_ROOT = REPO_ROOT / "runs" / "2026-05-07_phase4_min_supervision_boundary"

STAGE1A_RUNS = {
    2: PREVIOUS_RUN_ROOT
    / "stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-"
    "inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2",
    5: PREVIOUS_RUN_ROOT
    / "stage1a/seed5/2026-05-07_103539_395143_model-c-op0-19-adaptive_interface-"
    "inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5",
}

SEEDS = {
    2: 0,
    5: 3,
}

HANDOFFS = [
    {
        "label": "seed2_step30",
        "effective_seed": 2,
        "stage1_step": 30,
        "reason": "seed 2 lower midpoint between failed step 25 and retained step 60",
    },
    {
        "label": "seed2_step55",
        "effective_seed": 2,
        "stage1_step": 55,
        "reason": "seed 2 upper midpoint before retained step 60",
    },
    {
        "label": "seed5_step20",
        "effective_seed": 5,
        "stage1_step": 20,
        "reason": "seed 5 very-low below-boundary probe",
    },
    {
        "label": "seed5_step25",
        "effective_seed": 5,
        "stage1_step": 25,
        "reason": "seed 5 nearest lower neighbor below retained step 30",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def stage1_row(effective_seed: int, step: int) -> dict[str, str]:
    rows = read_rows(STAGE1A_RUNS[effective_seed] / "diagnostic_snapshots.csv")
    for row in rows:
        if int(row["step"]) == step:
            return row
    raise ValueError(f"missing seed {effective_seed} Stage 1A step {step}")


def checkpoint_for_stage1_row(effective_seed: int, row: dict[str, str]) -> Path:
    step = int(row["step"])
    return (
        STAGE1A_RUNS[effective_seed]
        / "checkpoint_snapshots"
        / f"step_{step:05d}_weights.pt"
    )


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


def common_args(*, cli_seed: int, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/overfit_one_batch.py",
        "--variant",
        "model-c",
        "--digits",
        "2",
        "--steps",
        "1000",
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
        "--answer-loss-weight",
        "1.0",
        "--adaptive-interface-loss-weight",
        "0.0",
        "--aux-operand-loss-weight",
        "0.0",
        "--input-proj-anchor-weight",
        "0.0",
        "--input-proj-lr",
        "0.0003",
        "--upstream-lr",
        "0.0003",
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
        "--snapshot-every",
        "50",
        "--checkpoint-every",
        "50",
        "--snapshot-samples",
        "256",
        "--log-every",
        "50",
    ]


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
    return Path(load_json(summary_paths[-1])["runs"][0]["run_dir"])


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


def selected_handoffs() -> list[dict[str, Any]]:
    selected = []
    for handoff in HANDOFFS:
        effective_seed = handoff["effective_seed"]
        row = stage1_row(effective_seed, handoff["stage1_step"])
        checkpoint = checkpoint_for_stage1_row(effective_seed, row)
        selected.append(
            {
                **handoff,
                "cli_seed": SEEDS[effective_seed],
                "stage1_run_dir": str(STAGE1A_RUNS[effective_seed]),
                "stage1_checkpoint": str(checkpoint),
                "stage1_metrics": row_to_metrics(row),
            }
        )
    return selected


def stage2(jobs: int) -> None:
    handoffs = selected_handoffs()
    tasks: list[tuple[str, list[str], Path]] = []
    for handoff in handoffs:
        effective_seed = handoff["effective_seed"]
        step = handoff["stage1_step"]
        label = handoff["label"]
        args = common_args(
            cli_seed=handoff["cli_seed"],
            checkpoint=Path(handoff["stage1_checkpoint"]),
        )
        tasks.append(
            (
                label,
                args,
                RUN_ROOT / "stage2" / f"seed{effective_seed}" / f"step{step}",
            )
        )
    manifest = load_manifest()
    manifest["stage2_handoffs"] = handoffs
    manifest.setdefault("stage2", {}).update(run_many(tasks, jobs=jobs))
    write_manifest(manifest)


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


def summarize_stage2(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    handoffs = {handoff["label"]: handoff for handoff in manifest["stage2_handoffs"]}
    rows = []
    for label, run_dir_text in manifest.get("stage2", {}).items():
        run_dir = Path(run_dir_text)
        handoff = handoffs[label]
        rows.append(
            {
                "label": label,
                "selected_checkpoint": str(run_dir / "final_weights.pt"),
                "effective_seed": handoff["effective_seed"],
                "cli_seed": handoff["cli_seed"],
                "stage1_checkpoint": handoff["stage1_checkpoint"],
                "stage1_step": handoff["stage1_step"],
                "stage1_operand_exact_match": handoff["stage1_metrics"][
                    "operand_exact_match"
                ],
                "reason": handoff["reason"],
                **fast_metrics(run_dir),
            }
        )
    return sorted(rows, key=lambda row: (row["effective_seed"], row["stage1_step"]))


def write_summary() -> None:
    manifest = load_manifest()
    summary = {
        "stage2_handoffs": manifest.get("stage2_handoffs", selected_handoffs()),
        "stage2": summarize_stage2(manifest) if manifest.get("stage2") else [],
    }
    write_json(RUN_ROOT / "summary.json", summary)
    print(RUN_ROOT / "summary.json")


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
    selections = [
        {
            "kind": row["label"],
            "checkpoint": row["selected_checkpoint"],
            "run_dir": row["run_dir"],
        }
        for row in summary["stage2"]
    ]
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
    parser.add_argument("stage", choices=["stage2", "summarize", "diagnostics"])
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "stage2":
        stage2(args.jobs)
    elif args.stage == "summarize":
        write_summary()
    elif args.stage == "diagnostics":
        diagnostics()


if __name__ == "__main__":
    main()
