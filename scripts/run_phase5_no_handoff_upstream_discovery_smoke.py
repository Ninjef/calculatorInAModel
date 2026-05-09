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


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "runs" / "2026-05-09_phase5_no_handoff_upstream_discovery_smoke"
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

SEED2_COMPLETION_SUMMARY = (
    REPO_ROOT
    / "runs/2026-05-08_phase5_upstream_assisted_partial_handoff_completion/summary.md"
)
SEED5_COMPLETION_SUMMARY = (
    REPO_ROOT
    / "runs/2026-05-08_phase5_cross_seed_upstream_assisted_completion/summary.md"
)

SEMANTIC_DECODER_CHECKPOINT_LOAD_SCOPE = "full_model"

CONDITIONS = {
    "seed0": {
        "label": "no_handoff_full_model_seed0",
        "stage": "stage1",
        "cli_seed": 0,
        "effective_seed_label": "previously associated with seed 2",
        "freeze_upstream_encoder": False,
        "upstream_lr": 0.00003,
        "input_proj_anchor_weight": 0.0,
        "description": (
            "answer-only no-handoff continuation from Stage 0B full-model "
            "checkpoint with upstream trainable"
        ),
    },
    "seed3": {
        "label": "no_handoff_full_model_seed3",
        "stage": "stage1",
        "cli_seed": 3,
        "effective_seed_label": "previously associated with seed 5",
        "freeze_upstream_encoder": False,
        "upstream_lr": 0.00003,
        "input_proj_anchor_weight": 0.0,
        "description": (
            "answer-only no-handoff continuation from Stage 0B full-model "
            "checkpoint with upstream trainable"
        ),
    },
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [jsonable(inner) for inner in value]
    return value


def load_json(path: Path) -> Any:
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
        metrics[key] = float(row[key])
    return metrics


def verify_starting_point() -> None:
    required = [
        STAGE0B_CHECKPOINT,
        STAGE0B_CHECKPOINT.parent / "metrics.json",
        SEED2_COMPLETION_SUMMARY,
        SEED5_COMPLETION_SUMMARY,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required no-handoff references: {missing}")


def stage0b_summary() -> dict[str, Any]:
    metrics = load_json(STAGE0B_CHECKPOINT.parent / "metrics.json")
    return {
        "checkpoint": str(STAGE0B_CHECKPOINT),
        "local_checkpoint_present": LOCAL_STAGE0B_CHECKPOINT.exists(),
        "recorded_absolute_checkpoint": str(RECORDED_STAGE0B_CHECKPOINT),
        "run_dir": str(STAGE0B_CHECKPOINT.parent),
        "eval_exact_match": metrics.get("exact_match"),
        "eval_correct": metrics.get("correct"),
        "eval_samples": metrics.get("samples"),
    }


def metric_snapshot_for_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    snapshots = [
        row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")
    ]
    return {
        "run_dir": str(run_dir),
        "selected_checkpoint": str(run_dir / "final_weights.pt"),
        "final_eval_exact_match": metrics.get("exact_match"),
        "final_eval_correct": metrics.get("correct"),
        "final_eval_samples": metrics.get("samples"),
        "snapshots": snapshots,
        "final_snapshot": snapshots[-1] if snapshots else {},
        "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
        "final_adaptive_interface_loss_weight": metrics.get(
            "final_adaptive_interface_loss_weight"
        ),
        "final_input_proj_anchor_weight": metrics.get("final_input_proj_anchor_weight"),
        "freeze_semantic_decoder": metrics.get("freeze_semantic_decoder"),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
        "semantic_decoder_checkpoint_load_scope": metrics.get(
            "semantic_decoder_checkpoint_load_scope"
        ),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        "input_proj_lr": metrics.get("input_proj_lr"),
        "upstream_lr": metrics.get("upstream_lr"),
    }


def counterfactual_value(summary: dict[str, Any], condition: str) -> float | None:
    for row in summary.get("counterfactual_exact_match", []):
        if row.get("condition") == condition:
            return row.get("exact_match")
    return None


def compact_canonical_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "normal_exact_match": summary.get("exact_match"),
        "operand_exact_match": summary.get("operand_exact_match"),
        "pair_exact_match": summary.get("pair_exact_match"),
        "calculator_result_accuracy": summary.get("calculator_result_accuracy"),
        "injection_zero_exact_match": counterfactual_value(summary, "injection_zero"),
        "forced_random_exact_match": counterfactual_value(summary, "forced_random"),
        "oracle_at_eval_exact_match": counterfactual_value(summary, "oracle_at_eval"),
        "mean_a_entropy": summary.get("mean_a_entropy"),
        "mean_b_entropy": summary.get("mean_b_entropy"),
        "mean_a_confidence": summary.get("mean_a_confidence"),
        "mean_b_confidence": summary.get("mean_b_confidence"),
    }


def compact_private_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "operand_exact_match": summary.get("operand_exact_match"),
        "pair_exact_match": summary.get("pair_exact_match"),
        "calculator_result_accuracy": summary.get("calculator_result_accuracy"),
        "mapped_operand_exact_match": summary.get("mapped_operand_exact_match"),
        "mapped_calculator_result_accuracy": summary.get(
            "mapped_calculator_result_accuracy"
        ),
        "a_best_affine_mod_exact": summary.get("a_best_affine_mod_mapping", {}).get(
            "exact"
        ),
        "b_best_affine_mod_exact": summary.get("b_best_affine_mod_mapping", {}).get(
            "exact"
        ),
    }


def compact_full_enum_summary(summary: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(summary, list):
        if len(summary) != 1:
            raise ValueError("expected exactly one full-enum summary")
        summary = summary[0]
    return {
        "mean_learned_minus_true_gap": summary.get("mean_learned_minus_true_gap"),
        "mean_learned_minus_best_gap": summary.get("mean_learned_minus_best_gap"),
        "learned_best_fraction": summary.get("learned_best_fraction"),
        "true_best_fraction": summary.get("true_best_fraction"),
        "best_matches_true_operands_fraction": summary.get(
            "best_matches_true_operands_fraction"
        ),
        "learned_result_matches_true_sum_fraction": summary.get(
            "learned_result_matches_true_sum_fraction"
        ),
    }


def find_single_summary(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.glob(f"**/{filename}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple {filename} files under {root}: {matches}")
    return matches[0]


def find_full_enum_summary(root: Path) -> Path | None:
    nested = find_single_summary(root, "full_enum_summary.json")
    if nested is not None:
        return nested
    return find_single_summary(root, "full_enum_summary_all.json")


def starting_point_summary() -> dict[str, Any]:
    return {
        "interpretation_label": "no_handoff_full_model_init",
        "stage0b_operand_aware_oracle_semantic_decoder": stage0b_summary(),
        "semantic_decoder_checkpoint_load_scope": (
            SEMANTIC_DECODER_CHECKPOINT_LOAD_SCOPE
        ),
        "semantic_decoder_checkpoint_load_scope_current_behavior": (
            "full_model_current_behavior"
        ),
        "loader_clarity": (
            "load_semantic_decoder_checkpoint defaults to full_model for "
            "backward compatibility; this runner passes full_model explicitly."
        ),
        "seed2_step55_upstream_assisted_completion_summary": str(
            SEED2_COMPLETION_SUMMARY
        ),
        "seed5_step25_cross_seed_completion_summary": str(SEED5_COMPLETION_SUMMARY),
    }


def common_args(condition: dict[str, Any]) -> list[str]:
    args = [
        sys.executable,
        "scripts/overfit_one_batch.py",
        "--variant",
        "model-c",
        "--digits",
        "2",
        "--steps",
        "2000",
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
        str(STAGE0B_CHECKPOINT),
        "--semantic-decoder-checkpoint-load-scope",
        SEMANTIC_DECODER_CHECKPOINT_LOAD_SCOPE,
        "--freeze-semantic-decoder",
        "--answer-loss-weight",
        "1.0",
        "--adaptive-interface-loss-weight",
        "0.0",
        "--aux-operand-loss-weight",
        "0.0",
        "--input-proj-anchor-weight",
        f"{condition['input_proj_anchor_weight']:g}",
        "--input-proj-lr",
        "0.0003",
        "--upstream-lr",
        f"{condition['upstream_lr']:g}",
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
        str(condition["cli_seed"]),
        "--snapshot-every",
        "50",
        "--checkpoint-every",
        "50",
        "--snapshot-samples",
        "256",
        "--log-every",
        "50",
    ]
    if condition["freeze_upstream_encoder"]:
        args.append("--freeze-upstream-encoder")
    return args


def run_command(args: list[str], run_root: Path, log_name: str) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / log_name
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
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


def run_conditions(condition_keys: list[str], jobs: int) -> None:
    verify_starting_point()
    manifest = load_manifest()
    manifest["starting_point"] = starting_point_summary()
    manifest["conditions"] = jsonable({key: CONDITIONS[key] for key in CONDITIONS})
    tasks: list[tuple[str, list[str], Path]] = []
    for key in condition_keys:
        condition = CONDITIONS[key]
        label = condition["label"]
        tasks.append(
            (
                label,
                common_args(condition),
                RUN_ROOT / condition["stage"] / label,
            )
        )
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_command, command, root, f"{label}.log"): label
            for label, command, root in tasks
        }
        for future in as_completed(futures):
            label = futures[future]
            run_dir = future.result()
            print(f"{label}: {run_dir}", flush=True)
            manifest.setdefault("runs", {})[label] = str(run_dir)
            write_manifest(manifest)
    write_manifest(manifest)
    write_summary()


def checkpoint_state_dict(path: Path) -> dict[str, Any]:
    import torch

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
    if name.startswith("calculator_hook.pair_proj."):
        return "calculator_hook.pair_proj"
    if name.startswith("calculator_hook.output_proj."):
        return "semantic_decoder"
    if name.startswith("answer_offset_emb.") or name.startswith("answer_decoder."):
        return "semantic_decoder"
    if name.startswith(("tok_emb.", "pos_emb.", "blocks.", "ln_f.", "lm_head.")):
        return "upstream_encoder"
    return "other"


def checkpoint_delta_summary(before_path: Path, after_path: Path) -> dict[str, Any]:
    import torch

    before = checkpoint_state_dict(before_path)
    after = checkpoint_state_dict(after_path)
    groups: dict[str, dict[str, Any]] = {}
    top_changes: dict[str, list[dict[str, float | str]]] = {}
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None or before_tensor.shape != after_tensor.shape:
            continue
        group = parameter_group_for_name(name)
        group_summary = groups.setdefault(
            group,
            {
                "tensor_count": 0,
                "element_count": 0,
                "changed_tensor_count": 0,
                "l2": 0.0,
                "mean_abs_numerator": 0.0,
                "max_abs": 0.0,
            },
        )
        delta = (after_tensor.float() - before_tensor.float()).reshape(-1)
        l2_sq = float(torch.dot(delta, delta).item())
        mean_abs_numerator = float(delta.abs().sum().item())
        max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
        group_summary["tensor_count"] += 1
        group_summary["element_count"] += int(delta.numel())
        group_summary["changed_tensor_count"] += int(max_abs > 0.0)
        group_summary["l2"] += l2_sq
        group_summary["mean_abs_numerator"] += mean_abs_numerator
        group_summary["max_abs"] = max(float(group_summary["max_abs"]), max_abs)
        top_changes.setdefault(group, []).append(
            {"name": name, "l2": math.sqrt(l2_sq), "max_abs": max_abs}
        )

    for group, group_summary in groups.items():
        group_summary["l2"] = math.sqrt(float(group_summary["l2"]))
        elements = int(group_summary["element_count"])
        group_summary["mean_abs"] = (
            float(group_summary["mean_abs_numerator"]) / elements if elements else 0.0
        )
        del group_summary["mean_abs_numerator"]
        group_summary["top_tensors_by_max_abs"] = sorted(
            top_changes[group],
            key=lambda row: float(row["max_abs"]),
            reverse=True,
        )[:5]
    return {
        "before_checkpoint": str(before_path),
        "after_checkpoint": str(after_path),
        "groups": groups,
    }


def collect_diagnostic_summary(run_dir: Path) -> dict[str, Any]:
    canonical = run_dir / "canonical_causal_diagnostics/diagnostic_summary.json"
    private = run_dir / "private_protocol_diagnostics/private_protocol_summary.json"
    full_enum = find_full_enum_summary(run_dir / "full_enum_action_loss")
    paths = {
        "canonical": str(canonical.parent),
        "private": str(private.parent),
        "full_enum": str(full_enum.parent) if full_enum is not None else "",
    }
    if not canonical.exists() or not private.exists() or full_enum is None:
        return {"complete": False, "paths": paths}
    return {
        "complete": True,
        "canonical": compact_canonical_summary(load_json(canonical)),
        "private": compact_private_summary(load_json(private)),
        "full_enum": compact_full_enum_summary(load_json(full_enum)),
        "paths": paths,
    }


def fast_metrics(run_dir: Path) -> dict[str, Any]:
    payload = metric_snapshot_for_run(run_dir)
    payload["parameter_delta_from_stage0b"] = checkpoint_delta_summary(
        STAGE0B_CHECKPOINT, run_dir / "final_weights.pt"
    )
    payload["diagnostics"] = collect_diagnostic_summary(run_dir)
    return payload


def best_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not snapshots:
        return None
    return max(
        snapshots,
        key=lambda row: (
            float(row.get("calculator_result_accuracy", 0.0)),
            float(row.get("pair_exact_match", 0.0)),
            float(row.get("operand_exact_match", 0.0)),
            int(row.get("step", 0)),
        ),
    )


def drift_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    saw_recovered_protocol = False
    for row in snapshots:
        calc = float(row.get("calculator_result_accuracy", 0.0))
        pair = float(row.get("pair_exact_match", 0.0))
        operand = float(row.get("operand_exact_match", 0.0))
        if saw_recovered_protocol and min(calc, pair, operand) < 0.999:
            candidates.append(row)
        if min(calc, pair, operand) >= 0.999:
            saw_recovered_protocol = True
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            float(row.get("calculator_result_accuracy", 1.0)),
            float(row.get("pair_exact_match", 1.0)),
            float(row.get("operand_exact_match", 1.0)),
        ),
    )


def summarize_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    labels = {condition["label"]: condition for condition in CONDITIONS.values()}
    for label, run_dir_text in sorted(manifest.get("runs", {}).items()):
        run_dir = Path(run_dir_text)
        snapshots = [
            row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")
        ]
        rows.append(
            {
                "label": label,
                "condition": jsonable(labels[label]),
                **fast_metrics(run_dir),
                "best_snapshot": best_snapshot(snapshots),
                "drift_snapshot": drift_snapshot(snapshots),
            }
        )
    return rows


def load_manifest() -> dict[str, Any]:
    path = RUN_ROOT / "manifest.json"
    if path.exists():
        return load_json(path)
    return {}


def write_manifest(manifest: dict[str, Any]) -> None:
    write_json(RUN_ROOT / "manifest.json", manifest)


def load_diagnostic_selection_summaries() -> list[dict[str, Any]]:
    path = RUN_ROOT / "diagnostic_selection_summaries.json"
    return load_json(path) if path.exists() else []


def run_has_discovery_positive(selection: dict[str, Any]) -> bool:
    if not selection.get("complete"):
        return False
    canonical = selection["canonical"]
    private = selection["private"]
    full_enum = selection["full_enum"]
    return (
        min(
            float(canonical.get("operand_exact_match", 0.0)),
            float(canonical.get("pair_exact_match", 0.0)),
            float(canonical.get("calculator_result_accuracy", 0.0)),
            float(private.get("operand_exact_match", 0.0)),
            float(private.get("pair_exact_match", 0.0)),
            float(private.get("calculator_result_accuracy", 0.0)),
        )
        >= 0.999
        and abs(float(full_enum.get("mean_learned_minus_true_gap", 1.0))) < 1e-9
        and abs(float(full_enum.get("mean_learned_minus_best_gap", 1.0))) < 1e-9
    )


def interpretation(summary: dict[str, Any]) -> list[str]:
    selections = summary.get("diagnostic_selections", [])
    if not selections:
        return [
            (
                "Training has run or been configured, but full canonical/private/"
                "full-enum diagnostics are still needed before any discovery claim."
            )
        ]
    positives = [row for row in selections if run_has_discovery_positive(row)]
    if positives:
        labels = ", ".join(row["kind"] for row in positives)
        return [
            (
                "Checkpoint-selected no-handoff discovery positive: selected "
                f"checkpoint(s) {labels} satisfy canonical/private protocol "
                "metrics and zero full-enum learned-minus-true/best gaps."
            ),
            (
                "This is no-handoff full-model initialization, not from-scratch "
                "random-upstream discovery, because the Stage 0B checkpoint is "
                "loaded with full_model scope."
            ),
        ]
    return [
        (
            "No selected checkpoint satisfied the no-handoff discovery criterion. "
            "Treat this as a clean smoke failure unless additional selected "
            "diagnostics contradict it."
        ),
        (
            "Do not upgrade answer exact or oracle-at-eval success into a "
            "calculator-use claim without learned protocol and full-enum agreement."
        ),
    ]


def write_summary() -> None:
    manifest = load_manifest()
    fresh_starting_point = starting_point_summary()
    if "starting_point" not in manifest:
        manifest["starting_point"] = fresh_starting_point
    else:
        manifest["starting_point"] = {
            **fresh_starting_point,
            **manifest["starting_point"],
        }
    summary = {
        "claim": (
            "Can answer-only training discover the calculator-query protocol "
            "without a Stage 1 supervised interface handoff, while preserving "
            "the strict Phase 4/5 semantic bottleneck?"
        ),
        "run_root": str(RUN_ROOT),
        "starting_point": manifest["starting_point"],
        "conditions": manifest.get("conditions", jsonable(CONDITIONS)),
        "runs": summarize_runs(manifest) if manifest.get("runs") else [],
        "diagnostic_selections": load_diagnostic_selection_summaries(),
    }
    summary["interpretation"] = interpretation(summary)
    write_json(RUN_ROOT / "summary.json", summary)
    write_summary_md(summary)
    print(RUN_ROOT / "summary.json")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def fast_gate_cells(row: dict[str, Any]) -> list[str]:
    snap = row["final_snapshot"]
    groups = row.get("parameter_delta_from_stage0b", {}).get("groups", {})
    upstream_delta = groups.get("upstream_encoder", {})
    input_delta = groups.get("calculator_hook.input_proj", {})
    semantic_delta = groups.get("semantic_decoder", {})
    return [
        row["label"],
        fmt(row.get("final_eval_exact_match")),
        fmt(snap.get("normal_exact_match")),
        fmt(snap.get("injection_zero_exact_match")),
        fmt(snap.get("forced_random_exact_match")),
        fmt(snap.get("oracle_exact_match")),
        fmt(snap.get("operand_exact_match")),
        fmt(snap.get("pair_exact_match")),
        fmt(snap.get("calculator_result_accuracy")),
        fmt(snap.get("mean_a_entropy")),
        fmt(snap.get("mean_b_entropy")),
        fmt(row.get("final_aux_operand_loss_weight")),
        fmt(row.get("final_adaptive_interface_loss_weight")),
        fmt(row.get("final_input_proj_anchor_weight")),
        fmt(row.get("freeze_semantic_decoder")),
        fmt(row.get("freeze_upstream_encoder")),
        fmt(row.get("semantic_decoder_checkpoint_load_scope")),
        fmt(input_delta.get("l2", 0.0)),
        fmt(upstream_delta.get("l2", 0.0)),
        fmt(semantic_delta.get("l2", 0.0)),
    ]


def diagnostic_cells(label: str, diagnostics: dict[str, Any]) -> str:
    if not diagnostics.get("complete"):
        return f"| {label} | pending | pending | pending | pending |"
    canonical = diagnostics["canonical"]
    private = diagnostics["private"]
    full_enum = diagnostics["full_enum"]
    return (
        "| "
        + " | ".join(
            [
                label,
                "/".join(
                    fmt(canonical.get(key))
                    for key in [
                        "operand_exact_match",
                        "pair_exact_match",
                        "calculator_result_accuracy",
                    ]
                ),
                "/".join(
                    fmt(private.get(key))
                    for key in [
                        "operand_exact_match",
                        "pair_exact_match",
                        "calculator_result_accuracy",
                    ]
                ),
                "/".join(
                    fmt(full_enum.get(key))
                    for key in [
                        "mean_learned_minus_true_gap",
                        "mean_learned_minus_best_gap",
                    ]
                ),
                fmt(full_enum.get("learned_best_fraction")),
            ]
        )
        + " |"
    )


def write_summary_md(summary: dict[str, Any]) -> None:
    starting = summary["starting_point"]
    stage0b = starting["stage0b_operand_aware_oracle_semantic_decoder"]
    lines = [
        "# Phase 5 No-Handoff Upstream Discovery Smoke",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Claim",
        "",
        summary["claim"],
        "",
        "## Starting point",
        "",
        f"- Interpretation label: `{starting['interpretation_label']}`",
        f"- Stage 0B checkpoint: `{stage0b['checkpoint']}`",
        f"- Stage 0B local checkpoint present: `{stage0b['local_checkpoint_present']}`",
        f"- Stage 0B eval exact: `{stage0b['eval_exact_match']}`",
        (
            "- semantic_decoder_checkpoint_load_scope: "
            f"`{starting['semantic_decoder_checkpoint_load_scope']}`"
        ),
        (
            "- semantic_decoder_checkpoint_load_scope_current_behavior: "
            f"`{starting['semantic_decoder_checkpoint_load_scope_current_behavior']}`"
        ),
        (
            "- Seed 2 step 55 upstream-assisted completion summary: "
            f"`{starting['seed2_step55_upstream_assisted_completion_summary']}`"
        ),
        (
            "- Seed 5 step 25 cross-seed completion summary: "
            f"`{starting['seed5_step25_cross_seed_completion_summary']}`"
        ),
        "",
        "## Fast gates",
        "",
        "| condition | final eval | snapshot normal | inj-zero | forced-random | oracle | operand | pair | calc | A ent | B ent | aux w | interface w | anchor w | freeze semantic | freeze upstream | load scope | input delta L2 | upstream delta L2 | semantic delta L2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        lines.append("| " + " | ".join(fast_gate_cells(row)) + " |")

    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "| condition | canonical operand/pair/calc | private operand/pair/calc | full-enum learned-true/best gaps | learned-best |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["runs"]:
        lines.append(diagnostic_cells(row["label"], row.get("diagnostics", {})))

    lines.extend(["", "## Selected checkpoints", ""])
    for row in summary["runs"]:
        lines.append(f"- `{row['label']}` final: `{row['selected_checkpoint']}`")
        if row.get("best_snapshot"):
            step = int(row["best_snapshot"]["step"])
            lines.append(
                f"- `{row['label']}` best by protocol fast gates: "
                f"`{Path(row['run_dir']) / 'checkpoint_snapshots' / f'step_{step:05d}_weights.pt'}`"
            )
        if row.get("drift_snapshot"):
            step = int(row["drift_snapshot"]["step"])
            lines.append(
                f"- `{row['label']}` first/worst post-recovery drift: "
                f"`{Path(row['run_dir']) / 'checkpoint_snapshots' / f'step_{step:05d}_weights.pt'}`"
            )
    if summary.get("diagnostic_selections"):
        lines.extend(
            [
                "",
                "## Diagnostic selections",
                "",
                "| selection | canonical operand/pair/calc | private operand/pair/calc | full-enum learned-true/best gaps | learned-best |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for selection in summary["diagnostic_selections"]:
            lines.append(diagnostic_cells(selection["kind"], selection))
        lines.extend(["", "Diagnostic checkpoint paths:", ""])
        for selection in summary["diagnostic_selections"]:
            status = "complete" if selection.get("complete") else "pending"
            lines.append(f"- `{selection['kind']}` ({status}): `{selection['checkpoint']}`")
    if summary.get("interpretation"):
        lines.extend(["", "## Interpretation", ""])
        for item in summary["interpretation"]:
            lines.append(f"- {item}")
    lines.append("")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def run_diagnostic_command(args: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/codex_pycache"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    with log_path.open("w") as log:
        subprocess.run(
            args,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def diagnostic_selections(summary: dict[str, Any]) -> list[dict[str, Any]]:
    selections = []
    for row in summary["runs"]:
        selections.append(
            {
                "kind": row["label"],
                "checkpoint": row["selected_checkpoint"],
                "run_dir": row["run_dir"],
                "reason": "final checkpoint",
                "canonical_dir_name": "canonical_causal_diagnostics",
                "private_dir_name": "private_protocol_diagnostics",
                "full_enum_dir_name": "full_enum_action_loss",
            }
        )
        best = row.get("best_snapshot")
        final_step = int(row.get("final_snapshot", {}).get("step", -1))
        if best is not None and int(best["step"]) != final_step:
            step = int(best["step"])
            checkpoint = (
                Path(row["run_dir"]) / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"
            )
            if checkpoint.exists():
                kind = f"{row['label']}_best_snapshot_step{step:05d}"
                selections.append(
                    {
                        "kind": kind,
                        "checkpoint": str(checkpoint),
                        "run_dir": row["run_dir"],
                        "reason": "best dense snapshot by learned protocol fast gates",
                        "snapshot": best,
                        "canonical_dir_name": f"{kind}_canonical_causal_diagnostics",
                        "private_dir_name": f"{kind}_private_protocol_diagnostics",
                        "full_enum_dir_name": f"{kind}_full_enum_action_loss",
                    }
                )
        drift = row.get("drift_snapshot")
        if drift is not None and int(drift["step"]) != final_step:
            step = int(drift["step"])
            checkpoint = (
                Path(row["run_dir"]) / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"
            )
            if checkpoint.exists():
                kind = f"{row['label']}_drift_snapshot_step{step:05d}"
                selections.append(
                    {
                        "kind": kind,
                        "checkpoint": str(checkpoint),
                        "run_dir": row["run_dir"],
                        "reason": "first/worst post-recovery protocol drift checkpoint",
                        "snapshot": drift,
                        "canonical_dir_name": f"{kind}_canonical_causal_diagnostics",
                        "private_dir_name": f"{kind}_private_protocol_diagnostics",
                        "full_enum_dir_name": f"{kind}_full_enum_action_loss",
                    }
                )
    return selections


def summarize_diagnostic_selection(selection: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(selection["run_dir"])
    canonical = run_dir / selection["canonical_dir_name"] / "diagnostic_summary.json"
    private = run_dir / selection["private_dir_name"] / "private_protocol_summary.json"
    full_enum = find_full_enum_summary(run_dir / selection["full_enum_dir_name"])
    payload = {
        **selection,
        "complete": canonical.exists() and private.exists() and full_enum is not None,
        "paths": {
            "canonical": str(canonical.parent),
            "private": str(private.parent),
            "full_enum": str(full_enum.parent) if full_enum is not None else "",
        },
    }
    if payload["complete"]:
        payload["canonical"] = compact_canonical_summary(load_json(canonical))
        payload["private"] = compact_private_summary(load_json(private))
        payload["full_enum"] = compact_full_enum_summary(load_json(full_enum))
    return payload


def diagnostics() -> None:
    if not (RUN_ROOT / "summary.json").exists():
        write_summary()
    summary = load_json(RUN_ROOT / "summary.json")
    if not summary.get("runs"):
        write_summary()
        summary = load_json(RUN_ROOT / "summary.json")
    selections = diagnostic_selections(summary)
    write_json(RUN_ROOT / "diagnostic_selections.json", selections)
    for selection in selections:
        run_dir = Path(selection["run_dir"])
        checkpoint = Path(selection["checkpoint"])
        canonical_dir = run_dir / selection["canonical_dir_name"]
        private_dir = run_dir / selection["private_dir_name"]
        full_enum_dir = run_dir / selection["full_enum_dir_name"]
        if (
            (canonical_dir / "diagnostic_summary.json").exists()
            and (private_dir / "private_protocol_summary.json").exists()
            and find_full_enum_summary(full_enum_dir) is not None
        ):
            print(f"{selection['kind']}: already complete", flush=True)
            continue
        print(f"{selection['kind']}: {checkpoint}", flush=True)
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
                str(canonical_dir),
            ],
            run_dir / f"{selection['canonical_dir_name']}.log",
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
                str(private_dir),
            ],
            run_dir / f"{selection['private_dir_name']}.log",
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
                str(full_enum_dir),
            ],
            run_dir / f"{selection['full_enum_dir_name']}.log",
        )
    write_json(
        RUN_ROOT / "diagnostic_selection_summaries.json",
        [summarize_diagnostic_selection(selection) for selection in selections],
    )
    write_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--condition",
        choices=sorted(CONDITIONS),
        action="append",
        help="Condition(s) to run; default runs both allowed seeds.",
    )
    run_parser.add_argument("--jobs", type=int, default=1)
    subparsers.add_parser("diagnostics")
    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        keys = args.condition if args.condition else ["seed0", "seed3"]
        run_conditions(keys, jobs=args.jobs)
    elif args.command == "diagnostics":
        diagnostics()
    elif args.command == "summarize":
        verify_starting_point()
        manifest = load_manifest()
        fresh_starting_point = starting_point_summary()
        if "starting_point" not in manifest:
            manifest["starting_point"] = fresh_starting_point
        else:
            manifest["starting_point"] = {
                **fresh_starting_point,
                **manifest["starting_point"],
            }
        manifest["conditions"] = jsonable({key: CONDITIONS[key] for key in CONDITIONS})
        write_manifest(manifest)
        write_summary()


if __name__ == "__main__":
    main()
