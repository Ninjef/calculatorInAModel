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
RUN_ROOT = REPO_ROOT / "runs" / "2026-05-08_phase5_upstream_unfreeze_stability_smoke"
PHASE4_RUN_ROOT = REPO_ROOT / "runs" / "2026-05-07_phase4_min_supervision_boundary"
PHASE4_SUMMARY = PHASE4_RUN_ROOT / "summary.json"

RETAINED_CHECKPOINT = (
    PHASE4_RUN_ROOT
    / "stage2/seed2/step60/2026-05-07_112933_781608_model-c-op0-19-adaptive_interface-"
    "inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/"
    "final_weights.pt"
)
SOURCE_STAGE1_CHECKPOINT = (
    PHASE4_RUN_ROOT
    / "stage1a/seed2/2026-05-07_103539_395099_model-c-op0-19-adaptive_interface-"
    "inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2/"
    "checkpoint_snapshots/step_00060_weights.pt"
)
SOURCE_STAGE1_RUN = SOURCE_STAGE1_CHECKPOINT.parents[1]

CONDITIONS = [
    {
        "label": "frozen_upstream_control",
        "stage": "stage1",
        "freeze_upstream_encoder": True,
        "upstream_lr": 0.00003,
        "description": "matched answer-only continuation with upstream encoder frozen",
    },
    {
        "label": "upstream_open_lr3e-05",
        "stage": "stage2",
        "freeze_upstream_encoder": False,
        "upstream_lr": 0.00003,
        "description": "answer-only continuation with upstream encoder trainable",
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


def source_stage1_metrics() -> dict[str, float | int]:
    for row in read_rows(SOURCE_STAGE1_RUN / "diagnostic_snapshots.csv"):
        if int(row["step"]) == 60:
            return row_to_metrics(row)
    raise ValueError("missing source Stage 1 step 60 metrics")


def prior_phase4_summary() -> dict[str, Any]:
    canonical = (
        RETAINED_CHECKPOINT.parent
        / "seed2_lowest_retained_canonical_causal_diagnostics/diagnostic_summary.json"
    )
    private = (
        RETAINED_CHECKPOINT.parent
        / "seed2_lowest_retained_private_protocol_diagnostics/private_protocol_summary.json"
    )
    full_enum = (
        RETAINED_CHECKPOINT.parent
        / "seed2_lowest_retained_full_enum_action_loss/model-c-2digit-seed2/"
        "full_enum_summary.json"
    )
    metrics = load_json(RETAINED_CHECKPOINT.parent / "metrics.json")
    return {
        "retained_checkpoint": str(RETAINED_CHECKPOINT),
        "source_stage1_checkpoint": str(SOURCE_STAGE1_CHECKPOINT),
        "source_stage1_operand_exact_match": source_stage1_metrics()[
            "operand_exact_match"
        ],
        "effective_seed": 2,
        "cli_seed": 0,
        "phase4_summary_json": str(PHASE4_SUMMARY),
        "retained_run_dir": str(RETAINED_CHECKPOINT.parent),
        "retained_fast_gate": {
            "final_snapshot": row_to_metrics(
                read_rows(RETAINED_CHECKPOINT.parent / "diagnostic_snapshots.csv")[-1]
            ),
            "final_aux_operand_loss_weight": metrics.get(
                "final_aux_operand_loss_weight"
            ),
            "final_adaptive_interface_loss_weight": metrics.get(
                "final_adaptive_interface_loss_weight"
            ),
            "final_input_proj_anchor_weight": metrics.get(
                "final_input_proj_anchor_weight"
            ),
            "freeze_semantic_decoder": metrics.get("freeze_semantic_decoder"),
            "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
            "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        },
        "retained_diagnostics": {
            "canonical": compact_canonical_summary(load_json(canonical)),
            "private": compact_private_summary(load_json(private)),
            "full_enum": compact_full_enum_summary(load_json(full_enum)),
            "paths": {
                "canonical": str(canonical.parent),
                "private": str(private.parent),
                "full_enum": str(full_enum.parent),
            },
        },
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
        str(RETAINED_CHECKPOINT),
        "--freeze-semantic-decoder",
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
        "0",
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


def run_stage(jobs: int) -> None:
    verify_starting_point()
    tasks: list[tuple[str, list[str], Path]] = []
    for condition in CONDITIONS:
        label = condition["label"]
        root = RUN_ROOT / condition["stage"] / label
        tasks.append((label, common_args(condition), root))

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

    manifest = load_manifest()
    manifest["starting_point"] = prior_phase4_summary()
    manifest["conditions"] = CONDITIONS
    manifest.setdefault("runs", {}).update(results)
    write_manifest(manifest)
    write_summary()


def verify_starting_point() -> None:
    missing = [
        path
        for path in [PHASE4_SUMMARY, RETAINED_CHECKPOINT, SOURCE_STAGE1_CHECKPOINT]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing required Phase 4 artifacts: {missing}")


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


def fast_metrics(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    snapshots = [row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")]
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
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        "input_proj_lr": metrics.get("input_proj_lr"),
        "upstream_lr": metrics.get("upstream_lr"),
        "parameter_delta_from_retained": checkpoint_delta_summary(
            RETAINED_CHECKPOINT, run_dir / "final_weights.pt"
        ),
    }


def summarize_runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    condition_by_label = {condition["label"]: condition for condition in CONDITIONS}
    rows = []
    for label, run_dir_text in sorted(manifest.get("runs", {}).items()):
        row = {
            "label": label,
            "condition": condition_by_label[label],
            **fast_metrics(Path(run_dir_text)),
        }
        row["diagnostics"] = collect_diagnostic_summary(Path(run_dir_text))
        rows.append(row)
    return rows


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


def counterfactual_value(summary: dict[str, Any], condition: str) -> float | None:
    for row in summary.get("counterfactual_exact_match", []):
        if row.get("condition") == condition:
            return row.get("exact_match")
    return None


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


def compact_full_enum_summary(summary: dict[str, Any]) -> dict[str, Any]:
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


def collect_diagnostic_summary(run_dir: Path) -> dict[str, Any]:
    canonical = run_dir / "canonical_causal_diagnostics/diagnostic_summary.json"
    private = run_dir / "private_protocol_diagnostics/private_protocol_summary.json"
    full_enum = find_single_summary(
        run_dir / "full_enum_action_loss", "full_enum_summary.json"
    )
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


def write_summary() -> None:
    manifest = load_manifest()
    summary = {
        "claim": (
            "Allowing upstream parameters to move from the Phase 4 seed 2 retained "
            "checkpoint tests protocol stability and transfer-readiness, not pure discovery."
        ),
        "run_root": str(RUN_ROOT),
        "starting_point": manifest.get("starting_point", prior_phase4_summary()),
        "conditions": CONDITIONS,
        "runs": summarize_runs(manifest) if manifest.get("runs") else [],
        "diagnostic_selections": load_diagnostic_selection_summaries(),
    }
    write_json(RUN_ROOT / "summary.json", summary)
    write_summary_md(summary)
    print(RUN_ROOT / "summary.json")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Upstream-Unfreeze Stability Smoke",
        "",
        f"Run root: `{summary['run_root']}`",
        "",
        "## Claim",
        "",
        summary["claim"],
        "",
        "## Starting point",
        "",
        f"- Retained checkpoint: `{summary['starting_point']['retained_checkpoint']}`",
        f"- Source Stage 1 checkpoint: `{summary['starting_point']['source_stage1_checkpoint']}`",
        f"- Source Stage 1 operand exact: `{summary['starting_point']['source_stage1_operand_exact_match']}`",
        "- Effective seed `2`, CLI seed `0`",
        "",
        "## Fast gates",
        "",
        "| condition | final eval | snapshot normal | inj-zero | forced-random | oracle | operand | pair | calc | A ent | B ent | upstream delta L2 | upstream delta max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        snap = row["final_snapshot"]
        upstream_delta = row["parameter_delta_from_retained"]["groups"].get(
            "upstream_encoder", {}
        )
        lines.append(
            "| "
            + " | ".join(
                [
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
                    fmt(upstream_delta.get("l2", 0.0)),
                    fmt(upstream_delta.get("max_abs", 0.0)),
                ]
            )
            + " |"
        )

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
        diagnostics = row.get("diagnostics", {})
        if not diagnostics.get("complete"):
            lines.append(f"| {row['label']} | pending | pending | pending | pending |")
            continue
        canonical = diagnostics["canonical"]
        private = diagnostics["private"]
        full_enum = diagnostics["full_enum"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["label"],
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

    lines.extend(["", "## Selected checkpoints", ""])
    for row in summary["runs"]:
        lines.append(f"- `{row['label']}`: `{row['selected_checkpoint']}`")
    if summary.get("diagnostic_selections"):
        lines.extend(["", "## Diagnostic selections", ""])
        for selection in summary["diagnostic_selections"]:
            status = "complete" if selection.get("complete") else "pending"
            lines.append(
                f"- `{selection['kind']}` ({status}): `{selection['checkpoint']}`"
            )
    lines.append("")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


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
        print(f"{selection['kind']}: {checkpoint}", flush=True)
        canonical_dir = run_dir / selection["canonical_dir_name"]
        private_dir = run_dir / selection["private_dir_name"]
        full_enum_dir = run_dir / selection["full_enum_dir_name"]
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
        drift = worst_snapshot(row["snapshots"])
        if drift is not None and drift["step"] != row["final_snapshot"].get("step"):
            step = int(drift["step"])
            checkpoint = Path(row["run_dir"]) / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"
            if checkpoint.exists():
                kind = f"{row['label']}_worst_snapshot_step{step:05d}"
                selections.append(
                    {
                        "kind": kind,
                        "checkpoint": str(checkpoint),
                        "run_dir": row["run_dir"],
                        "reason": "worst non-final dense snapshot by calculator-result accuracy",
                        "snapshot": drift,
                        "canonical_dir_name": f"{kind}_canonical_causal_diagnostics",
                        "private_dir_name": f"{kind}_private_protocol_diagnostics",
                        "full_enum_dir_name": f"{kind}_full_enum_action_loss",
                    }
                )
    return selections


def worst_snapshot(snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not snapshots:
        return None
    worst = min(
        snapshots,
        key=lambda row: (
            float(row.get("calculator_result_accuracy", 1.0)),
            float(row.get("operand_exact_match", 1.0)),
        ),
    )
    if float(worst.get("calculator_result_accuracy", 1.0)) >= 0.999:
        return None
    return worst


def summarize_diagnostic_selection(selection: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(selection["run_dir"])
    canonical = run_dir / selection["canonical_dir_name"] / "diagnostic_summary.json"
    private = run_dir / selection["private_dir_name"] / "private_protocol_summary.json"
    full_enum = find_single_summary(
        run_dir / selection["full_enum_dir_name"], "full_enum_summary.json"
    )
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


def find_single_summary(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.glob(f"**/{filename}"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"multiple {filename} files under {root}: {matches}")
    return matches[0]


def load_diagnostic_selection_summaries() -> list[dict[str, Any]]:
    selections_path = RUN_ROOT / "diagnostic_selections.json"
    if selections_path.exists():
        summaries = [
            summarize_diagnostic_selection(selection)
            for selection in load_json(selections_path)
        ]
        write_json(RUN_ROOT / "diagnostic_selection_summaries.json", summaries)
        return summaries
    path = RUN_ROOT / "diagnostic_selection_summaries.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def load_manifest() -> dict[str, Any]:
    path = RUN_ROOT / "manifest.json"
    if path.exists():
        return load_json(path)
    return {}


def write_manifest(manifest: dict[str, Any]) -> None:
    write_json(RUN_ROOT / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "summarize", "diagnostics"])
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.stage == "run":
        run_stage(args.jobs)
    elif args.stage == "summarize":
        write_summary()
    elif args.stage == "diagnostics":
        diagnostics()


if __name__ == "__main__":
    main()
