from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
RUN_ROOT = REPO_ROOT / "runs/2026-05-12_phase6_sum_only_interaction_decoder_gate"
EXISTING_SUM_CHECKPOINT = (
    REPO_ROOT
    / "runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/"
    "model-c-2digit-seed2/final_weights.pt"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }


def checkpoint_config(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    return payload.get("config", {})


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


def all_pair_specs(operand_max: int = 19) -> list[dict[str, int]]:
    return [
        {"sample": a * (operand_max + 1) + b, "true_a": a, "true_b": b}
        for a in range(operand_max + 1)
        for b in range(operand_max + 1)
    ]


def common_natural_args(
    *,
    seed: int,
    steps: int,
    run_root: Path,
    checkpoint: Path | None = None,
    load_scope: str = "semantic_decoder_only",
    estimator: str = "adaptive_interface",
    oracle_train: bool = False,
    freeze_upstream: bool = True,
    input_proj_lr: float = 0.03,
    upstream_lr: float = 0.0003,
    snapshot_every: int = 1,
    checkpoint_every: int = 1,
    snapshot_samples: int = 400,
    eval_samples: int = 400,
    batch_size: int = 64,
    lr: float = 0.003,
    read_position: str = "operand_spans",
    read_span_width: int = 2,
    n_layer: int = 2,
    n_head: int = 1,
    n_embd: int = 16,
    mlp_expansion: int = 1,
    answer_decoder_interaction: str = "none",
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
        str(batch_size),
        "--eval-samples",
        str(eval_samples),
        "--lr",
        f"{lr:g}",
        "--operand-max",
        "19",
        "--calculator-operand-vocab-size",
        "20",
        "--calculator-estimator",
        estimator,
        "--calculator-action-head",
        "independent_operands",
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
        read_position,
        "--calculator-read-span-width",
        str(read_span_width),
        "--calculator-bottleneck-mode",
        "answer_decoder",
        "--calculator-output-format",
        "sum",
        "--answer-decoder-interaction",
        answer_decoder_interaction,
        "--answer-format",
        "sum",
        "--n-layer",
        str(n_layer),
        "--n-head",
        str(n_head),
        "--n-embd",
        str(n_embd),
        "--mlp-expansion",
        str(mlp_expansion),
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
        str(max(snapshot_every, 50)),
    ]
    if checkpoint is not None:
        args.extend(
            [
                "--semantic-decoder-checkpoint",
                str(checkpoint),
                "--semantic-decoder-checkpoint-load-scope",
                load_scope,
            ]
        )
    if oracle_train:
        args.append("--oracle-train")
    else:
        args.append("--freeze-semantic-decoder")
    if freeze_upstream:
        args.append("--freeze-upstream-encoder")
    return args


def zero_step_semantic_gate_run(
    *,
    checkpoint: Path,
    root: Path,
    seed: int,
    read_position: str,
    read_span_width: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    mlp_expansion: int,
    answer_decoder_interaction: str = "none",
) -> Path:
    if not sorted(root.glob("*/summary_metrics.json")):
        run_command(
            common_natural_args(
                seed=seed,
                steps=0,
                run_root=root,
                checkpoint=checkpoint,
                load_scope="semantic_decoder_only",
                read_position=read_position,
                read_span_width=read_span_width,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                mlp_expansion=mlp_expansion,
                answer_decoder_interaction=answer_decoder_interaction,
                snapshot_every=1,
                checkpoint_every=1,
                snapshot_samples=400,
                eval_samples=400,
            ),
            log_path=root / "zero_step_gate.log",
        )
    return latest_run_dir(root)


def run_full_enum(checkpoint: Path, output_root: Path) -> dict[str, Any]:
    if not (output_root / checkpoint.parent.name / "full_enum_summary.json").exists():
        run_command(
            [
                sys.executable,
                "scripts/run_full_enum_action_loss_diagnostic.py",
                "--checkpoint",
                str(checkpoint),
                "--exhaustive-grid",
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
                str(output_root),
            ],
            log_path=output_root / "full_enum.log",
        )
    return load_json(output_root / checkpoint.parent.name / "full_enum_summary.json")


def run_canonical(checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    if not (output_dir / "diagnostic_summary.json").exists():
        run_command(
            [
                sys.executable,
                "-m",
                "scripts.run_causal_calculator_protocol_diagnostics",
                "--checkpoint",
                str(checkpoint),
                "--samples",
                "400",
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
                str(output_dir),
            ],
            log_path=output_dir.parent / f"{output_dir.name}.log",
        )
    return load_json(output_dir / "diagnostic_summary.json")


def counterfactual(summary: dict[str, Any], condition: str) -> float | None:
    for row in summary.get("counterfactual_exact_match", []):
        if row.get("condition") == condition:
            return row.get("exact_match")
    return None


def summarize_snapshot(run_dir: Path) -> dict[str, Any]:
    rows = read_rows(run_dir / "diagnostic_snapshots.csv")
    row = rows[0]
    keys = [
        "normal_exact_match",
        "injection_zero_exact_match",
        "forced_zero_exact_match",
        "forced_random_exact_match",
        "oracle_exact_match",
        "calculator_result_accuracy",
        "operand_exact_match",
        "pair_exact_match",
    ]
    return {key: float(row[key]) for key in keys if key in row and row[key] != ""}


def compact_full_enum(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "best_result_group_matches_true_sum_fraction",
        "best_result_matches_true_sum_fraction",
        "true_result_best_fraction",
        "learned_result_best_fraction",
        "mean_learned_result_minus_best_result_gap",
        "mean_true_result_minus_best_result_gap",
        "mean_same_true_sum_near_best_pair_count",
        "mean_same_best_sum_near_best_pair_count",
        "mean_effective_pair_count",
        "mean_effective_result_count",
        "true_best_fraction",
        "best_matches_true_operands_fraction",
    ]
    return {key: summary.get(key) for key in keys}


def gate_passes(snapshot: dict[str, Any], full_enum: dict[str, Any], sem_delta: dict[str, Any]) -> bool:
    return (
        float(snapshot.get("oracle_exact_match", 0.0)) >= 0.98
        and float(snapshot.get("injection_zero_exact_match", 1.0)) <= 0.10
        and float(snapshot.get("forced_random_exact_match", 1.0)) <= 0.10
        and float(full_enum.get("best_result_group_matches_true_sum_fraction", 0.0)) >= 0.98
        and float(sem_delta.get("l2", 1.0)) == 0.0
    )


def mismatch_summary(rows: list[dict[str, Any]], *, correct_key: str) -> dict[str, Any]:
    by_sum: dict[int, dict[str, int]] = {}
    by_bucket: dict[str, dict[str, int]] = {}
    position_misses: dict[int, int] = {}
    total = 0
    misses = 0
    examples = []
    for row in rows:
        total += 1
        if "true_sum" in row and row["true_sum"] != "":
            true_sum = int(row["true_sum"])
        else:
            prompt = str(row["prompt"])
            left, right = prompt.rstrip("=").split("+", 1)
            true_sum = int(left) + int(right)
        raw_correct = row[correct_key]
        correct = (
            raw_correct
            if isinstance(raw_correct, bool)
            else str(raw_correct).lower() in {"true", "1", "yes"}
        )
        miss = not correct
        misses += int(miss)
        sum_row = by_sum.setdefault(true_sum, {"count": 0, "misses": 0})
        sum_row["count"] += 1
        sum_row["misses"] += int(miss)
        bucket = "00-09" if true_sum < 10 else "10-19" if true_sum < 20 else "20-29" if true_sum < 30 else "30-38"
        bucket_row = by_bucket.setdefault(bucket, {"count": 0, "misses": 0})
        bucket_row["count"] += 1
        bucket_row["misses"] += int(miss)
        if miss:
            target = str(row["target_answer"])
            prediction = str(row["prediction"])
            for idx in range(max(len(target), len(prediction))):
                if (target[idx:idx + 1] or "") != (prediction[idx:idx + 1] or ""):
                    position_misses[idx] = position_misses.get(idx, 0) + 1
                    break
            if len(examples) < 12:
                examples.append(row)
    worst_sums = sorted(
        (
            {"sum": sum_value, **counts, "miss_rate": counts["misses"] / counts["count"]}
            for sum_value, counts in by_sum.items()
        ),
        key=lambda item: (-item["miss_rate"], -item["misses"], item["sum"]),
    )[:10]
    buckets = {
        name: {**counts, "miss_rate": counts["misses"] / counts["count"]}
        for name, counts in sorted(by_bucket.items())
    }
    return {
        "total": total,
        "misses": misses,
        "exact": 1.0 - (misses / max(total, 1)),
        "worst_sums": worst_sums,
        "sum_buckets": buckets,
        "first_wrong_token_position_counts": position_misses,
        "example_misses": examples,
    }


def all_pair_condition_rows(
    checkpoint: Path,
    output_path: Path,
    *,
    oracle: bool = False,
    calculator_result_override: str = "add",
    injection_scale: float | None = None,
    force_true_result: bool = False,
) -> list[dict[str, Any]]:
    if output_path.exists():
        return [dict(row) for row in read_rows(output_path)]
    from scripts.diagnose_calculator_protocol import (  # noqa: WPS433
        decode_tokens,
        generate_answer,
        load_checkpoint,
        make_problem,
        pick_device,
        temporary_calculator_injection_scale,
        trim_after_eos,
    )
    from src.data import max_answer_tokens

    device = pick_device()
    model, _ = load_checkpoint(checkpoint, device=device, injection_scale=None)
    rows: list[dict[str, Any]] = []
    with temporary_calculator_injection_scale(model, injection_scale):
        for spec in all_pair_specs():
            a = int(spec["true_a"])
            b = int(spec["true_b"])
            prompt_ids, target = make_problem(a, b, 2, answer_format="sum")
            forced_class = a + b if force_true_result else None
            torch.manual_seed(100_000 + int(spec["sample"]))
            pred_ids, confidence = generate_answer(
                model,
                prompt_ids=prompt_ids,
                a=a,
                b=b,
                max_new_tokens=max_answer_tokens(2, answer_format="sum"),
                device=device,
                oracle=oracle,
                calculator_result_override=calculator_result_override,
                forced_calculator_result_class=forced_class,
            )
            prediction = decode_tokens(trim_after_eos(pred_ids))
            rows.append(
                {
                    "sample": spec["sample"],
                    "prompt": decode_tokens(prompt_ids),
                    "true_a": a,
                    "true_b": b,
                    "true_sum": a + b,
                    "target_answer": target,
                    "prediction": prediction,
                    "correct": prediction == target,
                    "prediction_confidence": confidence,
                }
            )
    write_rows(output_path, rows)
    return rows


def stage0_existing() -> dict[str, Any]:
    if not EXISTING_SUM_CHECKPOINT.exists():
        raise FileNotFoundError(EXISTING_SUM_CHECKPOINT)
    root = RUN_ROOT / "stage0_existing" / "semantic_only_seed0"
    run_dir = zero_step_semantic_gate_run(
        checkpoint=EXISTING_SUM_CHECKPOINT,
        root=root,
        seed=0,
        read_position="operand_spans",
        read_span_width=2,
        n_layer=2,
        n_head=1,
        n_embd=16,
        mlp_expansion=1,
    )
    checkpoint = run_dir / "final_weights.pt"
    canonical = run_canonical(checkpoint, RUN_ROOT / "stage0_existing" / "canonical_random400")
    full_enum = run_full_enum(checkpoint, RUN_ROOT / "stage0_existing" / "full_enum_all400")
    normal_rows = all_pair_condition_rows(
        checkpoint, RUN_ROOT / "stage0_existing" / "normal_all400_rows.csv"
    )
    oracle_rows = all_pair_condition_rows(
        checkpoint,
        RUN_ROOT / "stage0_existing" / "oracle_at_eval_all400_rows.csv",
        oracle=True,
    )
    injection_zero_rows = all_pair_condition_rows(
        checkpoint,
        RUN_ROOT / "stage0_existing" / "injection_zero_all400_rows.csv",
        injection_scale=0.0,
    )
    forced_zero_rows = all_pair_condition_rows(
        checkpoint,
        RUN_ROOT / "stage0_existing" / "forced_zero_all400_rows.csv",
        calculator_result_override="zero",
    )
    forced_random_rows = all_pair_condition_rows(
        checkpoint,
        RUN_ROOT / "stage0_existing" / "forced_random_all400_rows.csv",
        calculator_result_override="random",
    )
    forced_rows = all_pair_condition_rows(
        checkpoint,
        RUN_ROOT / "stage0_existing" / "forced_true_result_all400_rows.csv",
        force_true_result=True,
    )
    full_enum_rows = read_rows(
        RUN_ROOT
        / "stage0_existing"
        / "full_enum_all400"
        / checkpoint.parent.name
        / "full_enum_rows.csv"
    )
    full_enum_mismatch_rows = [
        {
            "sample": row["sample"],
            "prompt": row["prompt"],
            "true_a": row["true_a"],
            "true_b": row["true_b"],
            "true_sum": row["true_sum"],
            "target_answer": "",
            "prediction": f"best_result={row['best_result']}",
            "correct": str(row["best_result_group_matches_true_sum"]) == "True",
        }
        for row in full_enum_rows
    ]
    snapshot = summarize_snapshot(run_dir)
    sem_delta = checkpoint_delta_summary(EXISTING_SUM_CHECKPOINT, checkpoint)["groups"].get(
        "semantic_decoder", {"l2": 0.0, "max_abs": 0.0}
    )
    source_cfg = checkpoint_config(EXISTING_SUM_CHECKPOINT)
    gate_cfg = checkpoint_config(checkpoint)
    source_model = source_cfg.get("model", {})
    gate_model = gate_cfg.get("model", {})
    normal_summary = mismatch_summary(normal_rows, correct_key="correct")
    oracle_summary = mismatch_summary(oracle_rows, correct_key="correct")
    injection_zero_summary = mismatch_summary(injection_zero_rows, correct_key="correct")
    forced_zero_summary = mismatch_summary(forced_zero_rows, correct_key="correct")
    forced_random_summary = mismatch_summary(forced_random_rows, correct_key="correct")
    forced_true_summary = mismatch_summary(forced_rows, correct_key="correct")
    diagnosis = {
        "run_root": str(RUN_ROOT),
        "source_checkpoint": str(EXISTING_SUM_CHECKPOINT),
        "semantic_only_run_dir": str(run_dir),
        "semantic_only_checkpoint": str(checkpoint),
        "source_metadata_audit": {
            "answer_format": source_cfg.get("answer_format"),
            "calculator_output_format": source_cfg.get("calculator_output_format"),
            "calculator_read_position": source_cfg.get("calculator_read_position"),
            "calculator_read_span_width": source_cfg.get("calculator_read_span_width"),
            "calculator_bottleneck_mode": source_cfg.get("calculator_bottleneck_mode"),
            "semantic_decoder_checkpoint_load_scope": source_cfg.get("semantic_decoder_checkpoint_load_scope"),
            "n_layer": source_cfg.get("n_layer"),
            "n_head": source_cfg.get("n_head"),
            "n_embd": source_cfg.get("n_embd"),
            "model": {
                "calculator_read_position": source_model.get("calculator_read_position"),
                "calculator_read_span_width": source_model.get("calculator_read_span_width"),
                "calculator_bottleneck_mode": source_model.get("calculator_bottleneck_mode"),
                "calculator_output_format": source_model.get("calculator_output_format"),
            },
        },
        "gate_metadata_audit": {
            "answer_format": gate_cfg.get("answer_format"),
            "calculator_output_format": gate_cfg.get("calculator_output_format"),
            "calculator_read_position": gate_cfg.get("calculator_read_position"),
            "calculator_read_span_width": gate_cfg.get("calculator_read_span_width"),
            "calculator_bottleneck_mode": gate_cfg.get("calculator_bottleneck_mode"),
            "semantic_decoder_checkpoint_load_scope": gate_cfg.get("semantic_decoder_checkpoint_load_scope"),
            "model": {
                "calculator_read_position": gate_model.get("calculator_read_position"),
                "calculator_read_span_width": gate_model.get("calculator_read_span_width"),
                "calculator_bottleneck_mode": gate_model.get("calculator_bottleneck_mode"),
                "calculator_output_format": gate_model.get("calculator_output_format"),
            },
        },
        "snapshot": snapshot,
        "canonical_all400": {
            "normal_exact_match": normal_summary["exact"],
            "oracle_at_eval_exact_match": oracle_summary["exact"],
            "injection_zero_exact_match": injection_zero_summary["exact"],
            "forced_zero_exact_match": forced_zero_summary["exact"],
            "forced_random_exact_match": forced_random_summary["exact"],
            "random400_forced_result_sweep_true_sum_is_best_fraction": canonical.get(
                "forced_result_sweep_summary", {}
            ).get("true_sum_is_best_fraction"),
        },
        "forced_true_result_exact_match": forced_true_summary,
        "oracle_at_eval_miss_summary": oracle_summary,
        "full_enum_best_result_miss_summary": mismatch_summary(
            full_enum_mismatch_rows, correct_key="correct"
        ),
        "full_enum": compact_full_enum(full_enum),
        "semantic_decoder_delta": sem_delta,
        "forced_true_result_agrees_with_oracle_operand_injection": (
            forced_true_summary["exact"] == oracle_summary["exact"]
            and forced_true_summary["misses"] == oracle_summary["misses"]
        ),
        "passes_gate": (
            oracle_summary["exact"] >= 0.98
            and injection_zero_summary["exact"] <= 0.10
            and forced_random_summary["exact"] <= 0.10
            and float(full_enum.get("best_result_group_matches_true_sum_fraction", 0.0))
            >= 0.98
            and float(sem_delta.get("l2", 1.0)) == 0.0
        ),
    }
    write_json(RUN_ROOT / "stage0_existing_decoder_diagnosis.json", diagnosis)
    write_existing_diagnosis_md(diagnosis)
    write_summary()
    return diagnosis


def write_existing_diagnosis_md(diagnosis: dict[str, Any]) -> None:
    oracle = diagnosis["canonical_all400"]
    forced = diagnosis["forced_true_result_exact_match"]
    full_enum = diagnosis["full_enum"]
    lines = [
        "# Stage 0 Existing Sum-Only Decoder Diagnosis",
        "",
        f"Source checkpoint: `{diagnosis['source_checkpoint']}`",
        "",
        "| oracle-at-eval | forced true result | injection-zero | forced-random | full-enum best result=true | semantic delta | gate |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        "| "
        + " | ".join(
            [
                fmt(oracle["oracle_at_eval_exact_match"]),
                fmt(forced["exact"]),
                fmt(oracle["injection_zero_exact_match"]),
                fmt(oracle["forced_random_exact_match"]),
                fmt(full_enum["best_result_group_matches_true_sum_fraction"]),
                fmt(diagnosis["semantic_decoder_delta"].get("l2")),
                str(diagnosis["passes_gate"]),
            ]
        )
        + " |",
        "",
        "Worst oracle-at-eval sums:",
        "",
        "| sum | count | misses | miss rate |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in diagnosis["oracle_at_eval_miss_summary"]["worst_sums"][:8]:
        lines.append(
            f"| {row['sum']} | {row['count']} | {row['misses']} | {row['miss_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Worst full-enum best-result sums:",
            "",
            "| sum | count | misses | miss rate |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnosis["full_enum_best_result_miss_summary"]["worst_sums"][:8]:
        lines.append(
            f"| {row['sum']} | {row['count']} | {row['misses']} | {row['miss_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: oracle operands and forced true result classes are the same semantic-decoder test in the sum-only branch. Any miss here is a decoder/result-readout failure, not learned-interface evidence.",
            "",
        ]
    )
    (RUN_ROOT / "stage0_existing_decoder_diagnosis.md").write_text("\n".join(lines))


def train_oracle_candidate(
    *,
    name: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    read_position: str,
    read_span_width: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    mlp_expansion: int,
    answer_decoder_interaction: str,
) -> Path:
    root = RUN_ROOT / "stage0_candidates" / name / "oracle_train"
    if not sorted(root.glob("*/summary_metrics.json")):
        run_command(
            common_natural_args(
                seed=seed,
                steps=steps,
                run_root=root,
                checkpoint=None,
                estimator="ste",
                oracle_train=True,
                freeze_upstream=False,
                input_proj_lr=lr,
                upstream_lr=lr,
                batch_size=batch_size,
                lr=lr,
                read_position=read_position,
                read_span_width=read_span_width,
                n_layer=n_layer,
                n_head=n_head,
                n_embd=n_embd,
                mlp_expansion=mlp_expansion,
                answer_decoder_interaction=answer_decoder_interaction,
                snapshot_every=250,
                checkpoint_every=250,
                snapshot_samples=128,
                eval_samples=512,
            ),
            log_path=root / f"{name}.log",
        )
    return latest_run_dir(root)


def candidate_checkpoint_paths(run_dir: Path) -> list[Path]:
    cfg = load_json(run_dir / "config.json")
    steps = int(cfg["steps"])
    paths = sorted((run_dir / "checkpoint_snapshots").glob("step_*_weights.pt"))
    final = run_dir / "final_weights.pt"
    if final.exists() and (not paths or paths[-1].name != f"step_{steps:05d}_weights.pt"):
        paths.append(final)
    return paths


def evaluate_candidate(
    *,
    name: str,
    source_checkpoint: Path,
    seed: int,
    read_position: str,
    read_span_width: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
    mlp_expansion: int,
    answer_decoder_interaction: str,
) -> dict[str, Any]:
    root = RUN_ROOT / "stage0_candidates" / name / source_checkpoint.stem
    run_dir = zero_step_semantic_gate_run(
        checkpoint=source_checkpoint,
        root=root / "semantic_only_gate",
        seed=seed,
        read_position=read_position,
        read_span_width=read_span_width,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_expansion=mlp_expansion,
        answer_decoder_interaction=answer_decoder_interaction,
    )
    checkpoint = run_dir / "final_weights.pt"
    snapshot = summarize_snapshot(run_dir)
    full_enum = run_full_enum(checkpoint, root / "full_enum")
    sem_delta = checkpoint_delta_summary(source_checkpoint, checkpoint)["groups"].get(
        "semantic_decoder", {"l2": 0.0, "max_abs": 0.0}
    )
    source_metrics = {}
    if (source_checkpoint.parent / "metrics.json").exists():
        source_metrics = load_json(source_checkpoint.parent / "metrics.json")
    return {
        "name": name,
        "source_checkpoint": str(source_checkpoint),
        "semantic_only_run_dir": str(run_dir),
        "semantic_only_checkpoint": str(checkpoint),
        "oracle_train_eval_exact": source_metrics.get("exact_match"),
        "snapshot": snapshot,
        "full_enum": compact_full_enum(full_enum),
        "semantic_decoder_delta": sem_delta,
        "read_position": read_position,
        "read_span_width": read_span_width,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "mlp_expansion": mlp_expansion,
        "answer_decoder_interaction": answer_decoder_interaction,
        "passes_gate": gate_passes(snapshot, full_enum, sem_delta),
    }


def stage0_candidates() -> dict[str, Any]:
    stage0_existing()
    candidate_specs = [
        {
            "name": "tiny_operand_spans_dense",
            "seed": 0,
            "steps": 1000,
            "batch_size": 400,
            "lr": 0.003,
            "read_position": "operand_spans",
            "read_span_width": 2,
            "n_layer": 2,
            "n_head": 1,
            "n_embd": 16,
            "mlp_expansion": 1,
            "answer_decoder_interaction": "product",
        },
        {
            "name": "embd32_heads2_operand_spans",
            "seed": 1,
            "steps": 2000,
            "batch_size": 400,
            "lr": 0.003,
            "read_position": "operand_spans",
            "read_span_width": 2,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 32,
            "mlp_expansion": 1,
            "answer_decoder_interaction": "product",
        },
    ]
    rows: list[dict[str, Any]] = []
    selected = None
    for spec in candidate_specs:
        run_dir = train_oracle_candidate(**spec)
        evaluated: list[dict[str, Any]] = []
        for checkpoint in candidate_checkpoint_paths(run_dir):
            row = evaluate_candidate(
                name=spec["name"],
                source_checkpoint=checkpoint,
                seed=spec["seed"],
                read_position=spec["read_position"],
                read_span_width=spec["read_span_width"],
                n_layer=spec["n_layer"],
                n_head=spec["n_head"],
                n_embd=spec["n_embd"],
                mlp_expansion=spec["mlp_expansion"],
                answer_decoder_interaction=spec["answer_decoder_interaction"],
            )
            evaluated.append(row)
            rows.append(row)
        best = max(
            evaluated,
            key=lambda row: (
                float(row["snapshot"].get("oracle_exact_match", 0.0)),
                float(row["full_enum"].get("best_result_group_matches_true_sum_fraction") or 0.0),
            ),
        )
        if best["passes_gate"]:
            selected = best
            break
    summary = {
        "candidates": rows,
        "selected_passing_candidate": selected,
        "passes_stage0b": selected is not None,
    }
    write_json(RUN_ROOT / "stage0_candidate_summary.json", summary)
    write_summary()
    return summary


def selected_stage0_candidate() -> dict[str, Any] | None:
    path = RUN_ROOT / "stage0_candidate_summary.json"
    if not path.exists():
        return None
    return load_json(path).get("selected_passing_candidate")


def relaxed_args(candidate: dict[str, Any], *, run_root: Path, seed: int) -> list[str]:
    args = common_natural_args(
        seed=seed,
        steps=300,
        run_root=run_root,
        checkpoint=Path(candidate["source_checkpoint"]),
        load_scope="semantic_decoder_only",
        estimator="gumbel_concrete_interface",
        oracle_train=False,
        freeze_upstream=True,
        input_proj_lr=0.03,
        upstream_lr=0.0003,
        read_position=candidate["read_position"],
        read_span_width=int(candidate["read_span_width"]),
        n_layer=int(candidate["n_layer"]),
        n_head=int(candidate["n_head"]),
        n_embd=int(candidate["n_embd"]),
        mlp_expansion=int(candidate["mlp_expansion"]),
        answer_decoder_interaction=str(candidate["answer_decoder_interaction"]),
        snapshot_every=25,
        checkpoint_every=25,
        snapshot_samples=400,
        eval_samples=512,
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


def retention_args(candidate: dict[str, Any], *, run_root: Path, checkpoint: Path, seed: int) -> list[str]:
    return common_natural_args(
        seed=seed,
        steps=1000,
        run_root=run_root,
        checkpoint=checkpoint,
        load_scope="full_model",
        estimator="adaptive_interface",
        oracle_train=False,
        freeze_upstream=True,
        input_proj_lr=0.0003,
        upstream_lr=0.0003,
        read_position=candidate["read_position"],
        read_span_width=int(candidate["read_span_width"]),
        n_layer=int(candidate["n_layer"]),
        n_head=int(candidate["n_head"]),
        n_embd=int(candidate["n_embd"]),
        mlp_expansion=int(candidate["mlp_expansion"]),
        answer_decoder_interaction=str(candidate["answer_decoder_interaction"]),
        snapshot_every=50,
        checkpoint_every=50,
        snapshot_samples=400,
        eval_samples=512,
    )


def checkpoint_for_step(run_dir: Path, step: int) -> Path:
    cfg = load_json(run_dir / "config.json")
    if step == int(cfg.get("steps", -1)):
        return run_dir / "final_weights.pt"
    return run_dir / "checkpoint_snapshots" / f"step_{step:05d}_weights.pt"


def row_to_metrics(row: dict[str, str]) -> dict[str, float | int]:
    out: dict[str, float | int] = {"step": int(row["step"])}
    for key, value in row.items():
        if key == "step" or value == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def analyze_training_run(run_dir: Path, *, threshold: float) -> dict[str, Any]:
    snapshots = [row_to_metrics(row) for row in read_rows(run_dir / "diagnostic_snapshots.csv")]
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
            -int(row["step"]),
        ),
    )
    selected = qualifying[0] if qualifying else best
    metrics = load_json(run_dir / "metrics.json")
    checkpoint = checkpoint_for_step(run_dir, int(selected["step"]))
    return {
        "run_dir": str(run_dir),
        "selected_snapshot": selected,
        "best_snapshot": best,
        "selected_checkpoint": str(checkpoint),
        "passes_gate": bool(qualifying),
        "final_eval_exact": metrics.get("exact_match"),
        "final_objective_weights": {
            "answer_loss_weight": metrics.get("answer_loss_weight"),
            "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
            "final_adaptive_interface_loss_weight": metrics.get("final_adaptive_interface_loss_weight"),
            "final_local_target_loss_weight": metrics.get("final_local_target_loss_weight"),
            "final_expected_answer_loss_weight": metrics.get("final_expected_answer_loss_weight"),
            "final_relaxed_calculator_entropy_weight": metrics.get("final_relaxed_calculator_entropy_weight"),
            "final_input_proj_anchor_weight": metrics.get("final_input_proj_anchor_weight"),
        },
    }


def stage1() -> dict[str, Any]:
    candidate = selected_stage0_candidate()
    if candidate is None:
        raise RuntimeError("Stage 0B has no passing candidate; not running bridge")
    root = RUN_ROOT / "stage1_natural_bridge"
    if not sorted(root.glob("*/summary_metrics.json")):
        run_command(relaxed_args(candidate, run_root=root, seed=0), log_path=root / "stage1.log")
    row = analyze_training_run(latest_run_dir(root), threshold=0.95)
    row["selected_stage0_candidate"] = candidate
    step0 = Path(row["run_dir"]) / "checkpoint_snapshots/step_00000_weights.pt"
    if step0.exists():
        row["parameter_delta_step0_to_selected"] = checkpoint_delta_summary(
            step0, Path(row["selected_checkpoint"])
        )
    write_json(RUN_ROOT / "stage1_summary.json", row)
    write_summary()
    return row


def stage2() -> dict[str, Any]:
    candidate = selected_stage0_candidate()
    stage1_summary = load_json(RUN_ROOT / "stage1_summary.json")
    if candidate is None or not stage1_summary.get("passes_gate"):
        raise RuntimeError("Stage 1 has no passing bridge checkpoint; not running retention")
    root = RUN_ROOT / "stage2_retention"
    source = Path(stage1_summary["selected_checkpoint"])
    if not sorted(root.glob("*/summary_metrics.json")):
        run_command(
            retention_args(candidate, run_root=root, checkpoint=source, seed=0),
            log_path=root / "stage2.log",
        )
    row = analyze_training_run(latest_run_dir(root), threshold=0.98)
    row["source_checkpoint"] = str(source)
    step0 = Path(row["run_dir"]) / "checkpoint_snapshots/step_00000_weights.pt"
    if step0.exists():
        row["parameter_delta_step0_to_selected"] = checkpoint_delta_summary(
            step0, Path(row["selected_checkpoint"])
        )
    write_json(RUN_ROOT / "stage2_summary.json", row)
    write_summary()
    return row


def diagnostic_items() -> list[dict[str, Any]]:
    items = []
    for path, label in [
        (RUN_ROOT / "stage1_summary.json", "stage1_selected"),
        (RUN_ROOT / "stage2_summary.json", "stage2_selected"),
    ]:
        if path.exists():
            row = load_json(path)
            items.append({"label": label, "checkpoint": row["selected_checkpoint"]})
    return items


def diagnostics() -> None:
    items = diagnostic_items()
    for item in items:
        label = item["label"]
        checkpoint = Path(item["checkpoint"])
        canonical = run_canonical(checkpoint, RUN_ROOT / "diagnostics" / f"{label}_canonical")
        full_enum = run_full_enum(checkpoint, RUN_ROOT / "diagnostics" / f"{label}_full_enum")
        private_dir = RUN_ROOT / "diagnostics" / f"{label}_private"
        if not (private_dir / "private_protocol_summary.json").exists():
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
                log_path=RUN_ROOT / "diagnostics" / f"{label}_private.log",
            )
        item["canonical"] = {
            "normal_exact_match": canonical.get("exact_match"),
            "oracle_at_eval_exact_match": counterfactual(canonical, "oracle_at_eval"),
            "injection_zero_exact_match": counterfactual(canonical, "injection_zero"),
            "forced_random_exact_match": counterfactual(canonical, "forced_random"),
            "calculator_result_accuracy": canonical.get("calculator_result_accuracy"),
        }
        item["full_enum"] = compact_full_enum(full_enum)
        item["private"] = load_json(private_dir / "private_protocol_summary.json")
    write_json(RUN_ROOT / "diagnostic_summary.json", items)
    write_summary()


def write_summary() -> None:
    summary = {
        "run_root": str(RUN_ROOT),
        "stage0_existing": load_json(RUN_ROOT / "stage0_existing_decoder_diagnosis.json")
        if (RUN_ROOT / "stage0_existing_decoder_diagnosis.json").exists()
        else {},
        "stage0_candidates": load_json(RUN_ROOT / "stage0_candidate_summary.json")
        if (RUN_ROOT / "stage0_candidate_summary.json").exists()
        else {},
        "stage1": load_json(RUN_ROOT / "stage1_summary.json")
        if (RUN_ROOT / "stage1_summary.json").exists()
        else {},
        "stage2": load_json(RUN_ROOT / "stage2_summary.json")
        if (RUN_ROOT / "stage2_summary.json").exists()
        else {},
        "diagnostics": load_json(RUN_ROOT / "diagnostic_summary.json")
        if (RUN_ROOT / "diagnostic_summary.json").exists()
        else [],
    }
    labels = []
    selected = summary.get("stage0_candidates", {}).get("selected_passing_candidate")
    if selected:
        labels.append("sum_only_interaction_gate_positive")
    elif summary.get("stage0_candidates"):
        labels.append("sum_only_interaction_gate_negative")
    if summary.get("stage1", {}).get("passes_gate"):
        labels.append("natural_deterministic_concrete_result_positive")
    elif summary.get("stage1"):
        labels.append("natural_deterministic_concrete_result_negative")
    if summary.get("stage2", {}).get("passes_gate"):
        labels.append("natural_retention_positive")
    elif summary.get("stage2"):
        labels.append("natural_retention_negative")
    summary["interpretation_labels"] = labels
    write_json(RUN_ROOT / "summary.json", summary)
    write_summary_md(summary)


def write_summary_md(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 6 Sum-Only Interaction Decoder Gate",
        "",
        f"Run root: `{RUN_ROOT}`",
        "",
        f"Labels: `{', '.join(summary.get('interpretation_labels', []))}`",
        "",
        "## Stage 0A Existing Decoder",
        "",
    ]
    existing = summary.get("stage0_existing", {})
    if existing:
        oracle = existing["canonical_all400"]
        full_enum = existing["full_enum"]
        lines.extend(
            [
                "| oracle-at-eval | injection-zero | forced-random | best result=true | semantic delta | pass |",
                "| ---: | ---: | ---: | ---: | ---: | --- |",
                "| "
                + " | ".join(
                    [
                        fmt(oracle["oracle_at_eval_exact_match"]),
                        fmt(oracle["injection_zero_exact_match"]),
                        fmt(oracle["forced_random_exact_match"]),
                        fmt(full_enum["best_result_group_matches_true_sum_fraction"]),
                        fmt(existing["semantic_decoder_delta"].get("l2")),
                        str(existing["passes_gate"]),
                    ]
                )
                + " |",
                "",
            ]
        )
    lines.extend(
        [
            "## Stage 0B Candidates",
            "",
            "| candidate | interaction | checkpoint | oracle | best result=true | inj-zero | forced-random | semantic delta | pass |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary.get("stage0_candidates", {}).get("candidates", []):
        snap = row["snapshot"]
        fe = row["full_enum"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    row.get("answer_decoder_interaction", ""),
                    f"`{row['source_checkpoint']}`",
                    fmt(snap.get("oracle_exact_match")),
                    fmt(fe.get("best_result_group_matches_true_sum_fraction")),
                    fmt(snap.get("injection_zero_exact_match")),
                    fmt(snap.get("forced_random_exact_match")),
                    fmt(row["semantic_decoder_delta"].get("l2")),
                    str(row["passes_gate"]),
                ]
            )
            + " |"
        )
    for label in ["stage1", "stage2"]:
        row = summary.get(label, {})
        if row:
            selected = row["selected_snapshot"]
            lines.extend(
                [
                    "",
                    f"## {label.upper()}",
                    "",
                    "| selected step | answer exact | result acc | final eval | pass | checkpoint |",
                    "| ---: | ---: | ---: | ---: | --- | --- |",
                    "| "
                    + " | ".join(
                        [
                            fmt(selected.get("step")),
                            fmt(selected.get("normal_exact_match")),
                            fmt(selected.get("calculator_result_accuracy")),
                            fmt(row.get("final_eval_exact")),
                            str(row.get("passes_gate")),
                            f"`{row.get('selected_checkpoint')}`",
                        ]
                    )
                    + " |",
                ]
            )
    diagnostics = summary.get("diagnostics", [])
    if diagnostics:
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "| label | normal | result acc | learned-result best | learned-result gap | oracle | inj-zero | forced-random |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in diagnostics:
            canonical = item.get("canonical", {})
            full_enum = item.get("full_enum", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        item.get("label", ""),
                        fmt(canonical.get("normal_exact_match")),
                        fmt(canonical.get("calculator_result_accuracy")),
                        fmt(full_enum.get("learned_result_best_fraction")),
                        fmt(full_enum.get("mean_learned_result_minus_best_result_gap")),
                        fmt(canonical.get("oracle_at_eval_exact_match")),
                        fmt(canonical.get("injection_zero_exact_match")),
                        fmt(canonical.get("forced_random_exact_match")),
                    ]
                )
                + " |"
            )
    lines.append("")
    (RUN_ROOT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 sum-only semantic decoder gate.")
    parser.add_argument(
        "command",
        choices=[
            "stage0-existing",
            "stage0-candidates",
            "stage1",
            "stage2",
            "diagnostics",
            "summarize",
            "all",
        ],
    )
    args = parser.parse_args()
    if args.command == "stage0-existing":
        stage0_existing()
    elif args.command == "stage0-candidates":
        stage0_candidates()
    elif args.command == "stage1":
        stage1()
    elif args.command == "stage2":
        stage2()
    elif args.command == "diagnostics":
        diagnostics()
    elif args.command == "summarize":
        write_summary()
    elif args.command == "all":
        stage0_candidates()
        if selected_stage0_candidate() is not None:
            stage1_summary = stage1()
            if stage1_summary.get("passes_gate"):
                stage2()
            diagnostics()


if __name__ == "__main__":
    main()
