from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_calculator_protocol import pick_device  # noqa: E402
from scripts.overfit_one_batch import (  # noqa: E402
    action_loss_weights_from_losses,
    calculator_read_operand_logits,
    fixed_width_operands_from_batch,
    freeze_semantic_decoder_parameters,
    freeze_upstream_encoder_parameters,
    full_enum_action_pairs,
    load_semantic_decoder_checkpoint,
    make_model_config,
    make_range_batch,
    score_action_loss_candidates_chunked,
)
from src.model import TinyGPT  # noqa: E402


RUN_ROOT = REPO_ROOT / "runs" / "2026-05-11_phase6_strict_random_upstream_local_target"
RUNNER_SCRIPT = "scripts/run_phase6_strict_random_upstream_local_target.py"
DEFAULT_LOAD_SCOPE = "semantic_decoder_only"
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def log_command(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "commands.jsonl").open("a") as handle:
        handle.write(json.dumps({"command": command, "cwd": str(cwd)}) + "\n")


def run_command(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    log_command(command, cwd=cwd)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/codex_pycache")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not env.get("PYTHONPATH")
        else str(REPO_ROOT) + os.pathsep + env["PYTHONPATH"]
    )
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def semantic_decoder_names(state: dict[str, torch.Tensor]) -> list[str]:
    prefixes = (
        "calculator_hook.output_proj.",
        "answer_offset_emb.",
        "answer_decoder.",
    )
    return [name for name in state if name.startswith(prefixes)]


def parameter_group_for_name(name: str) -> str:
    if name.startswith("calculator_hook.input_proj."):
        return "calculator_hook.input_proj"
    if name.startswith("calculator_hook.pair_proj."):
        return "calculator_hook.pair_proj"
    if name.startswith(("calculator_hook.output_proj.", "answer_offset_emb.", "answer_decoder.")):
        return "semantic_decoder"
    if name.startswith(("tok_emb.", "pos_emb.", "blocks.", "ln_f.", "lm_head.")):
        return "upstream_encoder"
    return "other"


def checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    return payload.get("model_state_dict", payload)


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
        row["tensor_count"] += 1
        row["element_count"] += int(delta.numel())
        row["changed_tensor_count"] += int(delta.abs().max().item() > 0.0)
        row["l2"] += l2_sq
        row["mean_abs_numerator"] += float(delta.abs().sum().item())
        row["max_abs"] = max(float(row["max_abs"]), float(delta.abs().max().item()))
    for row in groups.values():
        row["l2"] = math.sqrt(float(row["l2"]))
        elements = int(row["element_count"])
        row["mean_abs"] = float(row.pop("mean_abs_numerator")) / elements if elements else 0.0
    return {"before": str(before_path), "after": str(after_path), "groups": groups}


def model_group_delta_summary(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, Any]:
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
        delta = (after_tensor.float().cpu() - before_tensor.float().cpu()).reshape(-1)
        row["l2"] += float(torch.dot(delta, delta).item())
        row["max_abs"] = max(float(row["max_abs"]), float(delta.abs().max().item()))
        row["changed_tensor_count"] += int(delta.abs().max().item() > 0.0)
        row["tensor_count"] += 1
    for row in groups.values():
        row["l2"] = math.sqrt(float(row["l2"]))
    return groups


def semantic_grad_summary(model: torch.nn.Module) -> dict[str, float | int]:
    l2_sq = 0.0
    max_abs = 0.0
    tensors_with_grad = 0
    for name, param in model.named_parameters():
        if parameter_group_for_name(name) != "semantic_decoder":
            continue
        if param.grad is None:
            continue
        grad = param.grad.detach().float().reshape(-1)
        tensors_with_grad += 1
        l2_sq += float(torch.dot(grad, grad).item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
    return {
        "semantic_decoder_grad_l2": math.sqrt(l2_sq),
        "semantic_decoder_grad_max_abs": max_abs,
        "semantic_decoder_tensors_with_grad": tensors_with_grad,
    }


def build_phase6_model(
    *,
    checkpoint: Path,
    load_scope: str,
    device: str,
    seed: int,
    estimator: str = "identifiable_full_enum_local_target",
    freeze_upstream: bool = True,
) -> TinyGPT:
    torch.manual_seed(seed + 2)
    cfg = make_model_config(
        2,
        "model-c",
        operand_vocab_size=20,
        calculator_estimator=estimator,
        calculator_action_head="independent_operands",
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        calculator_output_format="sum_left_operand",
        answer_format="sum_left_operand",
        n_layer=2,
        n_head=1,
        n_embd=16,
        mlp_expansion=1,
        calculator_hook_after_layer=1,
    )
    model = TinyGPT(cfg).to(device)
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope=load_scope)
    freeze_semantic_decoder_parameters(model)
    if freeze_upstream:
        freeze_upstream_encoder_parameters(model)
    return model


def strict_baseline_run_dir() -> Path | None:
    root = RUN_ROOT / "stage0" / f"oracle_wiring_gate_{DEFAULT_LOAD_SCOPE}"
    runs = all_model_runs(root)
    return runs[-1] if runs else None


def checkpoint_or_none(run_dir: Path | None) -> Path | None:
    if run_dir is None:
        return None
    checkpoint = run_dir / "final_weights.pt"
    return checkpoint if checkpoint.exists() else None


def hard_best_local_target_loss_and_metrics(
    model: torch.nn.Module,
    batch: Any,
    *,
    num_digits: int,
    temperature: float,
    min_probability_floor: float,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    classes = a_logits.shape[-1]
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    candidates = pairs.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    with torch.no_grad():
        full_losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
        weights = action_loss_weights_from_losses(
            full_losses,
            temperature=temperature,
            min_probability_floor=min_probability_floor,
        )
    best_idx = full_losses.argmin(dim=-1)
    best_pairs = pairs.index_select(0, best_idx)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_idx = true_a * classes + true_b
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    true_losses = full_losses.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    best_losses = full_losses.gather(1, best_idx.unsqueeze(-1)).squeeze(-1)
    true_pair_ranks = (full_losses < true_losses.unsqueeze(-1)).sum(dim=-1) + 1
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    true_pair_probs = weights.gather(1, true_idx.unsqueeze(-1)).squeeze(-1)
    local_ce = (F.cross_entropy(a_logits, best_pairs[:, 0]) + F.cross_entropy(b_logits, best_pairs[:, 1])) / 2
    aux_ce = (F.cross_entropy(a_logits, true_a) + F.cross_entropy(b_logits, true_b)) / 2
    metrics = {
        "samples": int(batch.x.shape[0]),
        "hard_best_pair_equals_true_pair": float((best_idx == true_idx).float().mean().item()),
        "hard_best_a_target_equals_true_a": float((best_pairs[:, 0] == true_a).float().mean().item()),
        "hard_best_b_target_equals_true_b": float((best_pairs[:, 1] == true_b).float().mean().item()),
        "hard_best_local_ce": float(local_ce.detach().item()),
        "direct_aux_ce_same_logits": float(aux_ce.detach().item()),
        "local_minus_aux_ce": float((local_ce.detach() - aux_ce.detach()).item()),
        "best_true_fraction": float((best_idx == true_idx).float().mean().item()),
        "tie_aware_true_best_fraction": float((true_losses <= best_losses + 1e-6).float().mean().item()),
        "mean_true_pair_rank": float(true_pair_ranks.float().mean().item()),
        "mean_target_entropy": float(entropy.mean().item()),
        "mean_effective_pairs": float(entropy.exp().mean().item()),
        "mean_true_pair_probability": float(true_pair_probs.mean().item()),
        "target_constructed_without_true_operands": True,
        "true_operands_used_only_for_reporting_and_aux_comparison": True,
        "temperature": temperature,
        "min_probability_floor": min_probability_floor,
        "chunk_size": chunk_size,
    }
    return local_ce, metrics


def parity_gate_passed() -> bool:
    gate_path = RUN_ROOT / "parity_gate.json"
    if not gate_path.exists():
        return False
    gate = load_json(gate_path)
    return bool(gate.get("passes_gate"))


def oracle_wiring_gate_passed() -> bool:
    gate_path = RUN_ROOT / "oracle_wiring_gate.json"
    if not gate_path.exists():
        return False
    gate = load_json(gate_path)
    return bool(gate.get("passes_gate"))


def cmd_oracle_wiring_gate(args: argparse.Namespace) -> None:
    log_command([sys.executable, RUNNER_SCRIPT, *sys.argv[1:]])
    load_scope = args.semantic_decoder_checkpoint_load_scope
    label = f"oracle_wiring_gate_{load_scope}"
    run_root = RUN_ROOT / "stage0" / label
    run_command(
        phase6_train_command(
            checkpoint=args.checkpoint or STAGE0B_CHECKPOINT,
            run_root=run_root,
            estimator="adaptive_interface",
            answer_loss_weight=0.0,
            local_target_loss_weight=0.0,
            input_proj_lr=args.input_proj_lr,
            upstream_lr=args.upstream_lr,
            steps=0,
            snapshot_every=1,
            checkpoint_every=1,
            target_mode="hard_best_pair",
            freeze_upstream=True,
            seed=args.seed,
            load_scope=load_scope,
        )
    )
    run_dir = latest_model_run(run_root)
    metrics = load_json(run_dir / "metrics.json")
    source = args.checkpoint or STAGE0B_CHECKPOINT
    deltas = checkpoint_delta_summary(source, run_dir / "final_weights.pt")
    semantic_delta = deltas["groups"].get("semantic_decoder", {"l2": 0.0, "max_abs": 0.0})
    counterfactuals = metrics.get("counterfactuals", {})
    gate = {
        "checkpoint": str(run_dir / "final_weights.pt"),
        "run_dir": str(run_dir),
        "source_checkpoint": str(source),
        "semantic_decoder_checkpoint_load_scope": load_scope,
        "built_in_eval_exact_match": metrics.get("exact_match"),
        "injection_zero_exact_match": counterfactuals.get("injection_zero_exact_match"),
        "forced_zero_exact_match": counterfactuals.get("forced_zero_exact_match"),
        "forced_random_exact_match": counterfactuals.get("forced_random_exact_match"),
        "oracle_at_eval_exact_match": counterfactuals.get("oracle_at_eval_exact_match"),
        "semantic_decoder_delta_l2": float(semantic_delta.get("l2", 0.0)),
        "semantic_decoder_delta_max_abs": float(semantic_delta.get("max_abs", 0.0)),
        "parameter_delta_from_source": deltas,
    }
    gate["passes_gate"] = (
        gate["oracle_at_eval_exact_match"] is not None
        and gate["oracle_at_eval_exact_match"] >= 0.99
        and gate["injection_zero_exact_match"] is not None
        and gate["injection_zero_exact_match"] <= 0.05
        and gate["forced_random_exact_match"] is not None
        and gate["forced_random_exact_match"] <= 0.10
        and gate["semantic_decoder_delta_l2"] == 0.0
        and gate["semantic_decoder_delta_max_abs"] == 0.0
    )
    write_json(RUN_ROOT / "oracle_wiring_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not gate["passes_gate"]:
        raise SystemExit("oracle wiring gate failed; do not proceed to local-target parity/training")


def cmd_compare_local_target_to_aux(args: argparse.Namespace) -> None:
    log_command([sys.executable, RUNNER_SCRIPT, *sys.argv[1:]])
    if not oracle_wiring_gate_passed() and not args.force:
        raise SystemExit(
            "oracle wiring gate has not passed; run oracle-wiring-gate first "
            "or pass --force for development only"
        )
    device = pick_device()
    checkpoint = args.checkpoint or STAGE0B_CHECKPOINT
    load_scope = args.semantic_decoder_checkpoint_load_scope
    model = build_phase6_model(
        checkpoint=checkpoint,
        load_scope=load_scope,
        device=device,
        seed=args.seed,
        estimator="identifiable_full_enum_local_target",
        freeze_upstream=True,
    )
    rng = random.Random(args.seed + 71_000)
    batch = make_range_batch(
        batch_size=args.samples,
        num_digits=2,
        operand_max=19,
        rng=rng,
        fixed_width=True,
        device=device,
        answer_format="sum_left_operand",
    )
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.train()
    local_ce, metrics = hard_best_local_target_loss_and_metrics(
        model,
        batch,
        num_digits=2,
        temperature=args.temperature,
        min_probability_floor=args.min_probability_floor,
        chunk_size=args.chunk_size,
    )
    optim = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=1e-3)
    optim.zero_grad(set_to_none=True)
    local_ce.backward()
    grad_summary = semantic_grad_summary(model)
    optim.step()
    after = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    deltas = model_group_delta_summary(before, after)
    semantic_delta = deltas.get("semantic_decoder", {"l2": 0.0, "max_abs": 0.0})
    metrics.update(grad_summary)
    metrics["semantic_decoder_delta_l2"] = float(semantic_delta.get("l2", 0.0))
    metrics["semantic_decoder_delta_max_abs"] = float(semantic_delta.get("max_abs", 0.0))
    metrics["parameter_delta_after_one_local_step"] = deltas
    metrics["checkpoint"] = str(checkpoint)
    metrics["semantic_decoder_checkpoint_load_scope"] = load_scope
    metrics["device"] = device
    metrics["passes_gate"] = (
        metrics["hard_best_pair_equals_true_pair"] >= 0.98
        and abs(metrics["local_minus_aux_ce"]) <= 1e-6
        and metrics["semantic_decoder_delta_l2"] == 0.0
        and metrics["semantic_decoder_delta_max_abs"] == 0.0
        and metrics["semantic_decoder_grad_l2"] == 0.0
        and metrics["semantic_decoder_grad_max_abs"] == 0.0
    )
    write_json(RUN_ROOT / "parity_gate.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["passes_gate"]:
        raise SystemExit("parity gate failed; do not proceed to Stage 1")


def phase6_train_command(
    *,
    checkpoint: Path,
    run_root: Path,
    estimator: str,
    answer_loss_weight: float,
    local_target_loss_weight: float,
    input_proj_lr: float,
    upstream_lr: float,
    steps: int,
    snapshot_every: int,
    checkpoint_every: int,
    target_mode: str,
    freeze_upstream: bool,
    seed: int,
    load_scope: str,
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
        "--calculator-action-head",
        "independent_operands",
        "--calculator-estimator",
        estimator,
        "--semantic-decoder-checkpoint",
        str(checkpoint),
        "--semantic-decoder-checkpoint-load-scope",
        load_scope,
        "--freeze-semantic-decoder",
        "--answer-loss-weight",
        str(answer_loss_weight),
        "--aux-operand-loss-weight",
        "0.0",
        "--adaptive-interface-loss-weight",
        str(local_target_loss_weight),
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
                "--action-loss-full-enum-min-probability-floor",
                "0.0",
                "--action-loss-full-enum-chunk-size",
                "64",
            ]
        )
    if freeze_upstream:
        command.append("--freeze-upstream-encoder")
    return command


def cmd_run_stage1(args: argparse.Namespace) -> None:
    if (not oracle_wiring_gate_passed() or not parity_gate_passed()) and not args.force:
        raise SystemExit(
            "oracle wiring and parity gates have not both passed; run "
            "oracle-wiring-gate and compare-local-target-to-aux first "
            "or pass --force for development only"
        )
    label = args.label
    if label is None:
        frozen = "frozen" if args.freeze_upstream_encoder else "upstream_open"
        label = (
            f"{args.semantic_decoder_checkpoint_load_scope}_{frozen}"
            f"_inlr{args.input_proj_lr:g}_local{args.local_target_loss_weight:g}"
        )
    run_command(
        phase6_train_command(
            checkpoint=args.checkpoint or STAGE0B_CHECKPOINT,
            run_root=RUN_ROOT / "stage1" / label,
            estimator="identifiable_full_enum_local_target",
            answer_loss_weight=args.answer_loss_weight,
            local_target_loss_weight=args.local_target_loss_weight,
            input_proj_lr=args.input_proj_lr,
            upstream_lr=args.upstream_lr,
            steps=args.steps,
            snapshot_every=args.snapshot_every,
            checkpoint_every=args.checkpoint_every,
            target_mode=args.target_mode,
            freeze_upstream=args.freeze_upstream_encoder,
            seed=args.seed,
            load_scope=args.semantic_decoder_checkpoint_load_scope,
        )
    )


def all_model_runs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("**/model-c-2digit-seed*/metrics.json"))


def latest_model_run(root: Path) -> Path:
    runs = all_model_runs(root)
    if not runs:
        raise FileNotFoundError(f"no metrics.json found under {root}")
    return runs[-1]


def row_score(row: dict[str, str]) -> float:
    return min(
        float(row["operand_exact_match"]),
        float(row["pair_exact_match"]),
        float(row["calculator_result_accuracy"]),
    )


def qualifying_stage1_snapshots(threshold: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for run_dir in all_model_runs(RUN_ROOT / "stage1"):
        snapshot_path = run_dir / "diagnostic_snapshots.csv"
        if not snapshot_path.exists():
            continue
        label = run_dir.parents[1].name
        metrics = load_json(run_dir / "metrics.json")
        for row in read_rows(snapshot_path):
            score = row_score(row)
            if score < threshold:
                continue
            checkpoint = run_dir / "checkpoint_snapshots" / f"step_{int(row['step']):05d}_weights.pt"
            if checkpoint.exists():
                selected.append(
                    {
                        "label": label,
                        "run_dir": str(run_dir),
                        "checkpoint": checkpoint,
                        "step": int(row["step"]),
                        "score": score,
                        "row": row,
                        "freeze_upstream_encoder": bool(metrics.get("freeze_upstream_encoder", True)),
                    }
                )
    return selected


def best_snapshot_for_run(run_dir: Path) -> Path | None:
    snapshot_path = run_dir / "diagnostic_snapshots.csv"
    if not snapshot_path.exists():
        return None
    rows = read_rows(snapshot_path)
    if not rows:
        return None
    best = max(rows, key=row_score)
    checkpoint = run_dir / "checkpoint_snapshots" / f"step_{int(best['step']):05d}_weights.pt"
    return checkpoint if checkpoint.exists() else None


def cmd_run_retention(args: argparse.Namespace) -> None:
    starts: list[tuple[str, Path, bool]]
    if args.checkpoint is not None:
        freeze_upstream = True if args.freeze_upstream_encoder is None else args.freeze_upstream_encoder
        starts = [(args.label or "manual", args.checkpoint, freeze_upstream)]
    else:
        qualifying = qualifying_stage1_snapshots(args.threshold)
        if not qualifying:
            print(f"No Stage 1 snapshot reached threshold {args.threshold:.3f}; retention skipped.")
            return
        first = min(qualifying, key=lambda row: (str(row["run_dir"]), int(row["step"])))
        best = max(qualifying, key=lambda row: (float(row["score"]), -int(row["step"])))
        starts = [
            (
                f"{first['label']}_first_gate_step{first['step']:05d}",
                first["checkpoint"],
                bool(first["freeze_upstream_encoder"]),
            )
        ]
        if best["checkpoint"] != first["checkpoint"]:
            starts.append(
                (
                    f"{best['label']}_best_gate_step{best['step']:05d}",
                    best["checkpoint"],
                    bool(best["freeze_upstream_encoder"]),
                )
            )
    for label, checkpoint, freeze_upstream in starts:
        retention_kind = "frozen_upstream_retention" if freeze_upstream else "upstream_open_retention"
        run_command(
            phase6_train_command(
                checkpoint=checkpoint,
                run_root=RUN_ROOT / "stage2" / retention_kind / label,
                estimator="adaptive_interface",
                answer_loss_weight=args.answer_loss_weight,
                local_target_loss_weight=0.0,
                input_proj_lr=args.input_proj_lr,
                upstream_lr=args.upstream_lr,
                steps=args.steps,
                snapshot_every=args.snapshot_every,
                checkpoint_every=args.checkpoint_every,
                target_mode="hard_best_pair",
                freeze_upstream=freeze_upstream,
                seed=args.seed,
                load_scope="full_model",
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
    checkpoints: dict[str, Path] = {}
    baseline_checkpoint = checkpoint_or_none(strict_baseline_run_dir())
    if baseline_checkpoint is not None:
        checkpoints["strict_semantic_decoder_only_baseline"] = baseline_checkpoint
    else:
        checkpoints["stage0b_source"] = STAGE0B_CHECKPOINT
    for run_dir in all_model_runs(RUN_ROOT / "stage1"):
        label = run_dir.parents[1].name
        checkpoints[f"stage1_{label}_final"] = run_dir / "final_weights.pt"
        best = best_snapshot_for_run(run_dir)
        if best is not None:
            checkpoints[f"stage1_{label}_best_snapshot"] = best
    for run_dir in all_model_runs(RUN_ROOT / "stage2"):
        label = "_".join(run_dir.parts[len((RUN_ROOT / "stage2").parts) : -2])
        checkpoints[f"stage2_{label}_final"] = run_dir / "final_weights.pt"
        best = best_snapshot_for_run(run_dir)
        if best is not None:
            checkpoints[f"stage2_{label}_best_snapshot"] = best
    return checkpoints


def cmd_diagnostics(args: argparse.Namespace) -> None:
    for label, checkpoint in selected_checkpoints().items():
        if args.only and label not in set(args.only):
            continue
        output_root = RUN_ROOT / "diagnostics" / label
        for command in diagnostic_commands(checkpoint, output_root):
            run_command(command)


def compact_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    snapshots = read_rows(run_dir / "diagnostic_snapshots.csv")
    best = max(snapshots, key=row_score) if snapshots else {}
    return {
        "run_dir": str(run_dir),
        "final_checkpoint": str(run_dir / "final_weights.pt"),
        "final_exact_match": metrics.get("exact_match"),
        "final_aux_operand_loss_weight": metrics.get("final_aux_operand_loss_weight"),
        "final_local_target_loss_weight": metrics.get("final_local_target_loss_weight"),
        "final_adaptive_interface_loss_weight": metrics.get("final_adaptive_interface_loss_weight"),
        "final_input_proj_anchor_weight": metrics.get("final_input_proj_anchor_weight"),
        "answer_loss_weight": metrics.get("answer_loss_weight"),
        "semantic_decoder_checkpoint_load_scope": metrics.get("semantic_decoder_checkpoint_load_scope"),
        "semantic_decoder_checkpoint": metrics.get("semantic_decoder_checkpoint"),
        "input_proj_lr": metrics.get("input_proj_lr"),
        "upstream_lr": metrics.get("upstream_lr"),
        "freeze_upstream_encoder": metrics.get("freeze_upstream_encoder"),
        "trainable_parameter_groups": metrics.get("trainable_parameter_groups"),
        "best_snapshot": best,
        "best_snapshot_score": row_score(best) if best else None,
    }


def compact_diagnostics(label: str) -> dict[str, Any]:
    root = RUN_ROOT / "diagnostics" / label
    canonical = root / "canonical" / "diagnostic_summary.json"
    private = root / "private" / "private_protocol_summary.json"
    full_enum_summaries = list((root / "full_enum").glob("*/full_enum_summary.json"))
    payload: dict[str, Any] = {}
    if canonical.exists():
        row = load_json(canonical)
        counters = {
            item["condition"]: item
            for item in row.get("counterfactual_exact_match", [])
        }
        payload["canonical"] = {
            "normal_exact_match": row.get("exact_match"),
            "injection_zero_exact_match": counters.get("injection_zero", {}).get("exact_match"),
            "forced_zero_exact_match": counters.get("forced_zero", {}).get("exact_match"),
            "forced_random_exact_match": counters.get("forced_random", {}).get("exact_match"),
            "oracle_at_eval_exact_match": counters.get("oracle_at_eval", {}).get("exact_match"),
            "operand_exact_match": row.get("operand_exact_match"),
            "pair_exact_match": row.get("pair_exact_match"),
            "calculator_result_accuracy": row.get("calculator_result_accuracy"),
        }
    if private.exists():
        row = load_json(private)
        payload["private"] = {
            "answer_exact_match": row.get("exact_match"),
            "operand_exact_match": row.get("operand_exact_match"),
            "pair_exact_match": row.get("pair_exact_match"),
            "calculator_result_accuracy": row.get("calculator_result_accuracy"),
        }
    if full_enum_summaries:
        row = load_json(full_enum_summaries[0])
        payload["full_enum"] = {
            "mean_learned_nll": row.get("mean_learned_nll"),
            "mean_true_nll": row.get("mean_true_nll"),
            "mean_best_full_enum_nll": row.get("mean_best_full_enum_nll"),
            "mean_learned_minus_true_gap": row.get("mean_learned_minus_true_gap"),
            "mean_learned_minus_best_gap": row.get("mean_learned_minus_best_gap"),
            "learned_best_fraction": row.get("learned_best_fraction"),
            "true_best_fraction": row.get("true_best_fraction"),
        }
    return payload


def cmd_summarize(args: argparse.Namespace) -> None:
    runs: dict[str, Any] = {}
    for stage in ["stage1", "stage2"]:
        for run_dir in all_model_runs(RUN_ROOT / stage):
            label = f"{stage}/" + "/".join(run_dir.parts[len((RUN_ROOT / stage).parts) : -2])
            row = compact_run(run_dir)
            if stage == "stage1":
                step0 = run_dir / "checkpoint_snapshots" / "step_00000_weights.pt"
                source = step0 if step0.exists() else (checkpoint_or_none(strict_baseline_run_dir()) or STAGE0B_CHECKPOINT)
            else:
                source = Path(load_json(run_dir / "metrics.json")["semantic_decoder_checkpoint"])
            row["parameter_delta_from_source"] = checkpoint_delta_summary(source, run_dir / "final_weights.pt")
            runs[label] = row
    diagnostics = {label: compact_diagnostics(label) for label in selected_checkpoints()}
    summary = {
        "run_root": str(RUN_ROOT),
        "stage0b_checkpoint": str(STAGE0B_CHECKPOINT),
        "default_semantic_decoder_checkpoint_load_scope": DEFAULT_LOAD_SCOPE,
        "oracle_wiring_gate": load_json(RUN_ROOT / "oracle_wiring_gate.json") if (RUN_ROOT / "oracle_wiring_gate.json").exists() else {},
        "parity_gate": load_json(RUN_ROOT / "parity_gate.json") if (RUN_ROOT / "parity_gate.json").exists() else {},
        "runs": runs,
        "diagnostics": diagnostics,
    }
    write_json(RUN_ROOT / "summary.json", summary)
    lines = [
        "# Phase 6 Strict Random-Upstream Local-Target Discovery Summary",
        "",
        f"Run root: `{RUN_ROOT}`",
        f"Stage 0B checkpoint: `{STAGE0B_CHECKPOINT}`",
        f"Default load scope: `{DEFAULT_LOAD_SCOPE}`",
        "",
        "## Oracle Wiring Gate",
    ]
    oracle_gate = summary["oracle_wiring_gate"]
    if oracle_gate:
        lines.append(
            "- "
            f"pass={oracle_gate['passes_gate']} scope={oracle_gate['semantic_decoder_checkpoint_load_scope']} "
            f"oracle={oracle_gate['oracle_at_eval_exact_match']:.3f}, "
            f"injection-zero={oracle_gate['injection_zero_exact_match']:.3f}, "
            f"forced-random={oracle_gate['forced_random_exact_match']:.3f}, "
            f"semantic delta L2={oracle_gate['semantic_decoder_delta_l2']:.1f}"
        )
    lines.extend([
        "",
        "## Parity Gate",
    ])
    gate = summary["parity_gate"]
    if gate:
        lines.append(
            "- "
            f"pass={gate['passes_gate']} hard-best=true {gate['hard_best_pair_equals_true_pair']:.3f}, "
            f"local CE {gate['hard_best_local_ce']:.6f}, aux CE {gate['direct_aux_ce_same_logits']:.6f}, "
            f"delta {gate['local_minus_aux_ce']:.3g}, semantic delta L2 {gate['semantic_decoder_delta_l2']:.1f}"
        )
    lines.extend(["", "## Runs"])
    for label, row in runs.items():
        best = row.get("best_snapshot") or {}
        lines.append(
            "- "
            f"{label}: final exact {row['final_exact_match']}, best step {best.get('step')} "
            f"normal/operand/pair/calc {float(best.get('normal_exact_match', 0.0)):.3f}/"
            f"{float(best.get('operand_exact_match', 0.0)):.3f}/"
            f"{float(best.get('pair_exact_match', 0.0)):.3f}/"
            f"{float(best.get('calculator_result_accuracy', 0.0)):.3f}; "
            f"answer={row['answer_loss_weight']}, local={row['final_local_target_loss_weight']}, "
            f"aux={row['final_aux_operand_loss_weight']}"
        )
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 strict random-upstream local-target discovery runner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    oracle = subparsers.add_parser("oracle-wiring-gate")
    oracle.add_argument("--checkpoint", type=Path, default=None)
    oracle.add_argument(
        "--semantic-decoder-checkpoint-load-scope",
        choices=["full_model", "semantic_decoder_only"],
        default=DEFAULT_LOAD_SCOPE,
    )
    oracle.add_argument("--input-proj-lr", type=float, default=0.03)
    oracle.add_argument("--upstream-lr", type=float, default=0.003)
    oracle.add_argument("--seed", type=int, default=0)

    parity = subparsers.add_parser("compare-local-target-to-aux")
    parity.add_argument("--checkpoint", type=Path, default=None)
    parity.add_argument(
        "--semantic-decoder-checkpoint-load-scope",
        choices=["full_model", "semantic_decoder_only"],
        default=DEFAULT_LOAD_SCOPE,
    )
    parity.add_argument("--samples", type=int, default=128)
    parity.add_argument("--seed", type=int, default=0)
    parity.add_argument("--temperature", type=float, default=0.25)
    parity.add_argument("--min-probability-floor", type=float, default=0.0)
    parity.add_argument("--chunk-size", type=int, default=64)
    parity.add_argument("--force", action="store_true")

    stage1 = subparsers.add_parser("run-stage1")
    stage1.add_argument("--checkpoint", type=Path, default=None)
    stage1.add_argument(
        "--semantic-decoder-checkpoint-load-scope",
        choices=["full_model", "semantic_decoder_only"],
        default=DEFAULT_LOAD_SCOPE,
    )
    stage1.add_argument("--label", type=str, default=None)
    stage1.add_argument("--answer-loss-weight", type=float, default=0.0)
    stage1.add_argument("--local-target-loss-weight", type=float, default=1.0)
    stage1.add_argument("--input-proj-lr", type=float, default=0.03)
    stage1.add_argument("--upstream-lr", type=float, default=0.003)
    stage1.add_argument("--steps", type=int, default=300)
    stage1.add_argument("--snapshot-every", type=int, default=25)
    stage1.add_argument("--checkpoint-every", type=int, default=25)
    stage1.add_argument("--seed", type=int, default=0)
    stage1.add_argument("--target-mode", choices=["hard_best_pair", "soft_pair"], default="hard_best_pair")
    stage1.add_argument("--freeze-upstream-encoder", action=argparse.BooleanOptionalAction, default=True)
    stage1.add_argument("--force", action="store_true")

    retention = subparsers.add_parser("run-retention")
    retention.add_argument("--checkpoint", type=Path, default=None)
    retention.add_argument("--label", type=str, default=None)
    retention.add_argument("--threshold", type=float, default=0.90)
    retention.add_argument("--answer-loss-weight", type=float, default=1.0)
    retention.add_argument("--input-proj-lr", type=float, default=0.0003)
    retention.add_argument("--upstream-lr", type=float, default=0.0003)
    retention.add_argument("--steps", type=int, default=1000)
    retention.add_argument("--snapshot-every", type=int, default=50)
    retention.add_argument("--checkpoint-every", type=int, default=50)
    retention.add_argument("--seed", type=int, default=0)
    retention.add_argument("--freeze-upstream-encoder", action=argparse.BooleanOptionalAction, default=None)

    diagnostics = subparsers.add_parser("diagnostics")
    diagnostics.add_argument("--only", nargs="*", default=None)

    subparsers.add_parser("summarize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "oracle-wiring-gate":
        cmd_oracle_wiring_gate(args)
    elif args.command == "compare-local-target-to-aux":
        cmd_compare_local_target_to_aux(args)
    elif args.command == "run-stage1":
        cmd_run_stage1(args)
    elif args.command == "run-retention":
        cmd_run_retention(args)
    elif args.command == "diagnostics":
        cmd_diagnostics(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
