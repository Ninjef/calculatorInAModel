from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_enum_action_loss_diagnostic import (  # noqa: E402
    all_pair_specs,
    batch_from_specs,
    full_enum_diagnostic,
    write_rows,
)
from scripts.overfit_one_batch import (  # noqa: E402
    adaptive_optimizer_param_groups,
    calculator_read_operand_logits,
    fixed_width_operands_from_batch,
    freeze_semantic_decoder_parameters,
    freeze_upstream_encoder_parameters,
    full_enum_action_pairs,
    load_semantic_decoder_checkpoint,
    make_model_config,
    masked_cross_entropy_per_example,
    pick_device,
    score_action_loss_candidates_chunked,
)
from src.model import TinyGPT  # noqa: E402


RUN_ROOT = REPO_ROOT / "runs/2026-05-12_phase6_closure_landscape_diagnostic"
IDENTIFIABLE_SUMMARY = (
    REPO_ROOT
    / "runs/2026-05-11_phase6_relaxed_bridge_replication_stochastic_upstream/summary.json"
)
NATURAL_SUMMARY = (
    REPO_ROOT / "runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/summary.json"
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
ONE_STEP_SEEDS = [6201, 6202, 6203]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parameter_group_for_name(name: str) -> str:
    if name.startswith("calculator_hook.input_proj."):
        return "calculator_hook.input_proj"
    if name.startswith(("calculator_hook.output_proj.", "answer_offset_emb.", "answer_decoder.")):
        return "semantic_decoder"
    if name.startswith(("tok_emb.", "pos_emb.", "blocks.", "ln_f.", "lm_head.")):
        return "upstream_encoder"
    return "other"


def state_dict_copy(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def model_group_delta_summary(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for name, before_tensor in before.items():
        after_tensor = after.get(name)
        if after_tensor is None or after_tensor.shape != before_tensor.shape:
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
    return groups


def input_proj_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for name, param in model.named_parameters():
        if not name.startswith("calculator_hook.input_proj."):
            continue
        if param.grad is None:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1).cpu())
        else:
            chunks.append(param.grad.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.empty(0)


def semantic_grad_summary(model: torch.nn.Module) -> dict[str, float | int]:
    l2_sq = 0.0
    max_abs = 0.0
    tensors = 0
    for name, param in model.named_parameters():
        if parameter_group_for_name(name) != "semantic_decoder" or param.grad is None:
            continue
        grad = param.grad.detach().float().reshape(-1)
        l2_sq += float(torch.dot(grad, grad).item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
        tensors += 1
    return {
        "semantic_decoder_grad_l2": math.sqrt(l2_sq),
        "semantic_decoder_grad_max_abs": max_abs,
        "semantic_decoder_tensors_with_grad": tensors,
    }


def answer_loss(model: TinyGPT, batch: Any) -> torch.Tensor:
    logits = model(batch.x)
    return masked_cross_entropy_per_example(logits, batch.y, batch.loss_mask).mean()


def make_strict_model(
    *,
    checkpoint: Path,
    device: str | torch.device,
    answer_format: str,
    calculator_output_format: str,
    answer_decoder_interaction: str,
    seed: int,
) -> TinyGPT:
    torch.manual_seed(seed)
    cfg = make_model_config(
        2,
        "model-c",
        operand_vocab_size=20,
        calculator_estimator="gumbel_concrete_interface",
        calculator_action_head="independent_operands",
        calculator_read_position="operand_spans",
        calculator_read_span_width=2,
        calculator_bottleneck_mode="answer_decoder",
        calculator_output_format=calculator_output_format,
        answer_decoder_interaction=answer_decoder_interaction,
        relaxed_calculator_temperature=2.0,
        relaxed_calculator_mode="deterministic",
        relaxed_calculator_hard_forward=True,
        answer_format=answer_format,  # type: ignore[arg-type]
        n_layer=2,
        n_head=1,
        n_embd=16,
        mlp_expansion=1,
        calculator_hook_after_layer=1,
    )
    model = TinyGPT(cfg).to(device)
    load_semantic_decoder_checkpoint(model, checkpoint, load_scope="semantic_decoder_only")
    freeze_semantic_decoder_parameters(model)
    freeze_upstream_encoder_parameters(model)
    return model


def policy_landscape_metrics(
    model: TinyGPT,
    batch: Any,
    *,
    temperature: float,
    chunk_size: int,
) -> dict[str, Any]:
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    classes = a_logits.shape[-1]
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    candidates = pairs.unsqueeze(0).expand(batch.x.shape[0], -1, -1)
    with torch.no_grad():
        losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=2)
    true_sum = true_a + true_b
    true_idx = true_a * classes + true_b
    best_idx = losses.argmin(dim=-1)
    best_pairs = pairs.index_select(0, best_idx)
    pair_sums = pairs[:, 0] + pairs[:, 1]
    result_count = (2 * classes) - 1
    result_losses = losses.new_full((batch.x.shape[0], result_count), float("inf"))
    for result_idx in range(result_count):
        mask = pair_sums == result_idx
        result_losses[:, result_idx] = losses[:, mask].min(dim=1).values
    best_result = result_losses.argmin(dim=-1)

    pair_probs = (
        torch.softmax(a_logits / temperature, dim=-1).unsqueeze(-1)
        * torch.softmax(b_logits / temperature, dim=-1).unsqueeze(-2)
    ).reshape(batch.x.shape[0], classes * classes)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_idx = learned_a * classes + learned_b
    entropy = -(pair_probs * pair_probs.clamp_min(1e-12).log()).sum(dim=-1)

    def result_mass(result: torch.Tensor) -> torch.Tensor:
        return (
            pair_probs
            * (pair_sums.unsqueeze(0) == result.unsqueeze(-1)).to(pair_probs.dtype)
        ).sum(dim=-1)

    sorted_probs = pair_probs.sort(dim=-1, descending=True).values
    return {
        "best_idx": best_idx,
        "best_pairs": best_pairs,
        "best_result": best_result,
        "best_pair_equals_true_pair_fraction": float((best_idx == true_idx).float().mean().item()),
        "best_result_group_equals_true_sum_fraction": float((best_result == true_sum).float().mean().item()),
        "true_pair_probability": float(pair_probs.gather(1, true_idx.unsqueeze(-1)).mean().item()),
        "best_pair_probability": float(pair_probs.gather(1, best_idx.unsqueeze(-1)).mean().item()),
        "true_result_group_probability": float(result_mass(true_sum).mean().item()),
        "best_result_group_probability": float(result_mass(best_result).mean().item()),
        "hard_learned_pair_exact": float((learned_idx == true_idx).float().mean().item()),
        "hard_learned_calculator_result_accuracy": float(
            ((learned_a + learned_b) == true_sum).float().mean().item()
        ),
        "hard_learned_best_pair_fraction": float((learned_idx == best_idx).float().mean().item()),
        "hard_learned_best_result_group_fraction": float(
            ((learned_a + learned_b) == best_result).float().mean().item()
        ),
        "mean_pair_policy_entropy": float(entropy.mean().item()),
        "mean_pair_policy_effective_count": float(entropy.exp().mean().item()),
        "top1_policy_mass": float(sorted_probs[:, :1].sum(dim=-1).mean().item()),
        "top3_policy_mass": float(sorted_probs[:, :3].sum(dim=-1).mean().item()),
        "top5_policy_mass": float(sorted_probs[:, :5].sum(dim=-1).mean().item()),
    }


def best_result_group_loss(model: TinyGPT, batch: Any, best_result: torch.Tensor) -> torch.Tensor:
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    classes = a_logits.shape[-1]
    pair_log_probs = (
        F.log_softmax(a_logits / 2.0, dim=-1).unsqueeze(-1)
        + F.log_softmax(b_logits / 2.0, dim=-1).unsqueeze(-2)
    ).reshape(batch.x.shape[0], classes * classes)
    pairs = full_enum_action_pairs(classes=classes, device=batch.x.device)
    pair_sums = pairs[:, 0] + pairs[:, 1]
    losses = []
    for i in range(batch.x.shape[0]):
        mask = pair_sums == best_result[i]
        losses.append(-torch.logsumexp(pair_log_probs[i, mask], dim=0))
    return torch.stack(losses).mean()


def gradient_cosines(
    model: TinyGPT,
    batch: Any,
    initial: dict[str, Any],
) -> dict[str, float]:
    relaxed_model = copy.deepcopy(model)
    relaxed = answer_loss(relaxed_model, batch)
    relaxed_model.zero_grad(set_to_none=True)
    relaxed.backward()
    relaxed_grad = input_proj_grad_vector(relaxed_model)

    pair_model = copy.deepcopy(model)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(pair_model, batch)
    best_pairs = initial["best_pairs"].to(batch.x.device)
    pair_ce = (F.cross_entropy(a_logits, best_pairs[:, 0]) + F.cross_entropy(b_logits, best_pairs[:, 1])) / 2
    pair_model.zero_grad(set_to_none=True)
    pair_ce.backward()
    pair_grad = input_proj_grad_vector(pair_model)

    result_model = copy.deepcopy(model)
    result_ce = best_result_group_loss(result_model, batch, initial["best_result"].to(batch.x.device))
    result_model.zero_grad(set_to_none=True)
    result_ce.backward()
    result_grad = input_proj_grad_vector(result_model)

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        if not a.numel() or not b.numel() or a.norm().item() == 0.0 or b.norm().item() == 0.0:
            return float("nan")
        return float(F.cosine_similarity(a, b, dim=0).item())

    return {
        "relaxed_answer_vs_hard_best_pair_ce": cosine(relaxed_grad, pair_grad),
        "relaxed_answer_vs_best_result_group_nll": cosine(relaxed_grad, result_grad),
        "relaxed_answer_grad_norm": float(relaxed_grad.norm().item()),
        "hard_best_pair_ce_grad_norm": float(pair_grad.norm().item()),
        "best_result_group_grad_norm": float(result_grad.norm().item()),
    }


def one_step_diagnostic(
    *,
    label: str,
    checkpoint: Path,
    answer_format: str,
    calculator_output_format: str,
    answer_decoder_interaction: str,
    seed: int,
    device: str,
) -> dict[str, Any]:
    specs = all_pair_specs(19)
    batch = batch_from_specs(
        specs,
        num_digits=2,
        fixed_width=True,
        device=device,
        answer_format=answer_format,  # type: ignore[arg-type]
    )
    model = make_strict_model(
        checkpoint=checkpoint,
        device=device,
        answer_format=answer_format,
        calculator_output_format=calculator_output_format,
        answer_decoder_interaction=answer_decoder_interaction,
        seed=seed,
    )
    model.train()
    before = state_dict_copy(model)
    initial = policy_landscape_metrics(model, batch, temperature=2.0, chunk_size=64)
    cosines = gradient_cosines(model, batch, initial)
    optim = torch.optim.AdamW(
        adaptive_optimizer_param_groups(
            model,
            lr=0.03,
            input_proj_lr=0.03,
            upstream_lr=0.0003,
            weight_decay=0.0,
        ),
        betas=(0.9, 0.95),
    )
    optim.zero_grad(set_to_none=True)
    loss = answer_loss(model, batch)
    loss.backward()
    semantic_grad = semantic_grad_summary(model)
    optim.step()
    after = state_dict_copy(model)
    post = policy_landscape_metrics(model, batch, temperature=2.0, chunk_size=64)
    deltas = model_group_delta_summary(before, after)
    clean_initial = {k: v for k, v in initial.items() if k not in {"best_idx", "best_pairs", "best_result"}}
    clean_post = {k: v for k, v in post.items() if k not in {"best_idx", "best_pairs", "best_result"}}
    return {
        "label": label,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "answer_format": answer_format,
        "calculator_output_format": calculator_output_format,
        "answer_decoder_interaction": answer_decoder_interaction,
        "samples": len(specs),
        "initial_answer_loss": float(loss.detach().item()),
        "initial": clean_initial,
        "post_one_step": clean_post,
        "deltas": {
            key: clean_post[key] - clean_initial[key]
            for key in clean_initial
            if isinstance(clean_initial[key], float) and isinstance(clean_post.get(key), float)
        },
        "gradient_cosines": cosines,
        "semantic_decoder_grad": semantic_grad,
        "parameter_delta_after_one_step": deltas,
    }


def compact_existing_evidence() -> dict[str, Any]:
    identifiable = load_json(IDENTIFIABLE_SUMMARY)
    natural = load_json(NATURAL_SUMMARY)
    return {
        "source_summaries": {
            "identifiable": str(IDENTIFIABLE_SUMMARY),
            "natural": str(NATURAL_SUMMARY),
        },
        "identifiable_deterministic_concrete": {
            "stage1_selected": identifiable.get("stage1_deterministic", []),
            "stage2_retention": identifiable.get("stage2_retention", []),
            "upstream_open_stage1": identifiable.get("stage4_upstream_open_stage1", []),
            "upstream_open_retention": identifiable.get("stage4_upstream_open_retention", []),
            "diagnostics": identifiable.get("diagnostics", []),
        },
        "natural_product_decoder": {
            "stage0_selected_candidate": natural.get("stage0_candidates", {}).get("selected_passing_candidate"),
            "stage1": natural.get("stage1", {}),
            "diagnostics": natural.get("diagnostics", []),
        },
    }


def natural_stage0_checkpoint() -> Path:
    summary = load_json(NATURAL_SUMMARY)
    selected = summary["stage0_candidates"]["selected_passing_candidate"]
    return Path(selected["source_checkpoint"])


def run_full_enum_pair(device: str) -> dict[str, Any]:
    outputs = []
    specs = all_pair_specs(19)
    configs = [
        {
            "label": "identifiable_sum_left_operand_stage0b",
            "checkpoint": STAGE0B_CHECKPOINT,
            "answer_format": "sum_left_operand",
            "calculator_output_format": "sum_left_operand",
        },
        {
            "label": "natural_sum_only_product_stage0",
            "checkpoint": natural_stage0_checkpoint(),
            "answer_format": "sum",
            "calculator_output_format": "sum",
        },
    ]
    for cfg in configs:
        output_dir = RUN_ROOT / "full_enum" / cfg["label"]
        rows, summary = full_enum_diagnostic(
            checkpoint=cfg["checkpoint"],
            samples=len(specs),
            batch_size=64,
            digits=2,
            operand_max=19,
            temperature=0.25,
            min_probability_floor=0.0,
            near_best_tolerance=1e-3,
            chunk_size=64,
            seed=0,
            device=device,
            answer_format=cfg["answer_format"],  # type: ignore[arg-type]
            sample_specs=specs,
        )
        summary["label"] = cfg["label"]
        summary["requested_calculator_output_format"] = cfg["calculator_output_format"]
        summary["exhaustive_grid"] = True
        summary["output_dir"] = str(output_dir)
        summary["device"] = device
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "full_enum_rows.csv", rows)
        write_json(output_dir / "full_enum_summary.json", summary)
        outputs.append(summary)
    return {row["label"]: row for row in outputs}


def summarize_one_step(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    metrics = [
        "true_pair_probability",
        "best_pair_probability",
        "true_result_group_probability",
        "best_result_group_probability",
        "hard_learned_pair_exact",
        "hard_learned_calculator_result_accuracy",
    ]
    payload: dict[str, Any] = {"runs": rows}
    payload["mean_initial"] = {
        key: mean(float(row["initial"][key]) for row in rows) for key in metrics
    }
    payload["mean_post_one_step"] = {
        key: mean(float(row["post_one_step"][key]) for row in rows) for key in metrics
    }
    payload["mean_deltas"] = {
        key: payload["mean_post_one_step"][key] - payload["mean_initial"][key]
        for key in metrics
    }
    payload["mean_gradient_cosines"] = {
        key: mean(float(row["gradient_cosines"][key]) for row in rows)
        for key in rows[0]["gradient_cosines"]
    }
    payload["mean_parameter_delta"] = {}
    for group in ["calculator_hook.input_proj", "upstream_encoder", "semantic_decoder"]:
        payload["mean_parameter_delta"][group] = {
            field: mean(
                float(row["parameter_delta_after_one_step"].get(group, {}).get(field, 0.0))
                for row in rows
            )
            for field in ["l2", "max_abs"]
        }
    return payload


def run_one_step_pair(device: str) -> dict[str, Any]:
    natural_checkpoint = natural_stage0_checkpoint()
    by_label: dict[str, list[dict[str, Any]]] = {
        "identifiable_sum_left_operand": [],
        "natural_sum_only_product": [],
    }
    for seed in ONE_STEP_SEEDS:
        by_label["identifiable_sum_left_operand"].append(
            one_step_diagnostic(
                label="identifiable_sum_left_operand",
                checkpoint=STAGE0B_CHECKPOINT,
                answer_format="sum_left_operand",
                calculator_output_format="sum_left_operand",
                answer_decoder_interaction="product",
                seed=seed,
                device=device,
            )
        )
        by_label["natural_sum_only_product"].append(
            one_step_diagnostic(
                label="natural_sum_only_product",
                checkpoint=natural_checkpoint,
                answer_format="sum",
                calculator_output_format="sum",
                answer_decoder_interaction="product",
                seed=seed,
                device=device,
            )
        )
    return {label: summarize_one_step(rows) for label, rows in by_label.items()}


def decision_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    fe_ident = summary["full_enum"]["identifiable_sum_left_operand_stage0b"]
    fe_nat = summary["full_enum"]["natural_sum_only_product_stage0"]
    grad_ident = summary["one_step"]["identifiable_sum_left_operand"]["mean_deltas"]
    grad_nat = summary["one_step"]["natural_sum_only_product"]["mean_deltas"]
    return {
        "label": "phase6_close_start_phase7",
        "supported": [
            "Phase 6 success is strongest in identifiable action landscapes.",
            "The natural product-decoder bridge is not blocked by oracle/readout health, but by a diffuse underidentified result-level landscape for independent operand heads.",
            "One-step deterministic Concrete gradients move natural result-group mass only weakly from the strict random-upstream initialization, while the landscape itself remains many-pair underidentified.",
        ],
        "key_evidence": {
            "identifiable_effective_pairs": fe_ident["mean_effective_pair_count"],
            "natural_effective_pairs": fe_nat["mean_effective_pair_count"],
            "identifiable_true_pair_probability_delta": grad_ident[
                "true_pair_probability"
            ],
            "natural_true_result_group_probability_delta": grad_nat[
                "true_result_group_probability"
            ],
            "natural_best_result_group_probability_delta": grad_nat[
                "best_result_group_probability"
            ],
        },
        "recommended_phase7_first_task": (
            "Start Phase 7 with a result-space or structured joint-pair "
            "interface objective for natural 0..19 addition, not operand_max=99 "
            "scaling."
        ),
    }


def write_markdown(summary: dict[str, Any]) -> None:
    fe_ident = summary["full_enum"]["identifiable_sum_left_operand_stage0b"]
    fe_nat = summary["full_enum"]["natural_sum_only_product_stage0"]
    one_ident = summary["one_step"]["identifiable_sum_left_operand"]
    one_nat = summary["one_step"]["natural_sum_only_product"]
    lines = [
        "# Phase 6 Closure Landscape Diagnostic",
        "",
        "## Existing Evidence",
        "",
        "- Identifiable deterministic Concrete replicated across effective seeds `2`, `4`, and `5`, then retained with all teacher/local/expected/relaxed objectives inactive.",
        "- Natural sum-only product decoder passed the oracle/readout gate, but the learned deterministic Concrete bridge selected only about `0.11` learned-result-best fraction with a learned-result minus best-result gap around `5.57`.",
        "",
        "## Paired Full-Enum Landscape",
        "",
        "| setting | best pair=true | tie-aware true best | best result=true | true pair rank | effective pairs | effective results | same-true-sum near-best | true pair prob | true result prob | top1/top3/top5 mass | learned result acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| identifiable sum_left_operand | "
        + " | ".join(
            [
                fmt(fe_ident["best_matches_true_operands_fraction"]),
                fmt(fe_ident["tie_aware_true_best_fraction"]),
                fmt(fe_ident["best_result_group_matches_true_sum_fraction"]),
                fmt(fe_ident["mean_true_pair_rank"]),
                fmt(fe_ident["mean_effective_pair_count"]),
                fmt(fe_ident["mean_effective_result_count"]),
                fmt(fe_ident["mean_same_true_sum_near_best_pair_count"]),
                fmt(fe_ident["mean_soft_target_true_pair_probability"]),
                fmt(fe_ident["mean_soft_target_true_result_group_probability"]),
                f"{fmt(fe_ident['mean_top1_target_mass'])}/{fmt(fe_ident['mean_top3_target_mass'])}/{fmt(fe_ident['mean_top5_target_mass'])}",
                fmt(fe_ident["learned_result_matches_true_sum_fraction"]),
            ]
        )
        + " |",
        "| natural sum-only product | "
        + " | ".join(
            [
                fmt(fe_nat["best_matches_true_operands_fraction"]),
                fmt(fe_nat["tie_aware_true_best_fraction"]),
                fmt(fe_nat["best_result_group_matches_true_sum_fraction"]),
                fmt(fe_nat["mean_true_pair_rank"]),
                fmt(fe_nat["mean_effective_pair_count"]),
                fmt(fe_nat["mean_effective_result_count"]),
                fmt(fe_nat["mean_same_true_sum_near_best_pair_count"]),
                fmt(fe_nat["mean_soft_target_true_pair_probability"]),
                fmt(fe_nat["mean_soft_target_true_result_group_probability"]),
                f"{fmt(fe_nat['mean_top1_target_mass'])}/{fmt(fe_nat['mean_top3_target_mass'])}/{fmt(fe_nat['mean_top5_target_mass'])}",
                fmt(fe_nat["learned_result_matches_true_sum_fraction"]),
            ]
        )
        + " |",
        "",
        "## One-Step Relaxed Gradient",
        "",
        "| setting | true pair prob delta | true result prob delta | best result prob delta | hard pair delta | hard calc/result delta | grad cos vs pair CE | grad cos vs result group | input/upstream/semantic delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in [
        ("identifiable sum_left_operand", one_ident),
        ("natural sum-only product", one_nat),
    ]:
        deltas = row["mean_deltas"]
        cos = row["mean_gradient_cosines"]
        pdelta = row["mean_parameter_delta"]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                [
                    fmt(deltas["true_pair_probability"]),
                    fmt(deltas["true_result_group_probability"]),
                    fmt(deltas["best_result_group_probability"]),
                    fmt(deltas["hard_learned_pair_exact"]),
                    fmt(deltas["hard_learned_calculator_result_accuracy"]),
                    fmt(cos["relaxed_answer_vs_hard_best_pair_ce"]),
                    fmt(cos["relaxed_answer_vs_best_result_group_nll"]),
                    f"{fmt(pdelta['calculator_hook.input_proj']['l2'])}/{fmt(pdelta['upstream_encoder']['l2'])}/{fmt(pdelta['semantic_decoder']['l2'])}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Closure Decision",
            "",
            "Phase 6 should close. The deterministic Concrete positive is real, replicated, and retained in the identifiable setting, but the natural sum-only negative is best explained as an underidentified/diffuse result-action landscape for independent operand heads rather than a broken decoder or broken relaxation implementation.",
            "",
            "Recommended Phase 7 first task: test a result-space interface parameterization or structured joint-pair objective for natural `0..19` addition before any `operand_max=99` scaling.",
        ]
    )
    (RUN_ROOT / "summary.md").write_text("\n".join(lines) + "\n")


def run_all() -> dict[str, Any]:
    if not STAGE0B_CHECKPOINT.exists():
        raise FileNotFoundError(STAGE0B_CHECKPOINT)
    if not NATURAL_SUMMARY.exists() or not IDENTIFIABLE_SUMMARY.exists():
        raise FileNotFoundError("required existing Phase 6 summaries are absent")
    device = pick_device()
    summary = {
        "run_root": str(RUN_ROOT),
        "device": device,
        "stage0_existing_evidence": compact_existing_evidence(),
        "full_enum": run_full_enum_pair(device),
        "one_step": run_one_step_pair(device),
    }
    summary["decision"] = decision_from_summary(summary)
    write_json(RUN_ROOT / "summary.json", summary)
    write_markdown(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 closure landscape diagnostic.")
    parser.add_argument("command", choices=["run", "summarize"], nargs="?", default="run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        summary = run_all()
    else:
        summary = load_json(RUN_ROOT / "summary.json")
        write_markdown(summary)
    print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
