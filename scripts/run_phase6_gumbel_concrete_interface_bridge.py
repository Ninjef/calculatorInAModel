from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.overfit_one_batch import (  # noqa: E402
    adaptive_optimizer_param_groups,
    calculator_read_operand_logits,
    evaluate,
    fixed_width_operands_from_batch,
    freeze_semantic_decoder_parameters,
    freeze_upstream_encoder_parameters,
    full_enum_action_pairs,
    load_semantic_decoder_checkpoint,
    make_model_config,
    make_range_batch,
    masked_cross_entropy_per_example,
    pick_device,
    score_action_loss_candidates_chunked,
    snapshot_row_from_model,
)
from src.model import TinyGPT  # noqa: E402


RUN_ROOT = REPO_ROOT / "runs" / "2026-05-11_phase6_gumbel_concrete_interface_bridge"
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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parameter_group_for_name(name: str) -> str:
    if name.startswith("calculator_hook.input_proj."):
        return "calculator_hook.input_proj"
    if name.startswith(("calculator_hook.output_proj.", "answer_offset_emb.", "answer_decoder.")):
        return "semantic_decoder"
    if name.startswith(("tok_emb.", "pos_emb.", "blocks.", "ln_f.", "lm_head.")):
        return "upstream_encoder"
    return "other"


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
        if parameter_group_for_name(name) != "semantic_decoder" or param.grad is None:
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


def input_proj_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    chunks = []
    for name, param in model.named_parameters():
        if not name.startswith("calculator_hook.input_proj."):
            continue
        if param.grad is None:
            chunks.append(torch.zeros_like(param.detach()).reshape(-1).cpu())
        else:
            chunks.append(param.grad.detach().float().reshape(-1).cpu())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def make_strict_model(
    *,
    checkpoint: Path,
    device: str | torch.device,
    temperature: float,
    mode: str,
    hard_forward: bool,
) -> TinyGPT:
    cfg = make_model_config(
        2,
        "model-c",
        operand_vocab_size=20,
        calculator_estimator="gumbel_concrete_interface",
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
        relaxed_calculator_temperature=temperature,
        relaxed_calculator_mode=mode,
        relaxed_calculator_hard_forward=hard_forward,
    )
    model = TinyGPT(cfg).to(device)
    load_semantic_decoder_checkpoint(
        model, checkpoint, load_scope="semantic_decoder_only"
    )
    freeze_semantic_decoder_parameters(model)
    freeze_upstream_encoder_parameters(model)
    return model


def full_enum_gate_metrics(
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
        full_losses = score_action_loss_candidates_chunked(
            model, batch, candidates, chunk_size=chunk_size
        )
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=2)
    true_idx = true_a * classes + true_b
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_idx = learned_a * classes + learned_b
    best_idx = full_losses.argmin(dim=-1)
    pair_probs = (
        torch.softmax(a_logits / temperature, dim=-1).unsqueeze(-1)
        * torch.softmax(b_logits / temperature, dim=-1).unsqueeze(-2)
    ).reshape(batch.x.shape[0], classes * classes)
    entropy = -(pair_probs * pair_probs.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "best_idx": best_idx,
        "best_pairs": pairs.index_select(0, best_idx),
        "best_true_fraction": float((best_idx == true_idx).float().mean().item()),
        "hard_learned_best_fraction": float(
            (learned_idx == best_idx).float().mean().item()
        ),
        "hard_learned_pair_exact": float(
            (learned_idx == true_idx).float().mean().item()
        ),
        "hard_learned_calc_accuracy": float(
            ((learned_a + learned_b) == (true_a + true_b)).float().mean().item()
        ),
        "best_pair_probability": float(
            pair_probs.gather(1, best_idx.unsqueeze(-1)).mean().item()
        ),
        "operand_entropy": float(entropy.mean().item()),
        "effective_pair_count": float(entropy.exp().mean().item()),
    }


def answer_loss(model: TinyGPT, batch: Any) -> torch.Tensor:
    logits = model(batch.x)
    return masked_cross_entropy_per_example(logits, batch.y, batch.loss_mask).mean()


def cmd_stage0_gradient_gate(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint or STAGE0B_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    torch.manual_seed(args.seed)
    device = pick_device()
    rng = random.Random(args.seed)
    batch = make_range_batch(
        batch_size=args.samples,
        num_digits=2,
        operand_max=19,
        rng=rng,
        fixed_width=True,
        device=device,
        answer_format="sum_left_operand",
    )
    model = make_strict_model(
        checkpoint=checkpoint,
        device=device,
        temperature=args.temperature,
        mode=args.mode,
        hard_forward=True,
    )
    model.train()
    before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    initial_gate = full_enum_gate_metrics(
        model, batch, temperature=args.temperature, chunk_size=args.chunk_size
    )
    initial_answer_loss = float(answer_loss(model, batch).detach().item())

    relaxed_grad_model = copy.deepcopy(model)
    relaxed_loss = answer_loss(relaxed_grad_model, batch)
    relaxed_grad_model.zero_grad(set_to_none=True)
    relaxed_loss.backward()
    relaxed_grad = input_proj_grad_vector(relaxed_grad_model)

    target_grad_model = copy.deepcopy(model)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(target_grad_model, batch)
    best_pairs = initial_gate["best_pairs"]
    local_ce = (
        F.cross_entropy(a_logits, best_pairs[:, 0])
        + F.cross_entropy(b_logits, best_pairs[:, 1])
    ) / 2
    target_grad_model.zero_grad(set_to_none=True)
    local_ce.backward()
    target_grad = input_proj_grad_vector(target_grad_model)
    grad_cosine = float(
        F.cosine_similarity(relaxed_grad, target_grad, dim=0).item()
        if relaxed_grad.numel() and target_grad.norm().item() > 0
        else float("nan")
    )

    optim = torch.optim.AdamW(
        adaptive_optimizer_param_groups(
            model,
            lr=args.lr,
            input_proj_lr=args.lr,
            upstream_lr=args.upstream_lr,
            weight_decay=0.0,
        ),
        betas=(0.9, 0.95),
    )
    optim.zero_grad(set_to_none=True)
    loss = answer_loss(model, batch)
    loss.backward()
    grad_summary = semantic_grad_summary(model)
    optim.step()
    after = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    deltas = model_group_delta_summary(before, after)
    post_gate = full_enum_gate_metrics(
        model, batch, temperature=args.temperature, chunk_size=args.chunk_size
    )
    snapshot = snapshot_row_from_model(
        model,
        step=1,
        num_digits=2,
        operand_max=19,
        samples=args.samples,
        seed=args.seed + 1000,
        device=device,
        answer_format="sum_left_operand",
    )
    payload = {
        "checkpoint": str(checkpoint),
        "device": device,
        "samples": args.samples,
        "temperature": args.temperature,
        "mode": args.mode,
        "hard_forward": True,
        "initial_answer_loss": initial_answer_loss,
        "initial": {
            key: value
            for key, value in initial_gate.items()
            if key not in {"best_idx", "best_pairs"}
        },
        "post_one_step": {
            key: value
            for key, value in post_gate.items()
            if key not in {"best_idx", "best_pairs"}
        },
        "best_pair_probability_delta": (
            post_gate["best_pair_probability"] - initial_gate["best_pair_probability"]
        ),
        "hard_learned_best_fraction_delta": (
            post_gate["hard_learned_best_fraction"]
            - initial_gate["hard_learned_best_fraction"]
        ),
        "gradient_cosine_relaxed_answer_vs_hard_best_ce": grad_cosine,
        "relaxed_answer_loss_grad_norm": float(relaxed_grad.norm().item()),
        "hard_best_ce_grad_norm": float(target_grad.norm().item()),
        "semantic_decoder_grad_l2": grad_summary["semantic_decoder_grad_l2"],
        "semantic_decoder_grad_max_abs": grad_summary[
            "semantic_decoder_grad_max_abs"
        ],
        "parameter_delta_after_one_step": deltas,
        "oracle_at_eval_exact_match": snapshot["oracle_exact_match"],
        "injection_zero_exact_match": snapshot["injection_zero_exact_match"],
        "forced_random_exact_match": snapshot["forced_random_exact_match"],
        "snapshot_normal_exact_match": snapshot["normal_exact_match"],
        "snapshot_operand_exact_match": snapshot["operand_exact_match"],
        "snapshot_pair_exact_match": snapshot["pair_exact_match"],
        "snapshot_calc_accuracy": snapshot["calculator_result_accuracy"],
    }
    semantic_delta = deltas.get("semantic_decoder", {"l2": 0.0, "max_abs": 0.0})
    input_delta = deltas.get("calculator_hook.input_proj", {"l2": 0.0})
    upstream_delta = deltas.get("upstream_encoder", {"l2": 0.0})
    payload["passes_gate"] = (
        float(semantic_delta.get("l2", 0.0)) == 0.0
        and float(input_delta.get("l2", 0.0)) > 0.0
        and float(upstream_delta.get("l2", 0.0)) == 0.0
        and (
            payload["best_pair_probability_delta"] > 0.0
            or payload["gradient_cosine_relaxed_answer_vs_hard_best_ce"] > 0.0
        )
    )
    output = args.output or (RUN_ROOT / "stage0" / "gradient_gate.json")
    write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_pass and not payload["passes_gate"]:
        raise SystemExit("Stage 0 gradient gate failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 Gumbel/Concrete interface bridge helpers."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("stage0-gradient-gate")
    gate.add_argument("--checkpoint", type=Path, default=None)
    gate.add_argument("--output", type=Path, default=None)
    gate.add_argument("--samples", type=int, default=128)
    gate.add_argument("--seed", type=int, default=6201)
    gate.add_argument("--temperature", type=float, default=2.0)
    gate.add_argument("--mode", choices=["deterministic", "gumbel"], default="deterministic")
    gate.add_argument("--chunk-size", type=int, default=64)
    gate.add_argument("--lr", type=float, default=0.03)
    gate.add_argument("--upstream-lr", type=float, default=0.0003)
    gate.add_argument("--require-pass", action="store_true")
    gate.set_defaults(func=cmd_stage0_gradient_gate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
