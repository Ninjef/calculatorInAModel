import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from scripts.overfit_one_batch import (  # noqa: E402
    adaptive_optimizer_param_groups,
    calculator_read_result_logits,
    fixed_width_operands_from_batch,
    full_enum_expected_answer_loss,
    result_boundary_target_loss,
    score_forced_result_classes_chunked,
    snapshot_row_from_model,
    trainable_parameter_summary,
)
from scripts.run_phase7_local_target_propagation_gate import (  # noqa: E402
    DEFAULT_SEMANTIC_DECODER,
    build_model,
    exhaustive_batch,
    pick_device,
    resolve_path,
    soft_target_loss,
    target_weights_logit_descent,
    target_weights_policy_reweighted,
    write_rows,
)
from src.data import (  # noqa: E402
    ANSWER_FORMATS,
    ArithmeticBatch,
    answer_target,
    make_loss_mask,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import masked_cross_entropy  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent


def random_range_batch(
    *,
    batch_size: int,
    digits: int,
    operand_max: int,
    answer_format: str,
    rng: random.Random,
    device: str,
) -> ArithmeticBatch:
    seq_len = max_sequence_length(digits, answer_format=answer_format)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for _ in range(batch_size):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        prompt = f"{a:0{digits}d}+{b:0{digits}d}="
        ids = tokenize(
            prompt
            + answer_target(
                a,
                b,
                digits,
                answer_format=answer_format,
                fixed_width=True,
            )
        )
        samples.append(pad_sequence(ids, seq_len))
        masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))
    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def prompt_keys_from_batch(batch: ArithmeticBatch) -> list[tuple[int, ...]]:
    return [tuple(row.tolist()) for row in batch.x.detach().cpu()]


def parse_branch_specs(text: str) -> list[str]:
    branches = [part.strip() for part in text.split(",") if part.strip()]
    if not branches:
        raise argparse.ArgumentTypeError("expected at least one branch")
    allowed_prefixes = (
        "hard_boundary",
        "expected_loss",
        "policy_reweighted_t",
        "sampled_policy_reweighted_t",
        "adaptive_policy_reweighted_t",
        "memory_policy_reweighted_t",
        "logit_descent_p",
    )
    for branch in branches:
        if not branch.startswith(allowed_prefixes):
            raise argparse.ArgumentTypeError(f"unknown branch {branch!r}")
    return branches


def parse_sampled_policy_reweighted_branch(branch: str) -> tuple[float, int, int]:
    prefix = "sampled_policy_reweighted_t"
    if not branch.startswith(prefix):
        raise ValueError(f"not a sampled policy-reweighted branch: {branch!r}")
    parts = branch.removeprefix(prefix).split("_")
    if len(parts) != 3 or not parts[1].startswith("k") or not parts[2].startswith("u"):
        raise ValueError(
            "sampled policy-reweighted branch must look like "
            "'sampled_policy_reweighted_t1_k8_u8'"
        )
    temperature = float(parts[0].replace("p", "."))
    top_k = int(parts[1].removeprefix("k"))
    uniform_samples = int(parts[2].removeprefix("u"))
    if temperature <= 0:
        raise ValueError("sampled policy-reweighted temperature must be positive")
    if top_k < 0 or uniform_samples < 0:
        raise ValueError("candidate counts must be non-negative")
    if top_k + uniform_samples < 1:
        raise ValueError("sampled policy-reweighted branch needs at least one candidate")
    return temperature, top_k, uniform_samples


def parse_adaptive_policy_reweighted_branch(branch: str) -> tuple[float, int, int, int]:
    prefix = "adaptive_policy_reweighted_t"
    if not branch.startswith(prefix):
        raise ValueError(f"not an adaptive policy-reweighted branch: {branch!r}")
    parts = branch.removeprefix(prefix).split("_")
    if (
        len(parts) != 4
        or not parts[1].startswith("u")
        or not parts[2].startswith("b")
        or not parts[3].startswith("r")
    ):
        raise ValueError(
            "adaptive policy-reweighted branch must look like "
            "'adaptive_policy_reweighted_t1_u8_b4_r2'"
        )
    temperature = float(parts[0].replace("p", "."))
    uniform_samples = int(parts[1].removeprefix("u"))
    beam = int(parts[2].removeprefix("b"))
    radius = int(parts[3].removeprefix("r"))
    if temperature <= 0:
        raise ValueError("adaptive policy-reweighted temperature must be positive")
    if uniform_samples < 1:
        raise ValueError("adaptive branch needs at least one uniform seed")
    if beam < 1:
        raise ValueError("adaptive branch needs a positive beam size")
    if radius < 0:
        raise ValueError("adaptive branch radius must be non-negative")
    return temperature, uniform_samples, beam, radius


def parse_memory_policy_reweighted_branch(
    branch: str,
) -> tuple[float, int, int, int, int]:
    prefix = "memory_policy_reweighted_t"
    if not branch.startswith(prefix):
        raise ValueError(f"not a memory policy-reweighted branch: {branch!r}")
    parts = branch.removeprefix(prefix).split("_")
    if len(parts) < 3 or not parts[1].startswith("u") or not parts[2].startswith("m"):
        raise ValueError(
            "memory policy-reweighted branch must look like "
            "'memory_policy_reweighted_t1_u8_m24' or "
            "'memory_policy_reweighted_t1_u2_m30_r4' or "
            "'memory_policy_reweighted_t1_u2_m30_reset50'"
        )
    temperature = float(parts[0].replace("p", "."))
    uniform_samples = int(parts[1].removeprefix("u"))
    memory_candidates = int(parts[2].removeprefix("m"))
    rescore_candidates = 0
    reset_interval = 0
    for suffix in parts[3:]:
        if suffix.startswith("reset"):
            if reset_interval != 0:
                raise ValueError("memory reset interval specified more than once")
            reset_interval = int(suffix.removeprefix("reset"))
        elif suffix.startswith("r"):
            if rescore_candidates != 0:
                raise ValueError("memory rescore count specified more than once")
            rescore_candidates = int(suffix.removeprefix("r"))
        else:
            raise ValueError(
                "memory policy-reweighted branch suffixes must be rN or resetN"
            )
    if temperature <= 0:
        raise ValueError("memory policy-reweighted temperature must be positive")
    if uniform_samples < 1:
        raise ValueError("memory branch needs at least one fresh uniform sample")
    if memory_candidates < 0:
        raise ValueError("memory candidate count must be non-negative")
    if rescore_candidates < 0:
        raise ValueError("memory rescore count must be non-negative")
    if rescore_candidates > memory_candidates:
        raise ValueError("memory rescore count cannot exceed memory candidate count")
    if reset_interval < 0:
        raise ValueError("memory reset interval must be non-negative")
    return (
        temperature,
        uniform_samples,
        memory_candidates,
        rescore_candidates,
        reset_interval,
    )


def sample_uniform_result_candidates(
    *,
    batch_size: int,
    result_vocab_size: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    sample_count = min(count, result_vocab_size)
    return (
        torch.rand(batch_size, result_vocab_size, device=device)
        .topk(k=sample_count, dim=-1)
        .indices
    )


@torch.no_grad()
def score_forced_candidate_result_classes(
    model,
    batch,
    candidates: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("candidate result chunk size must be positive")
    if candidates.ndim != 2:
        raise ValueError("candidates must be [batch, candidate_count]")
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    try:
        for start in range(0, candidates.shape[1], chunk_size):
            forced_chunk = candidates[:, start : start + chunk_size]
            width = forced_chunk.shape[1]
            expanded_x = batch.x.repeat_interleave(width, dim=0)
            expanded_y = batch.y.repeat_interleave(width, dim=0)
            expanded_mask = batch.loss_mask.repeat_interleave(width, dim=0)
            forced = forced_chunk.reshape(-1)
            logits = model(expanded_x, forced_calculator_result_class=forced)
            token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                expanded_y.reshape(-1),
                reduction="none",
            ).reshape_as(expanded_y)
            loss_mask = expanded_mask.to(token_loss.dtype)
            losses.append(
                ((token_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0))
                .reshape(batch.x.shape[0], width)
                .detach()
            )
    finally:
        if was_training:
            model.train()
    return torch.cat(losses, dim=-1)


def load_prompt_keyed_memory_tables(
    *,
    state: dict[str, Any],
    batch: ArithmeticBatch,
    result_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str], list[tuple[int, ...]]]:
    loss_map = state.setdefault("prompt_loss_table", {})
    seen_map = state.setdefault("prompt_seen_table", {})
    keys = prompt_keys_from_batch(batch)
    loss_rows: list[torch.Tensor] = []
    seen_rows: list[torch.Tensor] = []
    new_prompts = 0
    for key in keys:
        loss_row = loss_map.get(key)
        seen_row = seen_map.get(key)
        if loss_row is None or seen_row is None:
            new_prompts += 1
            loss_row = batch.x.new_full(
                (result_vocab_size,),
                0,
                dtype=torch.float32,
            ).fill_(float("inf"))
            seen_row = torch.zeros(
                result_vocab_size,
                dtype=torch.bool,
                device=batch.x.device,
            )
        else:
            loss_row = loss_row.to(batch.x.device)
            seen_row = seen_row.to(batch.x.device)
        loss_rows.append(loss_row)
        seen_rows.append(seen_row)
    state["prompt_memory_entries"] = len(loss_map)
    return (
        torch.stack(loss_rows, dim=0),
        torch.stack(seen_rows, dim=0),
        {
            "target_memory_key_mode": "prompt",
            "target_prompt_memory_entries": int(len(loss_map)),
            "target_new_prompt_fraction": float(new_prompts / max(1, len(keys))),
        },
        keys,
    )


def save_prompt_keyed_memory_tables(
    *,
    state: dict[str, Any],
    keys: list[tuple[int, ...]],
    loss_table: torch.Tensor,
    seen_table: torch.Tensor,
) -> None:
    loss_map = state.setdefault("prompt_loss_table", {})
    seen_map = state.setdefault("prompt_seen_table", {})
    for idx, key in enumerate(keys):
        loss_map[key] = loss_table[idx].detach().clone()
        seen_map[key] = seen_table[idx].detach().clone()
    state["prompt_memory_entries"] = len(loss_map)


def dense_unique_candidate_policy_reweighted_loss(
    *,
    model,
    batch,
    result_logits: torch.Tensor,
    candidates: torch.Tensor,
    candidate_losses: torch.Tensor,
    temperature: float,
    min_probability_floor: float,
    metrics: dict[str, float | str],
    digits: int,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    if candidates.shape != candidate_losses.shape:
        raise ValueError("candidates and candidate losses must have the same shape")
    result_vocab_size = result_logits.shape[-1]
    batch_size = batch.x.shape[0]
    row_idx = torch.arange(batch_size, device=batch.x.device)
    loss_table = result_logits.new_full(
        (batch_size, result_vocab_size),
        float("inf"),
    )
    for idx in range(candidates.shape[1]):
        classes = candidates[:, idx]
        current = loss_table[row_idx, classes]
        loss_table[row_idx, classes] = torch.minimum(current, candidate_losses[:, idx])

    candidate_mask = torch.isfinite(loss_table)
    log_probs = result_logits.log_softmax(dim=-1)
    target_logits = log_probs.detach() - (loss_table / temperature)
    target_logits = target_logits.masked_fill(~candidate_mask, -1.0e9)
    weights = torch.softmax(target_logits, dim=-1).detach()
    if min_probability_floor > 0:
        unique_counts = candidate_mask.sum(dim=-1)
        if bool((min_probability_floor * unique_counts >= 1.0).any().item()):
            raise ValueError("min probability floor times candidate count must be < 1")
        weights = torch.where(
            candidate_mask,
            weights.clamp_min(min_probability_floor),
            torch.zeros_like(weights),
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

    loss = -(weights * log_probs).sum(dim=-1).mean()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
    true_sum = true_a + true_b
    target_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    target_argmax = weights.argmax(dim=-1)
    true_candidate = candidate_mask[row_idx, true_sum]
    true_prob = weights[row_idx, true_sum]
    candidate_current_weights = torch.softmax(
        log_probs.detach().masked_fill(~candidate_mask, -1.0e9),
        dim=-1,
    )
    masked_losses = loss_table.masked_fill(~candidate_mask, 0.0)
    candidate_current_loss = (candidate_current_weights * masked_losses).sum(dim=-1)
    candidate_target_loss = (weights * masked_losses).sum(dim=-1)
    unique_counts = candidate_mask.sum(dim=-1).to(torch.float32)
    return loss, {
        **metrics,
        "target_scored_results": int(candidates.shape[1]),
        "target_unique_scored_results": float(unique_counts.mean().item()),
        "target_scored_fraction": float(candidates.shape[1] / result_vocab_size),
        "target_unique_scored_fraction": float(
            (unique_counts / result_vocab_size).mean().item()
        ),
        "target_true_candidate_coverage": float(true_candidate.float().mean().item()),
        "target_entropy": float(target_entropy.mean().item()),
        "target_effective_results": float(target_entropy.exp().mean().item()),
        "target_true_probability": float(true_prob.mean().item()),
        "target_argmax_accuracy": float((target_argmax == true_sum).float().mean().item()),
        "target_argmax_matches_current": float(
            (target_argmax == result_logits.detach().argmax(dim=-1)).float().mean().item()
        ),
        "target_candidate_expected_loss": float(candidate_target_loss.mean().item()),
        "current_candidate_expected_loss": float(candidate_current_loss.mean().item()),
        "target_candidate_expected_improvement": float(
            (candidate_current_loss - candidate_target_loss).mean().item()
        ),
    }


def sampled_policy_reweighted_loss(
    model,
    batch,
    *,
    branch: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    temperature, top_k, uniform_samples = parse_sampled_policy_reweighted_branch(branch)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    result_vocab_size = result_logits.shape[-1]
    candidate_parts: list[torch.Tensor] = []
    if top_k > 0:
        candidate_parts.append(
            result_logits.detach()
            .topk(k=min(top_k, result_vocab_size), dim=-1)
            .indices
        )
    if uniform_samples > 0:
        candidate_parts.append(
            sample_uniform_result_candidates(
                batch_size=batch.x.shape[0],
                result_vocab_size=result_vocab_size,
                count=uniform_samples,
                device=batch.x.device,
            )
        )
    candidates = torch.cat(candidate_parts, dim=-1)
    candidate_losses = score_forced_candidate_result_classes(
        model,
        batch,
        candidates,
        chunk_size=args.result_chunk_size,
    )
    selected_log_probs = result_logits.log_softmax(dim=-1).gather(1, candidates)
    target_logits = selected_log_probs.detach() - (candidate_losses / temperature)
    weights = torch.softmax(target_logits, dim=-1).detach()
    if args.min_probability_floor > 0:
        action_count = weights.shape[-1]
        if args.min_probability_floor * action_count >= 1.0:
            raise ValueError("min probability floor times candidate count must be < 1")
        weights = weights.clamp_min(args.min_probability_floor)
        weights = weights / weights.sum(dim=-1, keepdim=True)

    loss = -(weights * selected_log_probs).sum(dim=-1).mean()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    target_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    target_argmax = candidates.gather(1, weights.argmax(dim=-1, keepdim=True)).squeeze(-1)
    true_mask = candidates == true_sum.unsqueeze(-1)
    true_prob = (weights * true_mask.to(weights.dtype)).sum(dim=-1)
    unique_counts = torch.tensor(
        [row.unique().numel() for row in candidates.detach().cpu()],
        device=batch.x.device,
        dtype=torch.float32,
    )
    candidate_current_weights = torch.softmax(selected_log_probs.detach(), dim=-1)
    candidate_current_loss = (candidate_current_weights * candidate_losses).sum(dim=-1)
    candidate_target_loss = (weights * candidate_losses).sum(dim=-1)
    return loss, {
        "branch_loss_mode": branch,
        "branch_loss": float(loss.detach().item()),
        "target_family": "sampled_policy_reweighted",
        "target_temperature": float(temperature),
        "target_top_k": int(top_k),
        "target_uniform_samples": int(uniform_samples),
        "target_scored_results": int(candidates.shape[1]),
        "target_unique_scored_results": float(unique_counts.mean().item()),
        "target_scored_fraction": float(candidates.shape[1] / result_vocab_size),
        "target_true_candidate_coverage": float(true_mask.any(dim=-1).float().mean().item()),
        "target_entropy": float(target_entropy.mean().item()),
        "target_effective_results": float(target_entropy.exp().mean().item()),
        "target_true_probability": float(true_prob.mean().item()),
        "target_argmax_accuracy": float((target_argmax == true_sum).float().mean().item()),
        "target_argmax_matches_current": float(
            (target_argmax == result_logits.detach().argmax(dim=-1)).float().mean().item()
        ),
        "target_candidate_expected_loss": float(candidate_target_loss.mean().item()),
        "current_candidate_expected_loss": float(candidate_current_loss.mean().item()),
        "target_candidate_expected_improvement": float(
            (candidate_current_loss - candidate_target_loss).mean().item()
        ),
    }


def adaptive_policy_reweighted_loss(
    model,
    batch,
    *,
    branch: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    temperature, uniform_samples, beam, radius = parse_adaptive_policy_reweighted_branch(
        branch
    )
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    result_vocab_size = result_logits.shape[-1]
    initial_candidates = sample_uniform_result_candidates(
        batch_size=batch.x.shape[0],
        result_vocab_size=result_vocab_size,
        count=uniform_samples,
        device=batch.x.device,
    )
    initial_losses = score_forced_candidate_result_classes(
        model,
        batch,
        initial_candidates,
        chunk_size=args.result_chunk_size,
    )
    beam_count = min(beam, initial_candidates.shape[1])
    best_initial_idx = initial_losses.topk(k=beam_count, largest=False, dim=-1).indices
    centers = initial_candidates.gather(1, best_initial_idx)
    offsets = torch.arange(-radius, radius + 1, device=batch.x.device)
    expanded_candidates = (
        centers.unsqueeze(-1)
        .add(offsets.view(1, 1, -1))
        .clamp(0, result_vocab_size - 1)
        .reshape(batch.x.shape[0], -1)
    )
    expanded_losses = score_forced_candidate_result_classes(
        model,
        batch,
        expanded_candidates,
        chunk_size=args.result_chunk_size,
    )
    candidates = torch.cat([initial_candidates, expanded_candidates], dim=-1)
    candidate_losses = torch.cat([initial_losses, expanded_losses], dim=-1)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    initial_true_coverage = (
        (initial_candidates == true_sum.unsqueeze(-1)).any(dim=-1).float().mean().item()
    )
    loss, metrics = dense_unique_candidate_policy_reweighted_loss(
        model=model,
        batch=batch,
        result_logits=result_logits,
        candidates=candidates,
        candidate_losses=candidate_losses,
        temperature=temperature,
        min_probability_floor=args.min_probability_floor,
        digits=args.digits,
        metrics={
            "branch_loss_mode": branch,
            "target_family": "adaptive_policy_reweighted",
            "target_temperature": float(temperature),
            "target_uniform_samples": int(uniform_samples),
            "target_adaptive_beam": int(beam),
            "target_adaptive_radius": int(radius),
            "target_initial_true_candidate_coverage": float(initial_true_coverage),
        },
    )
    return loss, {
        "branch_loss": float(loss.detach().item()),
        **metrics,
    }


def memory_policy_reweighted_loss(
    model,
    batch,
    *,
    branch: str,
    args: argparse.Namespace,
    state: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float | str]]:
    (
        temperature,
        uniform_samples,
        memory_candidates,
        rescore_candidates,
        reset_interval,
    ) = parse_memory_policy_reweighted_branch(branch)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    result_vocab_size = result_logits.shape[-1]
    batch_size = batch.x.shape[0]
    call_index = int(state.get("target_loss_calls", 0))
    reset_count = int(state.get("reset_count", 0))
    did_reset = False
    if reset_interval > 0 and call_index > 0 and call_index % reset_interval == 0:
        if args.streaming_train_batch_size > 0:
            state["prompt_loss_table"] = {}
            state["prompt_seen_table"] = {}
            state["prompt_memory_entries"] = 0
        elif "loss_table" in state:
            state["loss_table"].fill_(float("inf"))
            state["seen_table"].zero_()
        reset_count += 1
        state["reset_count"] = reset_count
        did_reset = True
    state["target_loss_calls"] = call_index + 1
    if args.streaming_train_batch_size > 0:
        loss_table, seen_table, key_metrics, prompt_keys = load_prompt_keyed_memory_tables(
            state=state,
            batch=batch,
            result_vocab_size=result_vocab_size,
        )
    else:
        prompt_keys = []
        key_metrics = {
            "target_memory_key_mode": "row",
            "target_prompt_memory_entries": 0,
            "target_new_prompt_fraction": 0.0,
        }
        if "loss_table" not in state:
            state["loss_table"] = result_logits.new_full(
                (batch_size, result_vocab_size), float("inf")
            )
            state["seen_table"] = torch.zeros(
                (batch_size, result_vocab_size),
                device=batch.x.device,
                dtype=torch.bool,
            )
        loss_table = state["loss_table"]
        seen_table = state["seen_table"]
        if loss_table.shape != (batch_size, result_vocab_size):
            raise ValueError("memory branch state shape changed across batches")

    fresh_candidates = sample_uniform_result_candidates(
        batch_size=batch_size,
        result_vocab_size=result_vocab_size,
        count=uniform_samples,
        device=batch.x.device,
    )
    fresh_losses = score_forced_candidate_result_classes(
        model,
        batch,
        fresh_candidates,
        chunk_size=args.result_chunk_size,
    )
    row_idx = torch.arange(batch_size, device=batch.x.device).unsqueeze(1)
    loss_table[row_idx, fresh_candidates] = fresh_losses
    seen_table[row_idx, fresh_candidates] = True

    if memory_candidates > 0:
        memory_count = min(memory_candidates, result_vocab_size)
        ranked_losses = loss_table.masked_fill(~seen_table, float("inf"))
        memory_losses, memory_result_classes = ranked_losses.topk(
            k=memory_count,
            largest=False,
            dim=-1,
        )
        rescore_count = min(rescore_candidates, memory_count)
        if rescore_count > 0:
            rescore_candidates_tensor = memory_result_classes[:, :rescore_count]
            rescored_losses = score_forced_candidate_result_classes(
                model,
                batch,
                rescore_candidates_tensor,
                chunk_size=args.result_chunk_size,
            )
            loss_table[row_idx, rescore_candidates_tensor] = rescored_losses
            seen_table[row_idx, rescore_candidates_tensor] = True
            memory_losses = memory_losses.clone()
            memory_losses[:, :rescore_count] = rescored_losses
        candidates = torch.cat([fresh_candidates, memory_result_classes], dim=-1)
        candidate_losses = torch.cat([fresh_losses, memory_losses], dim=-1)
    else:
        rescore_count = 0
        candidates = fresh_candidates
        candidate_losses = fresh_losses
    if args.streaming_train_batch_size > 0:
        save_prompt_keyed_memory_tables(
            state=state,
            keys=prompt_keys,
            loss_table=loss_table,
            seen_table=seen_table,
        )
        key_metrics["target_prompt_memory_entries"] = int(
            state.get("prompt_memory_entries", 0)
        )

    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    observed_true_coverage = (
        seen_table.gather(1, true_sum.unsqueeze(-1)).squeeze(-1).float().mean().item()
    )
    observed_counts = seen_table.sum(dim=-1).to(torch.float32)
    loss, metrics = dense_unique_candidate_policy_reweighted_loss(
        model=model,
        batch=batch,
        result_logits=result_logits,
        candidates=candidates,
        candidate_losses=candidate_losses,
        temperature=temperature,
        min_probability_floor=args.min_probability_floor,
        digits=args.digits,
        metrics={
            "branch_loss_mode": branch,
            "target_family": "memory_policy_reweighted",
            "target_temperature": float(temperature),
            "target_uniform_samples": int(uniform_samples),
            "target_memory_candidates": int(memory_candidates),
            "target_memory_reset_interval": int(reset_interval),
            "target_memory_reset_count": int(reset_count),
            "target_memory_did_reset": float(did_reset),
            "target_fresh_scored_results": int(fresh_candidates.shape[1]),
            "target_rescored_results": int(rescore_count),
            "target_forced_scores_per_step": int(
                fresh_candidates.shape[1] + rescore_count
            ),
            "target_observed_results": float(observed_counts.mean().item()),
            "target_observed_fraction": float(
                (observed_counts / result_vocab_size).mean().item()
            ),
            "target_observed_true_candidate_coverage": float(
                observed_true_coverage
            ),
            **key_metrics,
        },
    )
    return loss, {
        "branch_loss": float(loss.detach().item()),
        **metrics,
    }


def branch_loss(
    model,
    batch,
    *,
    branch: str,
    args: argparse.Namespace,
    state: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    if branch == "hard_boundary":
        loss, metrics = result_boundary_target_loss(
            model,
            batch,
            num_digits=args.digits,
            target_mode="hard_best_result",
            temperature=args.boundary_target_temperature,
            min_probability_floor=0.0,
            chunk_size=args.result_chunk_size,
        )
        return loss, {
            "branch_loss_mode": branch,
            "branch_loss": float(loss.detach().item()),
            **{f"boundary_{key}": value for key, value in metrics.items()},
        }
    if branch == "expected_loss":
        loss, metrics = full_enum_expected_answer_loss(
            model,
            batch,
            num_digits=args.digits,
            policy_temperature=args.expected_policy_temperature,
            cost_normalization="none",
            entropy_weight=0.0,
            chunk_size=args.result_chunk_size,
        )
        return loss, {
            "branch_loss_mode": branch,
            "branch_loss": float(loss.detach().item()),
            **{f"expected_{key}": value for key, value in metrics.items()},
        }
    if branch.startswith("sampled_policy_reweighted_t"):
        return sampled_policy_reweighted_loss(model, batch, branch=branch, args=args)
    if branch.startswith("adaptive_policy_reweighted_t"):
        return adaptive_policy_reweighted_loss(model, batch, branch=branch, args=args)
    if branch.startswith("memory_policy_reweighted_t"):
        if state is None:
            raise ValueError("memory policy-reweighted branch requires state")
        return memory_policy_reweighted_loss(
            model, batch, branch=branch, args=args, state=state
        )

    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    full_losses = score_forced_result_classes_chunked(
        model,
        batch,
        chunk_size=args.result_chunk_size,
    ).detach()
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=args.digits)
    true_sum = true_a + true_b
    best_result = full_losses.argmin(dim=-1)
    if branch.startswith("policy_reweighted_t"):
        temperature = float(branch.removeprefix("policy_reweighted_t").replace("p", "."))
        weights = target_weights_policy_reweighted(
            result_logits,
            full_losses,
            temperature=temperature,
            min_probability_floor=args.min_probability_floor,
        )
        family_metrics: dict[str, float | str] = {
            "target_family": "policy_reweighted",
            "target_temperature": float(temperature),
        }
    elif branch.startswith("logit_descent_p"):
        proximity = float(branch.removeprefix("logit_descent_p").replace("p", "."))
        weights = target_weights_logit_descent(
            result_logits,
            full_losses,
            steps=args.logit_descent_steps,
            lr=args.logit_descent_lr,
            proximity_weight=proximity,
            temperature=args.logit_descent_target_temperature,
            min_probability_floor=args.min_probability_floor,
        )
        family_metrics = {
            "target_family": "logit_descent",
            "target_temperature": float(args.logit_descent_target_temperature),
            "target_descent_steps": int(args.logit_descent_steps),
            "target_descent_lr": float(args.logit_descent_lr),
            "target_proximity_weight": float(proximity),
        }
    else:
        raise ValueError(f"unknown branch {branch!r}")

    loss = soft_target_loss(model, batch, weights)
    target_argmax = weights.argmax(dim=-1)
    target_entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    result_pred = result_logits.detach().argmax(dim=-1)
    current_probs = result_logits.detach().softmax(dim=-1)
    target_expected_loss = (weights * full_losses).sum(dim=-1)
    current_expected_loss = (current_probs * full_losses).sum(dim=-1)
    true_prob = weights.gather(1, true_sum.unsqueeze(-1)).squeeze(-1)
    best_prob = weights.gather(1, best_result.unsqueeze(-1)).squeeze(-1)
    return loss, {
        "branch_loss_mode": branch,
        "branch_loss": float(loss.detach().item()),
        "target_entropy": float(target_entropy.mean().item()),
        "target_effective_results": float(target_entropy.exp().mean().item()),
        "target_true_probability": float(true_prob.mean().item()),
        "target_best_probability": float(best_prob.mean().item()),
        "target_argmax_accuracy": float((target_argmax == true_sum).float().mean().item()),
        "target_argmax_matches_best": float(
            (target_argmax == best_result).float().mean().item()
        ),
        "target_argmax_matches_current": float(
            (target_argmax == result_pred).float().mean().item()
        ),
        "target_expected_loss": float(target_expected_loss.mean().item()),
        "current_expected_loss": float(current_expected_loss.mean().item()),
        "target_expected_improvement": float(
            (current_expected_loss - target_expected_loss).mean().item()
        ),
        **family_metrics,
    }


def answer_only_loss(model, batch) -> tuple[torch.Tensor, dict[str, float | str]]:
    logits = model(batch.x)
    loss = masked_cross_entropy(logits, batch.y, batch.loss_mask)
    return loss, {
        "branch_loss_mode": "answer_only_retention",
        "branch_loss": float(loss.detach().item()),
    }


@torch.no_grad()
def exact_grid_policy_metrics(model, batch, *, digits: int) -> dict[str, float | str]:
    logits, diagnostics = model(batch.x, return_diagnostics=True)
    result_logits, _, _, _ = calculator_read_result_logits(model, batch)
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=digits)
    true_sum = true_a + true_b
    eq_mask = diagnostics["calculator_trace"]["eq_mask"]
    eq_pos = eq_mask.float().argmax(dim=1).long()
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    result_pred = diagnostics["calculator_trace"]["result_pred"][batch_idx, eq_pos]
    probs = result_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    true_prob = probs.gather(1, true_sum.unsqueeze(1)).squeeze(1)
    token_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.y.reshape(-1),
        reduction="none",
    ).reshape_as(batch.y)
    loss_mask = batch.loss_mask.to(token_loss.dtype)
    answer_nll = (token_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)
    return {
        "exact_grid_answer_nll": float(answer_nll.mean().item()),
        "exact_grid_calculator_result_accuracy": float(
            (result_pred == true_sum).float().mean().item()
        ),
        "exact_grid_result_true_probability": float(true_prob.mean().item()),
        "exact_grid_result_entropy": float(entropy.mean().item()),
        "exact_grid_result_effective_results": float(entropy.exp().mean().item()),
    }


def train_branch(
    *,
    branch: str,
    args: argparse.Namespace,
    device: str,
    eval_batch,
) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rng = random.Random(args.seed + 7919)
    model = build_model(args, device=device)
    initial_state = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(
        adaptive_optimizer_param_groups(
            model,
            lr=args.lr,
            input_proj_lr=args.input_proj_lr,
            upstream_lr=args.upstream_lr,
            weight_decay=args.weight_decay,
        )
    )
    rows: list[dict[str, Any]] = []

    def next_train_batch() -> ArithmeticBatch:
        if args.streaming_train_batch_size > 0:
            return random_range_batch(
                batch_size=args.streaming_train_batch_size,
                digits=args.digits,
                operand_max=args.operand_max,
                answer_format=args.answer_format,
                rng=train_rng,
                device=device,
            )
        return eval_batch

    def append_snapshot(
        step: int,
        train_metrics: dict[str, float | str],
        *,
        phase: str,
        phase_step: int,
    ) -> None:
        row = {
            "branch": branch,
            "phase": phase,
            "step": int(step),
            "phase_step": int(phase_step),
            **train_metrics,
            **exact_grid_policy_metrics(model, eval_batch, digits=args.digits),
        }
        run_sampled_controls = (
            step == 0
            or phase_step == 0
            or (phase == "target" and phase_step == args.steps)
            or (
                phase == "retention"
                and phase_step == args.retention_steps
            )
            or (
                args.control_eval_every > 0
                and step % args.control_eval_every == 0
            )
        )
        if run_sampled_controls:
            row.update(
                {
                    f"sampled_{key}": value
                    for key, value in snapshot_row_from_model(
                        model,
                        step=step,
                        num_digits=args.digits,
                        operand_max=args.operand_max,
                        samples=args.eval_samples,
                        seed=args.seed + 1000,
                        device=device,
                        answer_format=args.answer_format,
                    ).items()
                }
            )
        rows.append(row)

    branch_state: dict[str, Any] = {}
    train_batch = next_train_batch()
    initial_loss, initial_metrics = branch_loss(
        model, train_batch, branch=branch, args=args, state=branch_state
    )
    append_snapshot(0, initial_metrics, phase="target", phase_step=0)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_batch = next_train_batch()
        loss, metrics = branch_loss(
            model, train_batch, branch=branch, args=args, state=branch_state
        )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            append_snapshot(step, metrics, phase="target", phase_step=step)

    target_final = rows[-1]
    if args.retention_steps > 0:
        optimizer = torch.optim.AdamW(
            adaptive_optimizer_param_groups(
                model,
                lr=args.retention_lr,
                input_proj_lr=args.retention_input_proj_lr,
                upstream_lr=args.retention_upstream_lr,
                weight_decay=args.weight_decay,
            )
        )
        for retention_step in range(1, args.retention_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_batch = next_train_batch()
            loss, metrics = answer_only_loss(model, train_batch)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            absolute_step = args.steps + retention_step
            if (
                retention_step % args.retention_eval_every == 0
                or retention_step == args.retention_steps
            ):
                append_snapshot(
                    absolute_step,
                    metrics,
                    phase="retention",
                    phase_step=retention_step,
                )

    final = rows[-1]
    sampled_rows = [row for row in rows if "sampled_normal_exact_match" in row]
    best = max(sampled_rows, key=lambda row: float(row["sampled_normal_exact_match"]))
    calc_best = max(rows, key=lambda row: float(row["exact_grid_calculator_result_accuracy"]))
    return {
        "branch": branch,
        "steps": int(args.steps),
        "retention_steps": int(args.retention_steps),
        "trainable_parameter_groups": trainable_parameter_summary(model),
        "initial_state_tensors": int(len(initial_state)),
        "target_final": target_final,
        "retention_final": final if args.retention_steps > 0 else None,
        "final": final,
        "best_sampled_normal": best,
        "best_exact_grid_calc": calc_best,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = pick_device()
    eval_batch = exhaustive_batch(
        digits=args.digits,
        operand_max=args.operand_max,
        answer_format=args.answer_format,
        device=device,
    )
    branches = parse_branch_specs(args.branches)
    results = [
        train_branch(branch=branch, args=args, device=device, eval_batch=eval_batch)
        for branch in branches
    ]
    return {
        "diagnostic": "local_target_stage1_lift_gate",
        "device": device,
        "seed": int(args.seed),
        "batch_size": int(eval_batch.x.shape[0]),
        "train_mode": "streaming" if args.streaming_train_batch_size > 0 else "exhaustive_grid",
        "streaming_train_batch_size": int(args.streaming_train_batch_size),
        "operand_max": int(args.operand_max),
        "steps": int(args.steps),
        "retention_steps": int(args.retention_steps),
        "eval_every": int(args.eval_every),
        "retention_eval_every": int(args.retention_eval_every),
        "eval_samples": int(args.eval_samples),
        "semantic_decoder_checkpoint": str(resolve_path(args.semantic_decoder_checkpoint)),
        "branches": branches,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Short Stage 1 lift gate for Phase 7 local-target branches."
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--operand-max", type=int, default=19)
    parser.add_argument("--answer-format", choices=ANSWER_FORMATS, default="sum")
    parser.add_argument(
        "--semantic-decoder-checkpoint",
        type=Path,
        default=DEFAULT_SEMANTIC_DECODER,
    )
    parser.add_argument("--calculator-operand-vocab-size", type=int, default=20)
    parser.add_argument("--calculator-read-span-width", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=1)
    parser.add_argument("--n-embd", type=int, default=16)
    parser.add_argument("--mlp-expansion", type=int, default=1)
    parser.add_argument("--calculator-hook-after-layer", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--streaming-train-batch-size", type=int, default=0)
    parser.add_argument("--retention-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--retention-eval-every", type=int, default=25)
    parser.add_argument("--control-eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--input-proj-lr", type=float, default=1e-2)
    parser.add_argument("--upstream-lr", type=float, default=3e-4)
    parser.add_argument("--retention-lr", type=float, default=3e-3)
    parser.add_argument("--retention-input-proj-lr", type=float, default=1e-2)
    parser.add_argument("--retention-upstream-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--result-chunk-size", type=int, default=16)
    parser.add_argument("--boundary-target-temperature", type=float, default=1.0)
    parser.add_argument("--expected-policy-temperature", type=float, default=1.0)
    parser.add_argument("--min-probability-floor", type=float, default=0.0)
    parser.add_argument("--logit-descent-steps", type=int, default=25)
    parser.add_argument("--logit-descent-lr", type=float, default=1.0)
    parser.add_argument("--logit-descent-target-temperature", type=float, default=1.0)
    parser.add_argument(
        "--branches",
        default="hard_boundary,expected_loss,policy_reweighted_t1,logit_descent_p0p1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs/2026-05-29_phase7_local_target_stage1_lift_gate",
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.streaming_train_batch_size < 0:
        raise ValueError("--streaming-train-batch-size must be non-negative")
    if args.retention_steps < 0:
        raise ValueError("--retention-steps must be non-negative")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")
    if args.retention_eval_every < 1:
        raise ValueError("--retention-eval-every must be positive")
    if args.control_eval_every < 0:
        raise ValueError("--control-eval-every must be non-negative")
    if args.result_chunk_size < 1:
        raise ValueError("--result-chunk-size must be positive")

    summary = run(args)
    output_root = resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "local_target_stage1_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    all_rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        branch_dir = output_root / result["branch"]
        branch_dir.mkdir(parents=True, exist_ok=True)
        write_rows(branch_dir / "training_curve.csv", result["rows"])
        all_rows.extend(result["rows"])
    write_rows(output_root / "local_target_stage1_rows.csv", all_rows)
    for result in summary["results"]:
        final = result["final"]
        target_final = result["target_final"]
        best = result["best_sampled_normal"]
        if result["retention_final"] is None:
            print(
                f"{result['branch']}: final_normal={final['sampled_normal_exact_match']:.4f} "
                f"final_calc={final['exact_grid_calculator_result_accuracy']:.4f} "
                f"best_normal={best['sampled_normal_exact_match']:.4f}"
            )
        else:
            print(
                f"{result['branch']}: target_normal={target_final['sampled_normal_exact_match']:.4f} "
                f"target_calc={target_final['exact_grid_calculator_result_accuracy']:.4f} "
                f"retention_normal={final['sampled_normal_exact_match']:.4f} "
                f"retention_calc={final['exact_grid_calculator_result_accuracy']:.4f} "
                f"best_normal={best['sampled_normal_exact_match']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
