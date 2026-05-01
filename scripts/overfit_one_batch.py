import argparse
import csv
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import (
    EOS_ID,
    EQ_ID,
    ID_TO_TOKEN,
    ArithmeticBatch,
    make_loss_mask,
    detokenize,
    make_batch,
    max_sequence_length,
    pad_sequence,
    tokenize,
)
from src.model import GPTConfig, TinyGPT, masked_cross_entropy

DEFAULT_DIGITS = (1, 2, 3)
DEFAULT_STEPS = 1000
DEFAULT_EVAL_SAMPLES = 256
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-3
DEFAULT_SEED = 0
LOG_EVERY = 50


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    run_name: str
    seed: int
    num_digits: int
    steps: int
    batch_size: int
    eval_samples: int
    lr: float
    weight_decay: float
    grad_clip: float
    fixed_width: bool
    operand_max: int | None
    calculator_operand_vocab_size: int
    oracle_train: bool
    oracle_warmup_steps: int
    aux_operand_loss_weight: float
    aux_operand_loss_decay_steps: int
    aux_operand_loss_floor: float
    snapshot_every: int
    snapshot_samples: int
    calculator_estimator: str
    calculator_read_position: str
    calculator_injection_mode: str
    calculator_bottleneck_mode: str
    semantic_decoder_checkpoint: str | None
    adaptive_interface_loss_weight: float
    freeze_semantic_decoder: bool
    freeze_upstream_encoder: bool
    reinforce_baseline_beta: float
    reinforce_entropy_weight: float
    reinforce_entropy_decay_steps: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_expansion: int
    calculator_hook_after_layer: int
    model: dict[str, object]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def make_problem(
    a: int, b: int, num_digits: int, fixed_width: bool
) -> tuple[list[int], str]:
    if fixed_width:
        prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
    else:
        prompt = f"{a}+{b}="
    return tokenize(prompt), f"{a + b}<eos>"


@contextmanager
def temporary_calculator_injection_scale(
    model: TinyGPT, scale: float | None
) -> object:
    if scale is None or model.calculator_hook is None:
        yield
        return
    old_scale = model.calculator_hook.injection_scale
    model.calculator_hook.injection_scale = scale
    try:
        yield
    finally:
        model.calculator_hook.injection_scale = old_scale


def generate_answer(
    model: TinyGPT,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str | torch.device,
    oracle_operands: tuple[int, int] | None = None,
    calculator_result_override: str = "add",
) -> list[int]:
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -model.cfg.block_size :]
        oracle_tensor = None
        if oracle_operands is not None:
            oracle_tensor = make_oracle_operands_from_values(
                a=oracle_operands[0],
                b=oracle_operands[1],
                shape=ids_cond.shape,
                device=device,
            )
        logits = model(
            ids_cond,
            oracle_operands=oracle_tensor,
            calculator_result_override=calculator_result_override,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[0, len(prompt_ids) :].tolist()


def trim_after_eos(ids: list[int]) -> list[int]:
    if EOS_ID in ids:
        return ids[: ids.index(EOS_ID) + 1]
    return ids


def evaluate(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    fixed_width: bool,
    device: str | torch.device,
    oracle_train: bool,
    calculator_result_override: str = "add",
    injection_scale: float | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    max_answer_tokens = num_digits + 2
    exact = 0
    examples: list[dict[str, str | bool]] = []

    was_training = model.training
    model.eval()
    with temporary_calculator_injection_scale(model, injection_scale):
        for i in range(samples):
            a = rng.randint(0, operand_max)
            b = rng.randint(0, operand_max)
            prompt_ids, target = make_problem(a, b, num_digits, fixed_width=fixed_width)
            oracle_operands = (a, b) if oracle_train else None
            pred_ids = trim_after_eos(
                generate_answer(
                    model,
                    prompt_ids,
                    max_answer_tokens,
                    device,
                    oracle_operands=oracle_operands,
                    calculator_result_override=calculator_result_override,
                )
            )
            pred = decode_tokens(pred_ids)
            ok = pred == target
            exact += int(ok)
            if i < 8:
                examples.append(
                    {
                        "prompt": detokenize(prompt_ids),
                        "target": target,
                        "prediction": pred,
                        "correct": ok,
                    }
                )
    if was_training:
        model.train()

    return {
        "num_digits": num_digits,
        "samples": samples,
        "exact_match": exact / samples,
        "correct": exact,
        "examples": examples,
    }


@torch.no_grad()
def calculator_trace_rows(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
    oracle_train: bool,
    calculator_result_override: str = "add",
    injection_scale: float | None = None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rows: list[dict[str, object]] = []
    max_answer_tokens = num_digits + 2
    was_training = model.training
    model.eval()
    with temporary_calculator_injection_scale(model, injection_scale):
        for i in range(samples):
            a = rng.randint(0, operand_max)
            b = rng.randint(0, operand_max)
            prompt_ids, target = make_problem(a, b, num_digits, fixed_width=True)
            x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            oracle_operands = None
            if oracle_train:
                oracle_operands = make_oracle_operands_from_values(
                    a=a, b=b, shape=x.shape, device=device
                )
            logits, diagnostics = model(
                x,
                return_diagnostics=True,
                oracle_operands=oracle_operands,
                calculator_result_override=calculator_result_override,
            )
            probs = logits[:, -1, :].softmax(dim=-1)
            pred_ids = trim_after_eos(
                generate_answer(
                    model,
                    prompt_ids,
                    max_answer_tokens,
                    device,
                    oracle_operands=(a, b) if oracle_train else None,
                    calculator_result_override=calculator_result_override,
                )
            )
            pred = decode_tokens(pred_ids)
            eq_pos = prompt_ids.index(EQ_ID)
            trace = diagnostics.get("calculator_trace", {})

            def trace_value(name: str, default: float | int | bool) -> float | int | bool:
                if name not in trace:
                    return default
                value = trace[name][0, eq_pos]
                if value.dtype == torch.bool:
                    return bool(value.item())
                if value.dtype.is_floating_point:
                    return float(value.item())
                return int(value.item())

            rows.append(
                {
                    "sample": i,
                    "prompt": decode_tokens(prompt_ids),
                    "true_a": a,
                    "true_b": b,
                    "true_sum": a + b,
                    "target_answer": target,
                    "prediction": pred,
                    "correct": pred == target,
                    "first_token_confidence": float(probs.max().item()),
                    "a_pred": trace_value("a_pred", -1),
                    "b_pred": trace_value("b_pred", -1),
                    "calculator_result": trace_value("result_pred", -1),
                    "a_confidence": trace_value("a_confidence", float("nan")),
                    "b_confidence": trace_value("b_confidence", float("nan")),
                    "a_entropy": trace_value("a_entropy", float("nan")),
                    "b_entropy": trace_value("b_entropy", float("nan")),
                    "a_logp": trace_value("a_logp", float("nan")),
                    "b_logp": trace_value("b_logp", float("nan")),
                    "sampled_logp": trace_value("sampled_logp", float("nan")),
                    "injection_norm": trace_value("injection_norm", float("nan")),
                    "calculator_read_position_id": trace_value(
                        "calculator_read_position_id", -1
                    ),
                    "a_read_position": trace_value("a_read_position", -1),
                    "b_read_position": trace_value("b_read_position", -1),
                    "eq_read_position": trace_value("eq_read_position", -1),
                    "oracle_used": trace_value("oracle_used", False),
                }
            )
    if was_training:
        model.train()
    return rows


def summarize_trace_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    answer_correct = sum(int(row["correct"]) for row in rows)
    operand_rows = [row for row in rows if row["a_pred"] >= 0 and row["b_pred"] >= 0]
    operand_correct = sum(
        int(row["a_pred"] == row["true_a"] and row["b_pred"] == row["true_b"])
        for row in operand_rows
    )
    result_correct = sum(
        int(row["calculator_result"] == row["true_sum"]) for row in operand_rows
    )

    def mean_field(name: str) -> float:
        finite = [
            float(row[name])
            for row in operand_rows
            if isinstance(row[name], float) and row[name] == row[name]
        ]
        return sum(finite) / len(finite) if finite else float("nan")

    return {
        "samples": len(rows),
        "exact_match": answer_correct / max(len(rows), 1),
        "correct": answer_correct,
        "operand_exact_match": operand_correct / max(len(operand_rows), 1),
        "calculator_result_accuracy": result_correct / max(len(operand_rows), 1),
        "mean_a_confidence": mean_field("a_confidence"),
        "mean_b_confidence": mean_field("b_confidence"),
        "mean_a_entropy": mean_field("a_entropy"),
        "mean_b_entropy": mean_field("b_entropy"),
        "mean_sampled_logp": mean_field("sampled_logp"),
    }


def compact_distribution(values: list[object], *, limit: int = 12) -> str:
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return json.dumps(dict(ordered[:limit]), sort_keys=True)


def snapshot_row_from_model(
    model: TinyGPT,
    *,
    step: int,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
) -> dict[str, object]:
    normal_rows = calculator_trace_rows(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        device=device,
        oracle_train=False,
    )
    normal = summarize_trace_rows(normal_rows)

    injection_zero = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        device=device,
        oracle_train=False,
        injection_scale=0.0,
    )
    oracle = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        device=device,
        oracle_train=True,
    )
    forced_zero = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        device=device,
        oracle_train=False,
        calculator_result_override="zero",
    )
    forced_random = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=samples,
        seed=seed,
        fixed_width=True,
        device=device,
        oracle_train=False,
        calculator_result_override="random",
    )

    return {
        "step": step,
        "samples": samples,
        "normal_exact_match": normal["exact_match"],
        "injection_zero_exact_match": injection_zero["exact_match"],
        "oracle_exact_match": oracle["exact_match"],
        "forced_zero_exact_match": forced_zero["exact_match"],
        "forced_random_exact_match": forced_random["exact_match"],
        "operand_exact_match": normal["operand_exact_match"],
        "calculator_result_accuracy": normal["calculator_result_accuracy"],
        "mean_a_confidence": normal["mean_a_confidence"],
        "mean_b_confidence": normal["mean_b_confidence"],
        "mean_a_entropy": normal["mean_a_entropy"],
        "mean_b_entropy": normal["mean_b_entropy"],
        "learned_result_distribution": compact_distribution(
            [row["calculator_result"] for row in normal_rows]
        ),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_curve(path: Path, curve: list[dict[str, float | int]]) -> None:
    preferred = [
        "step",
        "loss",
        "answer_loss",
        "aux_operand_loss",
        "aux_operand_loss_weight",
        "policy_loss",
        "policy_baseline",
        "policy_advantage_mean",
        "sampled_logp",
        "operand_entropy",
        "entropy_weight",
        "adaptive_interface_loss",
        "adaptive_target_result_accuracy",
        "adaptive_learned_target_agreement",
        "adaptive_target_operand_exact_match",
    ]
    fieldnames = preferred + sorted(
        {key for row in curve for key in row.keys()} - set(preferred)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve)


def create_unique_dir(path: Path) -> Path:
    for attempt in range(100):
        candidate = path if attempt == 0 else path.with_name(f"{path.name}-{attempt}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a unique run directory for {path}")


def masked_cross_entropy_per_example(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    B, T, V = logits.shape
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mask_f = mask.to(loss.dtype)
    return (loss * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)


def make_oracle_operands_from_values(
    *,
    a: int,
    b: int,
    shape: tuple[int, int],
    device: str | torch.device,
) -> torch.Tensor:
    oracle = torch.zeros((*shape, 2), dtype=torch.long, device=device)
    oracle[..., 0] = a
    oracle[..., 1] = b
    return oracle


def make_oracle_operands_from_batch(
    x: torch.Tensor, *, num_digits: int
) -> torch.Tensor:
    powers = torch.tensor(
        [10**i for i in range(num_digits - 1, -1, -1)],
        dtype=torch.long,
        device=x.device,
    )
    a = (x[:, :num_digits].long() * powers).sum(dim=-1)
    b_start = num_digits + 1
    b_end = b_start + num_digits
    b = (x[:, b_start:b_end].long() * powers).sum(dim=-1)
    oracle = torch.zeros((*x.shape, 2), dtype=torch.long, device=x.device)
    oracle[..., 0] = a.unsqueeze(-1)
    oracle[..., 1] = b.unsqueeze(-1)
    return oracle


def fixed_width_operands_from_batch(
    x: torch.Tensor, *, num_digits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    powers = torch.tensor(
        [10**i for i in range(num_digits - 1, -1, -1)],
        dtype=torch.long,
        device=x.device,
    )
    a = (x[:, :num_digits].long() * powers).sum(dim=-1)
    b_start = num_digits + 1
    b_end = b_start + num_digits
    b = (x[:, b_start:b_end].long() * powers).sum(dim=-1)
    return a, b


@torch.no_grad()
def counterfactual_result_targets(
    model: TinyGPT, batch: ArithmeticBatch, *, forced_result_batch_size: int = 256
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("adaptive interface targets require a calculator hook")
    was_training = model.training
    model.eval()
    result_losses: list[torch.Tensor] = []
    result_vocab_size = model.cfg.calculator_result_vocab_size
    for start in range(0, result_vocab_size, forced_result_batch_size):
        forced_classes = torch.arange(
            start,
            min(start + forced_result_batch_size, result_vocab_size),
            device=batch.x.device,
        )
        expanded_x = batch.x.repeat_interleave(len(forced_classes), dim=0)
        expanded_y = batch.y.repeat_interleave(len(forced_classes), dim=0)
        expanded_mask = batch.loss_mask.repeat_interleave(len(forced_classes), dim=0)
        forced = forced_classes.repeat(batch.x.shape[0])
        logits = model(expanded_x, forced_calculator_result_class=forced)
        losses = masked_cross_entropy_per_example(
            logits, expanded_y, expanded_mask
        ).reshape(batch.x.shape[0], len(forced_classes))
        result_losses.append(losses)
    if was_training:
        model.train()
    losses = torch.cat(result_losses, dim=-1)
    targets = losses.argmin(dim=-1)
    return targets, losses


def select_adaptive_operand_targets(
    a_logits: torch.Tensor, b_logits: torch.Tensor, result_targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if a_logits.shape != b_logits.shape:
        raise ValueError("a_logits and b_logits must have the same shape")
    if a_logits.ndim != 2:
        raise ValueError("operand target selection expects [batch, classes] logits")
    classes = a_logits.shape[-1]
    a_idx = torch.arange(classes, device=a_logits.device).view(1, classes, 1)
    b_idx = torch.arange(classes, device=a_logits.device).view(1, 1, classes)
    sum_idx = a_idx + b_idx
    valid = sum_idx == result_targets.view(-1, 1, 1)
    pair_scores = (
        a_logits.log_softmax(dim=-1).unsqueeze(-1)
        + b_logits.log_softmax(dim=-1).unsqueeze(-2)
    )
    pair_scores = pair_scores.masked_fill(~valid, float("-inf"))
    best_pair = pair_scores.reshape(a_logits.shape[0], -1).argmax(dim=-1)
    a_target = best_pair // classes
    b_target = best_pair % classes
    return a_target, b_target


def calculator_read_operand_logits(
    model: TinyGPT, batch: ArithmeticBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.calculator_hook is None:
        raise ValueError("calculator operand logits require a calculator hook")
    B, T = batch.x.shape
    assert T <= model.cfg.block_size, (
        f"sequence length {T} > block_size {model.cfg.block_size}"
    )
    pos = torch.arange(T, device=batch.x.device)
    residual = model.tok_emb(batch.x) + model.pos_emb(pos)
    if model.cfg.calculator_hook_after_layer > 0:
        for i, block in enumerate(model.blocks, start=1):
            residual = block(residual)
            if i == model.cfg.calculator_hook_after_layer:
                break
    operand_logits = model.calculator_hook.input_proj(residual)
    a_logits_all, b_logits_all = operand_logits.split(
        model.cfg.calculator_operand_vocab_size, dim=-1
    )
    positions = model._calculator_read_positions(batch.x)
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    a_pos = positions["a"]
    b_pos = positions["b"]
    return (
        a_logits_all[batch_idx, a_pos],
        b_logits_all[batch_idx, b_pos],
        a_pos,
        b_pos,
    )


def adaptive_interface_loss(
    model: TinyGPT, batch: ArithmeticBatch, *, num_digits: int
) -> tuple[torch.Tensor, dict[str, float]]:
    result_targets, _ = counterfactual_result_targets(model, batch)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    a_targets, b_targets = select_adaptive_operand_targets(
        a_logits, b_logits, result_targets
    )
    loss = (
        torch.nn.functional.cross_entropy(a_logits, a_targets)
        + torch.nn.functional.cross_entropy(b_logits, b_targets)
    ) / 2
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    true_sum = true_a + true_b
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    learned_sum = learned_a + learned_b
    metrics = {
        "adaptive_target_result_accuracy": float(
            (result_targets == true_sum).float().mean().item()
        ),
        "adaptive_learned_target_agreement": float(
            (learned_sum == result_targets).float().mean().item()
        ),
        "adaptive_target_operand_exact_match": float(
            ((a_targets == true_a) & (b_targets == true_b)).float().mean().item()
        ),
    }
    return loss, metrics


@torch.no_grad()
def adaptive_interface_trace_rows(
    model: TinyGPT,
    *,
    num_digits: int,
    operand_max: int,
    samples: int,
    seed: int,
    device: str | torch.device,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    batch = make_range_batch(
        batch_size=samples,
        num_digits=num_digits,
        operand_max=operand_max,
        rng=rng,
        fixed_width=True,
        device=device,
    )
    was_training = model.training
    model.eval()
    result_targets, result_losses = counterfactual_result_targets(model, batch)
    a_logits, b_logits, _, _ = calculator_read_operand_logits(model, batch)
    a_targets, b_targets = select_adaptive_operand_targets(
        a_logits, b_logits, result_targets
    )
    true_a, true_b = fixed_width_operands_from_batch(batch.x, num_digits=num_digits)
    learned_a = a_logits.argmax(dim=-1)
    learned_b = b_logits.argmax(dim=-1)
    prompts = [decode_tokens(row.tolist()).split("=")[0] + "=" for row in batch.x]
    rows: list[dict[str, object]] = []
    for i in range(samples):
        rows.append(
            {
                "sample": i,
                "prompt": prompts[i],
                "true_a": int(true_a[i].item()),
                "true_b": int(true_b[i].item()),
                "true_sum": int((true_a[i] + true_b[i]).item()),
                "target_result": int(result_targets[i].item()),
                "target_a": int(a_targets[i].item()),
                "target_b": int(b_targets[i].item()),
                "learned_a": int(learned_a[i].item()),
                "learned_b": int(learned_b[i].item()),
                "learned_result": int((learned_a[i] + learned_b[i]).item()),
                "target_result_loss": float(
                    result_losses[i, result_targets[i]].item()
                ),
                "target_matches_true_sum": bool(
                    result_targets[i].item() == (true_a[i] + true_b[i]).item()
                ),
                "learned_matches_target_result": bool(
                    (learned_a[i] + learned_b[i]).item() == result_targets[i].item()
                ),
                "target_operands_match_true": bool(
                    a_targets[i].item() == true_a[i].item()
                    and b_targets[i].item() == true_b[i].item()
                ),
            }
        )
    if was_training:
        model.train()
    return rows


def summarize_adaptive_interface_rows(
    rows: list[dict[str, object]]
) -> dict[str, float | int]:
    if not rows:
        return {
            "samples": 0,
            "target_result_accuracy": 0.0,
            "learned_target_result_agreement": 0.0,
            "target_operand_exact_match": 0.0,
        }
    return {
        "samples": len(rows),
        "target_result_accuracy": sum(
            int(row["target_matches_true_sum"]) for row in rows
        )
        / len(rows),
        "learned_target_result_agreement": sum(
            int(row["learned_matches_target_result"]) for row in rows
        )
        / len(rows),
        "target_operand_exact_match": sum(
            int(row["target_operands_match_true"]) for row in rows
        )
        / len(rows),
    }


def load_semantic_decoder_checkpoint(model: TinyGPT, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def freeze_semantic_decoder_parameters(model: TinyGPT) -> None:
    if model.calculator_hook is not None:
        for param in model.calculator_hook.output_proj.parameters():
            param.requires_grad = False
    if model.answer_offset_emb is not None:
        for param in model.answer_offset_emb.parameters():
            param.requires_grad = False
    if model.answer_decoder is not None:
        for param in model.answer_decoder.parameters():
            param.requires_grad = False


def freeze_upstream_encoder_parameters(model: TinyGPT) -> None:
    for module in [model.tok_emb, model.pos_emb, model.blocks, model.ln_f, model.lm_head]:
        for param in module.parameters():
            param.requires_grad = False


def make_range_batch(
    *,
    batch_size: int,
    num_digits: int,
    operand_max: int,
    rng: random.Random,
    fixed_width: bool,
    device: str | torch.device,
) -> ArithmeticBatch:
    seq_len = max_sequence_length(num_digits)
    samples: list[list[int]] = []
    masks: list[list[int]] = []
    for _ in range(batch_size):
        a = rng.randint(0, operand_max)
        b = rng.randint(0, operand_max)
        if fixed_width:
            ids = tokenize(f"{a:0{num_digits}d}+{b:0{num_digits}d}={a + b}<eos>")
        else:
            ids = tokenize(f"{a}+{b}={a + b}<eos>")
        samples.append(pad_sequence(ids, seq_len))
        masks.append(pad_sequence(make_loss_mask(ids), seq_len, pad_id=0))

    tokens = torch.tensor(samples, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)
    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )


def auxiliary_operand_loss(
    model: TinyGPT, batch: ArithmeticBatch, num_digits: int
) -> torch.Tensor:
    if model.calculator_hook is None:
        raise ValueError("auxiliary operand loss requires a calculator hook")
    with torch.no_grad():
        targets = make_oracle_operands_from_batch(batch.x, num_digits=num_digits)
        eq_pos = (batch.x == EQ_ID).float().argmax(dim=-1).long()
        batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
        if model.cfg.calculator_read_position == "operands":
            a_pos = torch.full_like(eq_pos, num_digits - 1)
            b_pos = torch.full_like(eq_pos, (num_digits + 1) + (num_digits - 1))
        else:
            a_pos = eq_pos
            b_pos = eq_pos
        target_a = targets[batch_idx, a_pos, 0]
        target_b = targets[batch_idx, b_pos, 1]

    _, diagnostics = model(batch.x, return_diagnostics=True)
    residual = diagnostics["calculator_read_residual"]
    operand_logits = model.calculator_hook.input_proj(residual)
    a_logits, b_logits = operand_logits.split(
        model.cfg.calculator_operand_vocab_size, dim=-1
    )
    batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
    a_eq_logits = a_logits[batch_idx, a_pos]
    b_eq_logits = b_logits[batch_idx, b_pos]
    return (
        torch.nn.functional.cross_entropy(a_eq_logits, target_a)
        + torch.nn.functional.cross_entropy(b_eq_logits, target_b)
    ) / 2


def auxiliary_operand_weight(
    *, initial_weight: float, decay_steps: int, floor: float, step: int
) -> float:
    if initial_weight <= 0:
        return 0.0
    if decay_steps <= 0:
        return initial_weight
    decayed = initial_weight * max(0.0, 1.0 - (step / decay_steps))
    return max(floor, decayed)


def make_model_config(
    num_digits: int,
    variant: str,
    *,
    injection_scale: float = 1.0,
    operand_vocab_size: int | None = None,
    calculator_estimator: str = "ste",
    calculator_read_position: str = "eq",
    calculator_injection_mode: str = "add",
    calculator_bottleneck_mode: str = "none",
    n_layer: int = 4,
    n_head: int = 4,
    n_embd: int = 128,
    mlp_expansion: int = 4,
    calculator_hook_after_layer: int | None = None,
) -> GPTConfig:
    operand_vocab_size = operand_vocab_size or 10**num_digits
    calculator_enabled = variant in {"model-b", "model-c"}
    calculator_mode = "add" if variant == "model-c" else "off"
    if calculator_hook_after_layer is None:
        calculator_hook_after_layer = min(2, n_layer)
    return GPTConfig(
        block_size=max_sequence_length(num_digits) - 1,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        mlp_expansion=mlp_expansion,
        calculator_enabled=calculator_enabled,
        calculator_mode=calculator_mode,
        calculator_hook_after_layer=calculator_hook_after_layer,
        calculator_operand_vocab_size=operand_vocab_size,
        calculator_result_vocab_size=(2 * operand_vocab_size) - 1,
        calculator_injection_scale=injection_scale,
        calculator_injection_mode=calculator_injection_mode,
        calculator_estimator=calculator_estimator,
        calculator_read_position=calculator_read_position,
        calculator_bottleneck_mode=calculator_bottleneck_mode,
    )


def run_variant(
    *,
    num_digits: int,
    args: argparse.Namespace,
    base_run_dir: Path,
    device: str,
) -> dict[str, object]:
    seed = args.seed + num_digits
    torch.manual_seed(seed)
    rng = random.Random(seed)

    operand_max = args.operand_max
    if operand_max is None:
        operand_max = 10**num_digits - 1
    if operand_max >= 10**num_digits:
        raise ValueError("--operand-max must fit inside --digits")
    calculator_operand_vocab_size = args.calculator_operand_vocab_size
    if calculator_operand_vocab_size is None:
        calculator_operand_vocab_size = 10**num_digits
    if operand_max >= calculator_operand_vocab_size:
        raise ValueError(
            "--calculator-operand-vocab-size must be greater than --operand-max"
        )

    cfg = make_model_config(
        num_digits,
        args.variant,
        injection_scale=args.injection_scale,
        operand_vocab_size=calculator_operand_vocab_size,
        calculator_estimator=args.calculator_estimator,
        calculator_read_position=args.calculator_read_position,
        calculator_injection_mode=args.calculator_injection_mode,
        calculator_bottleneck_mode=args.calculator_bottleneck_mode,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=args.calculator_hook_after_layer,
    )
    model = TinyGPT(cfg).to(device)
    if args.semantic_decoder_checkpoint is not None:
        load_semantic_decoder_checkpoint(model, args.semantic_decoder_checkpoint)
    if args.calculator_estimator == "adaptive_interface" and args.freeze_semantic_decoder:
        freeze_semantic_decoder_parameters(model)
    if args.calculator_estimator == "adaptive_interface" and args.freeze_upstream_encoder:
        freeze_upstream_encoder_parameters(model)
    optim = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.variant}-{num_digits}digit-seed{seed}"
    run_dir = base_run_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_cfg = TrainConfig(
        variant=args.variant,
        run_name=run_name,
        seed=seed,
        num_digits=num_digits,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_samples=args.eval_samples,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        fixed_width=True,
        operand_max=args.operand_max,
        calculator_operand_vocab_size=calculator_operand_vocab_size,
        oracle_train=args.oracle_train,
        oracle_warmup_steps=args.oracle_warmup_steps,
        aux_operand_loss_weight=args.aux_operand_loss_weight,
        aux_operand_loss_decay_steps=args.aux_operand_loss_decay_steps,
        aux_operand_loss_floor=args.aux_operand_loss_floor,
        snapshot_every=args.snapshot_every,
        snapshot_samples=args.snapshot_samples,
        calculator_estimator=args.calculator_estimator,
        calculator_read_position=args.calculator_read_position,
        calculator_injection_mode=args.calculator_injection_mode,
        calculator_bottleneck_mode=args.calculator_bottleneck_mode,
        semantic_decoder_checkpoint=(
            str(args.semantic_decoder_checkpoint)
            if args.semantic_decoder_checkpoint is not None
            else None
        ),
        adaptive_interface_loss_weight=args.adaptive_interface_loss_weight,
        freeze_semantic_decoder=args.freeze_semantic_decoder,
        freeze_upstream_encoder=args.freeze_upstream_encoder,
        reinforce_baseline_beta=args.reinforce_baseline_beta,
        reinforce_entropy_weight=args.reinforce_entropy_weight,
        reinforce_entropy_decay_steps=args.reinforce_entropy_decay_steps,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        mlp_expansion=args.mlp_expansion,
        calculator_hook_after_layer=cfg.calculator_hook_after_layer,
        model=asdict(cfg),
    )
    (run_dir / "config.json").write_text(
        json.dumps(asdict(train_cfg), indent=2) + "\n"
    )

    curve: list[dict[str, float | int]] = []
    snapshots: list[dict[str, object]] = []
    final_loss = float("nan")
    policy_baseline: float | None = None
    model.train()
    for step in range(args.steps + 1):
        if args.operand_max is None:
            batch = make_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                rng=rng,
                fixed_width=True,
                device=device,
            )
        else:
            batch = make_range_batch(
                batch_size=args.batch_size,
                num_digits=num_digits,
                operand_max=operand_max,
                rng=rng,
                fixed_width=True,
                device=device,
            )
        oracle_operands = None
        use_oracle_for_step = args.oracle_train or (
            args.variant == "model-c" and step < args.oracle_warmup_steps
        )
        if use_oracle_for_step:
            oracle_operands = make_oracle_operands_from_batch(
                batch.x, num_digits=num_digits
            )
        use_reinforce = (
            args.variant == "model-c"
            and args.calculator_estimator == "reinforce"
            and not args.oracle_train
        )
        use_adaptive_interface = (
            args.variant == "model-c"
            and args.calculator_estimator == "adaptive_interface"
            and not args.oracle_train
        )
        if use_reinforce:
            logits, diagnostics = model(
                batch.x, oracle_operands=oracle_operands, return_diagnostics=True
            )
        else:
            diagnostics = {}
            logits = model(batch.x, oracle_operands=oracle_operands)
        per_example_answer_loss = masked_cross_entropy_per_example(
            logits, batch.y, batch.loss_mask
        )
        answer_loss = per_example_answer_loss.mean()
        loss = answer_loss
        policy_loss_value = None
        policy_advantage_mean = None
        sampled_logp_value = None
        operand_entropy_value = None
        entropy_weight = 0.0
        adaptive_interface_loss_value = None
        adaptive_metrics: dict[str, float] = {}
        if use_reinforce:
            if policy_baseline is None:
                policy_baseline = float(answer_loss.detach().item())
            trace = diagnostics["calculator_trace"]
            eq_mask = trace["eq_mask"]
            eq_counts = eq_mask.long().sum(dim=-1)
            if not torch.all(eq_counts == 1):
                raise ValueError("REINFORCE training expects one '=' token per example")
            eq_pos = eq_mask.float().argmax(dim=-1).long()
            batch_idx = torch.arange(batch.x.shape[0], device=batch.x.device)
            sampled_logp = trace["sampled_logp"][batch_idx, eq_pos]
            operand_entropy = (
                trace["a_entropy"][batch_idx, eq_pos]
                + trace["b_entropy"][batch_idx, eq_pos]
            )
            advantage = per_example_answer_loss.detach() - policy_baseline
            policy_loss = (advantage * sampled_logp).mean()
            if args.reinforce_entropy_decay_steps > 0:
                entropy_weight = args.reinforce_entropy_weight * max(
                    0.0, 1.0 - (step / args.reinforce_entropy_decay_steps)
                )
            else:
                entropy_weight = args.reinforce_entropy_weight
            entropy_loss = -entropy_weight * operand_entropy.mean()
            loss = loss + policy_loss + entropy_loss
            policy_loss_value = policy_loss.item()
            policy_advantage_mean = advantage.mean().item()
            sampled_logp_value = sampled_logp.mean().item()
            operand_entropy_value = operand_entropy.mean().item()
        if use_adaptive_interface:
            adaptive_loss, adaptive_metrics = adaptive_interface_loss(
                model, batch, num_digits=num_digits
            )
            adaptive_interface_loss_value = adaptive_loss.item()
            loss = loss + (args.adaptive_interface_loss_weight * adaptive_loss)
        aux_loss_value = None
        aux_weight = 0.0
        if args.aux_operand_loss_weight > 0:
            if args.variant != "model-c":
                raise ValueError("--aux-operand-loss-weight requires --variant model-c")
            aux_weight = auxiliary_operand_weight(
                initial_weight=args.aux_operand_loss_weight,
                decay_steps=args.aux_operand_loss_decay_steps,
                floor=args.aux_operand_loss_floor,
                step=step,
            )
            aux_loss = auxiliary_operand_loss(model, batch, num_digits)
            aux_loss_value = aux_loss.item()
            loss = loss + (aux_weight * aux_loss)

        if step % args.log_every == 0:
            loss_value = loss.item()
            curve_row: dict[str, float | int] = {
                "step": step,
                "loss": loss_value,
                "answer_loss": answer_loss.item(),
                "oracle_operands_used": int(use_oracle_for_step),
            }
            if aux_loss_value is not None:
                curve_row["aux_operand_loss"] = aux_loss_value
                curve_row["aux_operand_loss_weight"] = aux_weight
            if use_reinforce:
                curve_row["policy_loss"] = policy_loss_value
                curve_row["policy_baseline"] = policy_baseline
                curve_row["policy_advantage_mean"] = policy_advantage_mean
                curve_row["sampled_logp"] = sampled_logp_value
                curve_row["operand_entropy"] = operand_entropy_value
                curve_row["entropy_weight"] = entropy_weight
            if use_adaptive_interface:
                curve_row["adaptive_interface_loss"] = adaptive_interface_loss_value
                curve_row.update(adaptive_metrics)
            curve.append(curve_row)
            print(
                f"variant={args.variant} digits={num_digits} "
                f"step={step:5d} loss={loss_value:.4f} "
                f"answer_loss={answer_loss.item():.4f}"
                + (
                    " oracle_warmup=1"
                    if use_oracle_for_step and not args.oracle_train
                    else ""
                )
                + (
                    f" policy_loss={policy_loss_value:.4f}"
                    f" baseline={policy_baseline:.4f}"
                    f" entropy={operand_entropy_value:.4f}"
                    if use_reinforce
                    else ""
                )
                + (
                    f" aux_operand_loss={aux_loss_value:.4f}"
                    f" aux_weight={aux_weight:.4f}"
                    if aux_loss_value is not None
                    else ""
                )
                + (
                    f" adaptive_interface_loss={adaptive_interface_loss_value:.4f}"
                    f" target_acc={adaptive_metrics['adaptive_target_result_accuracy']:.3f}"
                    if use_adaptive_interface
                    else ""
                )
            )

        if (
            args.variant == "model-c"
            and args.snapshot_every > 0
            and step % args.snapshot_every == 0
        ):
            snapshot = snapshot_row_from_model(
                model,
                step=step,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=args.snapshot_samples,
                seed=seed + 30_000 + step,
                device=device,
            )
            snapshots.append(snapshot)
            print(
                f"snapshot step={step:5d} "
                f"normal={snapshot['normal_exact_match']:.3f} "
                f"zero_inj={snapshot['injection_zero_exact_match']:.3f} "
                f"oracle={snapshot['oracle_exact_match']:.3f} "
                f"operand={snapshot['operand_exact_match']:.3f}"
            )
            model.train()

        if step == args.steps:
            final_loss = loss.item()
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if use_reinforce:
            answer_loss_value = float(answer_loss.detach().item())
            assert policy_baseline is not None
            policy_baseline = (
                args.reinforce_baseline_beta * policy_baseline
                + (1.0 - args.reinforce_baseline_beta) * answer_loss_value
            )

    metrics = evaluate(
        model,
        num_digits=num_digits,
        operand_max=operand_max,
        samples=args.eval_samples,
        seed=seed + 10_000,
        fixed_width=True,
        device=device,
        oracle_train=args.oracle_train,
    )
    metrics["final_loss"] = final_loss
    metrics["operand_max"] = operand_max
    metrics["calculator_operand_vocab_size"] = calculator_operand_vocab_size
    metrics["parameter_count"] = model.num_params()
    metrics["run_dir"] = str(run_dir)
    metrics["variant"] = args.variant
    metrics["oracle_train"] = args.oracle_train
    metrics["oracle_warmup_steps"] = args.oracle_warmup_steps
    metrics["aux_operand_loss_floor"] = args.aux_operand_loss_floor
    metrics["calculator_estimator"] = args.calculator_estimator
    metrics["calculator_read_position"] = args.calculator_read_position
    metrics["calculator_injection_mode"] = args.calculator_injection_mode
    metrics["calculator_bottleneck_mode"] = args.calculator_bottleneck_mode
    metrics["semantic_decoder_checkpoint"] = (
        str(args.semantic_decoder_checkpoint)
        if args.semantic_decoder_checkpoint is not None
        else None
    )
    metrics["adaptive_interface_loss_weight"] = args.adaptive_interface_loss_weight
    metrics["freeze_semantic_decoder"] = args.freeze_semantic_decoder
    metrics["freeze_upstream_encoder"] = args.freeze_upstream_encoder

    save_curve(run_dir / "training_curve.csv", curve)
    if snapshots:
        write_rows(run_dir / "diagnostic_snapshots.csv", snapshots)
    if args.variant == "model-c":
        trace_rows = calculator_trace_rows(
            model,
            num_digits=num_digits,
            operand_max=operand_max,
            samples=min(args.eval_samples, 128),
            seed=seed + 20_000,
            device=device,
            oracle_train=args.oracle_train,
        )
        trace_summary = summarize_trace_rows(trace_rows)
        metrics["diagnostic_summary"] = trace_summary
        counterfactual_samples = min(args.eval_samples, 128)
        metrics["counterfactuals"] = {
            "samples": counterfactual_samples,
            "injection_zero_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                device=device,
                oracle_train=False,
                injection_scale=0.0,
            )["exact_match"],
            "oracle_at_eval_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                device=device,
                oracle_train=True,
            )["exact_match"],
            "forced_zero_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                device=device,
                oracle_train=False,
                calculator_result_override="zero",
            )["exact_match"],
            "forced_random_exact_match": evaluate(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=counterfactual_samples,
                seed=seed + 21_000,
                fixed_width=True,
                device=device,
                oracle_train=False,
                calculator_result_override="random",
            )["exact_match"],
        }
        write_rows(run_dir / "calculator_trace_rows.csv", trace_rows)
        (run_dir / "diagnostic_summary.json").write_text(
            json.dumps(trace_summary, indent=2) + "\n"
        )
        if args.calculator_estimator == "adaptive_interface":
            adaptive_rows = adaptive_interface_trace_rows(
                model,
                num_digits=num_digits,
                operand_max=operand_max,
                samples=min(args.eval_samples, 128),
                seed=seed + 22_000,
                device=device,
            )
            adaptive_summary = summarize_adaptive_interface_rows(adaptive_rows)
            metrics["adaptive_interface_diagnostic_summary"] = adaptive_summary
            write_rows(run_dir / "adaptive_interface_trace_rows.csv", adaptive_rows)
            (run_dir / "adaptive_interface_summary.json").write_text(
                json.dumps(adaptive_summary, indent=2) + "\n"
            )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(train_cfg),
            "metrics": metrics,
        },
        run_dir / "final_weights.pt",
    )

    print(
        f"variant={args.variant} digits={num_digits} eval exact-match "
        f"{metrics['correct']}/{metrics['samples']} "
        f"({metrics['exact_match']:.3f}); saved {run_dir}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tiny addition models with optional latent calculator hook."
    )
    parser.add_argument(
        "--variant",
        choices=["model-a", "model-b", "model-c"],
        default="model-a",
        help="model-a is the raw baseline, model-b wires the hook off, model-c turns addition on.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIGITS),
        help="Digit counts to train/evaluate separately.",
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument(
        "--operand-max",
        type=int,
        default=None,
        help="Restrict generated operands to 0..N while keeping fixed-width formatting.",
    )
    parser.add_argument(
        "--calculator-operand-vocab-size",
        type=int,
        default=None,
        help=(
            "Override calculator operand classes. For true tiny-vocab runs, set this "
            "to operand_max + 1."
        ),
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--oracle-train",
        action="store_true",
        help="Feed true operands into Model C's calculator during training/eval.",
    )
    parser.add_argument(
        "--oracle-warmup-steps",
        type=int,
        default=0,
        help=(
            "For Model C, feed true operands for the first K training steps, then "
            "switch to learned operands. Final eval remains learned unless "
            "--oracle-train is also set."
        ),
    )
    parser.add_argument(
        "--aux-operand-loss-weight",
        type=float,
        default=0.0,
        help="Training-only diagnostic CE loss on learned calculator operand logits.",
    )
    parser.add_argument(
        "--aux-operand-loss-decay-steps",
        type=int,
        default=0,
        help="Linearly decay aux operand loss to zero over this many steps; 0 keeps it constant.",
    )
    parser.add_argument(
        "--aux-operand-loss-floor",
        type=float,
        default=0.0,
        help="Minimum aux operand loss weight after decay; only used with decay steps.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="For Model C, save lightweight calculator-dependence diagnostics every N steps.",
    )
    parser.add_argument(
        "--snapshot-samples",
        type=int,
        default=64,
        help="Samples per periodic diagnostic snapshot.",
    )
    parser.add_argument(
        "--injection-scale",
        type=float,
        default=1.0,
        help="Scale the calculator residual injection; default preserves existing behavior.",
    )
    parser.add_argument(
        "--calculator-estimator",
        choices=["ste", "reinforce", "adaptive_interface"],
        default="ste",
        help="Estimator for the learned calculator input interface.",
    )
    parser.add_argument(
        "--semantic-decoder-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint whose oracle-trained strict decoder/output interface should "
            "seed adaptive-interface training."
        ),
    )
    parser.add_argument(
        "--adaptive-interface-loss-weight",
        type=float,
        default=1.0,
        help="Weight for counterfactual adaptive-interface operand target loss.",
    )
    parser.add_argument(
        "--freeze-semantic-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze calculator output projection and strict answer decoder.",
    )
    parser.add_argument(
        "--freeze-upstream-encoder",
        action="store_true",
        help="Diagnostic: freeze transformer encoder and train only the interface.",
    )
    parser.add_argument(
        "--calculator-read-position",
        choices=["eq", "operands"],
        default="eq",
        help=(
            "Residual positions used for calculator operand logits. "
            "'eq' preserves existing behavior; 'operands' reads final A/B digits."
        ),
    )
    parser.add_argument(
        "--calculator-injection-mode",
        choices=["add", "replace"],
        default="add",
        help=(
            "How to apply the calculator injection. 'add' preserves the residual "
            "stream; 'replace' bottlenecks active '=' positions to the injection."
        ),
    )
    parser.add_argument(
        "--calculator-bottleneck-mode",
        choices=["none", "answer_decoder"],
        default="none",
        help=(
            "Optional stricter answer path. 'answer_decoder' predicts answer tokens "
            "only from calculator output plus answer-position metadata."
        ),
    )
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument(
        "--mlp-expansion",
        type=int,
        default=4,
        help="MLP hidden-size multiplier relative to n_embd.",
    )
    parser.add_argument(
        "--calculator-hook-after-layer",
        type=int,
        default=None,
        help=(
            "Transformer layer after which to inject the calculator. "
            "Default is 2 for depth >=2, otherwise 1."
        ),
    )
    parser.add_argument(
        "--reinforce-baseline-beta",
        type=float,
        default=0.95,
        help="Exponential moving average coefficient for the answer-loss baseline.",
    )
    parser.add_argument(
        "--reinforce-entropy-weight",
        type=float,
        default=0.01,
        help="Entropy bonus weight for sampled operand distributions.",
    )
    parser.add_argument(
        "--reinforce-entropy-decay-steps",
        type=int,
        default=0,
        help="Linearly decay entropy weight to zero over this many steps; 0 keeps it constant.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oracle_train and args.variant != "model-c":
        raise ValueError("--oracle-train is only meaningful with --variant model-c")
    if args.oracle_warmup_steps < 0:
        raise ValueError("--oracle-warmup-steps must be non-negative")
    if args.oracle_warmup_steps > 0 and args.variant != "model-c":
        raise ValueError("--oracle-warmup-steps requires --variant model-c")
    if args.aux_operand_loss_weight < 0:
        raise ValueError("--aux-operand-loss-weight must be non-negative")
    if args.aux_operand_loss_decay_steps < 0:
        raise ValueError("--aux-operand-loss-decay-steps must be non-negative")
    if args.aux_operand_loss_floor < 0:
        raise ValueError("--aux-operand-loss-floor must be non-negative")
    if (
        args.aux_operand_loss_floor > 0
        and args.aux_operand_loss_weight <= 0
    ):
        raise ValueError("--aux-operand-loss-floor requires --aux-operand-loss-weight")
    if args.aux_operand_loss_floor > args.aux_operand_loss_weight:
        raise ValueError("--aux-operand-loss-floor cannot exceed aux weight")
    if args.snapshot_every < 0:
        raise ValueError("--snapshot-every must be non-negative")
    if args.snapshot_every > 0 and args.variant != "model-c":
        raise ValueError("--snapshot-every requires --variant model-c")
    if args.snapshot_samples < 1:
        raise ValueError("--snapshot-samples must be positive")
    if args.calculator_estimator == "reinforce" and args.variant != "model-c":
        raise ValueError("--calculator-estimator reinforce requires --variant model-c")
    if args.calculator_estimator == "adaptive_interface" and args.variant != "model-c":
        raise ValueError(
            "--calculator-estimator adaptive_interface requires --variant model-c"
        )
    if args.calculator_estimator == "adaptive_interface":
        if args.oracle_train:
            raise ValueError("adaptive_interface is for learned operands, not --oracle-train")
        if args.calculator_bottleneck_mode != "answer_decoder":
            raise ValueError(
                "adaptive_interface requires --calculator-bottleneck-mode answer_decoder"
            )
        if args.semantic_decoder_checkpoint is None:
            raise ValueError(
                "adaptive_interface requires --semantic-decoder-checkpoint"
            )
    if (
        args.semantic_decoder_checkpoint is not None
        and not args.semantic_decoder_checkpoint.exists()
    ):
        raise ValueError("--semantic-decoder-checkpoint does not exist")
    if args.adaptive_interface_loss_weight < 0:
        raise ValueError("--adaptive-interface-loss-weight must be non-negative")
    if not 0 <= args.reinforce_baseline_beta < 1:
        raise ValueError("--reinforce-baseline-beta must be in [0, 1)")
    if args.reinforce_entropy_weight < 0:
        raise ValueError("--reinforce-entropy-weight must be non-negative")
    if args.reinforce_entropy_decay_steps < 0:
        raise ValueError("--reinforce-entropy-decay-steps must be non-negative")
    if args.n_layer < 1:
        raise ValueError("--n-layer must be positive")
    if args.n_head < 1:
        raise ValueError("--n-head must be positive")
    if args.n_embd < 1:
        raise ValueError("--n-embd must be positive")
    if args.n_embd % args.n_head != 0:
        raise ValueError("--n-embd must be divisible by --n-head")
    if args.mlp_expansion < 1:
        raise ValueError("--mlp-expansion must be positive")
    if (
        args.calculator_hook_after_layer is not None
        and not 0 <= args.calculator_hook_after_layer <= args.n_layer
    ):
        raise ValueError("--calculator-hook-after-layer must be within model depth")
    device = pick_device()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    suffix_parts = [args.variant]
    if args.oracle_train:
        suffix_parts.append("oracle")
    elif args.oracle_warmup_steps > 0:
        suffix_parts.append(f"oraclewarm{args.oracle_warmup_steps}")
    if args.operand_max is not None:
        suffix_parts.append(f"op0-{args.operand_max}")
    if args.calculator_estimator != "ste":
        suffix_parts.append(args.calculator_estimator)
    if args.calculator_injection_mode != "add":
        suffix_parts.append(args.calculator_injection_mode)
    if args.calculator_bottleneck_mode != "none":
        suffix_parts.append(args.calculator_bottleneck_mode)
    if args.aux_operand_loss_weight > 0:
        suffix_parts.append(f"aux{args.aux_operand_loss_weight:g}")
        if args.aux_operand_loss_decay_steps > 0:
            suffix_parts.append(f"auxdecay{args.aux_operand_loss_decay_steps}")
        if args.aux_operand_loss_floor > 0:
            suffix_parts.append(f"auxfloor{args.aux_operand_loss_floor:g}")
    suffix = "-".join(suffix_parts)
    base_run_dir = create_unique_dir(args.run_root / f"{timestamp}_{suffix}")

    print(f"device: {device}")
    print(f"variant: {args.variant}")
    print(f"oracle train: {args.oracle_train}")
    print(f"oracle warmup steps: {args.oracle_warmup_steps}")
    print(f"injection scale: {args.injection_scale}")
    print(f"calculator injection mode: {args.calculator_injection_mode}")
    print(f"calculator bottleneck mode: {args.calculator_bottleneck_mode}")
    print(
        "aux operand loss: "
        f"weight={args.aux_operand_loss_weight} "
        f"decay_steps={args.aux_operand_loss_decay_steps} "
        f"floor={args.aux_operand_loss_floor}"
    )
    print(
        "diagnostic snapshots: "
        f"every={args.snapshot_every} samples={args.snapshot_samples}"
    )
    print(f"calculator estimator: {args.calculator_estimator}")
    print(
        "architecture: "
        f"n_layer={args.n_layer} n_head={args.n_head} "
        f"n_embd={args.n_embd} mlp_expansion={args.mlp_expansion} "
        f"hook_after_layer={args.calculator_hook_after_layer}"
    )
    print(f"run root: {base_run_dir}")

    all_metrics = []
    for num_digits in args.digits:
        all_metrics.append(
            run_variant(
                num_digits=num_digits,
                args=args,
                base_run_dir=base_run_dir,
                device=device,
            )
        )

    summary = {"device": device, "variant": args.variant, "runs": all_metrics}
    (base_run_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print("summary:")
    for metrics in all_metrics:
        print(
            f"  {args.variant} {metrics['num_digits']}-digit: "
            f"exact-match={metrics['exact_match']:.3f}, "
            f"final_loss={metrics['final_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
