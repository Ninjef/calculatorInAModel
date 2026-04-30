import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.data import EQ_ID, VOCAB_SIZE


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 16
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_expansion: int = 4
    dropout: float = 0.0
    calculator_enabled: bool = False
    calculator_mode: str = "off"
    calculator_hook_after_layer: int = 2
    calculator_operand_vocab_size: int = 10
    calculator_result_vocab_size: int = 19
    calculator_injection_scale: float = 1.0
    calculator_estimator: str = "ste"
    calculator_read_position: str = "eq"


class HardAddSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_logits: Tensor, b_logits: Tensor) -> Tensor:
        a_idx = a_logits.argmax(dim=-1)
        b_idx = b_logits.argmax(dim=-1)
        result_idx = a_idx + b_idx
        result_size = a_logits.shape[-1] + b_logits.shape[-1] - 1
        ctx.save_for_backward(a_idx, b_idx)
        ctx.a_size = a_logits.shape[-1]
        ctx.b_size = b_logits.shape[-1]
        ctx.result_size = result_size
        return F.one_hot(result_idx, num_classes=result_size).to(a_logits.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a_idx, b_idx = ctx.saved_tensors
        a_offsets = torch.arange(ctx.a_size, device=grad_output.device)
        b_offsets = torch.arange(ctx.b_size, device=grad_output.device)
        grad_a_idx = a_offsets.unsqueeze(0) + b_idx.unsqueeze(-1)
        grad_b_idx = a_idx.unsqueeze(-1) + b_offsets.unsqueeze(0)
        grad_a = grad_output.gather(-1, grad_a_idx)
        grad_b = grad_output.gather(-1, grad_b_idx)
        return grad_a, grad_b


class CalculatorHook(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.calculator_mode not in {"off", "add"}:
            raise ValueError(f"unknown calculator mode: {cfg.calculator_mode}")
        if cfg.calculator_estimator not in {"ste", "reinforce"}:
            raise ValueError(f"unknown calculator estimator: {cfg.calculator_estimator}")
        if cfg.calculator_read_position not in {"eq", "operands"}:
            raise ValueError(
                "calculator_read_position must be one of {'eq', 'operands'}, "
                f"got {cfg.calculator_read_position!r}"
            )
        if cfg.calculator_operand_vocab_size < 1:
            raise ValueError("calculator operand vocab size must be positive")
        expected_result_size = (2 * cfg.calculator_operand_vocab_size) - 1
        if cfg.calculator_result_vocab_size != expected_result_size:
            raise ValueError(
                "calculator result vocab size must equal "
                "2 * operand_vocab_size - 1"
            )

        self.mode = cfg.calculator_mode
        self.estimator = cfg.calculator_estimator
        self.operand_vocab_size = cfg.calculator_operand_vocab_size
        self.result_vocab_size = cfg.calculator_result_vocab_size
        self.injection_scale = cfg.calculator_injection_scale
        self.read_position = cfg.calculator_read_position
        self.input_proj = nn.Linear(cfg.n_embd, 2 * self.operand_vocab_size)
        self.output_proj = nn.Linear(self.result_vocab_size, cfg.n_embd, bias=False)

    def forward(
        self,
        h: Tensor,
        tokens: Tensor,
        *,
        oracle_operands: Tensor | None = None,
        result_override: str = "add",
        forced_result_class: int | None = None,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if result_override not in {"add", "zero", "plus_one", "random"}:
            raise ValueError(f"unknown calculator result override: {result_override}")
        if forced_result_class is not None and not (
            0 <= forced_result_class < self.result_vocab_size
        ):
            raise ValueError(
                "forced_result_class must be in "
                f"[0, {self.result_vocab_size}), got {forced_result_class}"
            )
        eq_mask = (tokens == EQ_ID).unsqueeze(-1)
        trace: dict[str, Tensor] = {}
        if self.mode == "off" or not eq_mask.any():
            injection = h.new_zeros(h.shape)
            if return_trace:
                trace = self._empty_trace(h, tokens, injection)
                return injection, trace
            return injection

        operand_logits = self.input_proj(h)
        a_logits_all, b_logits_all = operand_logits.split(self.operand_vocab_size, dim=-1)
        if self.read_position == "eq":
            read_positions = self._eq_trace_positions(tokens)
            a_logits = a_logits_all
            b_logits = b_logits_all
        else:
            read_positions = self._operand_read_positions(tokens)
            a_logits = self._select_operand_logits(a_logits_all, read_positions["a"])
            b_logits = self._select_operand_logits(b_logits_all, read_positions["b"])
        a_logp = None
        b_logp = None
        if oracle_operands is None:
            if self.estimator == "ste":
                flat_a_logits = a_logits.reshape(-1, self.operand_vocab_size)
                flat_b_logits = b_logits.reshape(-1, self.operand_vocab_size)
                a_pred = a_logits.argmax(dim=-1)
                b_pred = b_logits.argmax(dim=-1)
                if forced_result_class is not None:
                    flat_result = self._forced_result(
                        a_pred, forced_result_class, dtype=h.dtype
                    )
                elif result_override == "add":
                    flat_result = HardAddSTE.apply(flat_a_logits, flat_b_logits)
                else:
                    flat_result = self._overridden_result(
                        a_pred, b_pred, result_override, dtype=h.dtype
                    )
            else:
                a_dist = torch.distributions.Categorical(logits=a_logits)
                b_dist = torch.distributions.Categorical(logits=b_logits)
                a_pred = a_dist.sample()
                b_pred = b_dist.sample()
                a_logp = a_dist.log_prob(a_pred)
                b_logp = b_dist.log_prob(b_pred)
                if forced_result_class is not None:
                    flat_result = self._forced_result(
                        a_pred, forced_result_class, dtype=h.dtype
                    )
                else:
                    flat_result = self._overridden_result(
                        a_pred, b_pred, result_override, dtype=h.dtype
                    )
        else:
            if oracle_operands.shape != (*h.shape[:2], 2):
                raise ValueError(
                    "oracle_operands must have shape "
                    f"{(*h.shape[:2], 2)}, got {tuple(oracle_operands.shape)}"
                )
            oracle_operands = oracle_operands.to(device=h.device, dtype=torch.long)
            a_pred = oracle_operands[..., 0]
            b_pred = oracle_operands[..., 1]
            if forced_result_class is not None:
                flat_result = self._forced_result(
                    a_pred, forced_result_class, dtype=h.dtype
                )
            else:
                flat_result = self._overridden_result(
                    a_pred, b_pred, result_override, dtype=h.dtype
                )
        result = flat_result.reshape(*h.shape[:2], self.result_vocab_size)
        unscaled_injection = self.output_proj(result) * eq_mask.to(h.dtype)
        injection = unscaled_injection * self.injection_scale
        if return_trace:
            trace = self._build_trace(
                a_logits=a_logits,
                b_logits=b_logits,
                a_pred=a_pred,
                b_pred=b_pred,
                a_logp=a_logp,
                b_logp=b_logp,
                result=result,
                tokens=tokens,
                read_positions=read_positions,
                unscaled_injection=unscaled_injection,
                scaled_injection=injection,
                oracle_used=oracle_operands is not None,
            )
            return injection, trace
        return injection

    def _eq_trace_positions(self, tokens: Tensor) -> dict[str, Tensor]:
        eq_mask = tokens == EQ_ID
        any_eq = eq_mask.any(dim=-1)
        eq_pos = eq_mask.float().argmax(dim=-1).long()
        eq_pos = torch.where(any_eq, eq_pos, torch.full_like(eq_pos, -1))
        return {"a": eq_pos, "b": eq_pos, "eq": eq_pos}

    def _operand_read_positions(self, tokens: Tensor) -> dict[str, Tensor]:
        B, T = tokens.shape
        eq_mask = tokens == EQ_ID
        eq_counts = eq_mask.long().sum(dim=-1)
        if not torch.all(eq_counts == 1):
            raise ValueError("calculator read position expects one '=' token per example")
        eq_pos = eq_mask.float().argmax(dim=-1).long()
        # Fixed-width prompt shape is A + B =, so final A/B digit positions are
        # determined by the unique '=' location.
        if torch.any((eq_pos - 1) % 2 != 0):
            raise ValueError(
                "calculator_read_position='operands' requires fixed-width A+B= prompts"
            )
        num_digits = (eq_pos - 1) // 2
        if torch.any(num_digits < 1):
            raise ValueError(
                "calculator_read_position='operands' requires at least one digit"
            )
        a_pos = num_digits - 1
        b_pos = (num_digits + 1) + (num_digits - 1)
        if torch.any(a_pos < 0) or torch.any(b_pos >= T):
            raise ValueError("computed operand read position is outside sequence")
        return {"a": a_pos.long(), "b": b_pos.long(), "eq": eq_pos}

    @staticmethod
    def _select_operand_logits(logits: Tensor, positions: Tensor) -> Tensor:
        B, T, C = logits.shape
        batch_idx = torch.arange(B, device=logits.device)
        selected = logits[batch_idx, positions]
        return selected.unsqueeze(1).expand(B, T, C)

    def _overridden_result(
        self, a_pred: Tensor, b_pred: Tensor, mode: str, *, dtype: torch.dtype
    ) -> Tensor:
        if mode == "add":
            result_idx = a_pred + b_pred
        elif mode == "zero":
            result_idx = torch.zeros_like(a_pred)
        elif mode == "plus_one":
            result_idx = (a_pred + b_pred + 1) % self.result_vocab_size
        elif mode == "random":
            result_idx = torch.randint(
                low=0,
                high=self.result_vocab_size,
                size=a_pred.shape,
                device=a_pred.device,
            )
        else:
            raise ValueError(f"unknown calculator result override: {mode}")
        return F.one_hot(result_idx.reshape(-1), num_classes=self.result_vocab_size).to(
            dtype=dtype
        )

    def _forced_result(
        self, like: Tensor, forced_result_class: int, *, dtype: torch.dtype
    ) -> Tensor:
        result_idx = torch.full_like(like, forced_result_class)
        return F.one_hot(result_idx.reshape(-1), num_classes=self.result_vocab_size).to(
            dtype=dtype
        )

    def _empty_trace(
        self, h: Tensor, tokens: Tensor, injection: Tensor
    ) -> dict[str, Tensor]:
        B, T = tokens.shape
        nan_float = h.new_full((B, T), float("nan"))
        neg_one = tokens.new_full((B, T), -1)
        return {
            "eq_mask": tokens == EQ_ID,
            "a_pred": neg_one,
            "b_pred": neg_one,
            "result_pred": neg_one,
            "a_confidence": nan_float,
            "b_confidence": nan_float,
            "a_entropy": nan_float,
            "b_entropy": nan_float,
            "a_logp": nan_float,
            "b_logp": nan_float,
            "sampled_logp": nan_float,
            "injection_norm": injection.norm(dim=-1),
            "unscaled_injection_norm": injection.norm(dim=-1),
            "oracle_used": torch.zeros((B, T), dtype=torch.bool, device=tokens.device),
            "calculator_read_position_id": tokens.new_full(
                (B, T), 0 if self.read_position == "eq" else 1
            ),
            "a_read_position": neg_one,
            "b_read_position": neg_one,
            "eq_read_position": neg_one,
        }

    @staticmethod
    def _entropy(probs: Tensor) -> Tensor:
        return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

    def _build_trace(
        self,
        *,
        a_logits: Tensor,
        b_logits: Tensor,
        a_pred: Tensor,
        b_pred: Tensor,
        a_logp: Tensor | None,
        b_logp: Tensor | None,
        result: Tensor,
        tokens: Tensor,
        read_positions: dict[str, Tensor],
        unscaled_injection: Tensor,
        scaled_injection: Tensor,
        oracle_used: bool,
    ) -> dict[str, Tensor]:
        a_probs = a_logits.softmax(dim=-1)
        b_probs = b_logits.softmax(dim=-1)
        if a_logp is None:
            a_logp = a_probs.gather(-1, a_pred.unsqueeze(-1)).squeeze(-1).log()
        if b_logp is None:
            b_logp = b_probs.gather(-1, b_pred.unsqueeze(-1)).squeeze(-1).log()
        read_position_id = 0 if self.read_position == "eq" else 1
        B, T = tokens.shape
        return {
            "eq_mask": tokens == EQ_ID,
            "a_pred": a_pred,
            "b_pred": b_pred,
            "result_pred": result.argmax(dim=-1),
            "a_confidence": a_probs.max(dim=-1).values,
            "b_confidence": b_probs.max(dim=-1).values,
            "a_entropy": self._entropy(a_probs),
            "b_entropy": self._entropy(b_probs),
            "a_logp": a_logp,
            "b_logp": b_logp,
            "sampled_logp": a_logp + b_logp,
            "injection_norm": scaled_injection.norm(dim=-1),
            "unscaled_injection_norm": unscaled_injection.norm(dim=-1),
            "oracle_used": torch.full(
                tokens.shape, oracle_used, dtype=torch.bool, device=tokens.device
            ),
            "calculator_read_position_id": tokens.new_full(
                tokens.shape, read_position_id
            ),
            "a_read_position": read_positions["a"].unsqueeze(-1).expand(B, T),
            "b_read_position": read_positions["b"].unsqueeze(-1).expand(B, T),
            "eq_read_position": read_positions["eq"].unsqueeze(-1).expand(B, T),
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:T, :T]
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden_size = cfg.mlp_expansion * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden_size)
        self.proj = nn.Linear(hidden_size, cfg.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.calculator_hook: CalculatorHook | None = None
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        if cfg.calculator_enabled:
            if not 0 <= cfg.calculator_hook_after_layer <= cfg.n_layer:
                raise ValueError("calculator hook layer must be within model depth")
            self.calculator_hook = CalculatorHook(cfg)
            self.calculator_hook.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: Tensor,
        *,
        return_diagnostics: bool = False,
        oracle_operands: Tensor | None = None,
        calculator_result_override: str = "add",
        forced_calculator_result_class: int | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Any]]:
        B, T = x.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        diagnostics: dict[str, Any] = {}
        if (
            self.calculator_hook is not None
            and self.cfg.calculator_hook_after_layer == 0
        ):
            if return_diagnostics:
                diagnostics["calculator_read_residual"] = h.detach()
                injection, trace = self.calculator_hook(
                    h,
                    x,
                    oracle_operands=oracle_operands,
                    result_override=calculator_result_override,
                    forced_result_class=forced_calculator_result_class,
                    return_trace=True,
                )
                diagnostics["calculator_trace"] = trace
                h = h + injection
            else:
                h = h + self.calculator_hook(
                    h,
                    x,
                    oracle_operands=oracle_operands,
                    result_override=calculator_result_override,
                    forced_result_class=forced_calculator_result_class,
                )
        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            if return_diagnostics:
                diagnostics.setdefault("layer_residuals", {})[i] = h.detach()
            if return_diagnostics and i == self.cfg.calculator_hook_after_layer:
                diagnostics["calculator_read_residual"] = h.detach()
            if (
                self.calculator_hook is not None
                and i == self.cfg.calculator_hook_after_layer
            ):
                if return_diagnostics:
                    injection, trace = self.calculator_hook(
                        h,
                        x,
                        oracle_operands=oracle_operands,
                        result_override=calculator_result_override,
                        forced_result_class=forced_calculator_result_class,
                        return_trace=True,
                    )
                    diagnostics["calculator_trace"] = trace
                    h = h + injection
                else:
                    h = h + self.calculator_hook(
                        h,
                        x,
                        oracle_operands=oracle_operands,
                        result_override=calculator_result_override,
                        forced_result_class=forced_calculator_result_class,
                    )
        h = self.ln_f(h)
        logits = self.lm_head(h)
        if return_diagnostics:
            return logits, diagnostics
        return logits

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        self.eval()
        ids = prompt_ids
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.cfg.block_size:]
            logits = self(ids_cond)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
        return ids


def masked_cross_entropy(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
        reduction="none",
    ).reshape(B, T)
    mask_f = mask.to(loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
