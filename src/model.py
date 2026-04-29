import math
from dataclasses import dataclass

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
    dropout: float = 0.0
    calculator_enabled: bool = False
    calculator_mode: str = "off"
    calculator_hook_after_layer: int = 2
    calculator_operand_vocab_size: int = 10
    calculator_result_vocab_size: int = 19


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
        if cfg.calculator_operand_vocab_size < 1:
            raise ValueError("calculator operand vocab size must be positive")
        expected_result_size = (2 * cfg.calculator_operand_vocab_size) - 1
        if cfg.calculator_result_vocab_size != expected_result_size:
            raise ValueError(
                "calculator result vocab size must equal "
                "2 * operand_vocab_size - 1"
            )

        self.mode = cfg.calculator_mode
        self.operand_vocab_size = cfg.calculator_operand_vocab_size
        self.result_vocab_size = cfg.calculator_result_vocab_size
        self.input_proj = nn.Linear(cfg.n_embd, 2 * self.operand_vocab_size)
        self.output_proj = nn.Linear(self.result_vocab_size, cfg.n_embd, bias=False)

    def forward(self, h: Tensor, tokens: Tensor) -> Tensor:
        eq_mask = (tokens == EQ_ID).unsqueeze(-1)
        if self.mode == "off" or not eq_mask.any():
            return h.new_zeros(h.shape)

        operand_logits = self.input_proj(h)
        a_logits, b_logits = operand_logits.split(self.operand_vocab_size, dim=-1)
        flat_result = HardAddSTE.apply(
            a_logits.reshape(-1, self.operand_vocab_size),
            b_logits.reshape(-1, self.operand_vocab_size),
        )
        result = flat_result.reshape(*h.shape[:2], self.result_vocab_size)
        injection = self.output_proj(result)
        return injection * eq_mask.to(injection.dtype)


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
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

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
        if cfg.calculator_enabled:
            if not 0 <= cfg.calculator_hook_after_layer <= cfg.n_layer:
                raise ValueError("calculator hook layer must be within model depth")
            self.calculator_hook: CalculatorHook | None = CalculatorHook(cfg)
        else:
            self.calculator_hook = None
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

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

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)
        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            if (
                self.calculator_hook is not None
                and i == self.cfg.calculator_hook_after_layer
            ):
                h = h + self.calculator_hook(h, x)
        h = self.ln_f(h)
        return self.lm_head(h)

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
