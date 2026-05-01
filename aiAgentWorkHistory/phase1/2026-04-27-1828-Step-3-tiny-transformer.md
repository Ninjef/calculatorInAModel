# 2026-04-27 — Step 3: Tiny transformer (Model A) + overfit smoke test

Task: Implement Step 3 from the research plan — a small decoder-only transformer and a "hello world" overfit test on a fixed batch of 8 examples.

## What I did

**Model.**
- Added `src/model.py` (~130 lines) with a nanoGPT-style decoder-only transformer:
  - `GPTConfig` dataclass — defaults: `vocab_size=VOCAB_SIZE` (13), `block_size=16`, `n_layer=4`, `n_head=4`, `n_embd=128`, `dropout=0.0`. ~799K params.
  - `CausalSelfAttention` — single QKV linear, multi-head split, triangular causal mask buffer, scaled dot-product softmax, output projection.
  - `MLP` — `Linear → GELU → Linear` with 4× expansion.
  - `Block` — pre-norm: `x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))`.
  - `TinyGPT` — token + learned positional embeddings, `n_layer` blocks, final `LayerNorm`, `lm_head` (no bias). nanoGPT-style weight init (`std=0.02`).
  - `generate(...)` — greedy argmax decoding loop (eval/no_grad).
  - `masked_cross_entropy(...)` — per-position cross-entropy summed over `loss_mask` and divided by mask count, matching the `loss_mask` contract from `src/data.py`.

**Overfit smoke test.**
- Added `scripts/overfit_one_batch.py`. Picks `mps` if available, builds `TinyGPT`, draws ONE `make_batch(batch_size=8, num_digits=2, ...)` and reuses it across 500 AdamW steps (`lr=3e-3`, `betas=(0.9, 0.95)`, `weight_decay=0.0`). Logs loss every 50 steps, applies `clip_grad_norm_(1.0)`. After training, decodes argmax predictions at the masked answer positions and asserts both `final_loss < 0.01` and `accuracy == 8/8`.
- Caught a bug in the first run: I was comparing predictions (extracted at masked positions) against full target slices that included trailing `<pad>` tokens. Fixed by extracting both target and prediction at the same masked positions.

**Tests.**
- Added `tests/test_model.py` with three fast unit tests (small config: `n_embd=32, n_layer=2, n_head=2`):
  - forward output shape `(B, T, VOCAB_SIZE)` and dtype `float32`,
  - causal-mask leak check: changing the last input token does not alter logits at earlier positions,
  - `masked_cross_entropy` ignores unmasked positions even when their loss would be huge.

**Docs.**
- Added the overfit script to the README's Run section.

## Verification

Ran:

```bash
python -m pytest
python scripts/overfit_one_batch.py
```

Results:
- `pytest`: 8 tests passed (5 data + 3 model) in ~1.1s.
- Overfit script on MPS: ~799K params; loss `2.65` → `0.0007` by step 50 → `0.0000` by step 200; final accuracy `8/8`; printed `OK`.

## Files changed

```
README.md
src/model.py                 (new)
tests/test_model.py          (new)
scripts/overfit_one_batch.py (new)
```

## Walkthrough exercises (not committed)

The user asked for hands-on exercises to build intuition. Suggested four manipulations to run interactively against the overfit script (revert each before moving on):

1. Overwrite `batch.y` at masked positions with random digits — loss still drops to ~0, demonstrating that 8-example "convergence" is memorization, not arithmetic.
2. Shrink to `GPTConfig(n_embd=4, n_layer=1, n_head=1)` — loss plateaus high; shows capacity matters.
3. Add a held-out eval batch from a fresh RNG — train accuracy 8/8, held-out ≈ 0/8; previews why Steps 4 and 7 exist.
4. Sanity: change `SEED` and `NUM_STEPS` and re-run to confirm robustness.

## Next suggested step

Step 4: real training run for Model A on 1/2/3-digit addition with held-out eval, saving baseline metrics for comparison against later Models B and C.
