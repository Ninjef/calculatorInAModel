# Overview

This task adds a calculator-required experiment to separate two questions that are currently entangled:

1. Can the model solve arithmetic through the normal transformer path or a private side-channel?
2. Can the model learn a usable latent protocol when the calculator is the only viable bridge from prompt encoding to answer decoding?

The current calculator hook is additive:

```text
h = h + calculator_injection
```

That is realistic and useful, but it leaves a bypass path. The next experiment should add a stricter bottleneck mode where the downstream answer path must rely on the calculator injection.

# Research Question

Can a tiny transformer learn to use the embedded calculator when normal residual bypass information is blocked at the calculator handoff?

More specifically:

- Does oracle calculator input still solve the task under a bottleneck?
- Does learned calculator input solve the task under the same bottleneck?
- If learned input fails, is the failure from input-protocol learning, output-side usability, or an over-severe bottleneck design?

# Proposed Mechanism

Add a calculator injection mode flag:

```bash
--calculator-injection-mode add
--calculator-injection-mode replace
```

Behavior:

- `add`: current default behavior. Preserve exact existing semantics.
- `replace`: at active calculator positions, replace the residual vector with the calculator injection instead of adding to it.

Conceptually:

```python
if mode == "add":
    h = h + injection
elif mode == "replace":
    h = torch.where(eq_mask, injection, h)
```

Only apply replacement at the `=` token positions where the calculator hook is active. Leave non-`=` token residuals unchanged.

This is intentionally the least invasive bottleneck. It preserves upstream computation of operands, but blocks the normal residual at the readout token from flowing downstream unchanged.

# Guardrail: Oracle First

Do not interpret learned failures until oracle replacement succeeds.

Required first control:

```bash
python3 scripts/overfit_one_batch.py \
  --variant model-c \
  --oracle-train \
  --digits 2 \
  --steps 1000 \
  --eval-samples 512 \
  --n-layer 2 \
  --n-head 1 \
  --n-embd 16 \
  --mlp-expansion 1 \
  --calculator-hook-after-layer 1 \
  --calculator-injection-mode replace
```

Success criterion:

- Oracle replacement should reach high exact match, roughly `>=0.95`.

If oracle replacement fails:

- Do not proceed to learned Model C as the main result.
- Diagnose whether replacement removes formatting/context needed by the decoder.
- Consider a slightly softer bottleneck such as concatenating or preserving a small learned non-arithmetic context projection.

# Primary Experiment Matrix

Use the current best 2-digit candidate first:

```text
digits=2
operand range=0..99
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
steps=1000
eval_samples=512
seeds=0,1,2 for promising settings
```

Run:

| Variant | Injection mode | Purpose |
| --- | --- | --- |
| Model A | none | ordinary transformer baseline |
| Model B | replace/off | verifies bottleneck without active calculator cannot solve through hook |
| Model C oracle | replace | verifies output/injection/downstream path is usable |
| Model C learned STE | replace | tests whether learned operands can solve when calculator is required |
| Model C learned STE | add | matched comparison to current side-channel behavior |

For any promising learned Model C run, also evaluate:

- injection scale zero;
- oracle operands at eval;
- forced zero result;
- random result;
- forced result-class sweep if result vocab size is small enough, or a sampled forced-class subset for 2-digit `199`-class runs.

# Required Metrics

Report:

- answer exact match;
- final loss;
- parameter count;
- true operand exact match;
- calculator result accuracy;
- operand confidence/entropy;
- injection-zero exact match;
- oracle-at-eval exact match;
- forced-zero/random result exact match;
- residual probe accuracy at the hook read position;
- representative trace rows.

For replacement-mode runs, explicitly report whether the model can still produce syntactically valid answers and `<eos>`.

# Implementation Notes

- Preserve all existing defaults. `add` must remain the default.
- Save `calculator_injection_mode` in every config and checkpoint.
- Add tests that:
  - `add` mode preserves current behavior;
  - `replace` mode changes only `=` positions;
  - Model B/off with `replace` still produces zero injection at `=`;
  - oracle operands with `replace` force the expected result class;
  - invalid injection modes raise clean errors.
- Keep this scoped. Do not introduce a full routing framework yet.

# Interpretation Rules

## Oracle replace succeeds, learned replace succeeds

Strong evidence that the model can learn a calculator protocol when the calculator is required.

Next step:

- compare learned protocol to true operands;
- run forced result sweeps;
- try seeds and 2-digit curriculum variants.

## Oracle replace succeeds, learned replace fails

The output side is usable, but the learned input protocol is the bottleneck.

Next step:

- try oracle warmup then learned operands;
- try low-weight auxiliary operand schedules;
- try smaller operand vocab curricula such as fixed-width 2-digit formatting with `operand_max=19` or `49`;
- inspect whether STE is saturating operand logits too early.

## Oracle replace fails

The bottleneck is too severe or missing context needed by the decoder.

Next step:

- avoid interpreting learned failure;
- test a softer bottleneck that preserves non-arithmetic context while blocking arithmetic bypass.

# Acceptance Criteria

Write a work history entry answering:

- Does oracle replacement work?
- Does learned replacement work?
- Does replacement mode actually make Model B/off fail as expected?
- Is the failure mode input-protocol learning, output-side decoding, or bottleneck severity?
- Should the next protocol-learning work use additive hooks, replacement hooks, or a softer bottleneck?

