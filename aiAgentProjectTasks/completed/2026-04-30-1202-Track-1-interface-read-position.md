# Track 1: Interface Scientist - Operand Read Position

## Mission

Test the most important current hypothesis:

> The calculator may be failing because it reads both operands from the `=` residual, while probes show operand information is more naturally available at operand-token positions.

This track owns the mainline implementation and first experimental pass for a probe-guided calculator interface.

## Tactical Work

Implement a narrow interface variant:

- Preserve current behavior as `calculator_read_position=eq`.
- Add `calculator_read_position=operands`.
- In `operands` mode:
  - read A logits from the final A digit position;
  - read B logits from the final B digit position;
  - keep calculator result injection at the `=` position.
- Thread the setting through `GPTConfig`, training scripts, diagnostic scripts, saved config, and tests.
- Keep existing Model A/B/C/oracle semantics unchanged.

The implementation should be minimal. Do not introduce broader routing, multiple read heads, learned read positions, or natural language tasks in this track.

## Experiments

Start with the existing recommended 2-digit regime:

```text
digits=2
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
```

Run the ladder:

- `operand_max=19`, `calculator_operand_vocab_size=20`
- if promising, `operand_max=49`, `calculator_operand_vocab_size=50`
- if still promising, `operand_max=99`, `calculator_operand_vocab_size=100`

For each rung, run:

- Model A
- Model B
- Model C learned operands
- Model C oracle operands

Start with one seed. Add 3-seed repeats only after `operand_max=19` shows a credible calculator-use signature.

## Required Metrics

For every serious run, report:

- answer exact match;
- true operand exact match at calculator read sites;
- calculator result accuracy;
- injection-zero exact match;
- forced-zero exact match;
- forced-random exact match;
- oracle-at-eval exact match;
- probe accuracy at A token, B token, and `=`;
- exact run path and seed.

## Success Criteria

A strong positive result:

- Oracle C remains near-perfect.
- Model B remains a healthy control.
- Model C learns a true or causally decodable operand/result protocol.
- Corrupting or removing calculator injection hurts answer accuracy.

A strong negative result:

- Oracle C works.
- Probes show operands are available at read sites.
- Model B is healthy.
- Model C still fails to learn a protocol under answer loss.

Either outcome is publishable-quality information for this sandbox.

## Guardrails

- Do not judge success by answer accuracy alone.
- Do not add auxiliary supervision in this track except as an explicitly labeled diagnostic after the pure learned-operand run.
- Do not scale to larger models before the `operand_max=19` result is understood.
- Preserve backwards compatibility with existing `eq` read behavior.
