# Strict Calculator-Required Bottleneck

## Mission

Implement and validate a stricter calculator-required setting where arithmetic success must flow through the calculator result, not through ordinary autoregressive residual context or answer-token bypasses.

The recent tracks answered the previous charter: operand-read sites help the interface and oracle use works, but answer-loss training still does not learn the intended operand protocol. The current `replace` mode is not enough because Model B/off still solves the task at high accuracy. The next task is to make the bottleneck real before comparing estimator families.

## Starting Point

Use the established small regime unless a narrower smoke test is needed first:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
```

Keep Model A/B/C/oracle controls and the Track 3/Track 4 diagnostic vocabulary intact.

## Required Design

Pick one strict bottleneck design and implement it cleanly. Good candidates:

- A two-phase encoder/decoder mode where the prompt encoder can compute calculator operands, but answer decoding receives only calculator result plus minimal formatting state.
- A masked or compressed post-`=` context mode that prevents answer-token positions from attending to or reconstructing operand identities through the normal residual stream.
- A small explicit answer decoder that consumes the projected calculator result and answer-position metadata, with no direct residual access to operand tokens.

Avoid treating the existing `calculator_injection_mode=replace` as sufficient. Track 2/3/4 already showed it is leaky.

## Validation Gate

Before any estimator sweep, the new setting must pass:

- Model B/off cannot solve arithmetic above the appropriate no-calculator control for this constrained setting.
- Oracle Model C reaches high exact match.
- Oracle Model C collapses under injection-zero, forced-zero, or forced-random calculator output.
- Track 3 diagnostics classify the setting as calculator-required rather than `non_bottleneck_leaky_by_design` or `invalid_or_leaky_bottleneck`.
- Track 4 action-loss diagnostics show true operand actions reduce target loss relative to random and shuffled actions.

## First Experiment

1. Implement the strict bottleneck behind an explicit config/CLI flag; preserve existing add/replace behavior.
2. Add focused tests proving the bottleneck blocks normal operand bypass while preserving oracle calculator use.
3. Run a tiny smoke matrix with Model B/off and oracle Model C.
4. Run Track 3 diagnostics on the bottleneck Model B/off and oracle checkpoints.
5. Run Track 4 action-loss diagnostics on the same checkpoints.
6. Only if those gates pass, run learned STE in the bottleneck setting.

## Required Reporting

Write a work history with:

- exact config and run paths;
- Model A/B/C/oracle exact match;
- true operand exact match and calculator result accuracy;
- injection-zero, forced-zero, forced-random, oracle-at-eval, and force-learned-result exact match;
- Track 3 classification and bottleneck label;
- Track 4 action-loss gaps for true vs learned/random/shuffled actions;
- a clear go/no-go recommendation for estimator comparisons.

## Success Criteria

A useful positive result is not high answer accuracy by itself. It is:

- Model B/off fails in the constrained setting;
- oracle succeeds and is calculator-dependent;
- true operand actions have a strong downstream loss advantage;
- learned STE has a fair, non-leaky bottleneck baseline to test against.

A useful negative result is:

- oracle cannot succeed, meaning the bottleneck removed too much needed formatting/context; or
- oracle succeeds but true-vs-random action-loss gaps are weak, meaning the downstream decoder still does not expose useful training signal for estimators.

Either outcome should sharpen the next step more than another non-bottleneck estimator sweep.
