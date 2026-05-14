# Multi-Sample Result-Space Policy-Gradient Gate

## Task

```text
aiAgentProjectTasks/2026-05-14-phase-7-eighth-task-Multi-sample-result-space-policy-gradient-gate.md
```

## Claim Tested

Can multi-sample score-function training over natural result requests discover
a hard calculator-result protocol from sampled answer loss, using per-prompt or
leave-one-out baselines to reduce variance?

This is explicitly not the Phase 1 REINFORCE repeat. Phase 1 used independent
A/B operand sampling, one sample per prompt, and a moving scalar baseline. This
gate used the Phase 7 natural result-space action (`0..38`), exact-grid
`0..19 x 0..19` coverage, `K=16` samples per prompt, and a PG-vs-boundary
gradient-agreement diagnostic against the supervised boundary-target ceiling.

## Code Changes

- `src/model.py`
  - Allowed `calculator_action_head=result_space` with
    `calculator_estimator=reinforce`.
  - Sampled result classes from `calculator_hook.result_proj` logits in
    result-space REINFORCE mode.
  - Mapped sampled result classes to the deterministic canonical valid
    calculator pair.
  - Added `result_logp` trace plumbing.
  - Made `sampled_logp` equal the sampled result log-probability for
    result-space policies instead of synthetic A/B canonical log-probs.

- `scripts/overfit_one_batch.py`
  - Added `--reinforce-baseline-mode` with `global_ema`, `per_prompt_mean`,
    and `leave_one_out`.
  - Added `--reinforce-num-samples-per-prompt`.
  - Added a K-sample REINFORCE loss path for sampled calculator actions.
  - Added `--reinforce-gradient-diagnostic-only`, which reports fixed-batch
    estimator metrics and PG-vs-boundary gradient cosine before any long run.
  - Routed result-space REINFORCE through the adaptive optimizer groups so
    result-head and upstream learning rates can be controlled separately.

- `tests/test_model.py`
  - Added a result-space REINFORCE trace test confirming `sampled_logp` is the
    sampled result log-probability.

## Validation Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
92 passed
```

## Stage 0 Command

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator reinforce --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --exhaustive-grid-batch --answer-loss-weight 0.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 0.0 --result-boundary-target-loss-weight 0.0 --input-proj-anchor-weight 0.0 --reinforce-baseline-mode leave_one_out --reinforce-num-samples-per-prompt 16 --reinforce-entropy-weight 0.0 --result-boundary-target-mode hard_best_result --result-boundary-target-temperature 1.0 --result-boundary-target-chunk-size 64 --input-proj-lr 0.01 --upstream-lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --steps 0 --batch-size 400 --eval-samples 64 --seed 2 --run-root runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate/stage0_gradient_gate --reinforce-gradient-diagnostic-only
```

## Run Paths

Run root:

```text
runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate
```

Stage 0 diagnostic:

```text
runs/2026-05-14_phase7_multisample_result_space_policy_gradient_gate/stage0_gradient_gate/2026-05-14_075911_897578_model-c-op0-19-fullgrid-reinforce-result_space-K16-leave_one_out-inlr0.01-uplr0.0003-graddiag-answer_decoder-adec-product/model-c-2digit-seed4/reinforce_gradient_diagnostic_summary.json
```

Selected checkpoint paths:

```text
none; Stage 0 alignment failed, so Stage 1 and Stage 2 were not run.
```

## Fast-Gate Table

| Gate | Value | Pass? |
| --- | ---: | --- |
| exact-grid prompts | `400` | yes |
| K samples per prompt | `16` | yes |
| PG result-proj grad L2 > 0 | `0.1551` | yes |
| PG upstream grad L2 > 0 | `0.0574` | yes |
| semantic decoder grad L2 == 0.0 | `0.0` | yes |
| result-proj PG-vs-boundary cosine > 0.0 | `-0.0945` | no |
| per-prompt/LOO advantage std < global EMA | `3.1542 / 3.4382 < 3.7048` | yes |

## Full Diagnostic Table

| Metric | Value |
| --- | ---: |
| answer loss | `7.9035` |
| policy-gradient objective | `0.001336` |
| policy loss | `0.001336` |
| advantage mean | `0.000000006` |
| advantage std | `3.4309` |
| sampled logp | `-3.6635` |
| result entropy | `3.6636` |
| sampled result accuracy | `0.0278` |
| PG result-proj grad L2 | `0.1551` |
| PG upstream grad L2 | `0.0574` |
| PG semantic decoder grad L2 | `0.0` |
| boundary result-proj grad L2 | `0.0897` |
| boundary upstream grad L2 | `0.0332` |
| boundary semantic decoder grad L2 | `0.0` |
| result-proj PG-vs-boundary cosine | `-0.0945` |
| upstream PG-vs-boundary cosine | `-0.1108` |
| result-proj PG/boundary relative norm | `1.7300` |
| upstream PG/boundary relative norm | `1.7289` |
| global EMA advantage std | `3.7048` |
| per-prompt mean advantage std | `3.1542` |
| leave-one-out advantage std | `3.4382` |
| boundary hard-best equals true sum | `1.0000` |
| boundary learned-best fraction | `0.0225` |
| boundary true result probability | `0.8003` |

## Final Objective Weights

Stage 0 diagnostic only:

| Objective | Weight |
| --- | ---: |
| answer loss | `0.0` |
| aux operand loss | `0.0` |
| adaptive interface loss | `0.0` |
| expected answer loss | `0.0` |
| result boundary target loss | `0.0` |
| input projection anchor | `0.0` |
| REINFORCE entropy bonus | `0.0` |

The boundary-target gradient was computed only for diagnostic cosine
comparison. It was not used for a model update.

## Parameter Movement Summary

No optimizer step was taken in Stage 0 diagnostic-only mode, so parameter delta
is `0.0` for all groups. Gradient summary:

| Group | PG grad L2 | Boundary grad L2 |
| --- | ---: | ---: |
| `calculator_hook.result_proj` | `0.1551` | `0.0897` |
| upstream | `0.0574` | `0.0332` |
| semantic decoder | `0.0` | `0.0` |

## Comparison To Prior Work

Compared to Phase 1 single-sample REINFORCE, this task used the natural Phase 7
result action space instead of independent A/B operand heads, exact-grid
coverage instead of sampled mini-batches, and multi-sample per-prompt
advantages instead of only a moving scalar baseline. The variance-control
piece behaved as hoped: both per-prompt mean and leave-one-out reduced
advantage standard deviation versus global EMA on the fixed grid.

Compared to the Phase 7 boundary-target ceiling, the signal is not currently
usable as a vanilla estimator. The boundary target still identifies the true
sum on the exact grid, but the sampled policy-gradient estimate points against
that result-head gradient at initialization (`cosine=-0.0945`).

## Decision

```text
multisample_result_space_policy_gradient_stage0_alignment_negative
```

Stage 1 and Stage 2 were intentionally skipped because the task instructed to
stop if the PG-vs-boundary cosine was negative or near zero. The recommended
next move is not a long vanilla result-space PG training run. Improve the
estimator family first, for example with actor-critic/NVIL-style learned
baselines that can be checked by the same gradient-agreement gate, or move to
surrogate/shadow calculator gradients, synthetic gradients/direct feedback
alignment, or a stricter decoder-phase bottleneck.
