# 2026-06-02 Op29 Proportional Half-Memory Refresh Gate

## Question

Can explicit prior-fit example accounting plus proportional half-memory replay
reduce the op29 post-fill refresh cost while preserving heldout source and
trusted frozen-policy additive handoff?

## Implementation

Added `--result-boundary-target-amortized-prior-fit-batch-fraction`. When set,
non-refresh prior fits use `ceil(fraction * current fit-memory entries)`.
Forced full-refresh updates still fit all entries.

Added cumulative prior fit-cost metrics:

- `result_boundary_target_amortized_prior_fit_examples`
- `result_boundary_target_amortized_prior_full_fit_examples`
- `result_boundary_target_amortized_prior_fit_effective_batch_size`

## Source

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_dualstop_val90_train98_pat100_src5000/2026-06-02_141801_001856_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-33b054986c/model-c-2digit-seed11
```

Recipe:

- op29 h128 numeric prior, four routed hooks, shared output projection.
- Prompt-keyed online hard memory, freeze when full.
- `1500` full-memory refresh updates after memory fill.
- Target-stratified proportional replay with fraction `0.5`.
- Eval-only validation split `0.2`.
- Dual stop guard: validation `>=0.9`, train prior `>=0.98`, patience `100`.

Results:

- Overall exact/calc `894/900 = 0.9933`.
- Train exact/calc `1.0000`.
- Heldout exact/calc `172/180 = 0.9556`.
- Heldout controls: injection-zero `0.0222`, forced-zero `0.0000`,
  forced-random `0.0056`.
- Diagnostic exact/calc `0.9922`/`0.9922`.
- Routed diagnostic hook calc: hook0 `0.9730`, hook1 `1.0000`, hook2
  `1.0000`, hook3 `1.0000`.
- Prior updates `3251`; stop did not fire.
- Prior fit examples `1,705,177`.
- Full-fit examples `1,080,000`.
- Final prior train/validation `0.9583`/`0.9635`.

Curve:

- Step `1500`: `1399` updates, `965,377` fit examples, `933,840` full-fit
  examples, train/validation prior `0.9611`/`0.9562`, effective batch `720`.
- Step `1700`: `1599` updates, `1,109,377` fit examples, `1,077,840`
  full-fit examples, train/validation prior `0.9542`/`0.9562`.
- Step `2000`: `1751` updates, `1,165,177` fit examples, full-fit examples
  capped at `1,080,000`, train/validation prior `0.9792`/`0.9927`,
  effective batch `360`.
- Step `3000`: `2251` updates, `1,345,177` fit examples, train/validation
  prior `0.9875`/`1.0000`.
- Step `4000`: `2751` updates, `1,525,177` fit examples, train/validation
  prior `0.9722`/`0.9781`.
- Step `5000`: `3251` updates, `1,705,177` fit examples, train/validation
  prior `0.9583`/`0.9635`, no stop.

## Trusted Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_dualstop_val90_train98_pat100_handoff600/2026-06-02_142244_295322_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Results:

- Final eval `900/900 = 1.0000`.
- Diagnostic exact/calc `1.0000`/`0.9922`.
- Final snapshot controls: injection-zero `0.0000`, forced-zero `0.0000`,
  forced-random `0.0156`, oracle-at-eval `1.0000`.
- Routed diagnostic hook calc: hook0 `0.9730`, hook1 `1.0000`, hook2
  `1.0000`, hook3 `1.0000`.

## Interpretation

Mixed-positive. Proportional half-memory replay preserves source and handoff,
and the new metrics make prior-fit example cost visible. But the update count
is still high (`3251`), the stop gate never fired, and example cost reduction
is modest: `1.705M` fit examples versus the `1.8M` lower bound implied by
`2500` full-memory refresh updates alone.

Do not run proportional-fraction or refresh-window ladders as novelty. The
next useful cost mechanism should add an explicit update cap/freeze after a
validated proportional phase, distill stable coreset coverage into the prior,
or move to many-calculator cost accounting/new credit assignment.
