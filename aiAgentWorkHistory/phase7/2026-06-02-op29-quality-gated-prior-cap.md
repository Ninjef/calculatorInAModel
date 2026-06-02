# 2026-06-02 Op29 Quality-Gated Prior Cap

## Question

Can a quality-gated prior update cap freeze the proportional op29 amortized
prior early enough to materially cut cost while preserving heldout source and
trusted additive handoff?

## Implementation

Added `--result-boundary-target-amortized-prior-quality-gate-update-cap`.

When positive, prior fitting stops after prompt memory is full once:

- total prior updates are at or above the cap;
- the configured stop metric meets
  `--result-boundary-target-amortized-prior-stop-train-accuracy`;
- the configured train requirement
  `--result-boundary-target-amortized-prior-stop-require-train-accuracy` is
  met.

This is an explicit cap/freeze rule, not a patience ladder.

## Source

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_qcap2000_dualstop_val90_train98_pat100_src5000/2026-06-02_143237_450578_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-b3b0395c67/model-c-2digit-seed11
```

Recipe:

- op29 h128 numeric prior, four routed hooks, shared output projection.
- Prompt-keyed online hard memory, freeze when full.
- `1500` full-memory refresh updates after memory fill.
- Target-stratified proportional replay with fraction `0.5`.
- Eval-only validation split `0.2`.
- Dual quality gate: validation `>=0.9`, train prior `>=0.98`.
- Quality-gated update cap `2000`.

Results:

- Overall exact/calc `896/900 = 0.9956`.
- Train exact/calc `1.0000`.
- Heldout exact/calc `173/180 = 0.9611`.
- Heldout controls: injection-zero `0.0222`, forced-zero `0.0000`,
  forced-random `0.0056`.
- Diagnostic exact/calc `0.9922`/`0.9922`.
- Routed diagnostic hook calc: hook0 `0.9730`, hook1 `1.0000`, hook2
  `1.0000`, hook3 `1.0000`.
- Prior froze at `2000` updates.
- Prior fit examples `1,254,817`.
- Full-fit examples `1,080,000`.
- Final prior train/validation `0.9861`/`0.9927`.

Cap curve:

- Step `1900`: `1701` updates, `1,147,177` fit examples, train/validation
  prior `0.9625`/`0.9562`.
- Step `2000`: `1751` updates, `1,165,177` fit examples, train/validation
  prior `0.9792`/`0.9927`; train requirement narrowly not met.
- Step `2200`: `1851` updates, `1,201,177` fit examples, train/validation
  prior `0.9806`/`0.9854`; quality met but cap not reached.
- Step `2500`: prior frozen at `2000` updates and `1,254,817` examples, with
  train/validation prior `0.9861`/`0.9927`.
- Step `5000`: unchanged prior cost and accuracy; source still passed.

## Trusted Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_qcap2000_dualstop_val90_train98_pat100_handoff600/2026-06-02_143648_261428_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Results:

- Final eval `900/900 = 1.0000`.
- Diagnostic exact/calc `1.0000`/`0.9922`.
- Final snapshot controls: injection-zero `0.0000`, forced-zero `0.0000`,
  forced-random `0.0078`, oracle-at-eval `1.0000`.
- Final snapshot learned calc `0.9844`.
- Routed diagnostic hook calc: hook0 `0.9730`, hook1 `1.0000`, hook2
  `1.0000`, hook3 `1.0000`.

## Interpretation

Positive-with-caveat. A single quality-gated cap materially improves the op29
prior-fit cost story while preserving both the source heldout gate and trusted
non-bottleneck handoff:

- updates: `3251 -> 2000` versus the uncapped proportional run;
- fit examples: `1.705M -> 1.255M`;
- heldout: `0.9556 -> 0.9611`;
- handoff: remains `1.0000`.

This is now the op29 numeric-prior cost lead, but it is not the final thesis:
it is one effective seed, still uses sparse answer-derived candidate scoring,
and still relies on staged frozen-policy handoff.

Do not run cap-value, proportional-fraction, or refresh-window ladders as
novelty. The next useful work should validate robustness on a fresh seed or
many-calculator cost axis, or replace the answer-derived scoring mechanism.
