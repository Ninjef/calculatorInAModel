# 2026-06-02 Op29 Quality-Gated Prior Cap Fresh Seed

## Question

Does the op29 quality-gated `2000`-update cap preserve the source and trusted
frozen-policy additive handoff gates on a fresh source seed?

This is a robustness check inside the integrated numeric-prior replay family,
not a new training algorithm.

## Runs

Source:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_qcap2000_dualstop_val90_train98_pat100_seed31_src5000/2026-06-02_144248_012237_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-b3b0395c67/model-c-2digit-seed33
```

Trusted frozen-policy additive handoff:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fitfrac50_targetstrat_val20_evalonly_fullrefresh1500_qcap2000_dualstop_val90_train98_pat100_seed31_handoff600/2026-06-02_144743_273571_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed33
```

## Setup

- Same capped op29 h128 proportional recipe as
  `2026-06-02-op29-quality-gated-prior-cap.md`.
- CLI seed `31`, effective model seed `33`.
- Streaming source with 20% heldout prompts.
- Target-stratified proportional prior replay fraction `0.5`.
- Post-memory-fill full refresh budget `1500`.
- Eval-only validation `0.2`.
- Quality gate: validation `>=0.9`, train prior `>=0.98`, update cap `2000`.

## Source Result

| Metric | Value |
| --- | ---: |
| Overall exact/calc | `885/900 = 0.9833` |
| Train exact/calc | `1.0000` / `1.0000` |
| Heldout exact/calc | `164/180 = 0.9111` |
| Heldout injection-zero | `0.0333` |
| Heldout forced-zero | `0.0000` |
| Heldout forced-random | `0.0167` |
| Diagnostic exact/calc | `0.9844` / `0.9844` |
| Prior updates | `2017` |
| Prior fit examples | `1,260,852` |
| Full-memory fit examples | `1,080,000` |
| Prior train/validation | `0.9889` / `0.9933` |
| Cap reached / quality met | `1.0` / `1.0` |

Routed diagnostic hook calculator-result accuracies:

```text
hook0 0.9767
hook1 1.0000
hook2 1.0000
hook3 0.9565
```

## Handoff Result

| Metric | Value |
| --- | ---: |
| Final eval | `900/900 = 1.0000` |
| Diagnostic exact/calc | `1.0000` / `0.9844` |
| Final snapshot normal | `1.0000` |
| Final snapshot learned calc | `1.0000` |
| Final injection-zero | `0.0000` |
| Final forced-zero | `0.0000` |
| Final forced-random | `0.0078` |
| Final oracle-at-eval | `1.0000` |

Routed diagnostic hook calculator-result accuracies:

```text
hook0 0.9767
hook1 1.0000
hook2 1.0000
hook3 0.9565
```

## Interpretation

Result:

```text
op29_quality_gated_prior_cap_fresh_seed_handoff_positive_with_source_variance
```

The capped recipe replicated the trusted non-bottleneck handoff on a fresh
source seed. The source heldout score is lower than the original capped run
(`0.9111` versus `0.9611`), but it remains above the op29 heldout gate and the
handoff reaches `1.0000` with low controls.

This strengthens the quality-gated cap as the current op29 numeric-prior cost
lead. It does not solve the thesis: the method still uses answer-derived sparse
candidate scoring, amortized-prior replay, and staged frozen-policy handoff.

Next work should move to explicit many-calculator cost accounting for this
recipe or a less-prescriptive/non-enumerative credit mechanism. Do not spend
mainline compute on cap-value, proportional-fraction, refresh-window, or
same-recipe seed ladders as novelty.
