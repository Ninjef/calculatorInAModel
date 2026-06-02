# Op29 Full-Refresh Stop-During-Refresh Gate

## Question

Can the existing eval-only validation stop end the op29 post-memory-fill
full-refresh prior window early while preserving the heldout source gate?

## Implementation

Added `--result-boundary-target-amortized-prior-full-refresh-allow-stop` in
`scripts/overfit_one_batch.py`.

Behavior:

- default remains unchanged;
- when enabled, the configured amortized-prior stop rule can operate during a
  post-memory-fill full-refresh window;
- if the stop rule fires during refresh, remaining refresh updates are cleared
  and the prior fit stays stopped;
- metrics and run suffix record the flag.

Validation:

```text
python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
```

## Exact-Matched Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_fullrefresh2500_stopduringrefresh_stopval90pat100_exactmatch_src5000/2026-06-02_132231_313409_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-44df8b2ce8/model-c-2digit-seed9
```

This run matches the positive op29 full-refresh source config except for the
new allow-stop flag:

- `operand_max=29`, `900` prompts, deterministic `20%` heldout;
- four `left_operand_mod` routed hooks with shared output projection;
- product semantic decoder, `operand_spans` readout;
- prompt-keyed online hard memory, topk8+unique24 sparse zero-improvement
  result-boundary scoring, freeze memory when full;
- numeric amortized prior h128, target-stratified fit batch `160`, eval-only
  validation fraction `0.2`, prior LR `0.01`;
- validation stop threshold `0.9`, patience `100`;
- full-refresh budget after memory full: `2500`;
- `answer_loss_weight=1.0`;
- no additive semantic distillation.

## Results

Source:

- overall exact/calc `858/900 = 0.9533`;
- train exact/calc `0.9889`;
- heldout exact/calc `147/180 = 0.8167`;
- prior train/heldout `0.9514`/`0.8167`;
- heldout controls: injection-zero `0.0278`, forced-zero `0.0000`,
  forced-random `0.0111`;
- prior updates `1140`;
- full-refresh budget configured `2500`;
- forced-result evals `342,528`;
- prompt memory entries `720/720`.

Training curve:

| Step | Prior updates | Refresh active | Refresh remaining | Converged steps | Train prior | Validation prior | Fit stopped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 790 | 1 | 1821 | 1 | 0.9153 | 0.9097 | 0 |
| 1100 | 990 | 1 | 1621 | 24 | 0.9403 | 0.9375 | 0 |
| 1200 | 1090 | 1 | 1521 | 50 | 0.9611 | 0.9514 | 0 |
| 1300 | 1140 | 0 | 0 | 100 | 0.9514 | 0.9583 | 1 |
| 5000 | 1140 | 0 | 0 | 100 | 0.9514 | 0.9583 | 1 |

Routed diagnostic summary:

- hook0 calculator-result accuracy `0.9756`;
- hook1 calculator-result accuracy `1.0000`;
- hook2 calculator-result accuracy `0.9412`;
- hook3 calculator-result accuracy `0.9130`.

No trusted frozen-policy additive handoff was run because heldout source/calc
missed the gate.

## Interpretation

Mixed-negative.

The early-stop rule materially reduced prior updates: `1140` instead of
`2755`, about `59%` fewer than the positive full-refresh run. But it also
dropped heldout exact/calc from `0.9167` to `0.8167`, below the source gate.
Train prompts remained strong and controls stayed low, so the failure is a
prior heldout-generalization miss rather than calculator wiring failure.

Do not rerun this as a validation threshold or patience ladder. A useful next
experiment needs a stronger cost-reduction mechanism that preserves coverage,
such as staged refresh then coreset replay, coverage-aware/proportional refresh,
or a dual train+validation/high-confidence stop/freeze transition with explicit
source, handoff, and cost gates.
