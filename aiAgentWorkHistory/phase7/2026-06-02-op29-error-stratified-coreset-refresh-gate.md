# Op29 Error-Stratified Coreset Refresh Gate

## Question

Can a shorter op29 full-refresh window plus error-focused coreset replay
preserve the source and trusted handoff gates while cutting prior updates?

## Implementation

Added `error_stratified` as a
`--result-boundary-target-amortized-prior-fit-sampling-mode`.

For non-full-fit prior updates, the mode:

- computes current prior predictions over the fit memory;
- selects currently misclassified memory entries target-stratified;
- fills any remaining batch slots target-stratified from the rest of memory.

This was intended as staged full refresh plus coreset replay, not a stop
threshold ladder.

Validation:

```text
python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
python3 scripts/overfit_one_batch.py --help | rg "error_stratified|stop-require-train"
```

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_errorstrat_fit160_val20_evalonly_fullrefresh1500_dualstop_val90_train98_pat100_src5000/2026-06-02_135902_100006_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-8bc45ae128/model-c-2digit-seed11
```

Setup:

- `operand_max=29`, deterministic `20%` heldout split;
- four `left_operand_mod` routed hooks with shared output projection;
- h128 numeric amortized prior;
- full-refresh budget reduced to `1500`;
- fit batch `160`, fit mode `error_stratified`;
- eval-only validation `0.2`;
- dual stop guard: validation `>=0.9`, train prior `>=0.98`, patience `100`;
- no additive semantic distillation.

Results:

- overall exact/calc `893/900 = 0.9922`;
- train exact/calc `1.0000`;
- heldout exact/calc `172/180 = 0.9556`;
- prior train/heldout `0.8806`/`0.8778`;
- heldout controls: injection-zero `0.0222`, forced-zero `0.0000`,
  forced-random `0.0056`;
- prior updates `3251`;
- forced-result evals `302,592`;
- prompt memory entries `720/720`.

Curve:

| Step | Prior updates | Refresh active | Train prior | Validation prior | Error fraction | Error selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 1403 | 1 | 0.9444 | 0.9781 | n/a | n/a |
| 1700 | 1601 | 0 | 0.6153 | 0.6350 | 0.3583 | 1.0000 |
| 1900 | 1701 | 0 | 0.7750 | 0.7810 | 0.2875 | 1.0000 |
| 2500 | 2001 | 0 | 0.8792 | 0.8832 | 0.1403 | 0.6313 |
| 3500 | 2501 | 0 | 0.8694 | 0.8905 | 0.1181 | 0.5313 |
| 5000 | 3251 | 0 | 0.8806 | 0.8978 | 0.1167 | 0.5250 |

The dual stop gate never fired.

## Trusted Additive Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_errorstrat_fit160_val20_evalonly_fullrefresh1500_dualstop_val90_train98_pat100_handoff600/2026-06-02_140337_366092_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Results:

- final eval `900/900 = 1.0000`;
- diagnostic exact/calc `1.0000`/`0.9844`;
- final 128-sample controls: injection-zero `0.0078`, forced-zero `0.0000`,
  forced-random `0.0391`, oracle-at-eval `1.0000`;
- routed diagnostic hook calculator-result accuracies: hook0 `0.9730`,
  hook1 `0.9778`, hook2 `1.0000`, hook3 `1.0000`.

## Interpretation

Mixed-negative for the cost-reduction hypothesis.

Error-stratified coreset replay preserves the source and handoff gates, so the
calculator policy itself stayed usable. But it fails the intended scalability
target: the prior never reached the stop gate, final prior memory accuracy was
weak, and prior updates increased to `3251`, above the full-refresh positive
(`2755`) and dual-guard positive (`2570`).

Do not run error-stratified batch-size, refresh-window, or threshold ladders as
novelty. The next cost mechanism should preserve global prior coverage more
explicitly: coverage-aware/proportional refresh with update caps, staged
refresh plus stable coreset distillation, or many-calculator cost accounting.
