# Op29 Dual-Guard Refresh Stop Cost Gate

## Question

Can a full-refresh stop rule require both validation quality and train-memory
prior coverage, avoiding the validation-only early-stop failure while reducing
the op29 full-refresh cost?

## Implementation

Added `--result-boundary-target-amortized-prior-stop-require-train-accuracy`.

When positive, a prior stop update counts only if:

- the configured stop metric passes
  `--result-boundary-target-amortized-prior-stop-train-accuracy`; and
- train-memory prior accuracy is at least the new required threshold.

This run uses the new guard while allowing stop during a post-memory-fill
full-refresh window:

- stop metric: validation accuracy;
- validation threshold: `0.9`;
- train-memory prior requirement: `0.98`;
- patience: `100`.

Validation:

```text
python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
```

One launch with `n_embd=16` failed before training because the op29 product
decoder checkpoint is `n_embd=32`; it is not counted as an experiment result.

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_fullrefresh2500_dualstop_val90_train98_pat100_src5000/2026-06-02_134640_075802_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-d4e7f05a3b/model-c-2digit-seed11
```

Setup:

- `operand_max=29`, `900` prompts, deterministic `20%` heldout;
- four `left_operand_mod` routed hooks with shared output projection;
- product semantic decoder, `operand_spans` readout;
- prompt-keyed online hard memory, topk8+unique24 sparse zero-improvement
  result-boundary scoring, freeze memory when full;
- numeric amortized prior h128, target-stratified fit batch `160`, eval-only
  validation fraction `0.2`, prior LR `0.01`;
- full-refresh budget after memory full: `2500`;
- `answer_loss_weight=1.0`;
- no additive semantic distillation.

Results:

- overall exact/calc `896/900 = 0.9956`;
- train exact/calc `1.0000`;
- heldout exact/calc `174/180 = 0.9667`;
- prior train/heldout `0.9972`/`0.9667`;
- heldout controls: injection-zero `0.0222`, forced-zero `0.0000`,
  forced-random `0.0056`;
- prior updates `2570`;
- forced-result evals `278,016`;
- prompt memory entries `720/720`.

Curve:

| Step | Prior updates | Refresh active | Refresh remaining | Converged steps | Train prior | Validation prior | Train req |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1500 | 1411 | 1 | 1179 | 0 | 0.9583 | 0.9708 | 0 |
| 1900 | 1811 | 1 | 779 | 0 | 0.9764 | 0.9854 | 0 |
| 2100 | 2011 | 1 | 579 | 14 | 0.9958 | 1.0000 | 1 |
| 2500 | 2411 | 1 | 179 | 28 | 0.9861 | 1.0000 | 1 |
| 2700 | 2570 | 0 | 0 | 100 | 0.9972 | 1.0000 | stopped |
| 5000 | 2570 | 0 | 0 | 100 | 0.9972 | 1.0000 | stopped |

## Trusted Additive Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_fullrefresh2500_dualstop_val90_train98_pat100_handoff600/2026-06-02_135116_516188_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed11
```

Setup:

- Loaded source `final_weights.pt` with
  `--semantic-decoder-checkpoint-load-scope compatible_model`.
- Switched to additive non-bottleneck mode:
  `--calculator-estimator ste`, `--calculator-bottleneck-mode none`,
  `--calculator-injection-mode add`.
- Froze the calculator policy and semantic decoder.
- Ran the trusted 600-step full-grid handoff gate.

Results:

- final eval `900/900 = 1.0000`;
- diagnostic exact/calc `1.0000`/`0.9844`;
- final 128-sample controls: injection-zero `0.0000`, forced-zero `0.0078`,
  forced-random `0.0234`, oracle-at-eval `1.0000`;
- routed diagnostic hook calculator-result accuracies: hook0 `0.9730`,
  hook1 `1.0000`, hook2 `1.0000`, hook3 `0.9412`.

## Interpretation

Positive-with-caveat.

The dual guard fixes the validation-only failure mode: it prevents stopping
while train-memory prior accuracy is still around `0.95`, and it preserves both
the heldout source gate and trusted non-bottleneck handoff.

The cost saving is modest, not transformative. Prior updates fell from `2755`
in the earlier full-refresh positive to `2570` here, a reduction of `185`
updates. Forced-result evals were also lower (`278,016` versus `294,912`), but
that came from this run's memory-fill dynamics rather than the stop guard
alone.

Do not run train-requirement threshold or patience ladders. The next useful
experiment should make a larger cost-structure change: staged full refresh then
coreset replay, coverage-aware/proportional refresh, or explicit
many-calculator cost accounting.
