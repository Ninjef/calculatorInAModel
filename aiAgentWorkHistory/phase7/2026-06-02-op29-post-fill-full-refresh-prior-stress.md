# 2026-06-02 - Op29 post-memory-fill full-refresh prior stress

Question: can changing online prior fit dynamics, rather than h128 capacity or
another constant-batch repeat, repair the op29 heldout source miss and preserve
trusted non-bottleneck handoff?

## Code Change

Added:

```text
--result-boundary-target-amortized-prior-full-refresh-after-memory-full-updates
```

When prompt hard memory first becomes full, the amortized prior can now run a
fixed number of forced full-memory fit updates before returning to the
configured fit batch and cadence. Convergence stopping is suppressed while the
full-refresh window is active, and training metrics record whether refresh is
active plus remaining refresh updates.

Validation:

```text
python3 -m py_compile scripts/overfit_one_batch.py scripts/diagnose_amortized_prior_from_trace.py
```

A direct function smoke simulated a full prompt memory and verified three
forced full-fit updates consumed the refresh counter:

```text
[(1.0, 2, 1), (1.0, 1, 2), (1.0, 0, 3)]
```

## Source Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_fullrefresh2500_stopval90pat100_src5000/2026-06-02_121235_381742_model-c-op0-29-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-16aa3b10c8/model-c-2digit-seed9
```

Key setup:

- `operand_max=29`, `900` total prompts, deterministic `20%` heldout.
- Four `left_operand_mod` routed hooks, shared output projection semantics,
  product answer decoder, `operand_spans` readout.
- Prompt-keyed online hard memory with topk8+unique24 sparse result-boundary
  scoring, freeze memory when full.
- Numeric amortized prior h128, target-stratified fit batch `160`,
  eval-only validation fraction `0.2`, validation stop threshold `0.9`,
  patience `100`.
- New full-refresh window:
  `--result-boundary-target-amortized-prior-full-refresh-after-memory-full-updates 2500`.

Source results:

- Overall exact/calc `884/900 = 0.9822`.
- Train exact/calc `0.9972`.
- Heldout exact/calc `155/180 = 0.9167`.
- Prior train/heldout accuracy `0.9958` / `0.9167`.
- Heldout controls: injection-zero `0.0278`, forced-zero `0.0000`,
  forced-random `0.0111`.
- Prior updates `2755`.
- Full-refresh updates configured: `2500`.
- Forced-result evals `294,912`.
- Prompt memory entries `720/720`.

Curve notes:

- Prompt memory was full by step `200`; full refresh was active with `2490`
  updates remaining at that snapshot.
- Refresh was nearly exhausted by step `2650`, with train/validation prior
  accuracy at `1.0`.
- Immediately after refresh ended, sampled continuation briefly dipped
  (`~0.76-0.81` train/validation prior around steps `2700-2750`), then
  recovered by final to `0.9958` train / `1.0` validation prior and stopped.
  This suggests the refresh-to-cheap-replay transition is itself a fit-dynamics
  object worth improving.

## Trusted Additive Handoff

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_op29_prior_h128_fit160_targetstrat_val20_evalonly_fullrefresh2500_stopval90pat100_handoff600/2026-06-02_124110_461471_model-c-op0-29-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Handoff setup:

- Loaded the source `final_weights.pt` with
  `--semantic-decoder-checkpoint-load-scope compatible_model`.
- Switched to additive non-bottleneck mode:
  `--calculator-estimator ste`, `--calculator-bottleneck-mode none`,
  `--calculator-injection-mode add`.
- Froze source policy and semantic decoder:
  `--freeze-semantic-decoder --freeze-calculator-policy`.
- Kept the op29 four-hook routed, `operand_spans`, product-decoder geometry.

Handoff results:

- Final eval `900/900 = 1.0000`.
- Diagnostic exact/calc `1.0000` / `0.9921875`.
- Final controls: injection-zero `0.0000`, forced-zero `0.0000`,
  forced-random `0.0078125`, oracle-at-eval `1.0000`.
- Routed diagnostic hook calculator-result accuracies:
  hook0 `1.0000`, hook1 `1.0000`, hook2 `1.0000`, hook3 `0.9565`.
- Step snapshots: step `200` normal `0.991`, step `400` normal `1.000`,
  step `600` normal `1.000`, with zero-injection near zero throughout.

## Interpretation

This is a real positive for fit dynamics. The earlier op29 constant-batch h64
and h128 sources missed heldout (`0.8444` and `0.8611`), while post-hoc
full-memory fits showed the targets were recoverable. The online
post-memory-fill full refresh closes that gap enough to clear both source
heldout and trusted non-bottleneck handoff.

It is not yet the scalable recipe. The run uses `2500` full-memory refresh
updates over `720` prompt-memory entries, for `2755` total prior updates. The
next high-leverage work is to preserve this gate while reducing or structuring
that cost: staged full refresh then coreset replay, coverage-aware or
proportional fitting, a better refresh-stop/freeze transition, or explicit
many-calculator cost accounting.

Do not rerun constant `fit_batch_size=160`, h128-only capacity bumps, random
fit-batch ladders, validation threshold/patience ladders, or the same
full-refresh pass as novelty.
