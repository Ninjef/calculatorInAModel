# 2026-05-31 - Prompt-keyed online hard memory streaming minibatch gate

## Question

Can the four-hook shared-output online-hard-memory plus additive semantic
distillation recipe train from stochastic minibatches, or was the prior
success dependent on backpropagating the fixed exhaustive grid every step?

## Implementation

Added two training-script controls:

- `--streaming-train-batch-size`: draws a fresh ranged training minibatch each
  step even when `--exhaustive-grid-batch` is enabled for evaluation and prompt
  count accounting.
- `--result-boundary-target-online-memory-key-mode prompt`: stores online hard
  memory by prompt-token tuple instead of fixed batch row.

Added a focused regression test showing that prompt-keyed memory can fill from
two different minibatches and freezes without rescoring after the expected
prompt count is reached.

## Runs

Short streaming source:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_src800/2026-05-30_213045_897295_model-c-op0-19-fullgrid-streamb64-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehard-5056870fb5/model-c-2digit-seed9
```

Matched-exposure streaming source:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_equalexposure_src5000/2026-05-30_213313_861629_model-c-op0-19-fullgrid-streamb64-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehard-5056870fb5/model-c-2digit-seed9
```

Trusted handoff:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_equalexposure_handoff600/2026-05-30_213727_469671_model-c-op0-19-fullgrid-hooks4-routeleft_operand_mod-adec-product/model-c-2digit-seed9
```

Shared settings:

- CLI seed `7`, effective seed `9`, op19 `400`-prompt grid.
- Four `left_operand_mod` routed hooks, active-only routed execution, shared
  output projection.
- `operand_spans` readout, span width `2`, product decoder, `n_embd=16`.
- Sparse zero-improvement online hard memory, topk8+unique24, freeze when full.
- Additive semantic distillation weight `1`, sample count `8`.

## Results

The 800-step batch64 source was undertrained, not target-limited:

- Final eval `253/400 = 0.6325`.
- Prompt memory filled all `400` entries and froze.
- Hard memory targets were true after fill (`best_true=1.0000`).
- Diagnostic calculator-result accuracy was `0.5781`.
- Final controls stayed causal: injection-zero `0.0703`, forced-zero `0.0078`,
  forced-random `0.0156`.

The matched-exposure batch64 source passed:

- Source final `400/400 = 1.0000`.
- Diagnostic calculator-result accuracy `1.0000`.
- All four hooks reached calculator-result accuracy `1.0000`.
- Prompt memory filled/froze all `400` entries; cumulative forced-result evals
  stopped at `173,568`.
- Final controls: injection-zero `0.0703`, forced-zero `0.0078`,
  forced-random `0.0156`.

Trusted frozen-policy additive handoff from the matched-exposure source passed:

- Final `400/400 = 1.0000`.
- Step-600 normal/calc `1.0000`.
- Final diagnostic calculator-result accuracy `1.0000`.
- Final controls: injection-zero `0.0781`, forced-zero `0.0078`,
  forced-random `0.0156`.
- All four hooks reached calculator-result accuracy `1.0000`.

## Interpretation

Prompt-keyed online hard memory plus additive semantic distillation can train
routed shared-output calculators from stochastic minibatches when the update
budget is exposure-matched to the fixed-grid baseline. This is a real
scalability improvement over requiring the exhaustive grid as the training
batch, but it does not solve fresh-prompt generalization and it shifts cost
into more optimizer updates.

Do not repeat the 800-step batch64 miss as a failure, and do not rerun the same
5000-step op19 streaming source/handoff as novelty. The next high-leverage gate
is fresh/heldout prompt memory or a cheaper streaming uptake mechanism.
