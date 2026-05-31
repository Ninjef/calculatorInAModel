# 2026-05-31 Integrated Amortized-Prior Source Replay

## Question

Can the numeric amortized-prior replay path be integrated into source training,
rather than remaining a post-hoc result-head repair?

## Context

The deterministic heldout prompt split showed prompt-keyed online hard memory is
transductive: train prompts reached `0.996875` exact/calc while heldout prompts
reached only `0.0875`. A follow-up diagnostic showed that a normalized numeric
operand prior trained from discovered train-memory targets reached `0.9125`
heldout target accuracy, and post-hoc result-head replay lifted heldout
exact/calc to `0.9125` while preserving train at `0.990625` when train replay
weight was `1`.

That result was not yet an end-to-end source-training method because the replay
was applied after the source run.

## Implementation

Added
`--result-boundary-target-amortized-prior-train-replay-weight` to
`scripts/overfit_one_batch.py`.

When the operand prior has enough prompt-memory entries and
`--result-boundary-target-amortized-prior-replay-batch-size` is positive:

- the existing heldout replay path samples heldout prompts and applies the base
  amortized-prior weight;
- the new train replay path samples from the streaming train pool and applies
  the base amortized-prior weight multiplied by the train replay weight;
- training curves record separate train and heldout replay losses,
  pseudo-target accuracies, confidences, and objective values.

The final config and metrics also record the train replay weight.

## Smoke

Ran a two-step CPU smoke with:

- `--streaming-train-heldout-fraction 0.2`
- prompt-keyed online hard memory
- numeric amortized prior
- replay batch size `8`
- train replay weight `1`
- additive semantic distillation enabled

Output run:

```text
runs/smoke_amortized_numeric_prior_train_replay_path/2026-05-31_082345_132651_model-c-op0-19-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk16-rbts4-rbtuniq-rbttopk2-rbto-008f34452a/model-c-2digit-seed9
```

The smoke completed and `training_curve.csv` contains both
`result_boundary_target_amortized_prior_train_replay_*` and
`result_boundary_target_amortized_prior_heldout_replay_*` fields. At step `2`,
the prior had only `5` entries, so this smoke is not evidence about final
learning quality.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "streaming_heldout_split or amortized_prior or prompt_keyed_online_hard_memory"
```

Focused tests passed: `3 passed, 151 deselected`.

## Interpretation

This is tooling progress for the active high-leverage experiment, not a solved
hypothesis. The next run should use the integrated train+heldout prior replay
path on the real streaming heldout source gate, then evaluate train and heldout
calculator accuracy before considering any trusted additive handoff.
