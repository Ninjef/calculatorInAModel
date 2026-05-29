# 2026-05-29 In-Training Additive Handoff Probe Logging

## Aim

Implement the next source-acquisition tool implied by the selector audits:
score source checkpoints with the real additive handoff behavior during source
training, without putting that probe on the gradient path.

This follows three selector negatives:

- frozen-state readout did not predict handoff;
- forced-result geometry and 25/50/100-step loss slope did not replace the
  500/600-step handoff gate;
- a simple leave-family-out ridge selector over early handoff traces did not
  beat raw early exact.

## Implementation

Added logging-only additive handoff probe flags to
`scripts/overfit_one_batch.py`:

```text
--additive-handoff-probe-every
--additive-handoff-probe-steps
--additive-handoff-probe-eval-every
--additive-handoff-probe-samples
--additive-handoff-probe-lr
--additive-handoff-probe-weight-decay
--additive-handoff-probe-seed
```

When enabled, the training loop:

1. clones the current model state into an additive, non-bottleneck
   result-space `ste` model;
2. loads only compatible weights;
3. freezes the calculator policy;
4. trains the additive downstream path for the configured probe steps;
5. logs snapshot-style exact-match, counterfactual, oracle, and learned-calc
   metrics to `additive_handoff_probe_rows.csv`;
6. records final/best probe rows in `metrics.json`.

The probe is diagnostic only. It does not alter the source model or contribute
to source gradients.

## Smoke Verification

Run root:

```text
runs/2026-05-29_phase7_additive_handoff_probe_logging_smoke
```

Smoke command shape:

```text
python3 scripts/overfit_one_batch.py \
  --variant model-c --digits 2 --steps 0 --operand-max 19 \
  --exhaustive-grid-batch --calculator-operand-vocab-size 20 \
  --calculator-estimator direct_feedback_alignment \
  --calculator-action-head result_space \
  --calculator-read-position operand_spans \
  --calculator-read-span-width 2 \
  --calculator-injection-mode add \
  --calculator-bottleneck-mode answer_decoder \
  --calculator-output-format sum \
  --answer-decoder-interaction product \
  --semantic-decoder-checkpoint <product decoder checkpoint> \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --result-policy-improvement-assignment-weight 1 \
  --freeze-semantic-decoder \
  --additive-handoff-probe-every 1 \
  --additive-handoff-probe-steps 1 \
  --additive-handoff-probe-eval-every 1 \
  --additive-handoff-probe-samples 20
```

The smoke completed and wrote:

```text
additive_handoff_probe_rows.csv
metrics.json
```

The emitted probe rows included source step `0`, probe steps `0/1`, normal
exact-match, injection-zero, oracle, forced-random, and learned calculator
accuracy. `metrics.json` included `final_additive_handoff_probe` and
`best_additive_handoff_probe_normal_exact_match`.

## Decision

Label:

```text
additive_handoff_probe_logging_implemented
```

The infrastructure now exists to score source checkpoints with the actual
additive handoff behavior during source training. This is not yet evidence
that a source recipe improves; it is the measurement tool needed for the next
source-acquisition experiment.

## Next Experiment

Run a fresh source acquisition with the current strongest no-decay source
recipe and enable logging-only handoff probes, for example every `200` source
steps with `500` probe steps. Select checkpoints by the probe exact-match
score and verify with the established 600-step or full handoff gate.

## Anti-Rerun Note

Do not repeat the one-step smoke as novelty. It only verified wiring. The next
useful run must use meaningful probe budgets on a real source-acquisition
lineage.

## Verification

- `python3 -m py_compile scripts/overfit_one_batch.py scripts/analyze_handoff_trace_selector.py`
- `git diff --check`
- zero-step source/probe smoke completed and wrote probe rows/metrics
