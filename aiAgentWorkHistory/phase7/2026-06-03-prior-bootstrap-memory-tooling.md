# 2026-06-03 - Prior-Bootstrap Prompt Memory Tooling

## Purpose

Add a materially different route-excluded shared-target mechanism after route
replay only marginally improved the op19 source. The goal is to let a shared
numeric prior create durable prompt-memory targets for routes whose direct
candidate-scored target discovery is disabled, instead of merely applying more
prior-replay pressure to model logits.

## Implementation

- Added `--result-boundary-target-amortized-prior-bootstrap-memory-routes`.
- Added `--result-boundary-target-amortized-prior-bootstrap-memory-min-confidence`.
- Added `--result-boundary-target-amortized-prior-bootstrap-memory-min-train-accuracy`.
- Added `--result-boundary-target-amortized-prior-bootstrap-memory-max-updates-per-step`.
- Bootstrapped entries are marked with `prior_bootstrap=1` and train the model
  through the existing prompt-memory target loss.
- Prior fitting excludes bootstrapped entries by default, so the prior remains
  trained on candidate-scored evidence rather than its own pseudo-labels.
- Bootstrap candidates are deduplicated by prompt key before applying the
  per-step cap, because streaming minibatches can contain duplicate prompts.
- Added runtime/final metrics for bootstrap candidates, updates, confidence,
  pseudo-accuracy, total bootstrap entries, and train-accuracy gate status.

## Smoke

Ran a tiny route-excluded smoke to verify the training path:

```bash
python3 -u scripts/overfit_one_batch.py \
  --variant model-c --digits 2 --steps 30 --batch-size 32 --eval-samples 40 \
  --operand-max 2 --exhaustive-grid-batch \
  --streaming-train-batch-size 8 --streaming-train-heldout-fraction 0.2 \
  --streaming-train-heldout-seed 91000 \
  --calculator-operand-vocab-size 20 --n-layer 4 --n-head 1 --n-embd 16 \
  --mlp-expansion 4 --calculator-hook-after-layer 2 \
  --calculator-read-position operand_spans --calculator-read-span-width 2 \
  --answer-format sum --calculator-output-format sum \
  --calculator-bottleneck-mode answer_decoder --answer-decoder-interaction product \
  --calculator-estimator gumbel_concrete_interface --calculator-action-head result_space \
  --calculator-hook-count 4 --calculator-hook-routing left_operand_mod \
  --share-calculator-output-proj \
  --semantic-decoder-checkpoint runs/2026-05-12_phase6_sum_only_interaction_decoder_gate/stage0_candidates/tiny_operand_spans_dense/oracle_train/2026-05-12_113346_842566_model-c-oracle-op0-19-answer_decoder-adec-product/model-c-2digit-seed2/checkpoint_snapshots/step_00500_weights.pt \
  --semantic-decoder-checkpoint-load-scope semantic_decoder_only \
  --freeze-semantic-decoder --answer-loss-weight 0.0 \
  --input-proj-lr 0.01 --upstream-lr 0.0003 \
  --result-boundary-target-loss-weight 1.0 \
  --result-boundary-target-mode zero_improvement \
  --result-boundary-target-online-hard-memory \
  --result-boundary-target-online-memory-key-mode prompt \
  --result-boundary-target-online-memory-freeze-when-full \
  --result-boundary-target-memory-update-exclude-routes 1 \
  --result-boundary-target-sample-count 4 \
  --result-boundary-target-unique-sampling \
  --result-boundary-target-policy-topk-count 2 \
  --result-boundary-target-chunk-size 16 \
  --result-boundary-target-amortized-prior-weight 1.0 \
  --result-boundary-target-amortized-prior-feature-mode numeric \
  --result-boundary-target-amortized-prior-hidden-size 16 \
  --result-boundary-target-amortized-prior-replay-batch-size 8 \
  --result-boundary-target-amortized-prior-fit-batch-size 0 \
  --result-boundary-target-amortized-prior-min-entries 1 \
  --result-boundary-target-amortized-prior-fit-every 1 \
  --result-boundary-target-amortized-prior-train-replay-weight 1.0 \
  --result-boundary-target-amortized-prior-bootstrap-memory-routes 1 \
  --result-boundary-target-amortized-prior-bootstrap-memory-min-confidence 0.0 \
  --result-boundary-target-amortized-prior-bootstrap-memory-max-updates-per-step 2 \
  --snapshot-every 30 --snapshot-samples 40 --checkpoint-every 0 \
  --log-every 5 --seed 6 \
  --run-root runs/2026-06-03_prior_bootstrap_memory_smoke
```

Run:

```text
runs/2026-06-03_prior_bootstrap_memory_smoke/2026-06-02_182454_691291_model-c-op0-2-fullgrid-streamb8-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk16-rbts4-rbtuniq-rbttopk2-rbton-5b7322ec44/model-c-2digit-seed8
```

Smoke result:

- Final exact-match was `0/40`; this was not a source-gate run.
- Final prompt-memory entries were `5` with expected direct entries `4`.
- Final prior-bootstrap entries were `3`.
- Training-curve step `0` recorded `2` unique bootstrap candidates and `2`
  updates with confidence `0.0398` and pseudo-accuracy `0.0`; the permissive
  confidence threshold was intentional for path validation.
- The smoke exposed duplicate prompt rows in streaming minibatches, so the
  helper now deduplicates candidate keys before counting and writing updates.

## Verification

```bash
python3 -m py_compile scripts/overfit_one_batch.py
python3 -m pytest tests/test_model.py -k "prior_bootstrap_memory or subset_arithmetic_batch_by_routes or prompt_keyed_online_hard_memory or streaming_heldout_split or amortized_prior or route_exclusion"
git diff --check
```

The focused pytest slice passed with `7 passed, 151 deselected`.

## Next

Run a real op19 route-excluded source gate with conservative bootstrap gates,
for example high prior train accuracy plus a confidence threshold, and require
heldout/excluded-route quality before any trusted handoff. Do not treat the
tiny smoke as evidence that prior-bootstrap targets improve source quality.
