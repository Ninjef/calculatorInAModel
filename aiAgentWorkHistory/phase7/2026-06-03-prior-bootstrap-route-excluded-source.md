# 2026-06-03 - Prior-Bootstrap Route-Excluded Source Gate

## Purpose

Test whether prior-bootstrap prompt memory can improve the op19 route-excluded
shared-prior source. This is the first source gate after adding the bootstrap
mechanism: the shared numeric prior may write prompt-memory targets for route 1,
whose direct candidate-scored target discovery remains disabled.

## Setup

Baseline family: op19 four-hook shared-output source with prompt-keyed hard
memory, route 1 excluded from target discovery, numeric amortized-prior replay,
full-memory prior fitting every 2 steps, and additive semantic distillation.

Delta from the no-bootstrap route-excluded source:

- `--result-boundary-target-amortized-prior-bootstrap-memory-routes 1`
- `--result-boundary-target-amortized-prior-bootstrap-memory-min-confidence 0.30`
- `--result-boundary-target-amortized-prior-bootstrap-memory-min-train-accuracy 0.75`
- `--result-boundary-target-amortized-prior-bootstrap-memory-max-updates-per-step 8`

No extra route-replay objective was used in this run.

## Command

```bash
python3 -u scripts/overfit_one_batch.py \
  --variant model-c --digits 2 --steps 5000 --batch-size 400 --eval-samples 400 \
  --operand-max 19 --exhaustive-grid-batch \
  --streaming-train-batch-size 64 --streaming-train-heldout-fraction 0.2 \
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
  --result-boundary-target-sample-count 24 \
  --result-boundary-target-unique-sampling \
  --result-boundary-target-policy-topk-count 8 \
  --result-boundary-target-chunk-size 64 \
  --result-boundary-target-amortized-prior-weight 1.0 \
  --result-boundary-target-amortized-prior-feature-mode numeric \
  --result-boundary-target-amortized-prior-hidden-size 64 \
  --result-boundary-target-amortized-prior-replay-batch-size 64 \
  --result-boundary-target-amortized-prior-fit-batch-size 0 \
  --result-boundary-target-amortized-prior-fit-every 2 \
  --result-boundary-target-amortized-prior-stop-train-accuracy 1.0 \
  --result-boundary-target-amortized-prior-stop-patience 100 \
  --result-boundary-target-amortized-prior-train-replay-weight 1.0 \
  --result-boundary-target-amortized-prior-bootstrap-memory-routes 1 \
  --result-boundary-target-amortized-prior-bootstrap-memory-min-confidence 0.30 \
  --result-boundary-target-amortized-prior-bootstrap-memory-min-train-accuracy 0.75 \
  --result-boundary-target-amortized-prior-bootstrap-memory-max-updates-per-step 8 \
  --additive-semantic-distill-weight 1.0 \
  --additive-semantic-distill-sample-count 8 \
  --additive-semantic-distill-temperature 1.0 \
  --snapshot-every 100 --snapshot-samples 400 --checkpoint-every 500 \
  --log-every 25 --seed 7 \
  --run-root runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_priorboot1c30tacc75_cap8_src5000
```

Run:

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_prior_fitfull_every2_stop1pat100_exclroute1_priorboot1c30tacc75_cap8_src5000/2026-06-02_183002_336175_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-520af39691/model-c-2digit-seed9
```

## Results

- Final eval exact/calculator-result accuracy: `308/400 = 0.7700`.
- Final loss: `0.1887894570827484`.
- Best snapshot normal/calculator-result accuracy: `0.7825` at step `4700`.
- Final snapshot normal/calculator-result accuracy: `0.7800`.
- Final snapshot controls: injection-zero `0.0475`, forced-zero `0.0025`,
  forced-random `0.0025`.
- Diagnostic 128-sample exact/calculator-result accuracy: `0.78125`.
- Diagnostic route accuracy: route 0 `0.7447`, route 1 `0.7714`, route 2
  `0.8182`, route 3 `0.8333`.
- Train prompts: exact/calculator-result accuracy `0.8125`.
- Heldout prompts: exact/calculator-result accuracy `0.5625`.
- Train prompt routes: hook0 `0.8660`, hook1 `0.6392`, hook2 `0.9322`,
  hook3 `0.8806`.
- Heldout prompt routes: hook0 `0.2609`, hook1 `0.7391`, hook2 `0.5238`,
  hook3 `0.8462`.
- Prompt-memory entries / direct expected entries: `300 / 223`.
- Prior-bootstrap entries: `77`.
- Forced-result evals: `37,896`, unchanged from the non-bootstrap route-excluded
  source because bootstrap did not add candidate scoring.
- Prior updates: `2,501`.
- Prior train/heldout accuracy: `0.7781` / `0.5625`.
- Prior train/heldout confidence: `0.3579` / `0.3253`.

Bootstrap timing:

- The train-accuracy gate did not open until late. Logged step `4500` still had
  prior train accuracy `0.7444` and `0` bootstrap entries in the row.
- Logged step `4600` had prior train accuracy `0.7534`, the gate open, and
  `62` total bootstrap entries. The row's current update had confidence
  `0.4341` and pseudo-accuracy `1.0`.
- Final bootstrap entries reached `77`.

## Interpretation

Mixed-negative. Prior-bootstrap prompt memory is wired and can write durable
excluded-route prompt targets, but this source missed the gate and did not
improve the shared-prior bottleneck. It underperformed both the earlier
route-excluded source (`0.7875` final / `0.8075` best snapshot) and the extra
route-replay source (`0.8175` final / `0.8075` best snapshot). Heldout stayed
at `0.5625`, excluded route 1 heldout stayed `0.7391`, and the prior itself
stayed weak.

No trusted handoff was run because the source gate missed.

Do not run bootstrap confidence, train-accuracy, cap, or same op19
route-excluded variants as novelty. The failure is not simply insufficient
post-hoc target writes; the shared prior needs better target formation before
route memory freezes, or the project needs a more different credit signal.
