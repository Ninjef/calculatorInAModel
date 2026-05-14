# Gradient-Friendly Result Decoder Alignment Gate

## Task

```text
aiAgentProjectTasks/2026-05-14-phase-7-tenth-task-Gradient-friendly-result-decoder-alignment-gate.md
```

## Claim Tested

Can a result-calibrated frozen answer decoder make the exact result-marginal
answer-loss gradient over natural result actions align with the known
boundary-target ceiling, and if so, can exact expected answer-loss training
discover the hard result request with the decoder frozen?

## Code Changes

- Added `scripts/run_phase7_gradient_friendly_result_decoder_gate.py`.
- The runner evaluates the baseline decoder plus two narrow decoder candidates:
  soft-result calibration and contrastive result-margin.
- Decoder candidates train only the semantic decoder tensors
  (`answer_offset_emb`, `answer_decoder`, and `calculator_hook.output_proj`).
- Downstream diagnostics instantiate a fresh `result_space`
  `full_enum_expected_answer_loss` model, load only the semantic decoder,
  freeze it, and reuse the existing exact result-marginal / sampled PG /
  boundary-target gradient diagnostic on the exhaustive `20 x 20` grid.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_full_enum_action_loss_diagnostic.py scripts/run_phase7_gradient_friendly_result_decoder_gate.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_model.py -q
```

Result:

```text
94 passed
```

## Stage 0 Command

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase7_gradient_friendly_result_decoder_gate.py --decoder-exhaustive-grid-batch --decoder-steps 300 --branches soft_calibration contrastive_margin
```

Run root:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589
```

Stage 0 summary:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589/stage0_gradient_friendly_decoder_gate_summary.json
```

## Stage 0 Results

| Decoder | Forced/oracle exact | Hard-best=true | Tie-aware true-best | Raw expected NLL | Best/true NLL | Learned NLL | Expected-best gap | Exact grad L2 result/upstream | Semantic grad L2 | Exact-vs-boundary result/upstream | PG-vs-exact result/upstream | PG-vs-boundary result/upstream |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | `1.000 / 1.000` | `1.000` | `1.000` | `7.8521` | `0.0004 / 0.0004` | `8.5497` | `7.8517` | `0.1465 / 0.0549` | `0.0` | `-0.0978 / -0.1231` | `0.9577 / 0.9736` | `-0.0945 / -0.1108` |
| soft calibration | `1.000 / 1.000` | `1.000` | `1.000` | `7.8441` | `0.0004 / 0.0004` | `8.5417` | `7.8436` | `0.1465 / 0.0547` | `0.0` | `-0.0911 / -0.1175` | `0.9579 / 0.9737` | `-0.0876 / -0.1044` |
| contrastive margin | `1.000 / 1.000` | `1.000` | `1.000` | `14.4892` | `0.0000 / 0.0000` | `15.7934` | `14.4892` | `0.2562 / 0.0824` | `0.0` | `0.1204 / 0.0484` | `0.9560 / 0.9640` | `0.0949 / 0.0410` |

Decision:

```text
gradient_friendly_decoder_alignment_pass
```

The contrastive-margin candidate passed the formal Stage 0 sign gate. The
upstream cosine was positive but weak, below the stronger optional `>0.10`
preference.

Selected decoder checkpoint:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589/contrastive_margin_best_weights.pt
```

## Stage 1 Command

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator full_enum_expected_answer_loss --calculator-action-head result_space --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum --answer-decoder-interaction product --semantic-decoder-checkpoint runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/2026-05-14_113835_814589/contrastive_margin_best_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --exhaustive-grid-batch --answer-loss-weight 0.0 --aux-operand-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --expected-answer-loss-weight 1.0 --expected-answer-loss-policy-temperature 1.0 --expected-answer-loss-cost-normalization none --expected-answer-loss-entropy-weight 0.0 --expected-answer-loss-chunk-size 64 --result-boundary-target-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.01 --upstream-lr 0.0003 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --steps 800 --batch-size 400 --eval-samples 400 --snapshot-every 25 --snapshot-samples 400 --checkpoint-every 25 --seed 2 --run-root runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/stage1_exact_marginal_discovery --log-every 50
```

Stage 1 run:

```text
runs/2026-05-14_phase7_gradient_friendly_result_decoder_gate/stage1_exact_marginal_discovery/2026-05-14_113930_411831_model-c-op0-19-fullgrid-full_enum_expected_answer_loss-result_space-inlr0.01-uplr0.0003-expanspolt1-expanschunk64-answer_decoder-adec-product/model-c-2digit-seed4
```

## Stage 1 Results

| Checkpoint | Normal / calc-result acc | Injection-zero | Forced-random | Oracle | Learned-best | Entropy | Learned-best NLL gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best sampled diagnostic, step `275` | `0.105 / 0.105` | `0.0325` | `0.0225` | `1.000` | not logged in snapshot | not logged in snapshot | not logged in snapshot |
| best training-curve learned-result, step `300` | sampled normal `0.068` | sampled zero `0.033` | not logged | `1.000` | `0.0750` | `0.0367` | `8.2436` |
| final, step `800` | `0.090 / 0.090` | `0.0375` | `0.0275` | `1.000` | `0.0750` | `0.00003` | `8.2003` |

Final exact-match in `metrics.json` was `0.085`, and final loss was `8.2003`.
The learned result policy collapsed onto a few wrong result classes instead of
discovering the true-sum request.

Decision:

```text
gradient_friendly_decoder_stage0_pass_stage1_exact_marginal_discovery_negative
```

## Interpretation

- Decoder/loss geometry can be made locally positive at initialization. The
  contrastive-margin decoder flipped both exact-vs-boundary cosines positive
  while keeping forced true/oracle exact accuracy at `1.0` and semantic decoder
  downstream gradient at `0.0`.
- Local positive alignment was not sufficient for discovery. Exact expected
  answer-loss training with the aligned decoder frozen still collapsed to a
  low-entropy wrong result policy.
- Ordinary expected-cost/score-function training should not be treated as
  rescued by this decoder branch. The next useful direction is an explicitly
  biased backward channel, such as synthetic gradients/direct feedback
  alignment or a learned shadow-gradient module, with the same fixed-grid
  boundary-ceiling diagnostic retained as a gate.
