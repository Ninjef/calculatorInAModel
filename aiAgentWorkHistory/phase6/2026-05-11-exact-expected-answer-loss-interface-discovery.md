# Exact Expected Answer-Loss Interface Discovery

## Claim Tested

Can the strict identifiable calculator interface be learned directly from exact
expected answer loss over the full `20 x 20` action space, without true-operand
labels, oracle operands, or hard-best local pseudo-label CE?

## Code Changes

- Added `calculator_estimator=full_enum_expected_answer_loss`.
- Added the direct expected-cost objective:
  `sum_{a,b} p_theta(a)p_theta(b) * stopgrad(answer_nll(a,b))`.
- Added metrics for expected NLL, best/true/learned NLLs, expected-minus-best
  gap, learned-minus-best/true gaps, policy entropy/effective pairs, best/true
  pair probability mass, learned hard pair exact, learned-best fraction, and
  learned calculator-result accuracy.
- Added CLI knobs:
  `--expected-answer-loss-weight`,
  `--expected-answer-loss-policy-temperature`,
  `--expected-answer-loss-cost-normalization`,
  `--expected-answer-loss-entropy-weight`,
  `--expected-answer-loss-entropy-decay-steps`, and
  `--expected-answer-loss-chunk-size`.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/diagnose_private_protocol.py scripts/run_full_enum_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
75 passed
```

## Stage 0 Gate

Output:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage0/expected_answer_loss_stage0_gate.json
```

Fixed 128-sample strict `semantic_decoder_only` batch:

| Metric | Value |
| --- | ---: |
| initial expected answer loss | `8.2412` |
| best / true / learned NLL | `0.0003 / 0.0003 / 8.2883` |
| entropy / effective pairs | `5.9915 / 399.998` |
| true-pair probability | `0.0025` |
| hard learned pair exact | `0.000` |
| one-step input-proj delta L2 | `1.0895` |
| upstream delta L2 | `0.0` |
| semantic decoder grad / delta L2 | `0.0 / 0.0` |
| post-step oracle / injection-zero / forced-random | `1.000 / 0.000 / 0.000` |

No aux, anchor, oracle, or hard-best local target was active.

## Stage 1 Runs

All branches:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
expected_answer_loss_weight=1.0
adaptive/local target weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.03
steps=300
```

Exact Stage 1 command template, with branch-specific policy/entropy values:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 512 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator full_enum_expected_answer_loss --semantic-decoder-checkpoint /Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 0.0 --expected-answer-loss-weight 1.0 --expected-answer-loss-policy-temperature <1.0-or-0.5> --expected-answer-loss-cost-normalization none --expected-answer-loss-entropy-weight <0.0-or-0.03> --expected-answer-loss-entropy-decay-steps <0-or-300> --expected-answer-loss-chunk-size 64 --adaptive-interface-loss-weight 0.0 --aux-operand-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum_left_operand --answer-format sum_left_operand --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 25 --checkpoint-every 25 --snapshot-samples 128 --run-root <branch-run-root> --log-every 25
```

Branch substitutions:

| Branch | Temperature | Entropy | Entropy decay | Run root |
| --- | ---: | ---: | ---: | --- |
| A | `1.0` | `0.0` | `0` | `runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_a` |
| B | `1.0` | `0.03` | `300` | `runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_b_entropy003` |
| C | `0.5` | `0.0` | `0` | `runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05` |

Run root:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery
```

| Branch | Policy temp | Entropy | Final expected NLL | Final entropy / effective pairs | Best fast operand/pair/calc | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `1.0` | `0.0` | `4.3363` | `0.366 / 1.448` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |
| B | `1.0` | `0.03` decayed | `4.3350` | `0.365 / 1.446` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |
| C | `0.5` | `0.0` | `4.1432` | `0.105 / 1.112` | `0.0156 / 0.0156 / 0.0313` | `0.0059` |

No Stage 1 branch reached the `>=0.90` fast-gate threshold, so Stage 2
hard-argmax answer-only retention was not run.

## Selected Diagnostics

Selected checkpoint:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05/2026-05-11_151203_589696_model-c-op0-19-full_enum_expected_answer_loss-inlr0.03-uplr0.0003-expanspolt0.5-expanschunk64-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Diagnostic outputs:

```text
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_canonical_sum_left_operand
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_private_sum_left_operand
runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_full_enum
```

Exact selected-diagnostic commands:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05/2026-05-11_151203_589696_model-c-op0-19-full_enum_expected_answer_loss-inlr0.03-uplr0.0003-expanspolt0.5-expanschunk64-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt --samples 256 --digits 2 --operand-max 19 --answer-format sum_left_operand --calculator-output-format sum_left_operand --seed 6101 --forced-result-sweep --forced-result-batch-size 64 --output-dir runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_canonical_sum_left_operand
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/diagnose_private_protocol.py --checkpoint runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05/2026-05-11_151203_589696_model-c-op0-19-full_enum_expected_answer_loss-inlr0.03-uplr0.0003-expanspolt0.5-expanschunk64-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt --digits 2 --operand-max 19 --answer-format sum_left_operand --calculator-output-format sum_left_operand --seed 6102 --output-dir runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_private_sum_left_operand
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_full_enum_action_loss_diagnostic.py --checkpoint runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/stage1_branch_c_temp05/2026-05-11_151203_589696_model-c-op0-19-full_enum_expected_answer_loss-inlr0.03-uplr0.0003-expanspolt0.5-expanschunk64-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt --samples 128 --batch-size 64 --digits 2 --operand-max 19 --temperature 1.0 --chunk-size 64 --seed 6103 --answer-format sum_left_operand --output-root runs/2026-05-11_phase6_expected_answer_loss_interface_discovery/diagnostics/branch_c_final_full_enum
```

| Diagnostic | Result |
| --- | ---: |
| canonical normal / oracle / injection-zero / forced-random | `0.0039 / 1.000 / 0.0156 / 0.0039` |
| canonical operand / pair / calc | `0.0039 / 0.0039 / 0.0430` |
| private answer / operand / pair / calc | `0.005 / 0.005 / 0.005 / 0.050` |
| full-enum learned / true / best NLL | `4.3075 / 0.0003 / 0.0003` |
| full-enum learned-minus-true / best gap | `4.3072 / 4.3072` |
| full-enum learned-best / true-best | `0.000 / 1.000` |

Parameter deltas versus each branch step `0`:

| Branch | input-proj L2 | upstream L2 | semantic decoder L2 |
| --- | ---: | ---: | ---: |
| A | `118.197` | `0.0` | `0.0` |
| B | `118.359` | `0.0` | `0.0` |
| C | `71.452` | `0.0` | `0.0` |

## Interpretation

The objective is wired correctly and produces a strong gradient into
`calculator_hook.input_proj`. It also lowers expected answer loss and collapses
the policy distribution. However, the collapsed hard argmax actions are wrong:
the learned hard pair remains near chance and the full-enum learned-best
fraction is `0.0`.

This is a negative for direct independent-head expected answer-loss discovery,
not a target-identifiability negative. The full-enum landscape still identifies
the true pair exactly (`true_best=1.000`).

## Recommendation

Do not run Stage 2 or broaden this exact branch. The next useful direction is a
Gumbel/Concrete hard-forward soft-backward bridge, or an upstream-open expected
loss branch only if testing frozen-readout limitations is more important than
following the clearest optimization failure mode.
