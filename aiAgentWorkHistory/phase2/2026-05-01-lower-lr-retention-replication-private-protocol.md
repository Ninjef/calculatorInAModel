# 2026-05-01 - Lower-LR retention replication and private-protocol decoding

Task: `aiAgentProjectTasks/2026-05-01-phase-2-sixth-task-Lower-LR-retention-replication-and-protocol-decoding.md`.

## Code changes

- Added `scripts/diagnose_private_protocol.py`, a checkpoint-first read-only diagnostic that evaluates all `0..19` operand pairs, writes true-vs-learned confusion matrices, per-operand/group summaries, majority and affine code mappings, read-vector intervention summaries, and example rows where operands/answers disagree.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/diagnose_private_protocol.py
```

## Setup

Starting Stage B checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Semantic decoder root checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Shared primary Stage C config:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
calculator_bottleneck_mode=answer_decoder
calculator_estimator=adaptive_interface
adaptive_interface_target_mode=hard_pair
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable_parameter_groups=[calculator_hook.input_proj]
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

All primary replication checkpoints and the long continuation have `final_aux_operand_loss_weight=0.0`, `freeze_upstream_encoder=true`, and only `calculator_hook.input_proj.weight` plus `.bias` trainable in `metrics.json`.

## Training commands and run paths

Common command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --adaptive-interface-target-mode hard_pair --aux-operand-loss-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0
```

| Variant | Extra args | Run path |
| --- | --- | --- |
| C-low-lr-rep-seed1 | `--seed 1 --steps 1000 --snapshot-every 100 --snapshot-samples 128 --log-every 50` | `runs/2026-05-01_122402_250571_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3` |
| C-low-lr-rep-seed2 | `--seed 2 --steps 1000 --snapshot-every 100 --snapshot-samples 128 --log-every 50` | `runs/2026-05-01_122359_647419_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4` |
| C-low-lr-rep-seed3 | `--seed 3 --steps 1000 --snapshot-every 100 --snapshot-samples 128 --log-every 50` | `runs/2026-05-01_122400_771781_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed5` |
| C-low-lr-long | `--seed 0 --steps 3000 --snapshot-every 250 --snapshot-samples 128 --log-every 100` | `runs/2026-05-01_123323_807024_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` |

Seed note: the CLI seeds `1`, `2`, and `3` became run-name seeds `3`, `4`, and `5` because `overfit_one_batch.py` adds `num_digits`.

## Built-in replication metrics

| Checkpoint | Eval exact | Learned-target agree | Target result acc | Aux final weight |
| --- | ---: | ---: | ---: | ---: |
| Stage B handoff | `0.4258` | `0.3828` | `0.9375` | `0.0000` |
| Drift control | `0.2734` | `0.1953` | `0.9375` | `0.0000` |
| Prior C-low-lr | `0.4492` | `0.4766` | `0.9375` | `0.0000` |
| C-low-lr-rep-seed1 | `0.5195` | `0.5078` | `0.9141` | `0.0000` |
| C-low-lr-rep-seed2 | `0.4902` | `0.4688` | `0.8516` | `0.0000` |
| C-low-lr-rep-seed3 | `0.4746` | `0.4688` | `0.9219` | `0.0000` |
| C-low-lr-long | `0.3613` | `0.3828` | `0.9375` | `0.0000` |

## Canonical causal diagnostics

Command pattern:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <run>/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir <run>/canonical_causal_diagnostics
```

| Checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval | Operand exact | Calc result acc | Classification | Bottleneck | Learned-best | True-sum-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Stage B handoff | `0.3750` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4219` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.3750` | `0.9219` |
| Drift control | `0.2500` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.2031` | `0.2656` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.2500` | `0.9219` |
| Prior C-low-lr | `0.4375` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4844` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.4375` | `0.9219` |
| C-low-lr-rep-seed1 | `0.4375` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4688` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.4375` | `0.9219` |
| C-low-lr-rep-seed2 | `0.4219` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4375` | `0.4375` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.4219` | `0.9219` |
| C-low-lr-rep-seed3 | `0.3750` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.3438` | `0.4063` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.3750` | `0.9219` |
| C-low-lr-long | `0.3438` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.3125` | `0.3594` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.3594` | `0.9219` |

Canonical MI summary:

| Checkpoint | MI learned A/true A | MI learned B/true B | MI learned result/true sum |
| --- | ---: | ---: | ---: |
| Stage B handoff | `3.8783` | `3.3621` | `3.9762` |
| Prior C-low-lr | `3.7773` | `3.3662` | `3.8496` |
| C-low-lr-rep-seed1 | `3.7773` | `3.2024` | `3.9845` |
| C-low-lr-rep-seed2 | `3.8193` | `3.2556` | `3.9475` |
| C-low-lr-rep-seed3 | `3.6268` | `3.2528` | `4.0059` |
| C-low-lr-long | `3.4861` | `3.1925` | `3.7026` |

## Canonical action-loss diagnostics

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint <stage-b> <drift> <prior-low-lr> <rep1> <rep2> <rep3> <long> --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

| Checkpoint | True NLL | Learned NLL | Random NLL | Shuffled NLL | Learned-true gap | Random-true gap | Shuffled-true gap | True best | Learned best | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B handoff | `0.1181` | `3.1823` | `9.9763` | `10.4317` | `3.0642` | `9.8583` | `10.3136` | `0.4844` | `0.0000` | `0.4375` | `0.5000` |
| Drift control | `0.1181` | `5.6422` | `9.9763` | `10.4317` | `5.5241` | `9.8583` | `10.3136` | `0.6875` | `0.0000` | `0.1875` | `0.2969` |
| Prior C-low-lr | `0.1181` | `3.1647` | `9.9763` | `10.4317` | `3.0466` | `9.8583` | `10.3136` | `0.5313` | `0.0000` | `0.4219` | `0.4531` |
| C-low-lr-rep-seed1 | `0.1181` | `2.5967` | `9.9763` | `10.4317` | `2.4786` | `9.8583` | `10.3136` | `0.4219` | `0.0000` | `0.5469` | `0.5781` |
| C-low-lr-rep-seed2 | `0.1181` | `3.0182` | `9.9763` | `10.4317` | `2.9001` | `9.8583` | `10.3136` | `0.4375` | `0.0000` | `0.5469` | `0.5625` |
| C-low-lr-rep-seed3 | `0.1181` | `2.7538` | `9.9763` | `10.4317` | `2.6357` | `9.8583` | `10.3136` | `0.4219` | `0.0000` | `0.5000` | `0.5781` |
| C-low-lr-long | `0.1181` | `4.5651` | `9.9763` | `10.4317` | `4.4470` | `9.8583` | `10.3136` | `0.5469` | `0.0000` | `0.3594` | `0.4375` |

## Private-protocol decoding

Command pattern:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/diagnose_private_protocol.py --checkpoint <run>/final_weights.pt --digits 2 --operand-max 19 --output-dir <run>/private_protocol_diagnostics
```

All-pair protocol summary:

| Checkpoint | Answer exact | Operand exact | Calc result acc | Majority-mapped operand exact | Majority-mapped calc acc | Result-code majority acc | Wrong operands/right answer | Right operands/wrong answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B handoff | `0.4500` | `0.4575` | `0.4875` | `0.4775` | `0.4850` | `0.4975` | `12` | `15` |
| Prior C-low-lr | `0.4875` | `0.4900` | `0.5250` | `0.4900` | `0.5250` | `0.5300` | `13` | `14` |
| C-low-lr-rep-seed1 | `0.5000` | `0.5100` | `0.5425` | `0.5200` | `0.5575` | `0.5475` | `12` | `16` |

Best affine modulo-20 mapping was identity-like in every diagnosed checkpoint:

| Checkpoint | A affine exact | A scale/offset | B affine exact | B scale/offset |
| --- | ---: | --- | ---: | --- |
| Stage B handoff | `0.8500` | `1 / 0` | `0.5400` | `1 / 0` |
| Prior C-low-lr | `0.8500` | `1 / 0` | `0.5700` | `1 / 0` |
| C-low-lr-rep-seed1 | `0.8500` | `1 / 0` | `0.5975` | `1 / 0` |

Group summary for best replication (`C-low-lr-rep-seed1`):

| Group | Count | Operand exact | Calc result acc | Answer exact | Mapped operand exact | Mapped result exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | `400` | `0.5100` | `0.5425` | `0.5000` | `0.5200` | `0.5575` |
| Carry (`sum >= 10`) | `345` | `0.4928` | `0.5304` | `0.4812` | `0.4812` | `0.5246` |
| No carry | `55` | `0.6182` | `0.6182` | `0.6182` | `0.7636` | `0.7636` |
| Large operand | `300` | `0.4767` | `0.5200` | `0.4800` | `0.4167` | `0.4667` |
| Small operands | `100` | `0.6100` | `0.6100` | `0.5600` | `0.8300` | `0.8300` |
| Symmetric | `20` | `0.7000` | `0.7000` | `0.6500` | `0.6500` | `0.6500` |

Read-vector intervention summary for best replication:

| Condition | Answer exact | Prediction changed vs normal | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: |
| Normal | `0.5000` | `0.0000` | `0.5100` | `0.5425` |
| Corrupt A read vector | `0.0350` | `0.8975` | `0.0225` | `0.0325` |
| Corrupt B read vector | `0.0475` | `0.9500` | `0.0425` | `0.0475` |
| Swap A read vector | `0.5000` | `0.0000` | `0.5100` | `0.5425` |
| Swap B read vector | `0.5000` | `0.0000` | `0.5100` | `0.5425` |

Interpretation:

- The retained interface is not a clean private permutation. The best affine mapping is identity-like, and majority mapping improves only modestly.
- The strongest structure is asymmetric: learned A is mostly true (`0.85` affine exact), while B remains much noisier (`0.54` to `0.5975`).
- Learned result classes remain informative about true sums (`MI` near `3.67` to `4.01` bits), but result-code majority mapping explains only about half the all-pair space.
- Errors concentrate in carry and large-operand regions. Small/no-carry cases are much more operand-like, and simple mapping helps them more.
- Corrupting either read vector collapses performance, while swap interventions are no-ops in the existing intervention implementation for this operand-read setup. This supports causal use of read vectors but not a clean A/B-swappable protocol.

## Parameter deltas

Input-proj L2 / max-abs deltas:

| Checkpoint | vs semantic weight | vs semantic bias | vs Stage B weight | vs Stage B bias |
| --- | --- | --- | --- | --- |
| Stage B handoff | `32.7283 / 2.5145` | `5.1038 / 1.0202` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| Drift control | `40.3043 / 3.4755` | `9.5512 / 1.8400` | `13.6511 / 1.1142` | `4.4675 / 0.8399` |
| Prior C-low-lr | `33.9883 / 2.7473` | `5.9916 / 1.1929` | `3.1749 / 0.2437` | `0.9003 / 0.2122` |
| C-low-lr-rep-seed1 | `34.0233 / 2.7486` | `5.9811 / 1.2011` | `3.2605 / 0.2540` | `0.8932 / 0.2231` |
| C-low-lr-rep-seed2 | `34.0285 / 2.7526` | `5.9672 / 1.1907` | `3.1640 / 0.2475` | `0.8783 / 0.2203` |
| C-low-lr-rep-seed3 | `34.0221 / 2.7475` | `5.9604 / 1.1879` | `3.2178 / 0.2451` | `0.8734 / 0.2212` |
| C-low-lr-long | `36.8621 / 3.1831` | `7.7808 / 1.4653` | `9.0463 / 0.7219` | `2.7066 / 0.5722` |

## Decision

Robust positive retention result for 1000-step lower-LR Stage C:

- All three primary replications have `final_aux_operand_loss_weight=0.0`, no anchor, frozen upstream, and train only `calculator_hook.input_proj`.
- All three beat the drift control on built-in eval exact, learned-target agreement, action-loss learned-minus-true gap, action-loss operand exact, and action-loss calculator-result accuracy.
- At least one replication is close to or better than the original C-low-lr result; seed1 is best by eval exact (`0.5195`) and action learned-true gap (`2.4786`).
- Injection-zero remains `0.0`, oracle-at-eval remains `0.9063`, and true-sum forced-result best remains `0.9219`.

Limits:

- Canonical classification did not improve beyond `causally_useful_opaque_private_code` / `strict_bottleneck_unvalidated`.
- Learned-best action-loss fraction remains `0.0` for every checkpoint.
- The 3000-step continuation regresses (`eval exact=0.3613`, action learned-true gap `4.4470`, canonical operand exact `0.3125`), so lower LR is a retention window, not a monotonic long-run solution.

Go/no-go:

- Go for more input-proj-only stabilization research around 1000-step lower-LR continuations and perhaps stop/early-selection criteria.
- No-go for upstream unfreezing now. The retained interface is causal and partly operand-like, but still opaque/private by canonical diagnostics and still not action-loss optimal.
- Anchor fallback was not run because the primary lower-LR replications were stable; keep it reserved for future stabilization if lower-LR runs regress.
