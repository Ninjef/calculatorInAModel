# 2026-05-01 - Post-supervision retention stabilization under strict bottleneck

Task: `aiAgentProjectTasks/2026-05-01-phase-2-fifth-task-Post-supervision-retention-stabilization.md`.

## Code changes

- Added optional checkpoint-relative `calculator_hook.input_proj` anchoring to `scripts/overfit_one_batch.py`:
  - `--input-proj-anchor-checkpoint`
  - `--input-proj-anchor-weight`
  - `--input-proj-anchor-decay-steps`
- The anchor loads only `calculator_hook.input_proj.weight` and `.bias`, applies a training-only mean-squared L2 retention loss, logs anchor weight/loss in `training_curve.csv`, and records final anchor metadata and deltas in `metrics.json`.
- Anchor variants were not run because the pure lower-LR Stage C runs already preserved/improved the Stage B handoff, satisfying the task's stop-early instruction.
- Added tests for anchor loss/decay/metadata in `tests/test_model.py`.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q tests/test_model.py -k "anchor or training_cli_supports_oracle_warmup_and_snapshots or adaptive_optimizer_groups"
```

Result: `4 passed, 44 deselected`.

## Setup

Stage B starting handoff checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Semantic decoder root checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Shared Stage C config:

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

All Stage C variants have `final_aux_operand_loss_weight=0.0` in `metrics.json`.

## Stage C commands and run paths

Common command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --seed 0 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --adaptive-interface-target-mode hard_pair --aux-operand-loss-weight 0.0 --snapshot-every 100 --snapshot-samples 128 --log-every 50
```

| Variant | Extra args | Run path |
| --- | --- | --- |
| C-control-repeat | `--steps 500 --input-proj-lr 0.003 --upstream-lr 0.003 --adaptive-interface-loss-weight 1.0` | `runs/2026-05-01_114850_422866_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2` |
| C-low-lr | `--steps 1000 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0` | `runs/2026-05-01_114850_238336_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` |
| C-very-low-lr | `--steps 1000 --input-proj-lr 0.0001 --upstream-lr 0.0001 --adaptive-interface-loss-weight 1.0` | `runs/2026-05-01_114849_026560_model-c-op0-19-adaptive_interface-inlr0.0001-uplr0.0001-answer_decoder/model-c-2digit-seed2` |
| C-adaptive-low-weight | `--steps 1000 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 0.3` | `runs/2026-05-01_114850_842904_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` |

## Built-in metric table

| Variant | Eval exact | Trace exact | Operand exact | Calc result acc | Target acc | Learned-target agreement | Aux CE / weight | Mean entropy A+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Stage B handoff | `0.4258` | `0.4922` | `0.4844` | `0.5078` | `0.9375` | `0.3828` | `2.7502 / 0.0000` | `5.9727` |
| Prior Stage C drift | `0.2734` | `0.3438` | `0.2578` | `0.3516` | `0.9375` | `0.1953` | `2.7613 / 0.0000` | `5.9644` |
| C-control-repeat | `0.2734` | `0.3438` | `0.2578` | `0.3516` | `0.9375` | `0.1953` | `2.7613 / 0.0000` | `5.9643` |
| C-low-lr | `0.4492` | `0.5703` | `0.5547` | `0.5859` | `0.9375` | `0.4766` | `2.7547 / 0.0000` | `5.9691` |
| C-very-low-lr | `0.4199` | `0.4688` | `0.4531` | `0.4766` | `0.9375` | `0.3672` | `2.7510 / 0.0000` | `5.9723` |
| C-adaptive-low-weight | `0.4395` | `0.5703` | `0.5781` | `0.5859` | `0.9375` | `0.4609` | `2.7560 / 0.0000` | `5.9698` |

## Snapshot drift readout

Final 128-sample built-in snapshots:

| Variant | Step | Normal | Injection-zero | Oracle | Operand exact | Calc result acc | Mean confidence A/B | Entropy A+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| C-control-repeat | 500 | `0.2656` | `0.0000` | `0.9375` | `0.1875` | `0.2813` | `0.0636 / 0.0646` | `5.9644` |
| C-low-lr | 1000 | `0.4688` | `0.0078` | `0.9453` | `0.4609` | `0.4922` | `0.0632 / 0.0653` | `5.9697` |
| C-very-low-lr | 1000 | `0.4141` | `0.0078` | `0.9453` | `0.4062` | `0.4375` | `0.0631 / 0.0651` | `5.9723` |
| C-adaptive-low-weight | 1000 | `0.4922` | `0.0078` | `0.9453` | `0.4922` | `0.5156` | `0.0632 / 0.0651` | `5.9700` |

Diagnosis:

- True-operand protocol drift is strongly reduced by lower LR. `C-low-lr` improves final built-in operand exact over Stage B (`0.5547` vs `0.4844`) and over prior Stage C (`0.2578`).
- Private-code drift remains in the canonical classification: all diagnosed checkpoints are still `causally_useful_opaque_private_code` with `strict_bottleneck_unvalidated`, not a clean true-operand protocol.
- Objective drift is improved: `C-low-lr` learned-target agreement is `0.4766`, beating Stage B `0.3828` and prior Stage C `0.1953`.
- Confidence drift is not the main failure mode here. Entropy stays high and stable around `5.96-5.97`, with mean operand confidence near `0.064`.

## Canonical causal diagnostics

Commands:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py --checkpoint <checkpoint>/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir <run>/canonical_causal_diagnostics
```

| Checkpoint | Normal | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval | Operand exact | Calc result acc | Classification | Bottleneck | Learned-best | True-sum-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Stage B handoff | `0.3750` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4219` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.3750` | `0.9219` |
| C-control-repeat | `0.2500` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.2031` | `0.2656` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.2500` | `0.9219` |
| C-low-lr | `0.4375` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4844` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.4375` | `0.9219` |
| C-adaptive-low-weight | `0.4375` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4688` | `0.4844` | `causally_useful_opaque_private_code` | `strict_bottleneck_unvalidated` | `0.4375` | `0.9219` |

## Canonical action-loss diagnostics

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint <stage-b> <control> <c-low-lr> <c-adaptive-low-weight> --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

| Checkpoint | True NLL | Learned NLL | Random NLL | Shuffled NLL | Learned-true gap | Random-true gap | Shuffled-true gap | True best | Learned best | Operand exact | Calc result acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B handoff | `0.1181` | `3.1823` | `9.9763` | `10.4317` | `3.0642` | `9.8583` | `10.3136` | `0.4844` | `0.0000` | `0.4375` | `0.5000` |
| C-control-repeat | `0.1181` | `5.6422` | `9.9763` | `10.4317` | `5.5241` | `9.8583` | `10.3136` | `0.6875` | `0.0000` | `0.1875` | `0.2969` |
| C-low-lr | `0.1181` | `3.1647` | `9.9763` | `10.4317` | `3.0466` | `9.8583` | `10.3136` | `0.5313` | `0.0000` | `0.4219` | `0.4531` |
| C-adaptive-low-weight | `0.1181` | `3.4270` | `9.9763` | `10.4317` | `3.3089` | `9.8583` | `10.3136` | `0.4844` | `0.0000` | `0.4844` | `0.5000` |

## Parameter deltas

Input-proj L2 / max-abs deltas:

| Checkpoint | vs semantic weight | vs semantic bias | vs Stage B weight | vs Stage B bias |
| --- | --- | --- | --- | --- |
| Stage B handoff | `32.7283 / 2.5145` | `5.1038 / 1.0202` | `0.0000 / 0.0000` | `0.0000 / 0.0000` |
| C-control-repeat | `40.3043 / 3.4755` | `9.5512 / 1.8400` | `13.6511 / 1.1142` | `4.4675 / 0.8399` |
| C-low-lr | `33.9883 / 2.7473` | `5.9916 / 1.1929` | `3.1749 / 0.2437` | `0.9003 / 0.2122` |
| C-very-low-lr | `33.1285 / 2.5967` | `5.4011 / 1.0844` | `1.1427 / 0.0839` | `0.3053 / 0.0795` |
| C-adaptive-low-weight | `33.8827 / 2.7462` | `5.9613 / 1.1902` | `3.1840 / 0.2409` | `0.8723 / 0.2112` |

## Comparison baselines

| Run | Eval exact | Operand exact | Calc result acc | Learned-target agree | Learned-true gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Failed adaptive baseline `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` | `0.0098` | `0.0000` | `0.0156` | `0.0156` | `14.3265` | `0.0000` |
| Lower-LR baseline `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | `0.0664` | `0.0156` | `0.0469` | `0.0703` | `10.4115` | `0.0000` |
| Best previous aux stabilizer `runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2` | `0.0391` | `0.0156` | `0.0469` | `0.0547` | `9.8871` | `0.0000` |
| Stage B retention target | `0.4258` | `0.4844` | `0.5078` | `0.3828` | `3.0642` | `0.0000` |
| Prior Stage C drift | `0.2734` | `0.2578` | `0.3516` | `0.1953` | `5.5241` | `0.0000` |
| C-low-lr, 1000 steps | `0.4492` | `0.5547` | `0.5859` | `0.4766` | `3.0466` | `0.0000` |

## Interpretation and recommendation

Strong positive result:

- `C-low-lr` is a no-supervision Stage C checkpoint after 1000 steps with `final_aux_operand_loss_weight=0.0`.
- It preserves/improves Stage B on built-in eval exact, trace exact, operand exact, calculator result accuracy, learned-target agreement, and action-loss learned-minus-true gap.
- It beats the prior Stage C drift checkpoint decisively: operand exact `0.5547` vs `0.2578`, learned-target agreement `0.4766` vs `0.1953`, learned-true gap `3.0466` vs `5.5241`.
- Counterfactuals still show calculator dependence: canonical injection-zero `0.0`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.9063`.
- True-sum forced-result sweep remains high at `0.9219`.

Limits:

- The canonical causal label remains `causally_useful_opaque_private_code` and `strict_bottleneck_unvalidated`; this is retention of a useful interface, not proof of a clean true-operand protocol.
- Learned actions are still never action-loss best (`learned_best_fraction=0.0`).
- Snapshot metrics fluctuate by held-out sample, so the result should be replicated before treating `0.0003` as fully stable.

Go/no-go:

- Go on pure input-proj Stage C stabilization: lower LR is enough to preserve the Stage B handoff in this run without anchors or true-operand supervision.
- No-go on upstream unfreezing as the immediate next move. First replicate `C-low-lr` across seeds and/or longer Stage C, then consider upstream unfreezing only if the canonical opaque/private-code label remains the blocking issue.
- Do not prioritize anchor variants yet. Keep the new anchor knob available as a fallback if replicated lower-LR runs regress.
