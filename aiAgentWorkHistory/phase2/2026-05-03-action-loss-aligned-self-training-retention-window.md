# 2026-05-03 - Action-loss-aligned self-training under retention window

Task: `aiAgentProjectTasks/2026-05-03-phase-2-seventh-task-Action-loss-aligned-self-training-under-retention-window.md`.

## Code changes

- Added `--checkpoint-every` to `scripts/overfit_one_batch.py`. When used with `--snapshot-every`, it saves `checkpoint_snapshots/step_XXXXX_weights.pt` containing `model_state_dict`, the training `config`, the snapshot row, and the step.
- Added `calculator_estimator=action_loss_weighted_interface`. It trains only `calculator_hook.input_proj` toward a soft target over learned/top-k/local/random candidate actions ranked by frozen answer-decoder NLL. The target uses answer loss only; it does not use true operands.
- Added `scripts/run_action_loss_candidate_diagnostic.py` to estimate whether candidate pools contain actions that beat the current learned action before running self-training.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
```

Result: `53 passed`.

## Dense Stage C runs

All primary dense runs used the strict phase-2 setup: `digits=2`, `operand_max=19`, `calculator_operand_vocab_size=20`, `n_layer=2`, `n_head=1`, `n_embd=16`, `mlp_expansion=1`, hook after layer `1`, read position `operands`, bottleneck `answer_decoder`, estimator `adaptive_interface`, Stage B handoff as semantic decoder checkpoint, `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`, `answer_loss_weight=1.0`, `aux_operand_loss_weight=0.0`, `input_proj_anchor_weight=0.0`, `input_proj_lr=0.0003`, `upstream_lr=0.0003`, `snapshot_every=50`, and `checkpoint_every=50`.

Common command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --adaptive-interface-target-mode hard_pair --aux-operand-loss-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0 --steps 1500 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 128 --log-every 50
```

| Dense run | Extra args | Run path | Final eval exact | Best built-in normal snapshot | Best action-loss snapshot |
| --- | --- | --- | ---: | --- | --- |
| C-low-lr-dense-seed1 | `--seed 1` | `runs/2026-05-03_112750_450950_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3` | `0.4688` | step `550`, normal `0.5781` | step `100`, gap `2.1383` |
| C-low-lr-dense-seed2 | `--seed 2` | `runs/2026-05-03_114747_070474_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed4` | `0.4707` | step `450`, normal `0.5938` | step `550`, gap `2.2018` |
| C-low-lr-dense-seed3 | `--seed 3` | `runs/2026-05-03_114747_345486_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed5` | `0.4180` | step `1300`, normal `0.5547` | step `1050`, gap `2.4347` |

Proof constraints: all dense `metrics.json` files have `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_upstream_encoder=true`, and `trainable_parameter_groups=[calculator_hook.input_proj]`.

## Checkpoint selection

Canonical action-loss diagnostics used:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py --checkpoint <checkpoints> --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

| Run | Final learned-true gap | Selected learned-true gap | Selected operand exact | Selected calc result acc | Learned best |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense seed1 | `2.8325` | `2.1383` | `0.5625` | `0.5938` | `0.0000` |
| Dense seed2 | `2.8509` | `2.2018` | `0.5469` | `0.6094` | `0.0000` |
| Dense seed3 | `3.3582` | `2.4347` | `0.5625` | `0.5938` | `0.0000` |

Selection by canonical action-loss learned-minus-true gap beat the final checkpoint in all three dense runs. Built-in normal-exact selection also beat final eval in all three runs, but did not always pick the same checkpoint as action-loss selection.

Selected dense checkpoint causal diagnostics for seed1 step `100`:

- Normal `0.3750`, injection-zero `0.0000`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.9063`.
- Classification stayed `causally_useful_opaque_private_code`; bottleneck stayed `strict_bottleneck_unvalidated`.
- Forced-result learned-best fraction `0.3750`; true-sum-best fraction `0.9219`.
- Private protocol all-pair operand exact `0.5275`, calculator-result accuracy `0.5425`, answer exact `0.5025`.
- Group behavior: no-carry/small operands are much easier than carry/large operands. No-carry operand exact `0.8000`; small-operands operand exact `0.6800`; carry operand exact `0.4841`; large-operand exact `0.4767`.

## Candidate-action diagnostic

Command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_action_loss_candidate_diagnostic.py --checkpoint <checkpoints> --samples 128 --batch-size 64 --random-actions 8 --topk 2 --local-radius 1 --digits 2 --operand-max 19
```

| Checkpoint | Better fraction | Mean best improvement | Best result acc |
| --- | ---: | ---: | ---: |
| Stage B handoff | `0.570` | `2.8486` | `0.914` |
| Prior best lower-LR replication | `0.508` | `2.2891` | `0.883` |
| Dense seed1 built-in selected step 550 | `0.531` | `2.7905` | `0.891` |
| Dense seed2 built-in selected step 450 | `0.523` | `2.9544` | `0.891` |
| Dense seed3 built-in selected step 1300 | `0.570` | `3.5481` | `0.852` |

The candidate pool often contains better answer-loss actions, so the objective is not candidate-limited at this pool size.

## Action-loss self-training

Self-training command stem:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --batch-size 64 --eval-samples 512 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator action_loss_weighted_interface --semantic-decoder-checkpoint runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --aux-operand-loss-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --adaptive-interface-loss-weight 1.0 --action-loss-candidate-random 4 --action-loss-candidate-topk 1 --action-loss-candidate-local-radius 1 --action-loss-candidate-temperature 1.0 --steps 1000 --snapshot-every 100 --snapshot-samples 128
```

| Run | Extra args | Run path | Final eval exact | Learned-true gap | Operand exact | Calc result acc | Learned best |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| C-actionloss-selftrain-seed1 | `--seed 1` | `runs/2026-05-03_154205_948410_model-c-op0-19-action_loss_weighted_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-answer_decoder/model-c-2digit-seed3` | `0.4941` | `3.0137` | `0.4531` | `0.5000` | `0.0000` |
| C-actionloss-selftrain-seed2 | `--seed 2` | `runs/2026-05-03_154959_116705_model-c-op0-19-action_loss_weighted_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-answer_decoder/model-c-2digit-seed4` | `0.5449` | `1.9814` | `0.6094` | `0.6406` | `0.0000` |
| C-actionloss-selftrain-seed3 | `--seed 3` | `runs/2026-05-03_154958_668369_model-c-op0-19-action_loss_weighted_interface-inlr0.0003-uplr0.0003-alrand4-altop1-alloc1-alt1-answer_decoder/model-c-2digit-seed5` | `0.4102` | `3.2365` | `0.4375` | `0.4844` | `0.0000` |

Best self-training checkpoint, seed2:

- Canonical causal normal `0.4688`, injection-zero `0.0000`, forced-zero `0.0156`, forced-random `0.0625`, oracle-at-eval `0.9063`.
- Classification improved to `semantically_decodable_private_calculator_code`, bottleneck still `strict_bottleneck_unvalidated`.
- Forced-result learned-best fraction `0.4688`, true-sum-best fraction `0.9219`.
- Private protocol all-pair answer exact `0.5375`, operand exact `0.5500`, calculator-result accuracy `0.5775`.
- Best affine mapping remained identity-like: A exact `0.9000`, B exact `0.6025`.
- Group behavior remained uneven: no-carry operand exact `0.6364`, small-operands `0.5900`, carry `0.5362`, large-operand `0.5367`.

Proof constraints: all self-training runs have `final_aux_operand_loss_weight=0.0`, `final_input_proj_anchor_weight=0.0`, `freeze_upstream_encoder=true`, and `trainable_parameter_groups=[calculator_hook.input_proj]`.

## Decision

Checkpoint selection is useful. Action-loss-selected snapshots beat final checkpoints in all three dense runs on canonical learned-minus-true gap, without increasing aux supervision, unfreezing upstream, or adding an anchor.

Candidate action search contains real answer-NLL signal, but the current self-training objective is not robust. One of three self-training runs was strong and improved over the best prior replication on learned-minus-true gap (`1.9814` vs `2.4786`) and operand/result structure (`0.6094`/`0.6406` canonical action-loss rows; all-pair `0.5500`/`0.5775`). The other two runs did not improve, and learned-best action-loss fraction stayed `0.0` for every run.

Recommendation: do not unfreeze upstream yet. Next path should be checkpoint selection plus a better action-loss objective or candidate search, likely with lower-variance targets, candidate replay, or selecting/continuing from action-loss-selected snapshots. The first answer-NLL-derived training signal exists, but it is not reliable enough to justify upstream unfreezing.
