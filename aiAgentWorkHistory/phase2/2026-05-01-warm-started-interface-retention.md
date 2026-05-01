# 2026-05-01 - Warm-started interface retention under strict bottleneck

Task: `aiAgentProjectTasks/2026-05-01-phase-2-fourth-task-Warm-started-interface-retention.md`.

## Code changes

- Added `--answer-loss-weight` to `scripts/overfit_one_batch.py` so Stage A can be a clean supervised interface warm start instead of mixing answer loss into the setup objective.
- Recorded `trainable_parameter_groups` in `config.json` and `metrics.json`.
- Recorded final aux operand CE, final aux weight, and `aux_operand_loss_grad_upstream` in `metrics.json`.
- Preserved the pre-existing `--aux-operand-loss-grad-upstream` worktree change and covered the new metadata in `tests/test_model.py`.

Verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q tests/test_model.py -k "training_cli_supports_oracle_warmup_and_snapshots or aux_operand_weight"
```

Result: `2 passed, 45 deselected`.

## Setup

Semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Shared strict-bottleneck config:

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
freeze_upstream_encoder=true for retained primary stages
trainable_parameter_groups=[calculator_hook.input_proj]
```

## Setup attempts before the retained run

Two mixed-objective warm starts failed to make the setup condition real:

| Attempt | Run | Trainable | Objective | Outcome |
| --- | --- | --- | --- | --- |
| Input-proj only, mixed answer+aux | `runs/2026-05-01_111104_301738_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1/model-c-2digit-seed2` | `calculator_hook.input_proj` | answer weight `1.0`, aux `1.0`, adaptive `0.0` | final snapshot operand exact `0.016`, eval exact `0.059`; no warm start |
| Upstream trainable, mixed answer+aux | `runs/2026-05-01_111426_155793_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1/model-c-2digit-seed2` | input proj + upstream | answer weight `1.0`, aux `1.0`, adaptive `0.0`, aux gradients upstream | diverged; final aux CE `377.7`, operand exact `0.000` |
| Upstream trainable, lower LR | `runs/2026-05-01_111741_001165_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux1/model-c-2digit-seed2` | input proj + upstream | answer weight `1.0`, aux `1.0`, adaptive `0.0`, aux gradients upstream | stable but no warm start; final snapshot operand exact `0.000`, eval exact `0.049` |

Conclusion from setup attempts: answer loss during Stage A interfered with the direct true-operand warm start. The retained run therefore used `--answer-loss-weight 0.0` only for Stage A.

## Primary staged run

Stage A command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 500 --batch-size 64 --eval-samples 512 --seed 0 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt --freeze-semantic-decoder --freeze-upstream-encoder --input-proj-lr 0.003 --upstream-lr 0.003 --answer-loss-weight 0.0 --adaptive-interface-loss-weight 0.0 --adaptive-interface-target-mode hard_pair --aux-operand-loss-weight 1.0 --aux-operand-loss-decay-steps 0 --aux-operand-loss-floor 0.0 --snapshot-every 250 --snapshot-samples 64 --log-every 50
```

Stage B command: same architecture, loading Stage A `final_weights.pt`, with `--answer-loss-weight 1.0 --adaptive-interface-loss-weight 1.0 --aux-operand-loss-weight 1.0 --aux-operand-loss-decay-steps 500`.

Stage C command: same architecture, loading Stage B `final_weights.pt`, with `--answer-loss-weight 1.0 --adaptive-interface-loss-weight 1.0 --aux-operand-loss-weight 0.0`.

Run paths:

| Stage | Role | Run path | Supervision at end | Trainable parameters |
| --- | --- | --- | ---: | --- |
| A | warm-start handoff | `runs/2026-05-01_112203_955074_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1/model-c-2digit-seed2` | `1.0` | `calculator_hook.input_proj.weight`, `.bias` |
| B | supervision-zero handoff | `runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2` | `0.0` | `calculator_hook.input_proj.weight`, `.bias` |
| C | final adaptive-only | `runs/2026-05-01_112843_524620_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2` | `0.0` | `calculator_hook.input_proj.weight`, `.bias` |

## Checkpoint metrics

Built-in final metrics and 128-sample trace summaries:

| Stage | Eval exact | Trace exact | Operand exact | Calc result acc | Target result acc | Learned-target agreement | Adaptive CE | Aux CE / weight | Mean entropy A+B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| A | `0.3809` | `0.3594` | `0.2500` | `0.3750` | `0.9375` | `0.2969` | `2.7615` | `2.7462 / 1.0` | `5.9786` |
| B | `0.4258` | `0.4922` | `0.4844` | `0.5078` | `0.9375` | `0.3828` | `2.7629` | `2.7502 / 0.0` | `5.9727` |
| C | `0.2734` | `0.3438` | `0.2578` | `0.3516` | `0.9375` | `0.1953` | `2.7613` | `2.7613 / 0.0` | `5.9644` |

Canonical causal diagnostics, 64 samples:

| Stage | Normal | Injection-zero | Forced-zero | Forced-random | Oracle-at-eval | Operand exact | Calc result acc | Classification | Learned-best | True-sum-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| A | `0.2656` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.2188` | `0.2969` | `causally_useful_opaque_private_code` | `0.2656` | `0.9219` |
| B | `0.3750` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.4219` | `0.4219` | `causally_useful_opaque_private_code` | `0.3750` | `0.9219` |
| C | `0.2500` | `0.0000` | `0.0156` | `0.0625` | `0.9063` | `0.2031` | `0.2656` | `causally_useful_opaque_private_code` | `0.2500` | `0.9219` |

Canonical action-loss diagnostics, 64 samples:

| Stage | True NLL | Learned NLL | Random NLL | Shuffled NLL | Learned-true gap | Random-true gap | Shuffled-true gap | True best | Learned best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `0.1181` | `4.2483` | `9.9763` | `10.4317` | `4.1302` | `9.8583` | `10.3136` | `0.6563` | `0.0000` |
| B | `0.1181` | `3.1823` | `9.9763` | `10.4317` | `3.0642` | `9.8583` | `10.3136` | `0.4844` | `0.0000` |
| C | `0.1181` | `5.6422` | `9.9763` | `10.4317` | `5.5241` | `9.8583` | `10.3136` | `0.6875` | `0.0000` |

Parameter movement versus the original semantic decoder checkpoint:

| Stage | `input_proj.weight` L2/max | `input_proj.bias` L2/max | Frozen semantic deltas |
| --- | --- | --- | --- |
| A | `28.0190 / 1.4945` | `0.3532 / 0.1450` | output proj, answer offset, answer decoder all `0.0` |
| B | `32.7283 / 2.5145` | `5.1038 / 1.0202` | output proj, answer offset, answer decoder all `0.0` |
| C | `40.3043 / 3.4755` | `9.5512 / 1.8400` | output proj, answer offset, answer decoder all `0.0` |

## Comparison to prior baselines

| Run | Eval exact | Operand exact | Calc result acc | Learned-target agree | Learned-true gap | Learned-best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Failed adaptive baseline `runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2` | `0.0098` | `0.0000` | `0.0156` | `0.0156` | `14.3265` | `0.0000` |
| Lower-LR baseline `runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2` | `0.0664` | `0.0156` | `0.0469` | `0.0703` | `10.4115` | `0.0000` |
| Best previous aux stabilizer `runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2` | `0.0391` | `0.0156` | `0.0469` | `0.0547` | `9.8871` | `0.0000` |
| This Stage B zero handoff | `0.4258` | `0.4844` | `0.5078` | `0.3828` | `3.0642` | `0.0000` |
| This Stage C final adaptive-only | `0.2734` | `0.2578` | `0.3516` | `0.1953` | `5.5241` | `0.0000` |

## Interpretation

Positive evidence:

- A clean aux-only Stage A made a nontrivial operand protocol using `input_proj` alone.
- Stage B reached true-operand supervision weight exactly `0.0` while improving operand exact and calculator-result accuracy.
- Stage C final evaluation was after supervision weight `0.0` and remained materially above prior adaptive-interface baselines on eval exact, operand exact, calculator-result accuracy, learned-target agreement, and learned-minus-true action-loss gap.
- Counterfactuals show calculator dependence: Stage C normal `0.25` versus injection-zero `0.0`, forced-zero `0.0156`, forced-random `0.0625`, and oracle-at-eval `0.9063`.

Limits:

- The protocol degraded during Stage C: trace operand exact dropped from `0.4844` at Stage B to `0.2578` final, and action-loss learned-true gap worsened from `3.0642` to `5.5241`.
- Learned actions still never became action-loss best (`learned_best_fraction=0.0`).
- Canonical causal classification remains `causally_useful_opaque_private_code`, not a clean validated true-operand protocol.
- The forced-result sweep still strongly prefers true sums (`0.9219`) over learned classes (`0.25` final).

Go/no-go recommendation:

- Go for follow-up retention work: warm-started `input_proj` can preserve a materially better calculator-dependent interface after direct supervision is removed.
- No-go on claiming solved adaptive-interface learning: Stage C shows partial retention with degradation, not stable true-operand protocol maintenance.
- Next run should extend Stage C monitoring with lower `input_proj_lr` during adaptive-only continuation, and only then test unfreezing upstream after the retained input-proj protocol remains stable.
