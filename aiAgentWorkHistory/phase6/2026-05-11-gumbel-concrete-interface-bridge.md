# Gumbel/Concrete Hard-Forward Interface Bridge

## Claim Tested

Can answer loss through a hard-forward / soft-backward relaxed calculator
signal train the strict identifiable calculator-query protocol without true
operand labels, oracle operands during training, hard-best local CE, or exact
expected answer-loss optimization?

## Code Changes

- Added `calculator_estimator=gumbel_concrete_interface`.
- Added deterministic and Gumbel relaxed operand distributions in
  `src/model.py`.
- Added differentiable sum-distribution construction by convolving independent
  operand distributions.
- For `calculator_output_format=sum_left_operand`, the relaxed soft signal is
  `concat(p_sum, p_a)`.
- The default relaxed path uses hard-forward / soft-backward:
  `hard_signal.detach() + soft_signal - soft_signal.detach()`.
- Added overfit CLI knobs:
  `--relaxed-calculator-temperature`,
  `--relaxed-calculator-final-temperature`,
  `--relaxed-calculator-temperature-decay-steps`,
  `--relaxed-calculator-mode deterministic|gumbel`,
  `--relaxed-calculator-hard-forward`,
  `--relaxed-calculator-entropy-weight`, and
  `--relaxed-calculator-entropy-decay-steps`.
- Added `scripts/run_phase6_gumbel_concrete_interface_bridge.py` for the Stage 0
  one-step gradient gate.
- Added a unit test proving the relaxed signal is hard in the forward pass but
  still gives nonzero gradients to operand logits.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/run_phase6_gumbel_concrete_interface_bridge.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
76 passed
```

## Run Root

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge
```

The repo-local Stage 0B checkpoint was absent, so all runs used the recorded
Phase 4 checkpoint:

```text
/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

## Stage 0 Gradient Gate

Output:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage0/gradient_gate_temp2.json
```

Command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/run_phase6_gumbel_concrete_interface_bridge.py stage0-gradient-gate --samples 128 --temperature 2.0 --mode deterministic --chunk-size 64 --output runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage0/gradient_gate_temp2.json
```

| Metric | Value |
| --- | ---: |
| oracle / injection-zero / forced-random | `1.000 / 0.000 / 0.000` |
| initial answer loss | `10.8585` |
| initial hard pair / calc | `0.000 / 0.0078` |
| entropy / effective pairs | `5.9915 / 399.999` |
| full-enum best=true | `1.000` |
| best-pair probability delta after one step | `+0.0000209` |
| gradient cosine, relaxed answer vs hard-best CE | `+0.2345` |
| input-proj / upstream / semantic delta L2 | `1.0896 / 0.0 / 0.0` |
| semantic decoder grad L2 | `0.0` |

Gate decision: pass.

## Stage 1 Frozen-Upstream Relaxed Training

Branch A reached the fast gate, so Branches B-D were skipped per the task's
early-stop rule.

Command shape:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 300 --batch-size 64 --eval-samples 512 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator gumbel_concrete_interface --semantic-decoder-checkpoint <stage0b> --semantic-decoder-checkpoint-load-scope semantic_decoder_only --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --adaptive-interface-loss-weight 0.0 --aux-operand-loss-weight 0.0 --expected-answer-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.03 --upstream-lr 0.0003 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum_left_operand --answer-format sum_left_operand --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --relaxed-calculator-temperature 2.0 --relaxed-calculator-final-temperature 0.5 --relaxed-calculator-temperature-decay-steps 300 --relaxed-calculator-mode deterministic --relaxed-calculator-hard-forward --relaxed-calculator-entropy-weight 0.0 --snapshot-every 25 --checkpoint-every 25 --snapshot-samples 128 --run-root runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage1_branch_a_temp2_to_05 --log-every 25
```

| Branch | Mode | Temperature | Best snapshot | Final snapshot | Final eval |
| --- | --- | --- | ---: | ---: | ---: |
| A | deterministic | `2.0 -> 0.5` | step `200`: `1.000 / 1.000 / 1.000 / 1.000` | step `300`: `0.789 / 0.789 / 0.789 / 0.789` | `0.873` |

Stage 1 selected checkpoint:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage1_branch_a_temp2_to_05/2026-05-11_154207_114387_model-c-op0-19-gumbel_concrete_interface-inlr0.03-uplr0.0003-rtemp2-rfinal0.5-rdecay300-answer_decoder-sum_left_operand/model-c-2digit-seed2/checkpoint_snapshots/step_00200_weights.pt
```

Stage 1 parameter deltas from step `0` to step `200`:

```text
calculator_hook.input_proj L2/max = 26.7822 / 2.4764
upstream L2/max = 0.0 / 0.0
semantic decoder L2/max = 0.0 / 0.0
```

## Stage 2 Relaxation-Off Retention

Two retention runs were launched, from the first qualifying checkpoint
`step_00175` and the best qualifying checkpoint `step_00200`.

Command shape:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --steps 1000 --batch-size 64 --eval-samples 512 --operand-max 19 --calculator-operand-vocab-size 20 --calculator-estimator adaptive_interface --semantic-decoder-checkpoint <stage1-checkpoint> --semantic-decoder-checkpoint-load-scope full_model --freeze-semantic-decoder --freeze-upstream-encoder --answer-loss-weight 1.0 --adaptive-interface-loss-weight 0.0 --aux-operand-loss-weight 0.0 --expected-answer-loss-weight 0.0 --input-proj-anchor-weight 0.0 --input-proj-lr 0.0003 --upstream-lr 0.0003 --calculator-read-position operand_spans --calculator-read-span-width 2 --calculator-bottleneck-mode answer_decoder --calculator-output-format sum_left_operand --answer-format sum_left_operand --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --snapshot-every 50 --checkpoint-every 50 --snapshot-samples 128 --log-every 50
```

| Source | Step 0 snapshot | Best snapshot | Final snapshot | Final eval |
| --- | ---: | ---: | ---: | ---: |
| first qualifying step `175` | `0.883 / 0.883 / 0.883 / 0.883` | step `50`: `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |
| best step `200` | `0.992 / 0.992 / 0.992 / 0.992` | step `50`: `1.000 / 1.000 / 1.000 / 1.000` | `1.000 / 1.000 / 1.000 / 1.000` | `1.000` |

Selected retained checkpoint:

```text
runs/2026-05-11_phase6_gumbel_concrete_interface_bridge/stage2_retention_best_step200/2026-05-11_154745_323390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt
```

Selected diagnostics:

| Diagnostic | Result |
| --- | ---: |
| canonical normal / oracle / injection-zero / forced-random | `1.000 / 1.000 / 0.0039 / 0.0313` |
| canonical operand / pair / calc | `1.000 / 1.000 / 1.000` |
| private answer / operand / pair / calc | `1.000 / 1.000 / 1.000 / 1.000` |
| full-enum learned / true / best NLL | `0.0002 / 0.0002 / 0.0002` |
| full-enum learned-minus-true / best gap | `0.0 / 0.0` |
| full-enum learned-best / true-best | `1.000 / 1.000` |

Final objective weights:

```text
relaxed calculator objective inactive
aux_operand_loss_weight=0.0
adaptive/local target weight=0.0
expected_answer_loss_weight=0.0
input_proj_anchor_weight=0.0
semantic decoder delta=0.0
upstream delta=0.0
```

## Interpretation

This is a strong positive for the hard-forward / soft-backward relaxed bridge.
The deterministic relaxed answer-loss path trained the strict
`semantic_decoder_only` interface to an exact hard calculator-query protocol
without direct true-operand supervision, oracle operands during training, or
hard-best CE. The retained protocol stayed exact after the relaxation was fully
off.

Compared with the exact expected answer-loss negative, the key difference is
that the answer loss flowed through the same hard-forward calculator signal
used by the answer decoder instead of optimizing detached full-enum expected
cost over a policy that could collapse to the wrong hard argmax.

## Recommendation

Treat this as the first strong Phase 6 discovery-positive branch. Next useful
work should stress-test stability across seeds or test whether the same bridge
can tolerate carefully opened upstream parameters. Do not rerun oracle-only
controls.
