# 2026-05-01 - Adaptive calculator interface under strict bottleneck

Task: `aiAgentProjectTasks/2026-05-01-phase-2-first-task-Adaptive-calculator-interface-bottleneck.md`

## Implementation

- Used the existing `calculator_estimator=adaptive_interface` path in `scripts/overfit_one_batch.py`.
- Vectorized `counterfactual_result_targets` so each adaptive batch evaluates forced calculator result classes in packed chunks instead of one forward pass per class.
- Added `scripts/run_track4_action_loss_diagnostic.py` as a compatibility wrapper for the existing phase-prefixed Track 4 script, because tests and older docs import the unprefixed path.

## Headline Run

Strict phase-2 starting regime:

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
```

Semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Adaptive run command:

```bash
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/overfit_one_batch.py --variant model-c --digits 2 --operand-max 19 --calculator-operand-vocab-size 20 --steps 1000 --batch-size 64 --eval-samples 512 --seed 0 --n-layer 2 --n-head 1 --n-embd 16 --mlp-expansion 1 --calculator-hook-after-layer 1 --calculator-read-position operands --calculator-bottleneck-mode answer_decoder --calculator-estimator adaptive_interface --semantic-decoder-checkpoint runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt --snapshot-every 200 --snapshot-samples 64 --log-every 50
```

Output:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
```

Training freeze settings:

- `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder` frozen: yes.
- Upstream encoder frozen: no.
- `calculator_hook.input_proj` trainable: yes.

Parameter movement versus the oracle semantic checkpoint:

| Component | L2 delta | Max abs delta |
| --- | ---: | ---: |
| `calculator_hook.input_proj.weight` | 13.1145 | 0.9286 |
| `calculator_hook.input_proj.bias` | 2.6793 | 0.7138 |
| `calculator_hook.output_proj.weight` | 0.0000 | 0.0000 |
| `answer_offset_emb.weight` | 0.0000 | 0.0000 |
| `answer_decoder.weight` | 0.0000 | 0.0000 |

## Results

| Metric | Value |
| --- | ---: |
| Eval exact match, 512 samples | 0.0098 |
| Diagnostic exact match, 128 samples | 0.0156 |
| Operand exact match | 0.0000 |
| Calculator result accuracy | 0.0156 |
| Adaptive target result accuracy | 0.9375 |
| Learned-target result agreement | 0.0156 |
| Adaptive target operand exact match | 0.1172 |
| Final answer loss | 14.5775 |
| Final adaptive interface CE loss | 193.0297 |

Counterfactuals from built-in 128-sample eval:

| Condition | Exact |
| --- | ---: |
| normal | 0.0156 |
| injection-zero | 0.0078 |
| forced-zero | 0.0000 |
| forced-random | 0.0156 |
| oracle-at-eval | 0.9531 |

Track 3 command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/diagnose_calculator_protocol.py --checkpoint runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/final_weights.pt --digits 2 --operand-max 19 --samples 64 --forced-result-sweep --forced-result-batch-size 64 --leakage-control-exact-match 0.004 --output-dir runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track3_diagnostics
```

Track 3 output:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track3_diagnostics
```

Track 3 classification:

| Category | Bottleneck label | Learned class best | True-sum class best |
| --- | --- | ---: | ---: |
| `calculator_ignored_or_bypassed` | `strict_bottleneck_unvalidated` | 0.0156 | 0.9219 |

The learned calculator result collapsed to class `5` on all 64 Track 3 prompts.

Track 4 command:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. python3 scripts/run_track4_action_loss_diagnostic.py --checkpoint runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/final_weights.pt --samples 64 --random-actions 16 --digits 2 --operand-max 19 --no-work-history
```

Track 4 output:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2/track4_action_loss
```

Track 4 summary:

| Metric | Value |
| --- | ---: |
| Mean true-action NLL | 0.1181 |
| Mean random-action NLL | 9.9763 |
| Mean shuffled-action NLL | 10.4317 |
| Mean learned-forced NLL | 14.4446 |
| Random minus true gap | 9.8583 |
| Shuffled minus true gap | 10.3136 |
| Learned minus true gap | 14.3265 |
| True best fraction | 0.9844 |
| Learned best fraction | 0.0000 |

## Interpretation

This is a useful negative result. The adaptive target selector saw the right downstream result class most of the time: true-sum class was best on `0.9219` of Track 3 prompts and adaptive target result accuracy was `0.9375` on the built-in sample. The downstream semantic decoder remains valid: oracle-at-eval recovers `0.9531`, and Track 4 shows true actions massively beat random and shuffled actions.

The failure is in optimization of the moving interface. `input_proj` changed substantially, but it moved into a collapsed, overconfident constant protocol rather than tracking the adaptive targets. Final adaptive CE exploded to `193.0297`, learned-target agreement stayed near chance, operand exact was zero, and answer accuracy remained at chance.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track3_causal_diagnostics.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_track4_action_loss_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache PYTHONPATH=. pytest -q
```

Result: `49 passed`.

## Recommendation

No-go on claiming adaptive-interface success in this exact setup. Go for targeted variants that stabilize the CE target, such as lower learning rate for `input_proj`, lower adaptive loss weight, entropy regularization, clipping/calibrating adaptive CE, freezing upstream for a diagnostic run, or mixing direct true-operand aux loss only as a stabilizer. Keep the same strict oracle decoder and the same Track 3/Track 4 gates.
