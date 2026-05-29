# 2026-05-29 Bottleneck-to-Additive Continuous Anchor Gate

## Aim

Replace the discrete behavior-gated anchor jump with continuous retention
control keyed to current calculator-result accuracy.

## Code

Changed:

- `scripts/overfit_one_batch.py`
- `tests/test_model.py`

The new `--result-policy-anchor-gate-mode linear` mode ramps the effective
anchor weight from the scheduled/base weight to the gate weight as the selected
metric falls below threshold over `--result-policy-anchor-gate-band`.

## Runs

Run root:

```text
runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_continuous_gated_unfreeze
```

Shared configuration:

- `model-c`, 2-digit exact-grid natural `0..19`.
- `calculator_action_head=result_space`.
- `calculator_bottleneck_mode=none`.
- Continued from adapted weak-source frozen-policy handoff checkpoints.
- Full model checkpoint load, no frozen policy.
- LR `3e-4`, 400 steps, snapshots every 50.
- Base result-policy KL anchor `0.01`.
- Gate metric `current_argmax_accuracy`.
- Gate threshold `0.85`.
- Gate mode `linear`.
- Gate band `0.10`.
- Gate weight `0.1`.

Specific cells:

| Cell | Run directory |
| --- | --- |
| `src4_add2` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_continuous_gated_unfreeze/source_seed4_additive_seed2_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_lineargate0.85_band0.1_to0.1_steps400/2026-05-28_190535_295164_model-c-op0-19-fullgrid/model-c-2digit-seed4` |
| `src5_add5` | `runs/2026-05-29_phase7_bottleneck_to_additive_transfer_policy_anchor_continuous_gated_unfreeze/source_seed5_additive_seed5_adapted_unfreeze_lr3e-4_anchor0.01_calcacc_lineargate0.85_band0.1_to0.1_steps400/2026-05-28_190535_295103_model-c-op0-19-fullgrid/model-c-2digit-seed7` |

## Results

| Run | Final eval | Best normal | Last normal | Last inj-zero | Last forced-random | Last oracle | Last calc | Last anchor weight | Final agreement | Final anchor acc | Gate active rows | Mean effective weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `src4_linear_gate` | `0.8375` | `0.8375` at `400` | `0.8375` | `0.0575` | `0.1175` | `0.8925` | `0.7675` | `0.0550` | `0.8975` | `0.8000` | `9/9` | `0.0385` |
| `src5_linear_gate` | `0.9725` | `0.9525` at `400` | `0.9525` | `0.0000` | `0.0550` | `0.9300` | `0.7600` | `0.0910` | `0.8925` | `0.7600` | `9/9` | `0.0833` |

## Conclusion

The continuous gate worked mechanically and is a useful adaptive-retention
knob. It improved `src4_add2` beyond fixed anchor `0.1` while using much lower
average anchor weight, but `src5_add5` landed slightly below fixed `0.1` and
the discrete accuracy-gated runs.

Label:

```text
bottleneck_to_additive_continuous_anchor_gate_partial
```

## Anti-Rerun Note

Do not repeat base `0.01`, `current_argmax_accuracy` threshold `0.85`, linear
band `0.10`, gate weight `0.1`, LR `3e-4`, 400-step full unfreeze from the
same adapted `src4_add2/src5_add5` checkpoints as novelty.
