# 2026-05-29 Conjunctive Recovery Trigger

## Question

Can the hard seed-17 adaptive recovery transition improve if recovery requires
both the best forced-loss readiness signal and a source-policy accuracy gate?

This tests whether the smoothed forced-loss trigger fired too early because it
ignored source policy maturity.

## Code Change

Added optional secondary trigger support to `scripts/overfit_one_batch.py`:

- `--late-source-recovery-secondary-trigger-metric`
- `--late-source-recovery-secondary-trigger-threshold`
- `--late-source-recovery-secondary-trigger-mode`
- `--late-source-recovery-secondary-trigger-ema-beta`
- `--late-source-recovery-secondary-trigger-patience`

The secondary trigger is opt-in. Existing one-trigger and fixed-step recovery
runs keep the previous behavior.

## Runs

Source:

```text
runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_adaptive_conj_forcedloss005_ema08_pat10_acc070_steps631_cpu/2026-05-29_155830_899753_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed19
```

Handoff:

```text
runs/2026-05-29_phase7_scheduled_source_adaptive_recovery_replication/seed17_handoff600_from_adaptive_conj_forcedloss005_ema08_pat10_acc070_cpu/2026-05-29_160309_184624_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed19
```

Trigger configuration:

```text
primary: additive_forced_true_loss <= 0.05, EMA beta 0.8, patience 10, min step 500
secondary: result_policy_argmax_result_accuracy >= 0.70, raw, patience 1
```

## Results

The conjunctive trigger never fired.

| Metric | Value |
| --- | ---: |
| source final exact-match | `0.6100` |
| primary forced-loss final EMA | `0.0055` |
| primary trigger count | `132` |
| secondary source-accuracy final value | `0.6325` |
| secondary trigger count | `0` |
| handoff final exact-match | `0.6825` |
| handoff step-600 normal | `0.6925` |
| handoff injection-zero | `0.0400` |
| handoff forced-random | `0.0500` |
| handoff learned calculator accuracy | `0.6075` |

## Comparison

| Seed-17 branch | Source final | Handoff final | Trigger |
| --- | ---: | ---: | --- |
| raw source-accuracy trigger | `0.6100` | `0.6825` | none |
| raw forced-loss trigger | `0.7225` | `0.7625` | step `500` |
| fixed step-600 control | `0.7450` | `0.7675` | step `600` |
| forced-loss EMA/patience | `0.7625` | `0.8025` | step `509` |
| forced-loss EMA plus source accuracy `>=0.70` | `0.6100` | `0.6825` | none |

## Interpretation

The primary forced-loss signal was ready, but the hard source-accuracy gate was
too conservative. On seed 14, source accuracy thresholding could fire early and
raise zero/random controls; on seed 17, a high hard gate recreates the no-fire
failure mode.

Do not repeat this exact threshold combination as novelty. Future work should
return to scalable assignment or train source objectives against
handoff/readout geometry more directly unless a genuinely different transition
signal is proposed up front.

## Verification

```text
python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. pytest tests/test_model.py -q -k late_source_recovery
```
