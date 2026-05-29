# 2026-05-29 Frozen-State Readout Probe

## Goal

Find a cheaper proxy for the short additive handoff probe. The previous audit
showed that 400/600-step additive handoff progress predicts final handoff, but
that still requires partial downstream training. This task tested whether a
small linear probe on frozen additive-compatible source states can predict the
same handoff quality.

## Periodic Review

The ledger ruled out repeating full 800-step frozen-policy transfer cells,
source-normal-accuracy checkpoint selection, or the short trace audit as
novelty. The allowed direction was a cheaper readout/linear proxy for handoff
geometry.

## Probe

For each source checkpoint:

1. Load the bottleneck source checkpoint into an additive-compatible model with
   `calculator_bottleneck_mode=none`, `answer_decoder_interaction=none`, and
   `calculator_estimator=ste`.
2. Load only compatible checkpoint tensors, matching the frozen-policy transfer
   setup.
3. Run the exact `0..19` grid once with `return_diagnostics=True`.
4. Train a tiny linear classifier on frozen states to predict the true sum
   class, using a deterministic `320/80` train/eval split.

Answer-position features were discarded after inspection because teacher-forced
full-sequence inputs leak answer tokens at those positions. The clean probe
uses only the `=` read position.

## Results

| Source | Known final additive handoff | Read-`=` probe eval | Layer-2 `=` probe eval |
| --- | ---: | ---: | ---: |
| `src2_final` | `0.9525` | `0.5875` | `0.5750` |
| `src2_step1300` | `0.8675` | `0.5000` | `0.5125` |
| `src5_step1500` | `0.6975` | `0.3125` | `0.3125` |
| `src5_final` | `0.5550` | `0.2625` | `0.2625` |
| `src4_final` | `0.3025` | `0.1625` | `0.1625` |

Correlations with known final additive handoff:

| Probe | Correlation |
| --- | ---: |
| read-`=` residual linear probe | `0.9643` |
| layer-2 `=` residual linear probe | `0.9659` |

The probe also ranks the `src2` source-selection failure correctly:
`src2_final` scores above `src2_step1300`, even though `src2_step1300` had
higher source normal/calculator accuracy.

## Conclusion

Label:

```text
bottleneck_to_additive_frozen_state_readout_probe_partial
```

A frozen-state linear readout probe is a promising cheaper proxy for handoff
geometry. In this small audit, it tracked known final handoff better than
source normal/calculator accuracy and did not require hundreds of downstream
adaptation steps.

This is not yet a validated selector. The sample is small, the probe trains on
the exact grid, and it still uses supervised sum labels. It should be validated
against new source checkpoints before replacing the 400/600-step handoff probe.

## Anti-Regression Note

Do not repeat this exact five-checkpoint frozen-state readout probe as novelty.
Next useful tests are:

- use the readout probe to select among unseen source checkpoints, then confirm
  with a short or full additive transfer;
- turn the one-off probe into a reusable diagnostic script if it remains useful;
- optimize source acquisition for the readout-probe score instead of source
  action accuracy alone.

## Verification

No code changed. The probe was run as an analysis command over existing source
checkpoints and used only frozen model states.
