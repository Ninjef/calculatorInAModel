# 2026-05-29 Frozen-State Readout Probe

## Correction

The original scratch analysis for this task used a hardcoded token id for `=`.
The real `EQ_ID` is `11`, while the scratch command used `12`, which selected
a wrong/leaky token position. The reusable script added later fixed this and
invalidated the initial positive interpretation.

The corrected decision is:

```text
bottleneck_to_additive_frozen_state_readout_probe_negative
```

Safe non-answer frozen-state probes did not reliably predict handoff quality.

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

## Corrected Results

| Source | Known final additive handoff | Read-`=` | Read-pair | Layer-1 pair | Layer-2 pair | Best safe probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `src2_final` | `0.9525` | `0.0250` | `0.5125` | `0.5125` | `0.5500` | `0.5500` |
| `src2_step1300` | `0.8675` | `0.0250` | `0.5000` | `0.5000` | `0.5000` | `0.5000` |
| `src5_step1500` | `0.6975` | `0.1625` | `0.3375` | `0.3375` | `0.3375` | `0.3375` |
| `src5_final` | `0.5550` | `0.1750` | `0.3250` | `0.3250` | `0.3375` | `0.3375` |
| `src4_final` | `0.3025` | `0.0125` | `0.5000` | `0.5000` | `0.5000` | `0.5000` |

Correlations with known final additive handoff:

| Probe | Correlation |
| --- | ---: |
| read-`=` residual linear probe | `-0.1218` |
| read-pair residual linear probe | `0.2118` |
| layer-1 pair residual linear probe | `0.2118` |
| layer-2 pair residual linear probe | `0.2865` |
| best safe probe per checkpoint | `0.2865` |

The probe also ranks the `src2` source-selection failure correctly:
`src2_final` scores above `src2_step1300`, even though `src2_step1300` had
higher source normal/calculator accuracy.

## Conclusion

Label:

```text
bottleneck_to_additive_frozen_state_readout_probe_negative
```

The initial positive was an artifact. Correct non-answer probes do not reliably
predict handoff quality: `src4_final` scores near the strong `src2` sources on
pair probes while transferring poorly.

The reusable script remains useful infrastructure, but simple frozen-state
linear sum separability should not replace the 400/600-step handoff probe.

## Anti-Regression Note

Do not repeat this exact five-checkpoint frozen-state readout probe as novelty.
Do not use the wrong-token/leaky answer-position scratch result.
Next useful tests are:

- build a better non-leaky geometry proxy;
- use 400/600-step handoff probes for checkpoint selection until a cheaper
  proxy is proven;
- optimize source acquisition for early additive handoff slope rather than
  source action accuracy alone.

## Verification

`scripts/run_frozen_state_readout_probe.py` now reproduces the corrected safe
probe and avoids checkpoint-snapshot output collisions.
