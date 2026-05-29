# 2026-05-29 Handoff Probe Selector on `src4`

## Goal

Apply the validated 600-step handoff-probe selector to the weak `src4` source,
where the final source checkpoint previously produced poor frozen additive
handoff despite high learned calculator-result accuracy.

## Periodic Review

The ledger ruled out:

- repeating the old final-source `src4_add2/src4_add4` frozen 800-step handoffs
  as novelty;
- selecting by source normal/calculator accuracy alone;
- relying on the corrected frozen-state readout probe, which failed.

The allowed direction was to acquire source snapshots and use the 600-step
handoff probe to select a source checkpoint.

## Runs

Run root:

```text
runs/2026-05-29_phase7_handoff_probe_selector_src4
```

Source reproduction:

```text
source_seed4_snapshots_steps1600
```

Configuration:

- bottleneck source, `calculator_bottleneck_mode=answer_decoder`;
- `calculator_estimator=direct_feedback_alignment`;
- `calculator_action_head=result_space`;
- frozen product semantic decoder;
- `result_policy_improvement_assignment_weight=10`;
- exact-grid natural `0..19`;
- CLI seed `4`;
- 1600 steps;
- `--checkpoint-every 100`.

Selector candidates:

| Candidate | Source normal/calc | Selection role |
| --- | ---: | --- |
| final / step `1600` | `0.8700` | old final-source baseline |
| step `1000` | `0.8150` | lower-source-accuracy candidate |
| step `1200` | `0.7550` | lower-source-accuracy candidate |

All transfer cells used:

- additive path, `calculator_bottleneck_mode=none`;
- `calculator_estimator=ste`;
- `calculator_action_head=result_space`;
- compatible checkpoint load from the bottleneck source;
- `--freeze-calculator-policy`;
- answer loss weight `1`;
- exact-grid natural `0..19`;
- additive CLI seed `2`.

## Results

| Setup | Source normal | Normal @ 400 | Normal @ 600 | Normal @ 800 | Injection-zero @ end | Oracle @ end | Calc @ end | Final eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final-source old baseline | `0.8700` | `0.2400` | `0.2675` | `0.3150` | `0.0000` | `0.3125` | `0.8725` | `0.3025` |
| final-source continued old baseline | `0.8700` | `0.4975` | `0.5300` | `0.5675` | `0.0025` | `0.5725` | `0.8725` | `0.6050` |
| step `1000` probe | `0.8150` | `0.3875` | `0.5450` | n/a | `0.0025` | `0.5325` | `0.8075` | `0.5225` |
| step `1200` probe | `0.7550` | `0.5450` | `0.6250` | n/a | `0.0000` | `0.6075` | `0.7875` | `0.6425` |
| step `1200` full | `0.7550` | `0.5450` | `0.6250` | `0.7625` | `0.0000` | `0.7650` | `0.7775` | `0.7800` |

## Conclusion

Label:

```text
bottleneck_to_additive_handoff_probe_selector_src4_positive
```

The 600-step handoff probe selected `src4` step `1200`, a lower-source-accuracy
checkpoint (`0.7550`) than the final source (`0.8700`), and full transfer
confirmed a much better frozen additive handoff: `0.7800` versus the old
final-source `0.3025`.

This also beats the old one-extra-800-step frozen continuation from the final
source (`0.6050`) while preserving calculator dependence: injection-zero stayed
near `0`, oracle tracked normal, and learned calculator-result accuracy stayed
high (`0.7775`).

The result strengthens the claim that source handoff quality is a geometry/time
property, not source action accuracy. The 600-step probe is now validated on
both `src5` and `src4`, but it is still a partial downstream training procedure
and therefore not yet the scalable final solution.

## Anti-Regression Note

Do not repeat the same `src4` step `1000/1200/final`, additive seed `2`,
frozen-policy handoff-probe comparison as novelty. Next useful tests are:

- use 600-step probe score during source acquisition rather than after the
  fact;
- reduce the cost of the 600-step probe;
- test whether probe-selected sources reduce or eliminate later anchoring and
  long-adaptation needs.

## Verification

No code changed in this task. The source reproduction, two 600-step probes, and
one full 800-step confirmation completed and wrote metrics under the run root
above.
