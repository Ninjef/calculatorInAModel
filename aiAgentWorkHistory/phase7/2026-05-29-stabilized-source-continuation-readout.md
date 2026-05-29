# 2026-05-29 Stabilized Source Continuation and Readout

## Aim

The no-decay entropy/diversity source produced high bottleneck calculator
accuracy and weak initial additive handoff. This task tests whether that weak
handoff is a real source failure or whether the frozen calculator signal can be
unlocked by the selected-source continuation/readout recipe.

## Lineage

Source and handoff run root:

```text
runs/2026-05-29_phase7_source_acquisition_stabilization_floor
```

Starting point:

```text
handoff_final_additive_seed9
```

The starting handoff came from the no-decay final source checkpoint:

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final-source handoff | `0.6500` | `0.7025` at step `800` | `0.0000` | `0.0859` | `0.7422` | `0.8750` |

## Readout Compatibility Check

Before running the full continuation chain, a direct 600-step
policy-backbone-frozen readout was run from the handoff checkpoint.

```text
readout_from_final_handoff_seed9_steps600
```

Result:

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct 600-step readout | `0.8000` | `0.8425` at step `600` | `0.0234` | `0.0938` | `0.8281` | `0.8750` |

The direct readout improved the weak handoff but stayed below gate, so the
standard 800-step frozen-policy continuation was still needed.

## Continuation and Readout

Continuation:

```text
continuation_from_final_handoff_seed9_steps800
```

Readout:

```text
readout_from_continuation_seed9_steps600
```

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 800-step continuation | `0.9050` | `0.9350` at step `800` | `0.0078` | `0.0938` | `0.9141` | `0.8750` |
| 600-step readout | `0.9575` | `0.9625` at step `400` | `0.0156` | `0.0859` | `0.9375` | `0.8750` |

## Decision

Label:

```text
stabilized_source_continuation_readout_positive
```

The no-decay stabilized source clears the non-bottleneck gate once the
final-source handoff receives the standard 800-step frozen-policy continuation
and 600-step readout. Controls remain far below normal, so the final answer
path remains calculator-dependent despite the additive residual path.

## Interpretation

This overturns the initial "transfer negative" read of the no-decay source.
The first 800-step handoff was weak, but the frozen calculator policy retained
a high-quality result signal (`0.8750` calc), and continuation made that signal
usable for the additive readout.

The useful selector for this family is not bottleneck source accuracy alone:
source step `1400` had higher source normal (`0.9100`) but much worse handoff
than final. It is also not initial handoff alone, since the final-source
handoff started at only `0.6500` but finished the recipe at `0.9575`.

## Anti-Rerun Note

Do not repeat this exact no-decay final-source handoff plus direct readout,
800-step continuation, and 600-step readout chain as novelty.

Next useful tests:

- replicate no-decay stabilized source continuation/readout on another fresh
  seed;
- reduce continuation cost for no-decay stabilized sources;
- identify a cheaper proxy for continuation/readout slope that can replace
  full downstream probes during source acquisition.

## Verification

Direct readout, frozen-policy continuation, and post-continuation readout all
completed and wrote metrics under the run root above.
