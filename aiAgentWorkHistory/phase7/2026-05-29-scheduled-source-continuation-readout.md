# 2026-05-29 Scheduled Source Continuation/Readout

## Purpose

Test whether the scheduled forced-true step-600 source/handoff lineage can
clear the high non-bottleneck gate after the standard continuation/readout
recipe.

Starting point:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/handoff600_step600
```

The starting handoff from the scheduled source step-600 checkpoint reached:

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 600-step handoff | `0.7725` | `0.7150` @ step `500` | `0.0469` | `0.0313` | `0.7344` | `0.5391` |

## Runs

Continuation:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/continuation_from_step600_handoff_steps800
```

Post-continuation readout:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/readout_from_continuation_steps600
```

Direct readout control:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/direct_readout_from_step600_handoff_steps600
```

Extended readout:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/readout_extended_from_continuation_read600_steps1000
```

Recipes:

- Continuation used `800` frozen-policy additive steps from the handoff final
  checkpoint.
- Readout used policy-backbone freeze at LR `3e-4`.
- The extended readout loaded the 600-step readout final checkpoint and ran
  another `1000` stable-policy readout steps.

## Results

| Stage | Final eval | Best snapshot | Injection-zero | Forced-random | Oracle | Learned calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| handoff step-600 source | `0.7725` | `0.7150` @ step `500` | `0.0469` | `0.0313` | `0.7344` | `0.5391` |
| 800-step continuation | `0.7775` | `0.7750` @ step `800` | `0.0391` | `0.0313` | `0.6484` | `0.5391` |
| direct 600-step readout | `0.7875` | `0.8050` @ step `600` | `0.0391` | `0.0391` | `0.6406` | `0.5391` |
| 600-step readout after continuation | `0.8175` | `0.8075` @ step `600` | `0.0547` | `0.0391` | `0.6250` | `0.5391` |
| extended readout, +1000 steps | `0.8475` | `0.8350` @ step `1000` | `0.0547` | `0.0313` | `0.6328` | `0.5391` |

## Decision

```text
scheduled_source_continuation_readout_partial_below_gate
```

## Interpretation

Mixed-positive, below gate.

The scheduled source step-600 lineage is causally calculator-dependent: controls
remain far below normal through handoff, continuation, and readout. It also
improves with stable readout adaptation, reaching `0.8475` after the extended
readout.

However, the standard continuation/readout recipe does not clear the high
non-bottleneck gate for this lineage. The main limitation appears to be the
calculator signal quality at handoff: learned calc remains around `0.5391`,
and oracle-at-eval drops during readout rather than climbing. This differs from
the no-decay stabilized positive lineage, where continuation/readout worked
with learned calc around `0.8750`.

The scheduled source branch improved additive geometry and handoff, but the
current source checkpoint is not yet a full non-bottleneck solution. Next work
should improve source policy accuracy while preserving the scheduled geometry
benefit, or test whether a later/fresh scheduled source can combine high source
calc with strong handoff.

## Anti-Rerun Note

Do not repeat this exact scheduled step-600 handoff -> 800 continuation ->
600 readout -> extra 1000 readout chain as novelty.

Allowed next tests:

- Add policy-retention/source-accuracy pressure during scheduled source
  training while keeping the forced-true geometry objective.
- Replicate scheduled source on a fresh seed only if the explicit question is
  whether higher learned-calc handoff emerges naturally.
- Try continuation/readout only after a scheduled source checkpoint shows both
  strong handoff geometry and materially higher learned calculator accuracy.

