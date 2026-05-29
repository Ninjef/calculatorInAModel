# 2026-05-29 Scheduled Source Low-LR Recovery

## Purpose

Follow up the scheduled step-600 continuation/readout miss. The prior lineage
had useful additive geometry but learned calculator accuracy stayed around
`0.5391`, and readout plateaued at `0.8475`.

Question:

Can a gentle late-source recovery phase raise source calculator accuracy while
preserving enough scheduled forced-true additive geometry to clear the high
non-bottleneck continuation/readout gate?

## Tooling Note

Two MPS launches from loaded checkpoints wrote only configs and did not produce
training curves in a useful time window. I added `--device {auto,cpu,mps}` to
`scripts/overfit_one_batch.py`, preserving the default MPS-first behavior while
allowing bounded CPU diagnostics. The CPU runs below completed normally.

## Source Recovery

Source checkpoint:

```text
runs/2026-05-29_phase7_additive_forced_true_schedule/op19_long/2026-05-29_131421_175000_model-c-op0-19-fullgrid-direct_feedback_alignment-answer_decoder-adec-product/model-c-2digit-seed15/checkpoint_snapshots/step_00600_weights.pt
```

High-LR smoke:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/smoke_low_aux0p1_from_step600_steps5_cpu
```

Configuration:

- CPU device
- loaded scheduled step-600 source with `full_model` scope
- bottleneck source mode
- source stabilization still active:
  - improvement assignment weight `10`
  - entropy `0.05`
  - batch diversity `0.1`
- forced-true additive weight reduced from `0.5` to `0.1`
- LR `0.003`
- `5` steps

Result: source normal/calc collapsed from `0.5800` at step `0` to `0.1700`
by step `5`; final eval was `0.1550`. This ruled out simply continuing from
the checkpoint at the original LR.

Low-LR recovery:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/lr3e4_low_aux0p1_from_step600_steps30_cpu
```

Changed LR to `0.0003` for all trainable groups and ran `30` steps.

| Step | Source normal / calc | Injection-zero | Oracle | Forced-random | Forced-true additive loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.5800` | `0.0675` | `1.0000` | `0.0175` | `0.0268` |
| `10` | `0.7450` | `0.0500` | `1.0000` | `0.0300` | `0.0233` |
| `20` | `0.7700` | `0.0350` | `1.0000` | `0.0175` | `0.0221` |
| `30` | `0.7950` | `0.0675` | `1.0000` | `0.0125` | `0.0206` |

Final eval was `0.7900`.

## Handoff Verification

Trusted 600-step frozen-policy additive handoff:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/handoff600_from_lr3e4_low_aux_step30_cpu
```

Configuration:

- loaded low-LR recovered step-30 source checkpoint with `compatible_model`
  scope
- additive non-bottleneck mode
- frozen calculator policy
- frozen semantic decoder
- answer loss weight `1`
- LR `0.003`
- `600` steps

| Handoff step | Normal | Injection-zero | Forced-random | Oracle | Learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `0.7925` | `0.0575` | `0.0200` | `0.9900` | `0.7925` |
| `100` | `0.8525` | `0.0425` | `0.0250` | `0.9600` | `0.8225` |
| `600` | `0.8475` | `0.0300` | `0.0175` | `0.8875` | `0.7750` |

Final eval was `0.8425`, improving over the prior scheduled step-600 handoff
final eval `0.7725`.

## Continuation / Readout

800-step continuation from handoff final:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/continuation800_from_recovered_handoff600_cpu
```

Result:

- final eval `0.8900`
- best snapshot `0.9175` at step `700`
- final snapshot controls: normal `0.8525`, injection-zero `0.0200`, oracle
  `0.8675`

600-step readout from continuation final:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/readout600_from_recovered_continuation_cpu
```

Result:

- final eval `0.9320`
- best snapshot `0.9300` at step `100`
- step-600 snapshot controls: normal `0.9025`, injection-zero `0.0200`,
  oracle `0.8700`

Zero-step controls on the readout final checkpoint:

```text
runs/2026-05-29_phase7_scheduled_source_policy_recovery/eval_readout_final_controls_cpu
```

| Normal snapshot | Final eval | Injection-zero | Forced-random | Oracle | Learned calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.9225` | `0.9320` | `0.0300` | `0.0325` | `0.9050` | `0.7925` |

## Decision

```text
scheduled_source_low_lr_recovery_clears_readout_gate
```

## Interpretation

Positive.

The scheduled source branch can clear the high non-bottleneck gate when the
source policy is gently recovered after the geometry-forming phase. This
supports the diagnosis from the previous run: the below-gate readout plateau
was mainly a learned-calculator-quality bottleneck.

The original LR is unsafe for this continuation: a 5-step smoke at `0.003`
collapsed the source policy. The useful late phase used LR `0.0003`, reduced
forced-true weight `0.1`, and kept the hard-improvement source objective
active.

This is still not the final project result. It remains prescriptive and uses a
hand-selected checkpoint plus manual late-source recovery. But it is a genuine
non-bottleneck calculator-dependent gate clear for the scheduled source
geometry branch.

## Anti-Rerun Note

Do not repeat this exact seed-13 chain as novelty:

```text
scheduled step-600 source -> 30-step low-LR aux=0.1 recovery -> 600-step
handoff -> 800-step continuation -> 600-step readout
```

Next useful work:

- replicate the gentle recovery phase on a fresh scheduled source seed;
- automate the late-source transition instead of manually selecting step `600`;
- test whether a trust-region or behavior-gated source phase can replace the
  manual LR drop while preserving the same handoff/readout behavior.

## Verification

Completed:

- `python3 -m py_compile scripts/overfit_one_batch.py`
- `PYTHONPATH=. pytest tests/test_model.py -q -k 'pick_device or result_policy_anchor_weight_schedule'`

Full-suite verification is recorded in the final closeout for this turn.
