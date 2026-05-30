# 2026-05-30 Forced-Margin Second Fresh-Seed Stability

## Question

Does the automated one-negative forced-margin recovery recipe remain a
trusted non-bottleneck handoff positive on another fresh source seed?

This is a stability check of the current staged-transfer benchmark, not a
forced-margin knob-tuning run. The configuration matches the prior automated
recovery recipe: one sampled negative, forced-margin weight `0.5`, start step
`50`, late recovery at step `600` with LR multiplier `0.1` and forced-margin
weight override `0.1`.

## Runs

Fresh source:

```text
runs/2026-05-30_phase7_forced_margin_auto_recovery_seed19/fresh_seed19_source630_cpu
```

Trusted handoff:

```text
runs/2026-05-30_phase7_forced_margin_auto_recovery_seed19/handoff600_from_seed19_step630_cpu
```

The CLI seed was `19`; the run directory records the effective model seed as
`21`, matching the project convention seen in the previous seed-16/effective-18
run.

## Results

Source snapshots:

| Step | Source calc | Snapshot normal | Injection-zero | Oracle | Late recovery |
| ---: | ---: | ---: | ---: | ---: | --- |
| `570` | `0.6400` | `0.6325` | `0.0550` | `1.0000` | off |
| `600` | `0.5625` | `0.5750` | `0.0525` | `1.0000` | on |
| `630` | `0.8325` | `0.8400` | `0.0525` | `1.0000` | on |

Final source eval was `0.8600`. The 128-sample diagnostic summary reported
learned calculator accuracy `0.8516`.

Trusted frozen-policy 600-step additive handoff:

| Handoff step | Normal | Injection-zero | Oracle | Learned calc | Forced-random |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `100` | `0.5600` | `0.0025` | `0.6525` | `0.8300` | `0.0175` |
| `200` | `0.8150` | `0.0000` | `0.9350` | `0.8175` | `0.0175` |
| `300` | `0.8825` | `0.0000` | `0.9550` | `0.8250` | `0.0325` |
| `400` | `0.9000` | `0.0000` | `0.9675` | `0.8450` | `0.0300` |
| `500` | `0.9100` | `0.0000` | `0.9350` | `0.8550` | `0.0350` |
| `600` | `0.9050` | `0.0000` | `0.9400` | `0.8425` | `0.0350` |

Final handoff eval was `0.8975`. The 128-sample diagnostic summary reported
normal `0.9063` and learned calculator accuracy `0.8516`.

## Decision

```text
automated_forced_margin_recovery_second_fresh_seed_mixed_positive
```

Interpretation:

- The late recovery mechanism replicated: source calculator accuracy rose
  sharply in the late window (`0.5625 -> 0.8325`).
- The trusted frozen-policy additive handoff cleared the functional gate with
  low controls: final eval `0.8975`, step-600 normal `0.9050`, zero-injection
  `0.0000`, forced-random `0.0350`.
- The recipe is less seed-stable than the prior very strong run
  (`0.9875` final / `0.9800` step-600 normal). Treat it as a benchmark with
  real variance, not a solved source recipe.
- Do not rerun the same CLI seed-19/effective-seed-21 source plus handoff as
  novelty. Further forced-margin work needs broader scale/stability testing or
  removal of hard assignment / true-result forcing.
