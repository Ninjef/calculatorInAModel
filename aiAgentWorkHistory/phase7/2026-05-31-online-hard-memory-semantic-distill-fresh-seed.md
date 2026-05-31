# 2026-05-31 Online Hard Memory Semantic-Distill Fresh Seed

## Question

Does online hard memory plus additive semantic distillation replicate on a fresh
seed, including the trusted frozen-policy additive handoff?

The previous run fixed the source-only online-hard-memory handoff failure on
one op19 fixed-grid seed. This test repeats the same method on CLI seed `7`
without changing weights, sample counts, or run length.

## Source Run

```text
runs/ohm_semdist_fresh_src800/2026-05-30_194915_191676_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rbtonlinehardmem-rbtmem-44b570a4cc/model-c-2digit-seed9
```

Settings:

- CLI seed `7`, effective model seed `9`.
- op19 exhaustive `400`-prompt grid.
- `result_boundary_target_mode=zero_improvement`.
- `topk8+unique24` sparse candidate scoring.
- Online hard memory with freeze-when-full.
- Additive semantic distillation weight `1`, sample count `8`.
- Source mode `calculator_bottleneck_mode=answer_decoder`.
- `800` source steps.

Results:

| Step | Normal | Zero-inj | Calc | Distill agree | Forced evals |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0200 | 0.0275 | 0.0200 | 0.0340 | active |
| 200 | 0.4725 | 0.0400 | 0.4725 | 0.4410 | 76,800 |
| 400 | 0.8825 | 0.0475 | 0.8825 | 0.6450 | 76,800 |
| 600 | 0.9875 | 0.0450 | 0.9875 | 0.7317 | 76,800 |
| 800 | 1.0000 | 0.0575 | 1.0000 | 0.7403 | 76,800 |

Final source metrics:

- Final eval: `400/400 = 1.0000`.
- Diagnostic calculator-result accuracy: `1.0000`.
- 128-sample counterfactuals: injection-zero `0.0703`, forced-zero `0.0078`,
  forced-random `0.0156`.

## Trusted Handoff

```text
runs/ohm_semdist_fresh_handoff600/2026-05-30_195503_831954_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed9
```

Settings:

- Loaded source final checkpoint with `compatible_model`.
- Additive non-bottleneck mode: `calculator_bottleneck_mode=none`.
- `calculator_estimator=ste`.
- `--freeze-semantic-decoder`.
- `--freeze-calculator-policy`.
- `answer_loss_weight=1`.
- `600` downstream/readout steps.

Results:

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1275 | 0.0050 | 0.0000 | 0.0075 | 1.0000 |
| 100 | 0.1850 | 0.0425 | 0.0000 | 0.0250 | 1.0000 |
| 200 | 0.3500 | 0.0250 | 0.0050 | 0.0175 | 1.0000 |
| 300 | 0.4900 | 0.0400 | 0.0100 | 0.0550 | 1.0000 |
| 400 | 0.5075 | 0.0275 | 0.0125 | 0.0275 | 1.0000 |
| 500 | 0.5625 | 0.0250 | 0.0125 | 0.0350 | 1.0000 |
| 600 | 0.6625 | 0.0250 | 0.0325 | 0.0225 | 1.0000 |

Final handoff metrics:

- Final eval: `259/400 = 0.6475`.
- Final 128-sample counterfactuals: injection-zero `0.0469`,
  forced-zero `0.0156`, forced-random `0.0391`.
- Diagnostic calculator-result accuracy: `1.0000`.

## Continuation Diagnostic

```text
runs/ohm_semdist_fresh_handoff_continue600/2026-05-30_195749_735842_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed9
```

The handoff was continued for another `600` steps with the calculator policy
still frozen.

| Continuation step | Normal | Zero-inj | Forced zero | Forced random | Calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6475 | 0.0550 | 0.0075 | 0.0200 | 1.0000 |
| 200 | 0.7375 | 0.0525 | 0.0025 | 0.0225 | 1.0000 |
| 400 | 0.8400 | 0.0500 | 0.0025 | 0.0250 | 1.0000 |
| 600 | 0.8500 | 0.0325 | 0.0075 | 0.0225 | 1.0000 |

Final continuation metrics:

- Final eval: `329/400 = 0.8225`.
- Final 128-sample counterfactuals: injection-zero `0.0391`,
  forced-zero `0.0078`, forced-random `0.0156`.
- Diagnostic calculator-result accuracy: `1.0000`.

## Alternate Handoff Seed

```text
runs/ohm_semdist_fresh_handoff_altseed600/2026-05-30_200445_067695_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed6
```

The same fresh source checkpoint was handed off with CLI seed `4` / effective
seed `6`, matching the first clean positive's downstream seed while preserving
the fresh source checkpoint.

| Step | Normal | Zero-inj | Forced zero | Forced random | Calc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1300 | 0.0050 | 0.0000 | 0.0050 | 1.0000 |
| 200 | 0.3625 | 0.0250 | 0.0050 | 0.0175 | 1.0000 |
| 400 | 0.4725 | 0.0175 | 0.0225 | 0.0325 | 1.0000 |
| 600 | 0.6325 | 0.0300 | 0.0125 | 0.0100 | 1.0000 |

Final alternate-handoff metrics:

- Final eval: `253/400 = 0.6325`.
- Final 128-sample counterfactuals: injection-zero `0.0391`,
  forced-zero `0.0000`, forced-random `0.0312`.
- Diagnostic calculator-result accuracy: `1.0000`.

## Interpretation

The source mechanism replicated. The fresh source reached perfect
calculator-result accuracy and low controls at the same fixed-grid op19 scale.

The trusted handoff did not replicate as a pass. Calculator accuracy remained
perfect and causal controls were low, so the miss is not policy collapse. The
continuation shows the source is usable by the additive path, but the readout
geometry is less handoff-friendly than the first seed.

The alternate handoff seed also missed, so the likely failure is source/readout
geometry rather than downstream seed luck. Further downstream-seed repeats from
this source are low value unless they are part of a designed variance audit.

This updates the method status from clean positive to mixed-positive:
online-hard-memory plus semantic distillation is a strong answer-derived source
and geometry mechanism, but robust 600-step non-bottleneck handoff across seeds
is not solved.

## Decision

```text
online_hard_memory_semantic_distill_fresh_seed_mixed_positive
```
