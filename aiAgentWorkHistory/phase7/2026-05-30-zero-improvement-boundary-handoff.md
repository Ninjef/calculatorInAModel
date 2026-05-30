# 2026-05-30 Zero-Improvement Boundary Handoff

## Question

Does the less-prescriptive full-enum zero-improvement result-boundary target
produce a source policy that transfers into the trusted additive
non-bottleneck frozen-policy handoff gate?

This directly follows the 200-step source gate. The source target is
answer-derived from improvement over zero calculator injection, not direct
true-result forcing.

## Runs

800-step source:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_handoff/source800_full_enum_cpu/2026-05-30_173156_596931_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Trusted handoff from source800:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_handoff/handoff600_from_source800_cpu/2026-05-30_173502_548730_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed4
```

1600-equivalent source continuation:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_handoff/source1600_continue_full_enum_cpu/2026-05-30_173618_410386_model-c-op0-19-fullgrid-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rtemp1-rfinal1-answer_decoder-adec-product/model-c-2digit-seed4
```

Trusted handoff from mature source:

```text
runs/2026-05-30_phase7_zero_improvement_boundary_handoff/handoff600_from_source1600_cpu/2026-05-30_173922_629095_model-c-op0-19-fullgrid-adec-product/model-c-2digit-seed4
```

## Source Results

| Source | Final eval | Last snapshot normal/calc | Injection-zero | Forced-random | Learned-best/calc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 800-step source | `0.9150` | `0.8950` | `0.0375` | `0.0275` | `0.8975` |
| 1600-equivalent source | `0.9850` | `0.9525` | `0.0375` | `0.0275` | `0.9725` |

The continuation confirms that zero-improvement can train a strong bottleneck
source when given enough full-enum source budget.

## Handoff Results

| Handoff | Final eval | Step-600 normal | Injection-zero | Forced-random | Oracle | Frozen calc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| from source800 | `0.3650` | `0.3900` | `0.0550` | `0.0875` | `0.3850` | `0.9050` |
| from source1600 | `0.6775` | `0.7150` | `0.0100` | `0.0525` | `0.6975` | `0.9725` |

The mature-source handoff is causal: normal accuracy is far above
injection-zero and forced-random controls, while the frozen calculator policy
stays accurate. But it misses the trusted handoff gate and remains below the
old hard-best result-boundary source handoff (`0.8825` final / `0.8425`
step-600 normal).

## Decision

```text
zero_improvement_mature_source_handoff_mixed_positive
```

Do not rerun the same source length or handoff length as novelty. The useful
lesson is that source maturity helps, but the zero-improvement source/readout
geometry is weaker than the old hard-best boundary source for frozen-policy
additive handoff.

Next useful work needs a new mechanism:

- handoff-aware zero-improvement source shaping;
- an additive/readout geometry auxiliary that keeps the less-prescriptive
  target while improving transfer;
- or a scalable proposal mechanism that preserves mature source quality at
  lower scorer cost.
