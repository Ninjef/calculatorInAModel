# 2026-05-07 - Minimum supervision and partial completion boundary

Task: measure whether answer loss only preserves an already learned
calculator-query protocol, or can complete and stabilize a partially learned
protocol after direct operand supervision is removed.

## Claim

Answer loss can complete a partially learned calculator protocol, but the
minimum handoff quality is seed-dependent. The compact decayed-aux curricula did
not reliably bootstrap the interface when answer loss was mixed in before a
useful protocol formed.

## Shared setup

- Run root: `runs/2026-05-07_phase4_min_supervision_boundary`
- Stage 0B checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- Trainable parameters: `calculator_hook.input_proj` only

Added runner:

- `scripts/run_phase4_min_supervision_boundary.py`

## Stage 1A aux-only warm starts

| Effective seed | CLI seed | >=0.25 | >=0.50 | >=0.75 | >=0.90 | >=0.95 | First 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `0` | step `25` / `0.320` | step `60` / `0.641` | step `65` / `0.820` | step `75` / `0.965` | step `75` / `0.965` | step `95` |
| `4` | `2` | step `35` / `0.363` | step `40` / `0.520` | step `65` / `0.816` | step `80` / `0.914` | step `85` / `0.969` | step `110` |
| `5` | `3` | step `35` / `0.422` | step `55` / `0.625` | step `65` / `0.859` | step `95` / `0.949` | step `100` / `0.969` | step `105` |

Final Stage 1A checkpoints reached `1.000` normal/operand/pair/calculator
result for all three seeds.

## Stage 2 aux-zero handoff ladder

All Stage 2 runs used `answer_loss_weight=1.0`,
`aux_operand_loss_weight=0.0`, `adaptive_interface_loss_weight=0.0`,
`input_proj_anchor_weight=0.0`, frozen semantic decoder, frozen upstream
encoder, and only `calculator_hook.input_proj` trainable.

| Effective seed | Handoff | Stage 1 operand | Final operand/pair/calc | Status |
| ---: | ---: | ---: | ---: | --- |
| `2` | step `25` | `0.320` | `0.773` | failed below |
| `2` | step `60` | `0.641` | `1.000` | lowest retained |
| `2` | step `65` | `0.820` | `1.000` | retained |
| `2` | step `75` | `0.965` | `1.000` | retained |
| `4` | step `30` | `0.188` | `0.699` | failed below |
| `4` | step `35` | `0.363` | `0.980` | lowest retained |
| `4` | step `40` | `0.520` | `1.000` | retained |
| `4` | step `65` | `0.816` | `1.000` | retained |
| `4` | step `80` | `0.914` | `1.000` | retained |
| `4` | step `85` | `0.969` | `1.000` | retained |
| `5` | step `30` | `0.230` | `0.980` | lowest retained |
| `5` | step `35` | `0.422` | `0.992` | retained |
| `5` | step `55` | `0.625` | `1.000` | retained |
| `5` | step `65` | `0.859` | `1.000` | retained |
| `5` | step `95` | `0.949` | `1.000` | retained |
| `5` | step `100` | `0.969` | `1.000` | retained |

The one-extra-lower expansion produced a failed lower neighbor for seed `4` and
an additional retained lower checkpoint for seed `5`; a failed lower neighbor
for seed `5` remains unmeasured.

## Stage 1B decayed-aux curricula

Final operand exact at step `300`:

| Effective seed | decay 25 | decay 50 | decay 100 |
| ---: | ---: | ---: | ---: |
| `2` | `0.387` | `0.230` | `0.230` |
| `4` | `0.324` | `0.250` | `0.352` |
| `5` | `0.355` | `0.352` | `0.445` |

All final checkpoints had `final_aux_operand_loss_weight=0.0`, but none reached
a retention-quality protocol. The best run was effective seed `5`, decay `100`.

## Selected diagnostics

Diagnostics were run on each seed's lowest retained checkpoint, nearest failed
below where measured, and the best decayed-aux checkpoint.

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-best | Full-enum gaps |
| --- | ---: | ---: | ---: | ---: |
| seed `2` lowest retained | `1.000` / `1.000` / `1.000` | `1.000` / `1.000` / `1.000` | `1.000` | `0.000` / `0.000` |
| seed `2` failed below | `0.809` / `0.809` / `0.809` | `0.785` / `0.785` / `0.790` | `0.719` | `1.163` / `1.163` |
| seed `4` lowest retained | `0.996` / `0.996` / `0.996` | `0.993` / `0.993` / `0.993` | `1.000` | `0.000` / `0.000` |
| seed `4` failed below | `0.730` / `0.730` / `0.730` | `0.705` / `0.705` / `0.705` | `0.641` | `1.846` / `1.846` |
| seed `5` lowest retained | `0.992` / `0.992` / `0.992` | `0.988` / `0.988` / `0.988` | `0.977` | `0.098` / `0.098` |
| best decayed-aux | `0.457` / `0.457` / `0.465` | `0.445` / `0.445` / `0.450` | `0.438` | `2.241` / `2.241` |

Retained selections were classified as `intended_true_operand_calculator_use`.
The failed and decayed selections remained partial/private-code regimes with
positive full-enum gaps.

## Interpretation

Answer loss does more than preserve a perfect protocol. It can complete and
stabilize materially imperfect handoffs after direct operand supervision is
exactly removed.

The measured boundary is seed-dependent:

- seed `2`: handoff `0.320` failed; `0.641` retained.
- seed `4`: handoff `0.188` failed; `0.363` retained.
- seed `5`: handoff `0.230` retained; failed lower neighbor not yet measured.

The decayed-aux result is the useful negative. Mixing answer loss from the
start while decaying the teacher by step `25`, `50`, or `100` did not produce a
retained protocol, even though answer loss could complete a measured partial
handoff after aux was already removed.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_phase4_min_supervision_boundary.py
python3 scripts/run_phase4_min_supervision_boundary.py stage1a --jobs 3
python3 scripts/run_phase4_min_supervision_boundary.py stage2 --jobs 3
python3 scripts/run_phase4_min_supervision_boundary.py stage2-lower --jobs 2
python3 scripts/run_phase4_min_supervision_boundary.py stage1b --jobs 3
python3 scripts/run_phase4_min_supervision_boundary.py diagnostics
```

## Recommendation

Next, run a narrower lower-boundary probe around Stage 1 operand exact `0.20`
to `0.35`, especially to find a failed lower neighbor for seed `5`.
