# Phase 6 Closure Landscape Diagnostic

## Task

```text
aiAgentProjectTasks/2026-05-12-phase-6-eleventh-task-Phase-6-closure-landscape-diagnostic-and-next-phase-decision.md
```

## Code Added

- Extended `scripts/run_full_enum_action_loss_diagnostic.py` with
  result-group soft-target probabilities:
  - `soft_target_true_result_group_probability`
  - `soft_target_best_result_group_probability`
  - corresponding summary means.
- Added `scripts/run_phase6_closure_landscape_diagnostic.py`.
- The closure runner:
  - compacts existing Phase 6 evidence from the deterministic Concrete and
    natural product-decoder summaries;
  - runs paired all-400 full-enum landscapes for identifiable
    `sum_left_operand` and natural sum-only product decoder;
  - runs paired one-step deterministic hard-forward / soft-backward Concrete
    gradient diagnostics across seeds `6201`, `6202`, and `6203`;
  - writes `summary.json` and `summary.md`.

## Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/run_full_enum_action_loss_diagnostic.py scripts/run_phase6_closure_landscape_diagnostic.py
PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX=/tmp/codex_pycache OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 scripts/run_phase6_closure_landscape_diagnostic.py run
```

## Results

Run root:

```text
runs/2026-05-12_phase6_closure_landscape_diagnostic
```

Primary outputs:

```text
runs/2026-05-12_phase6_closure_landscape_diagnostic/summary.json
runs/2026-05-12_phase6_closure_landscape_diagnostic/summary.md
```

### Existing Evidence Table

No completed training was rerun.

- Identifiable deterministic Concrete replicated across effective seeds `2`,
  `4`, and `5`; relaxation-off retention completed all three to exact fast
  protocol metrics with aux/adaptive/local/expected/relaxed/anchor weights
  exactly `0.0` and semantic decoder delta `0.0`.
- Upstream-open deterministic stress selected a `0.961` fast protocol and
  retained to exact protocol metrics with upstream frozen and open.
- Natural product decoder passed the all-400 oracle/readout gate, but the
  natural deterministic Concrete bridge selected only `0.135` fast result
  accuracy, `0.1175` canonical result accuracy, learned-result best fraction
  `0.1100`, and mean learned-result minus best-result gap `5.5657`.

### Paired Full-Enum Landscape

At the Phase 6 local-target temperature `0.25`:

| Setting | Best pair=true | Best result=true | Effective pairs | Effective results | Same-true-sum near-best | True pair prob | True result prob |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifiable `sum_left_operand` | `1.0000` | `1.0000` | `1.0839` | `1.0462` | `1.0000` | `0.9879` | `0.9931` |
| Natural sum-only product | `0.0975` | `1.0000` | `13.3573` | `1.0011` | `13.3500` | `0.0975` | `0.9999` |

The identifiable landscape is essentially single-pair. The natural landscape
is result-identifiable but pair-underidentified: the correct result group gets
nearly all mass, spread across the same-sum diagonal.

### One-Step Relaxed Gradient

Mean deltas across seeds `6201`, `6202`, and `6203`:

| Setting | True pair prob delta | True result prob delta | Best result prob delta | Hard pair delta | Hard calc/result delta | Input/upstream/semantic delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identifiable `sum_left_operand` | `+8.54e-06` | `+2.88e-05` | `+2.88e-05` | `+0.0033` | `+0.0033` | `1.0893 / 0.0 / 0.0` |
| Natural sum-only product | `+1.31e-05` | `+7.72e-06` | `+7.72e-06` | `+0.0075` | `+0.0142` | `1.0892 / 0.0 / 0.0` |

The natural one-step result-group signal is positive but tiny from the strict
random-upstream initialization. The semantic decoder and upstream stayed fixed
exactly in both settings.

## Interpretation

```text
phase6_close_start_phase7
identifiable_landscape_positive
natural_sum_only_underidentified_boundary
```

Phase 6 should close. The deterministic Concrete positive is real and
replicated in the identifiable task, but the natural sum-only failure after the
product decoder gate is best explained as an underidentification and action
parameterization boundary for independent operand heads, not as a broken
decoder or broken relaxation implementation.

## Recommended Phase 7 Direction

Start Phase 7 with natural `0..19` result-space or structured-action work:

- result-space interface parameterization;
- joint-pair or structured action head;
- communication-constrained natural task;
- result-level target-propagation/local critic.

Do not start Phase 7 with `operand_max=99` scaling.
