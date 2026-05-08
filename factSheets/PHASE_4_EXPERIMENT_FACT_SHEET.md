# Phase 4 Experiment Fact Sheet

## Direction

Phase 4 tests whether a calculator-query protocol can be taught and retained
when the downstream answer target makes operand identity useful.

## Current Findings

Phase 4 now has a seed-robust learned-interface positive, but not yet a full
upstream-discovery result.

Established positives:

- The `sum_left_operand` answer target plus
  `calculator_output_format=sum_left_operand` gives a strict answer-decoder
  setting where true operand identity matters.
- The frozen Stage 0B upstream representation contains enough operand
  information for `calculator_hook.input_proj` to read out true two-digit
  operands when the hook reads `operand_spans`.
- With direct operand supervision, the operand-span interface learns the true
  calculator-query protocol across effective seeds `2`, `4`, and `5`.
- After direct operand supervision is exactly removed
  (`final_aux_operand_loss_weight=0.0`), answer loss retains the true learned
  protocol across all three seeds.
- Retention works from the first gated warm-start checkpoints, not only from
  final high-confidence Stage 1 checkpoints.
- Selected aux-zero checkpoints show normal/oracle/operand/pair/calculator
  result accuracy `1.000`, injection-zero near `0.0`, forced-random near
  chance, private all-pair exact `1.000`, identity learned A/B mappings, and
  full-enum learned-minus-true and learned-minus-best gaps `0.0`.

Interpretive boundary:

- This is protocol teaching and retention, not proof that the whole upstream
  model independently discovered calculator use from answer loss. In the
  positive runs, the semantic decoder and upstream encoder were frozen and the
  only trainable group was `calculator_hook.input_proj`.

Most important next question:

```text
Does answer loss merely preserve an already-taught calculator protocol, or can
it complete and stabilize a partially learned protocol after the teacher signal
is removed?
```

Next work should prioritize reduced-supervision curricula and partial-handoff
boundaries before upstream unfreezing or new estimators.

## Do Not Rediscover Oracle Success

This is the most important interpretive rule for Phase 4:

```text
Oracle calculator success is not progress on the research question.
```

The project has known since Phase 1 that downstream answer components can emit
the right answer when the calculator path is given correct values. Oracle
train/eval, oracle-at-eval recovery, injection-zero controls, and forced-random
controls are sanity checks for wiring and bottleneck dependence only. They
should not be described as evidence that the model has learned to use a
calculator.

The only result that matters for the core thesis is learned-interface behavior:

- learned operand/pair exact match;
- learned calculator-result accuracy;
- private all-pair protocol decoding;
- learned-vs-true and learned-vs-best action-loss gaps;
- retention when `aux_operand_loss_weight` or other direct teacher signals are
  exactly `0.0`;
- replication across checkpoints/seeds.

Do not spend time rerunning oracle-only controls unless code has changed in a
way that could invalidate the wiring. Once a wiring gate passes for a
configuration, the next work item should move directly to upstream/interface
teaching and retention.

The first implemented identifiable target keeps the prompt shape:

```text
AA+BB=
```

and adds an opt-in answer format:

```text
SSSAA<eos>
```

where `SSS` is the zero-padded sum and `AA` is the zero-padded left operand.

Example:

```text
07+12=01907<eos>
```

## 2026-05-06 Implementation Facts

- Added `--answer-format sum | sum_left_operand`.
- Default `sum` behavior is preserved.
- `sum_left_operand` is digit-only and needs no vocabulary changes.
- The model architecture is unchanged; only block/sequence length follows the answer format.
- Training and checkpoint diagnostics can now be run with `--answer-format sum_left_operand`.
- Calculator operand read positions remain anchored to the fixed-width prompt operands, independent of the longer answer suffix.

## Validation

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/data.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
66 passed
```

## 2026-05-06 Operand-Aware Calculator Signal

Claim tested:

```text
For answer_format=sum_left_operand, the strict answer decoder needs an
operand-aware calculator output signal; sum-only is an intentional negative
control.
```

Implementation facts:

- Added `--calculator-output-format sum | sum_left_operand`; default is `sum`.
- `sum` preserves the old calculator signal: `one_hot(a+b)`.
- `sum_left_operand` projects `concat(one_hot(a+b), one_hot(a))`.
- Oracle operands use oracle `a`; learned operands use learned `a_pred`, with a
  small straight-through one-hot helper for the independent learned A head.
- The `sum_left_operand` strict decoder path now includes a calculator-signal /
  answer-offset interaction. The old `sum` decoder path remains additive for
  compatibility.
- Training configs, metrics, and diagnostic summaries serialize
  `calculator_output_format`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_phase1_track4_action_loss_diagnostic.py scripts/run_action_loss_diagnostic.py scripts/run_full_enum_action_loss_diagnostic.py scripts/diagnose_private_protocol.py scripts/run_action_loss_candidate_diagnostic.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
71 passed
```

Stage 0A negative control:

- Command used oracle operands with `--answer-format sum_left_operand` and
  `--calculator-output-format sum`.
- Run path:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_163931_767398_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2`
- Eval exact: `0.043` (`22/512`).
- Injection-zero exact: `0.0`; forced-random exact: `0.0`.
- Trace operand exact and calculator-result accuracy under oracle operands:
  `1.0`.

Stage 0B operand-aware oracle semantic decoder:

- Command used oracle operands with `--answer-format sum_left_operand` and
  `--calculator-output-format sum_left_operand`.
- Run path:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2`
- Selected checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- Eval exact: `1.000` (`512/512`), final loss `0.000224`.
- Canonical diagnostics (`samples=256`): normal exact `1.0`,
  injection-zero exact `0.0`, forced-zero exact `0.0039`,
  forced-random exact `0.0273`, oracle-at-eval exact `1.0`,
  operand exact `1.0`, calculator-result accuracy `1.0`.
- Canonical classification: `valid_oracle_calculator_use` /
  `calculator_required_bottleneck`.
- Full-enum diagnostic (`samples=128`): true-best fraction `1.0`, learned-best
  fraction `0.0078`, mean learned-minus-true gap `9.3793`. This is expected for
  the oracle semantic decoder because the learned interface head is still
  untrained/random.

Interpretation:

- Stage 0B is not evidence that the model learned to use a calculator.
- Stage 0B only says the downstream wiring is no longer blocking the actual
  experiment.
- The meaningful next result must train/evaluate the learned upstream
  calculator interface and report whether it survives with direct supervision
  exactly removed.

Go recommendation:

- Go to Stage 1 supervised interface warm start using the selected Stage 0B
  checkpoint above.

## 2026-05-06 Learned Interface Warm Start and Teacher-Zero Retention

Claim tested:

```text
A supervised interface warm start can teach the learned calculator query
protocol for the sum_left_operand decoder, and answer loss can retain that
protocol after direct operand supervision is set exactly to 0.0.
```

Mechanics finding:

- The requested `calculator_read_position=operands` reads only the final digit
  position for each fixed-width operand. Under the frozen Stage 0B upstream,
  this did not expose the full two-digit operand identity well enough.
- Original-read Stage 1 at the requested LR reached only `0.273` eval exact and
  best snapshot operand exact `0.355`.
- A higher-LR/final-digit continuation improved to `0.727` eval exact and
  snapshot operand exact `0.715`, still below the warm-start gate.
- Added opt-in `calculator_read_position=operand_spans` plus
  `--calculator-read-span-width 2`, preserving existing `eq` and `operands`
  behavior. The trainable group remains only `calculator_hook.input_proj`.

Stage 1 selected warm start:

- Run path:
  `runs/2026-05-06_192430_233405_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed3`
- Selected checkpoint:
  `runs/2026-05-06_192430_233405_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed3/final_weights.pt`
- Config: `answer_loss_weight=0.0`, `aux_operand_loss_weight=1.0`,
  `adaptive_interface_loss_weight=0.0`, `freeze_semantic_decoder=true`,
  `freeze_upstream_encoder=true`, `calculator_read_position=operand_spans`,
  `calculator_read_span_width=2`.
- Trainable parameters: `calculator_hook.input_proj` only (`1320` params).
- Fast gate: final eval exact `1.000`; snapshots from step `200` onward had
  normal exact `1.000`, operand exact `1.000`, and oracle exact `1.000`.

Stage 2 selected teacher-zero retention:

- Run path:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3`
- Selected checkpoint:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/final_weights.pt`
- Config: `answer_loss_weight=1.0`, `aux_operand_loss_weight=0.0`,
  `final_aux_operand_loss_weight=0.0`, `adaptive_interface_loss_weight=0.0`,
  `final_adaptive_interface_loss_weight=0.0`,
  `final_input_proj_anchor_weight=0.0`, `freeze_semantic_decoder=true`,
  `freeze_upstream_encoder=true`.
- Trainable parameters: `calculator_hook.input_proj` only (`1320` params).
- Final eval exact: `1.000` (`512/512`), final loss `0.000198`.
- Fast-gate snapshots stayed stable through step `1000`: normal exact `1.000`,
  operand exact `1.000`; final snapshot injection-zero exact `0.004`.

Selected Stage 2 canonical diagnostics:

- Output:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/canonical_causal_diagnostics`
- Samples: `256`.
- Normal exact `1.000`; injection-zero exact `0.000`; forced-zero exact
  `0.0039`; forced-random exact `0.0313`; oracle-at-eval exact `1.000`.
- Operand exact `1.000`; pair exact `1.000`; calculator-result accuracy
  `1.000`.
- Classification: `intended_true_operand_calculator_use`.
- Forced-result sweep: learned-class best fraction `1.000`, true-sum best
  fraction `1.000`, learned-minus-true target-logprob gap `0.0`.

Selected Stage 2 private protocol diagnostics:

- Output:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/private_protocol_diagnostics`
- All `20 x 20` pairs evaluated.
- Exact match `1.000`; operand exact `1.000`; pair exact `1.000`;
  calculator-result accuracy `1.000`.
- Best affine mappings for learned A and B were identity: exact `1.000`,
  offset `0`, scale `1`.

Selected Stage 2 full-enum action-loss diagnostics:

- Output:
  `runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/full_enum_action_loss/model-c-2digit-seed3`
- Samples: `128`; action pairs: `400`.
- Learned-best fraction `1.000`; true-best fraction `1.000`;
  best-matches-true-operands fraction `1.000`.
- Mean learned NLL, true NLL, and best NLL were all `0.0002419`.
- Mean learned-minus-true gap `0.0`; mean learned-minus-best gap `0.0`.

Interpretation:

- This is not oracle-only success. The selected Stage 2 checkpoint has direct
  teacher signals exactly removed and still emits the true learned calculator
  operands.
- The positive result depends on exposing the full fixed-width operand spans to
  the interface head. The original final-digit-only readout is an insufficient
  warm-start mechanic for two-digit operands under the frozen Stage 0B upstream.

Go recommendation:

- Go to seed replication with `calculator_read_position=operand_spans` before
  broadening objectives. The next task should confirm whether this
  interface-readable protocol teaching result holds across at least two seeds.

## 2026-05-07 Operand-Span Retention Replication and Boundary

Claim tested:

```text
Aux-zero retention of the operand-span calculator-query protocol is robust
across seeds, and answer loss can preserve the protocol even from the earliest
warm-start checkpoint that clears the Stage 1 gate.
```

Shared config:

- Stage 0B checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- `answer_format=sum_left_operand`,
  `calculator_output_format=sum_left_operand`,
  `calculator_read_position=operand_spans`,
  `calculator_read_span_width=2`,
  `calculator_bottleneck_mode=answer_decoder`.
- `freeze_semantic_decoder=true`, `freeze_upstream_encoder=true`.
- Trainable parameters were limited to `calculator_hook.input_proj`
  (`1320` params) for every Stage 1 and Stage 2 run.

Stage 1 aux-only warm starts:

| Effective seed | CLI seed | Run path | First gated handoff | First perfect | Final exact |
| --- | ---: | --- | --- | --- | ---: |
| `2` | `0` | `runs/2026-05-07_070737_999460_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2` | `checkpoint_snapshots/step_00100_weights.pt` | step `150` | `1.000` |
| `4` | `2` | `runs/2026-05-07_070738_192155_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed4` | `checkpoint_snapshots/step_00150_weights.pt` | step `150` | `1.000` |
| `5` | `3` | `runs/2026-05-07_070737_995829_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5` | `checkpoint_snapshots/step_00100_weights.pt` | step `150` | `1.000` |

Stage 1 final fast gates:

| Effective seed | Normal | Injection-zero | Forced-random | Oracle | Operand | Pair | Calculator result | A/B entropy | Aux |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `2` | `1.000` | `0.000` | `0.016` | `1.000` | `1.000` | `1.000` | `1.000` | `0.466/0.403` | `1.0` |
| `4` | `1.000` | `0.000` | `0.043` | `1.000` | `1.000` | `1.000` | `1.000` | `0.464/0.397` | `1.0` |
| `5` | `1.000` | `0.004` | `0.020` | `1.000` | `1.000` | `1.000` | `1.000` | `0.446/0.398` | `1.0` |

Stage 2A aux-zero retention from earliest gated handoff:

| Effective seed | CLI seed | Run path | Selected checkpoint | Aux | Normal | Injection-zero | Forced-random | Operand | Calculator result | A/B entropy |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `2` | `0` | `runs/2026-05-07_092659_995383_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `final_weights.pt` | `0.0` | `1.000` | `0.000` | `0.016` | `1.000` | `1.000` | `2.918/2.904` |
| `4` | `2` | `runs/2026-05-07_074429_578037_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed4` | `final_weights.pt` | `0.0` | `1.000` | `0.000` | `0.043` | `1.000` | `1.000` | `2.829/2.793` |
| `5` | `3` | `runs/2026-05-07_092657_329340_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `final_weights.pt` | `0.0` | `1.000` | `0.004` | `0.020` | `1.000` | `1.000` | `2.914/2.900` |

Stage 2B aux-zero retention from final Stage 1 handoff:

| Effective seed | CLI seed | Run path | Selected checkpoint | Aux | Normal | Injection-zero | Forced-random | Operand | Calculator result | A/B entropy |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `2` | `0` | `runs/2026-05-07_082239_879644_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `final_weights.pt` | `0.0` | `1.000` | `0.000` | `0.016` | `1.000` | `1.000` | `0.462/0.404` |
| `4` | `2` | `runs/2026-05-07_082241_865137_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed4` | `final_weights.pt` | `0.0` | `1.000` | `0.000` | `0.043` | `1.000` | `1.000` | `0.462/0.395` |
| `5` | `3` | `runs/2026-05-07_082240_648656_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `final_weights.pt` | `0.0` | `1.000` | `0.004` | `0.020` | `1.000` | `1.000` | `0.446/0.395` |

Selected Stage 2A diagnostics:

- Canonical diagnostics on all three selected Stage 2A finals classified each
  checkpoint as `intended_true_operand_calculator_use`.
- Canonical metrics for all selected Stage 2A finals: normal exact `1.000`,
  injection-zero exact `0.000`, forced-random exact `0.03125`,
  oracle-at-eval exact `1.000`, operand exact `1.000`, pair exact `1.000`,
  calculator-result accuracy `1.000`, forced-result learned-class best
  fraction `1.000`, true-sum best fraction `1.000`, and learned-minus-true
  target-logprob gap `0.0`.
- Private all-pair diagnostics on all selected Stage 2A finals: exact
  `1.000`, operand exact `1.000`, pair exact `1.000`, calculator-result
  accuracy `1.000`, learned A identity mapping exact `1.000`, and learned B
  identity mapping exact `1.000`.
- Full-enum action-loss diagnostics on all selected Stage 2A finals:
  learned-best fraction `1.000`, true-best fraction `1.000`,
  learned-minus-true gap `0.0`, learned-minus-best gap `0.0`, effective
  action pairs about `30.2`.

Interpretation:

- The previous one-seed positive replicated strongly: three of three seeds
  learned the operand-span protocol under direct supervision, and three of
  three retained it after direct operand supervision was exactly removed.
- The boundary result is stronger than expected: final-handoff retention works,
  but it was not required. Even the first gated handoff checkpoints retained
  perfectly under aux-zero answer-only training.
- Stage 2A retained despite high final A/B entropy near `2.8-2.9`, while
  Stage 2B retained with sharper A/B entropy near `0.4`. Hard argmax protocol
  correctness is therefore robust even when the retained distributions remain
  relatively soft.

Go recommendation:

- Go to reduced-supervision curricula. Interface-only aux-zero retention is now
  seed-robust across at least three effective seeds.
- No-go on upstream unfreezing as the next move; keep it reserved until the
  reduced-supervision boundary is known.
- No-go on new estimators for the next task. The current best signal is to find
  how little direct operand supervision is needed before aux-zero retention
  survives.

## 2026-05-07 Minimum Supervision and Partial Completion Boundary

Claim tested:

```text
Answer loss can complete and stabilize a partially learned calculator-query
protocol after direct operand supervision is removed, but the boundary is
seed-dependent and not reached by short decayed-aux curricula.
```

Run root:

```text
runs/2026-05-07_phase4_min_supervision_boundary
```

Implementation:

- Added `scripts/run_phase4_min_supervision_boundary.py` to run the Stage 1A
  warm-start ladder, Stage 2 aux-zero continuations, lower-handoff expansion,
  Stage 1B decayed-aux curricula, summaries, and selected diagnostics.
- Shared config preserved the Phase 4 robust-positive setup:
  `answer_format=sum_left_operand`,
  `calculator_output_format=sum_left_operand`,
  `calculator_read_position=operand_spans`,
  `calculator_read_span_width=2`, strict `answer_decoder` bottleneck, frozen
  semantic decoder, frozen upstream encoder, and trainable
  `calculator_hook.input_proj` only.

Stage 1A aux-only warm-start thresholds:

| Effective seed | CLI seed | >=0.25 | >=0.50 | >=0.75 | >=0.90 | >=0.95 | First 1.0 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `0` | step `25` / `0.320` | step `60` / `0.641` | step `65` / `0.820` | step `75` / `0.965` | step `75` / `0.965` | step `95` |
| `4` | `2` | step `35` / `0.363` | step `40` / `0.520` | step `65` / `0.816` | step `80` / `0.914` | step `85` / `0.969` | step `110` |
| `5` | `3` | step `35` / `0.422` | step `55` / `0.625` | step `65` / `0.859` | step `95` / `0.949` | step `100` / `0.969` | step `105` |

Stage 2 aux-zero completion boundary:

| Effective seed | Lowest retained handoff | Final operand/pair/calc | Nearest failed below | Final operand/pair/calc |
| ---: | --- | ---: | --- | ---: |
| `2` | step `60`, Stage 1 operand `0.641` | `1.000` | step `25`, Stage 1 operand `0.320` | `0.773` fast gate, `0.809` canonical |
| `4` | step `35`, Stage 1 operand `0.363` | `0.980` fast gate, `0.996` canonical | step `30`, Stage 1 operand `0.188` | `0.699` fast gate, `0.730` canonical |
| `5` | step `30`, Stage 1 operand `0.230` | `0.980` fast gate, `0.992` canonical | not established by the one-extra-lower expansion | n/a |

All selected Stage 2 retention runs had:

- `final_aux_operand_loss_weight=0.0`;
- `final_adaptive_interface_loss_weight=0.0`;
- `final_input_proj_anchor_weight=0.0`;
- `freeze_semantic_decoder=true`;
- `freeze_upstream_encoder=true`;
- trainable parameters limited to `calculator_hook.input_proj`.

The retained checkpoints kept injection-zero and forced-random near chance while
oracle-at-eval stayed `1.000`, so these are not answer-decoder-only shortcuts.
Canonical diagnostics classified the retained selections as
`intended_true_operand_calculator_use`.

Selected full diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-best | Full-enum gaps |
| --- | ---: | ---: | ---: | ---: |
| seed `2` lowest retained | `1.000` / `1.000` / `1.000` | `1.000` / `1.000` / `1.000` | `1.000` | `0.000` / `0.000` |
| seed `2` failed below | `0.809` / `0.809` / `0.809` | `0.785` / `0.785` / `0.790` | `0.719` | `1.163` / `1.163` |
| seed `4` lowest retained | `0.996` / `0.996` / `0.996` | `0.993` / `0.993` / `0.993` | `1.000` | `0.000` / `0.000` |
| seed `4` failed below | `0.730` / `0.730` / `0.730` | `0.705` / `0.705` / `0.705` | `0.641` | `1.846` / `1.846` |
| seed `5` lowest retained | `0.992` / `0.992` / `0.992` | `0.988` / `0.988` / `0.988` | `0.977` | `0.098` / `0.098` |

Stage 1B decayed-aux curricula:

| Effective seed | decay 25 | decay 50 | decay 100 |
| ---: | ---: | ---: | ---: |
| `2` | `0.387` | `0.230` | `0.230` |
| `4` | `0.324` | `0.250` | `0.352` |
| `5` | `0.355` | `0.352` | `0.445` |

All Stage 1B final checkpoints had `final_aux_operand_loss_weight=0.0`, but no
decayed-aux curriculum approached retention. The best decayed checkpoint
(`seed5`, decay `100`) reached only `0.445` fast-gate operand exact and full
diagnostics stayed partial: canonical operand/pair/calc `0.457` / `0.457` /
`0.465`, private operand/pair/calc `0.445` / `0.445` / `0.450`, full-enum
learned-best `0.438`, and learned-minus-true/best gaps `2.241`.

Interpretation:

- Answer loss does more than preserve a perfect protocol. It can complete a
  partially learned protocol from materially imperfect handoffs.
- The practical aux-zero completion boundary is seed-dependent in this compact
  ladder: seed `2` needed the `0.64` handoff, seed `4` retained from `0.36` but
  failed from `0.19`, and seed `5` retained from `0.23`.
- Decaying the teacher before a useful protocol forms is not equivalent to
  handing off after a measured partial protocol. The answer loss can complete a
  partially learned interface, but it did not reliably bootstrap the interface
  while aux and answer losses were mixed from the start.

Go recommendation:

- Go to a narrower boundary probe around the `0.20` to `0.35` Stage 1 operand
  range, especially to find a failed lower neighbor for seed `5`.
- Keep upstream frozen and avoid new estimators until this completion boundary
  is sharper.

## 2026-05-08 Boundary Closure Before Phase Wrap

Claim tested:

```text
The partial-handoff completion boundary from 2026-05-07 can be closed without
introducing a new estimator, unfreezing upstream, or adding a new objective.
```

Run root:

```text
runs/2026-05-08_phase4_boundary_closure
```

Implementation:

- Added `scripts/run_phase4_boundary_closure.py` as a compact reproducible
  runner for the selected Stage 2 handoffs and diagnostics.
- Reused the existing dense Stage 1A snapshots from
  `runs/2026-05-07_phase4_min_supervision_boundary`; Stage 1A was not rerun.
- Shared setup stayed fixed: `answer_format=sum_left_operand`,
  `calculator_output_format=sum_left_operand`,
  `calculator_read_position=operand_spans`,
  `calculator_read_span_width=2`, strict `answer_decoder` bottleneck, frozen
  semantic decoder, frozen upstream encoder, and trainable
  `calculator_hook.input_proj` only.
- All closure Stage 2 continuations used `answer_loss_weight=1.0`,
  `aux_operand_loss_weight=0.0`,
  `adaptive_interface_loss_weight=0.0`, and
  `input_proj_anchor_weight=0.0`.

New closure fast gates:

| Effective seed | CLI seed | Handoff | Stage 1 operand | Final operand/pair/calc | Injection-zero | Forced-random | Oracle | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `2` | `0` | step `30` | `0.395` | `0.809` / `0.809` / `0.809` | `0.000` | `0.016` | `1.000` | failed |
| `2` | `0` | step `55` | `0.438` | `0.844` / `0.844` / `0.844` | `0.000` | `0.016` | `1.000` | nearest failed below step `60` |
| `5` | `3` | step `20` | `0.027` | `0.734` / `0.734` / `0.734` | `0.004` | `0.020` | `1.000` | failed |
| `5` | `3` | step `25` | `0.078` | `0.855` / `0.855` / `0.855` | `0.004` | `0.020` | `1.000` | nearest failed below step `30` |

All four closure runs ended with:

- `final_aux_operand_loss_weight=0.0`;
- `final_adaptive_interface_loss_weight=0.0`;
- `final_input_proj_anchor_weight=0.0`;
- `freeze_semantic_decoder=true`;
- `freeze_upstream_encoder=true`;
- trainable parameters limited to `calculator_hook.input_proj` (`1320`
  params).

Closure diagnostics:

| Selection | Canonical operand/pair/calc | Private operand/pair/calc | Full-enum learned-best | Full-enum gaps |
| --- | ---: | ---: | ---: | ---: |
| seed `2`, step `30` | `0.848` / `0.848` / `0.848` | `0.828` / `0.828` / `0.828` | `0.758` | `0.839` / `0.839` |
| seed `2`, step `55` | `0.855` / `0.855` / `0.855` | `0.845` / `0.845` / `0.845` | `0.852` | `0.705` / `0.705` |
| seed `5`, step `20` | `0.727` / `0.727` / `0.727` | `0.723` / `0.723` / `0.723` | `0.711` | `0.995` / `0.995` |
| seed `5`, step `25` | `0.828` / `0.828` / `0.828` | `0.848` / `0.848` / `0.850` | `0.883` | `0.349` / `0.349` |

Closed boundary:

| Effective seed | Nearest failed lower handoff | Lowest retained handoff | Interpretation |
| ---: | --- | --- | --- |
| `2` | step `55`, Stage 1 operand `0.438`, final/private/full-enum all below retention | step `60`, Stage 1 operand `0.641`, prior diagnostics exact | narrower fail/retain bracket |
| `4` | step `30`, Stage 1 operand about `0.19`, prior failed diagnostics | step `35`, Stage 1 operand `0.363`, prior retained diagnostics | already bracketed |
| `5` | step `25`, Stage 1 operand `0.078`, final/private/full-enum all below retention | step `30`, Stage 1 operand about `0.20`, prior retained diagnostics high | failed lower neighbor established |

Interpretation:

- Seed `2` no longer has a loose `0.320` failed vs `0.641` retained boundary.
  The nearest measured failure is now Stage 1 operand `0.438`, immediately
  below the retained step `60` handoff.
- Seed `5` is permissive, but not unbounded. The step `25` handoff failed under
  fast gates, private all-pair decoding, and full-enum action-loss diagnostics,
  while the previously measured step `30` handoff remained the retained upper
  neighbor.
- No new retained checkpoint showed high answer exact with weak operand/pair
  exact. The new diagnostics are failed-boundary evidence, not oracle-only or
  answer-only progress.

Recommendation:

- Wrap Phase 4. The remaining boundary is seed-dependent, but it is now
  bracketed tightly enough for the phase conclusion: answer loss can complete a
  partially taught calculator-query protocol after direct operand supervision
  is exactly removed, but only above a seed-dependent handoff quality.
- Next phase should move beyond frozen-interface boundary closure and decide
  whether to test upstream discovery, transfer, or a larger identifiable task.
