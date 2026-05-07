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
