# 2026-05-07 - Operand-span retention replication

Task: replicate the Phase 4 `operand_spans` protocol-teaching result across
seeds and test whether answer-only aux-zero retention works from the earliest
Stage 1 handoff or only from a heavily trained warm start.

## Claim

Aux-zero retention of the learned calculator-query protocol is seed-robust, and
the first gated supervised warm-start checkpoint is enough for answer loss to
preserve the protocol.

## Shared setup

- Stage 0B checkpoint:
  `/Users/jarnold/Documents/Codex/2026-05-06/please-work-in-this-repo-users-9/runs/2026-05-06_164330_870116_model-c-oracle-op0-19-answer_decoder-sum_left_operand/model-c-2digit-seed2/final_weights.pt`
- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `calculator_bottleneck_mode=answer_decoder`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- Trainable parameters: `calculator_hook.input_proj` only (`1320` params)

## Stage 1 aux-only warm starts

| Effective seed | CLI seed | Run path | First gated handoff | First perfect | Final exact |
| --- | ---: | --- | --- | --- | ---: |
| `2` | `0` | `runs/2026-05-07_070737_999460_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed2` | `checkpoint_snapshots/step_00100_weights.pt` | step `150` | `1.000` |
| `4` | `2` | `runs/2026-05-07_070738_192155_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed4` | `checkpoint_snapshots/step_00150_weights.pt` | step `150` | `1.000` |
| `5` | `3` | `runs/2026-05-07_070737_995829_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed5` | `checkpoint_snapshots/step_00100_weights.pt` | step `150` | `1.000` |

Final fast gates:

- Seed `2`: normal `1.000`, injection-zero `0.000`, forced-random `0.016`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `0.466/0.403`.
- Seed `4`: normal `1.000`, injection-zero `0.000`, forced-random `0.043`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `0.464/0.397`.
- Seed `5`: normal `1.000`, injection-zero `0.004`, forced-random `0.020`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `0.446/0.398`.

## Stage 2A from earliest gated handoff

| Effective seed | CLI seed | Run path | Selected checkpoint | Aux |
| --- | ---: | --- | --- | ---: |
| `2` | `0` | `runs/2026-05-07_092659_995383_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `final_weights.pt` | `0.0` |
| `4` | `2` | `runs/2026-05-07_074429_578037_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed4` | `final_weights.pt` | `0.0` |
| `5` | `3` | `runs/2026-05-07_092657_329340_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `final_weights.pt` | `0.0` |

Final fast gates:

- Seed `2`: normal `1.000`, injection-zero `0.000`, forced-random `0.016`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `2.918/2.904`.
- Seed `4`: normal `1.000`, injection-zero `0.000`, forced-random `0.043`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `2.829/2.793`.
- Seed `5`: normal `1.000`, injection-zero `0.004`, forced-random `0.020`,
  oracle `1.000`, operand `1.000`, pair `1.000`, calculator-result `1.000`,
  A/B entropy `2.914/2.900`.

The corrected earliest seed `2` and seed `5` runs started from step `100`
checkpoints whose fresh step-0 snapshot sampled at `0.941` normal/operand
exact, then recovered to `1.000` by step `50` and retained through step `1000`.

## Stage 2B from final Stage 1 handoff

| Effective seed | CLI seed | Run path | Selected checkpoint | Aux |
| --- | ---: | --- | --- | ---: |
| `2` | `0` | `runs/2026-05-07_082239_879644_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed2` | `final_weights.pt` | `0.0` |
| `4` | `2` | `runs/2026-05-07_082241_865137_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed4` | `final_weights.pt` | `0.0` |
| `5` | `3` | `runs/2026-05-07_082240_648656_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed5` | `final_weights.pt` | `0.0` |

Final-handoff retention also held for all three seeds: normal/operand/pair and
calculator-result accuracy were all `1.000`; injection-zero was `0.000`,
`0.000`, `0.004`; forced-random was `0.016`, `0.043`, `0.020`.

## Selected diagnostics

Diagnostics were run on the Stage 2A earliest-handoff selections.

- Canonical causal diagnostics: all three selected checkpoints classified as
  `intended_true_operand_calculator_use`; normal exact `1.000`,
  injection-zero exact `0.000`, forced-random exact `0.03125`,
  oracle-at-eval exact `1.000`, operand exact `1.000`, pair exact `1.000`,
  calculator-result accuracy `1.000`, learned-minus-true target-logprob gap
  `0.0`.
- Private protocol diagnostics: all three selected checkpoints had all-pair
  exact `1.000`, operand exact `1.000`, pair exact `1.000`,
  calculator-result accuracy `1.000`, and identity affine mappings for both
  learned operands.
- Full-enum action-loss diagnostics: all three selected checkpoints had
  learned-best fraction `1.000`, true-best fraction `1.000`,
  learned-minus-true gap `0.0`, learned-minus-best gap `0.0`, and effective
  action pairs about `30.2`.

## Interpretation

The previous one-seed result replicated cleanly. Three of three seeds learned
the operand-span protocol under direct operand supervision, and three of three
retained the protocol when `final_aux_operand_loss_weight=0.0` with no anchor,
no adaptive-interface loss, frozen semantic decoder, frozen upstream encoder,
and only `calculator_hook.input_proj` trainable.

The boundary result is positive: final handoff is not required. The earliest
gated handoff retained, including the less confident step `100` handoffs for
effective seeds `2` and `5`.

## Recommendation

Go to reduced-supervision curricula next. Do not move to upstream unfreezing or
new estimators until the reduced-supervision boundary is known.
