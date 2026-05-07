# 2026-05-06 - Learned interface warm start and teacher-zero retention

Task: run the Phase 4 supervised interface warm start and teacher-zero
retention pipeline from the validated `sum_left_operand` Stage 0B semantic
decoder.

## Code change

Added an opt-in calculator read mode:

```text
calculator_read_position=operand_spans
calculator_read_span_width=2
```

This preserves existing `eq` and `operands` behavior, but lets
`calculator_hook.input_proj` read the full fixed-width A and B digit spans. The
original `operands` mode reads only the final digit positions.

Why this was needed:

- Requested final-digit-only Stage 1 (`input_proj_lr=0.003`) reached only
  final eval exact `0.273`; best snapshot operand exact `0.355`.
- Higher-LR final-digit continuation improved to final eval exact `0.727`;
  best snapshot operand exact `0.715`, still below the warm-start gate.
- With `operand_spans`, the interface crossed the gate cleanly while keeping
  semantic decoder and upstream frozen and keeping trainable parameters limited
  to `calculator_hook.input_proj`.

Validation:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile src/model.py scripts/overfit_one_batch.py scripts/diagnose_calculator_protocol.py scripts/run_non_bottleneck_protocol_experiments.py
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m pytest tests/test_data.py tests/test_model.py -q
```

Result:

```text
72 passed
```

## Stage 1 selected warm start

Run:

```text
runs/2026-05-06_192430_233405_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed3
```

Selected handoff:

```text
runs/2026-05-06_192430_233405_model-c-op0-19-adaptive_interface-inlr0.03-uplr0.003-answer_decoder-sum_left_operand-aux1/model-c-2digit-seed3/final_weights.pt
```

Config highlights:

- `calculator_read_position=operand_spans`
- `calculator_read_span_width=2`
- `answer_loss_weight=0.0`
- `aux_operand_loss_weight=1.0`
- `adaptive_interface_loss_weight=0.0`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- trainable groups: `calculator_hook.input_proj` only (`1320` params)

Fast gate:

- Final eval exact: `1.000` (`512/512`)
- Step `200` and later snapshots: normal exact `1.000`, operand exact `1.000`,
  oracle exact `1.000`

## Stage 2 selected aux-zero retention

Run:

```text
runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3
```

Selected checkpoint:

```text
runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/final_weights.pt
```

Config proof:

- `answer_loss_weight=1.0`
- `aux_operand_loss_weight=0.0`
- `final_aux_operand_loss_weight=0.0`
- `adaptive_interface_loss_weight=0.0`
- `final_adaptive_interface_loss_weight=0.0`
- `final_input_proj_anchor_weight=0.0`
- `freeze_semantic_decoder=true`
- `freeze_upstream_encoder=true`
- trainable groups: `calculator_hook.input_proj` only (`1320` params)

Fast gate:

- Final eval exact: `1.000` (`512/512`)
- Final loss: `0.000198`
- Final snapshot: normal exact `1.000`, operand exact `1.000`,
  injection-zero exact `0.004`, oracle exact `1.000`

Canonical diagnostics:

```text
runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/canonical_causal_diagnostics
```

- Samples `256`
- Normal exact `1.000`
- Injection-zero exact `0.000`
- Forced-zero exact `0.0039`
- Forced-random exact `0.0313`
- Oracle-at-eval exact `1.000`
- Operand exact `1.000`
- Pair exact `1.000`
- Calculator-result accuracy `1.000`
- Classification `intended_true_operand_calculator_use`

Private protocol:

```text
runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/private_protocol_diagnostics
```

- All `20 x 20` pairs
- Exact match `1.000`
- Operand exact `1.000`
- Pair exact `1.000`
- Calculator-result accuracy `1.000`
- Learned A/B identity affine mappings exact `1.000`

Full-enum action loss:

```text
runs/2026-05-06_195001_156276_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-sum_left_operand/model-c-2digit-seed3/full_enum_action_loss/model-c-2digit-seed3
```

- Samples `128`
- Learned-best fraction `1.000`
- True-best fraction `1.000`
- Best-matches-true-operands fraction `1.000`
- Mean learned-minus-true gap `0.0`
- Mean learned-minus-best gap `0.0`

## Interpretation

This is a positive one-seed protocol-teaching and teacher-zero retention result,
not oracle-only success. The selected Stage 2 checkpoint has direct operand
supervision exactly removed and still emits the true learned operands.

The important caveat is mechanical: the result required full-span readout for
two-digit operands. The original final-digit-only `operands` readout is not a
sufficient Stage 1 warm-start mechanism under this frozen upstream checkpoint.

## Recommendation

Go to seed replication with `calculator_read_position=operand_spans` before
trying broader objectives or upstream unfreezing.
