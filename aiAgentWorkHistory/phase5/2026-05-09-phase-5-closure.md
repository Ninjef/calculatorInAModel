# 2026-05-09 - Phase 5 closure

Task: finalize Phase 5 after the no-handoff upstream discovery smoke.

## Closure Decision

Phase 5 is complete.

Final conclusion:

```text
Upstream movement can preserve and complete an already partially taught
calculator-query protocol, but current answer-only training does not discover
the protocol without a supervised handoff or local teaching signal.
```

## What I Changed

- Moved the Phase 5 overarching task into the completed Phase 5 folder as a
  closure artifact:
  `aiAgentProjectTasks/completed/phase5/2026-05-09-phase-5-closure-Upstream-discovery-after-protocol-teaching.md`
- Rewrote that artifact from a planning document into a closure summary.
- Moved the Phase 5 third and fourth completed task files into
  `aiAgentProjectTasks/completed/phase5/` for consistency with the first two
  Phase 5 tasks.
- Added a Phase 5 closure section to
  `factSheets/PHASE_5_EXPERIMENT_FACT_SHEET.md`.

## Scientific State

Supported:

- Upstream-open continuations can preserve a previously retained protocol.
- Upstream-open answer-only continuations can complete failed partial handoffs.
- Completion replicated across seed `2` and seed `5`.
- Dense checkpoint selection and full diagnostics remain necessary.

Not supported:

- No-handoff answer-only discovery.
- Strict random-upstream discovery.
- All-snapshot stability.
- Oracle-at-eval success as a learned-use claim.

## Next Recommendation

Do not keep Phase 5 open for broad seed or LR sweeps. Start a new phase focused
on one explicit interface-discovery training signal:

- minimal full-enum/local-target target-prop style objective; or
- Gumbel-Softmax / Concrete relaxation.

The next phase should keep the strict Phase 4/5 setup fixed:

- `answer_format=sum_left_operand`
- `calculator_output_format=sum_left_operand`
- `calculator_read_position=operand_spans`
- `calculator_bottleneck_mode=answer_decoder`
- dense checkpoints
- canonical/private/full-enum diagnostics
- direct teacher weights exactly `0.0` for any discovery claim
