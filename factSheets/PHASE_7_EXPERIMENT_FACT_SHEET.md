# Phase 7 Experiment Fact Sheet

## Direction

Phase 7 tests natural `0..19` result-level calculator use.

The phase starts from the Phase 6 closure:

```text
Deterministic hard-forward / soft-backward Concrete can discover and retain an
identifiable `sum_left_operand` hard calculator protocol, but natural sum-only
addition failed because the answer identifies a result group rather than a
unique operand pair.
```

Phase 7 should therefore prioritize structured joint-pair or result-space
interfaces that match the result-level information available in natural answer
loss.

## Starting Guardrail

Oracle/readout success is a wiring gate only. Phase 7 progress must be judged
by learned calculator-result behavior under the hard calculator path:

- learned calculator-result accuracy;
- result-equivalent pair accuracy;
- private all-pair result accuracy;
- full-enum learned-result best fraction and learned-result gaps;
- injection-zero and forced-random controls;
- semantic decoder movement exactly `0.0`;
- auxiliary/direct operand supervision exactly `0.0`;
- all discovery-specific objective weights exactly `0.0` for retention claims.

Exact true operand-pair recovery is diagnostic only in natural sum-only
addition, because many valid calculator calls share the same correct result.

## First Recommended Track

Start with the Phase 7 overarching plan:

```text
aiAgentProjectTasks/2026-05-12-phase-7-overarching_plan-Natural-result-level-interface-discovery.md
```

The first task should gate or implement a natural joint-pair result-group
deterministic Concrete bridge before attempting larger operand ranges.
