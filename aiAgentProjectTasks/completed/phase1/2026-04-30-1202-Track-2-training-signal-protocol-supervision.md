# Track 2: Training-Signal Scientist - Protocol Supervision and Retention

Status: completed.

Completion report:

```text
aiAgentWorkHistory/2026-04-30-track-2-training-signal-protocol-supervision.md
```

Key outcome: auxiliary supervision did not induce or retain the intended true-operand protocol in the tested `operand_max=19` regime. Oracle replacement works, but the first `calculator_injection_mode=replace` bottleneck is leaky in autoregressive generation because later answer-token positions can still use normal residual context.

## Mission

Determine how much explicit training signal is needed for the model to learn, retain, and rely on the intended calculator protocol.

This track answers:

> Is the internal calculator protocol emergent from answer loss, or does it need explicit API-like shaping?

Use Track 1's `calculator_read_position=operands` interface once it exists. The Track 1 first rung should be treated as evidence that the non-bottleneck setup can bypass the calculator under answer loss alone. This track should therefore test protocol supervision in two settings:

- the existing non-bottleneck residual-injection setup, for continuity with prior runs;
- a calculator-required bottleneck setup, where the downstream answer path must use the calculator result or fail.

## Tactical Work

Focus on auxiliary operand supervision and curriculum design:

- Train the calculator operand head toward true A/B labels.
- Measure whether the protocol survives when auxiliary pressure is reduced or removed.
- Keep answer loss as the ultimate task, but protocol metrics as the success gate.
- Add a bottleneck variant where normal residual bypass is removed or compressed enough that arithmetic success requires the calculator path.
- Treat the bottleneck as the primary capability test and the non-bottleneck setup as a retention/emergence stress test.

Recommended schedules:

- constant small aux: `0.003`, `0.01`, `0.03`;
- aux decay to zero;
- aux decay to a small floor;
- supervised operand-head warmup, then answer-only continuation;
- oracle-operand warmup only if clearly separated from aux-head supervision.

If useful, add CLI/config support for cleaner schedule expression, but keep it simple and test-covered.

## Experiments

Start with:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
calculator_read_position=operands
```

Compare:

- pure Model C answer-loss baseline from Track 1;
- Model C with constant aux;
- Model C with decayed aux;
- Model C with aux warmup then no aux;
- optional oracle-warmup variants if the above clarifies rather than confuses.

Also run the same supervision schedules in a calculator-required bottleneck variant:

- oracle-bottleneck control: true operands go into the calculator and downstream answer generation must work;
- supervised-encoder bottleneck: A/B heads are directly trained, then evaluated with hard calculator outputs;
- supervised-to-answer continuation: start from a supervised encoder, remove or decay aux pressure, and test whether answer loss preserves the protocol;
- answer-loss-only bottleneck baseline: no aux, no oracle, no normal residual bypass.

Move to `operand_max=49` only after the `0..19` protocol behavior is clear.

## Required Metrics

Report:

- answer exact match;
- true operand exact match;
- calculator result accuracy;
- aux loss curve;
- answer loss curve;
- protocol retention after aux decay/removal;
- injection-zero exact match;
- forced-zero exact match;
- forced-random exact match;
- oracle-at-eval exact match;
- bottleneck mode and bypass/compression setting;
- oracle-bottleneck exact match;
- exact run path and seed.

For decay/warmup runs, include checkpoint snapshots before, during, and after the schedule transition.

## Success Criteria

A strong positive result:

- Aux supervision induces the intended protocol.
- The protocol remains after aux is reduced or removed.
- The calculator remains causally useful under counterfactual tests.
- In the bottleneck setting, answer accuracy collapses when the calculator result is removed or corrupted.

A useful partial result:

- Aux supervision induces protocol use, but the protocol disappears when aux is removed.
- This would imply that hidden tool use may require persistent protocol-level supervision.
- The bottleneck learns under supervision but not answer loss alone, implying capability exists but credit assignment is weak.

A strong negative result:

- Even direct operand-head supervision fails under the operand-read interface.
- Or the oracle-bottleneck control fails, which would point to implementation, capacity, or downstream decoding problems rather than weak answer-loss signal.

## Guardrails

- Do not present auxiliary-supervised success as pure emergence.
- Keep pure answer-loss and aux-supervised results clearly separated.
- Do not tune only for final answer exact match.
- Preserve comparability with Track 1's architecture and ladder.
- Do not treat non-bottleneck success as calculator reliance unless counterfactuals show dependence.
