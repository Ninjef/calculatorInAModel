# Phase 5 Closure: Upstream Discovery After Protocol Teaching

## Status

Phase 5 is complete.

Phase 5 asked whether the Phase 4 taught-and-retained calculator-query protocol
could survive, transfer, improve, or be discovered once upstream/model-side
parameters were allowed to move.

The completed Phase 5 answer is:

```text
Upstream movement can preserve and complete an already partially taught
calculator-query protocol, but current answer-only training does not discover
the protocol without a supervised handoff or local teaching signal.
```

This closes Phase 5 as a meaningful bridge phase, not as a pure-discovery
positive.

## Starting Point From Phase 4

Phase 4 established a real learned-interface positive:

```text
With an identifiable answer target and a frozen readable upstream
representation, answer loss can complete a partially taught calculator-query
protocol after direct operand supervision is exactly removed, but only above a
seed-dependent handoff quality.
```

Important Phase 4 facts carried into Phase 5:

- `answer_format=sum_left_operand` makes operand identity useful.
- `calculator_output_format=sum_left_operand` is required for the strict
  answer decoder to use the operand-aware signal.
- `calculator_read_position=operand_spans` exposes enough frozen upstream
  information for `calculator_hook.input_proj` to learn true two-digit operands.
- Direct operand supervision can teach the true protocol across multiple seeds.
- Answer loss can retain or complete the protocol after direct teacher weights
  are exactly `0.0`, but only from sufficiently good handoffs.

Phase 4 did not show answer-only discovery from scratch.

## Completed Phase 5 Tracks

### Track A: Upstream-Unfreeze Stability

Task:

```text
aiAgentProjectTasks/completed/phase5/2026-05-08-phase-5-first-task-Upstream-unfreeze-stability-smoke.md
```

Result:

- Starting from a retained Phase 4 seed `2`, step `60` checkpoint, opening
  upstream parameters with conservative LR preserved the true protocol at the
  final checkpoint.
- Upstream parameters moved measurably while the semantic decoder stayed frozen.
- The result was a cautious stability positive, not pure discovery.
- Dense snapshots showed transient protocol degradation, so final success was
  not all-snapshot stability.

### Track B1: Upstream-Assisted Completion, Seed 2

Task:

```text
aiAgentProjectTasks/completed/phase5/2026-05-08-phase-5-second-task-Upstream-assisted-partial-handoff-completion.md
```

Result:

- Starting from the known failed seed `2`, Stage 1 step `55` partial handoff,
  upstream-open answer-only continuation recovered to retained-protocol quality.
- The matched frozen-upstream continuation stayed partial.
- Direct teacher weights were exactly `0.0`.
- Semantic decoder movement stayed `0.0`; upstream and input-proj parameters
  moved measurably.
- This was upstream-assisted completion, not discovery from scratch.

### Track B2: Cross-Seed Completion Replication, Seed 5

Task:

```text
aiAgentProjectTasks/completed/phase5/2026-05-08-phase-5-third-task-Cross-seed-upstream-assisted-completion-replication.md
```

Result:

- Starting from the much lower seed `5`, Stage 1 step `25` failed handoff,
  upstream-open answer-only continuation reached an exact selected checkpoint
  at step `950`.
- The final checkpoint drifted mildly, and an optional anchor repeat also
  reached an exact selected checkpoint before drifting more strongly.
- This replicated completion across seeds, but made checkpoint selection and
  dense diagnostics mandatory.

### Track C: No-Handoff Full-Model Discovery Smoke

Task:

```text
aiAgentProjectTasks/completed/phase5/2026-05-09-phase-5-fourth-task-Controlled-no-handoff-upstream-discovery-smoke.md
```

Result:

- Added a backward-compatible
  `--semantic-decoder-checkpoint-load-scope full_model | semantic_decoder_only`
  flag.
- Ran the allowed two no-handoff full-model seeds from the Stage 0B checkpoint,
  with no Stage 1 supervised interface handoff and direct teacher weights
  exactly `0.0`.
- Oracle-at-eval stayed `1.0`, so the fixed semantic decoder/calculator path
  remained mechanically viable.
- Best checkpoints reached only partial protocol alignment:
  - seed `0`: best step `650`, canonical operand/pair/calc `0.4297`
  - seed `3`: best step `350`, canonical operand/pair/calc `0.4336`
- Full-enum learned-minus-true/best gaps remained strongly positive.
- Final checkpoints drifted close to chance learned actions.
- Because this branch did not produce a no-handoff discovery checkpoint, the
  optional strict random-upstream branch was correctly not run.

## Phase 5 Final Claims

Supported:

- Upstream-open continuations can preserve a previously learned true
  calculator-query protocol.
- Upstream-open answer-only continuations can complete failed partial handoffs
  after direct operand supervision is exactly removed.
- This completion replicated across seeds `2` and `5`.
- The strict Phase 4/5 bottleneck and diagnostics can separate real learned
  protocol behavior from oracle wiring and answer-only shortcuts.

Not supported:

- Pure answer-only discovery of the calculator-query protocol without a
  supervised handoff.
- All-snapshot stability of upstream-open continuations.
- Strict random-upstream discovery.
- Any claim that oracle-at-eval success is research progress rather than a
  wiring gate.

Best concise project-level conclusion:

```text
The architecture can support a learned calculator-query protocol, and answer
loss can preserve or complete that protocol once partially taught. However,
plain answer-only training has not yet discovered the protocol without a
supervised or local teaching signal.
```

## Go/No-Go Decision

No-go for more Phase 5 seed/LR sweeping.

Go to a new phase focused on one explicit interface-discovery training signal,
while keeping the Phase 4/5 identifiable task and diagnostics fixed.

Best next directions:

1. Minimal local-target / full-enum target-prop style objective.
2. Gumbel-Softmax / Concrete relaxation for the discrete calculator actions.
3. Later, revisit strict random-upstream discovery only after a local objective
   or relaxation can pass the full-model no-handoff benchmark.

## Reporting Contract To Keep

Future phases should preserve the Phase 5 diagnostic standard:

- learned operand exact;
- learned pair exact;
- learned calculator-result accuracy;
- private all-pair protocol decoding;
- full-enum learned-minus-true and learned-minus-best gaps;
- exact teacher weights at the selected checkpoint;
- oracle-at-eval only as a wiring gate, never as a learned-use claim;
- dense checkpoints whenever upstream or interface parameters are trainable.
