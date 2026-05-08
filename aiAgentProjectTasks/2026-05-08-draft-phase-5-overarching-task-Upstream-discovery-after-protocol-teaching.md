# Draft Phase 5 Overarching Task: Upstream Discovery After Protocol Teaching

This is a draft handoff for the next agent to review and revise after reading
`CLAUDE.md`, `factSheets/PHASE_4_EXPERIMENT_FACT_SHEET.md`, and the Phase 4
work histories.

## Phase 4 Closure

Phase 4 should be treated as completed.

Its core result was:

```text
With an identifiable answer target and a frozen readable upstream
representation, answer loss can complete a partially taught calculator-query
protocol after direct operand supervision is exactly removed, but only above a
seed-dependent handoff quality.
```

This is a real learned-interface positive, but it is not pure discovery from
answer loss alone.

Established Phase 4 facts:

- The `sum_left_operand` answer target makes operand identity useful.
- `calculator_output_format=sum_left_operand` is required for the strict
  answer decoder to use the operand-aware signal.
- `calculator_read_position=operand_spans` exposes enough frozen upstream
  information for `calculator_hook.input_proj` to learn true operands.
- With direct operand supervision, the interface learns the true operand
  protocol across effective seeds `2`, `4`, and `5`.
- After direct operand supervision is set exactly to `0.0`, answer loss can
  retain or complete the protocol from sufficiently good partial handoffs.
- The completion boundary is seed-dependent:
  - seed `2`: failed step `55` handoff at Stage 1 operand `0.438`; retained
    step `60` handoff at `0.641`.
  - seed `4`: failed step `30` handoff around `0.19`; retained step `35`
    handoff at `0.363`.
  - seed `5`: failed step `25` handoff at `0.078`; retained step `30` handoff
    around `0.20`.
- Decayed-aux curricula that mixed answer loss before a useful protocol formed
  did not reach retention quality.

The main limitation:

```text
Phase 4 taught a protocol into a frozen readable interface. It did not show
that the model discovers the calculator-query protocol from ordinary answer
loss without a direct or staged teacher.
```

## Phase 5 Mission

Phase 5 should test whether the project can move from protocol teaching toward
protocol discovery or transfer.

Recommended question:

```text
Can a model acquire or recover a calculator-query protocol with less direct
operand teaching by using the Phase 4 identifiable target and a staged,
diagnosable training setup?
```

Do not spend Phase 5 rediscovering Phase 4 retention. Use Phase 4 checkpoints
and diagnostics as baselines.

## Anti-Waste Rules

- Do not report oracle calculator success as progress. It is only a wiring
  check.
- Do not call high answer exact alone calculator-use success.
- Do not call a checkpoint retained unless learned operand exact, pair exact,
  calculator-result accuracy, private all-pair decoding, and full-enum gaps
  agree.
- Do not rerun broad Phase 4 ladders unless code changes invalidate them.
- Do not start with new estimators before making the discovery/transfer
  question explicit.

## Recommended Tracks

### Track A: Upstream-Unfreeze After Stable Interface

Start from a Phase 4 retained aux-zero checkpoint and unfreeze a narrow upstream
slice while keeping the calculator interface and diagnostics intact.

Questions:

- Does upstream unfreezing preserve the learned protocol or create drift?
- Can upstream learn representations that make lower-quality handoffs retain?
- Does a staged unfreeze improve private all-pair decoding or full-enum gaps
  without reintroducing direct operand supervision?

Constraints:

- Keep `answer_format=sum_left_operand`.
- Keep `calculator_output_format=sum_left_operand`.
- Keep strict `answer_decoder` bottleneck.
- Start from selected Phase 4 retained checkpoints, not random restarts.
- Use small learning rates and dense checkpoints.

### Track B: Less-Teacher Discovery Curriculum

Search for a curriculum that reduces direct operand teaching while still
crossing the Phase 4 handoff boundary.

Candidate curricula:

- Short aux warm start followed by answer-only continuation, with the handoff
  chosen by measured Stage 1 quality.
- Sparse or intermittent operand labels.
- Operand supervision on only one side, then answer-only completion.
- Transfer a learned `input_proj` from one seed or operand subset to another,
  then remove aux.

Required comparison:

- Compare against Phase 4 decayed-aux negatives. If a new curriculum succeeds,
  explain why it is not merely the failed Stage 1B setup with different
  bookkeeping.

### Track C: Broader Identifiable Tasks

Only after Track A or B has a clean result, broaden the task signal.

Candidates:

- `sum_diff`
- `sum_left_operand` transfer to held-out operand groups
- randomly requested operations such as `sum`, `diff`, `min`, `max`, or carry

Keep the first broadened task small enough that private all-pair and full-enum
diagnostics remain practical.

## First Concrete Phase 5 Task Candidate

Run an upstream-unfreeze smoke test from a Phase 4 retained checkpoint.

Suggested starting checkpoint:

```text
runs/2026-05-07_phase4_min_supervision_boundary/stage2/seed2/step60/.../model-c-2digit-seed2/final_weights.pt
```

If the exact path is needed, read it from:

```text
runs/2026-05-07_phase4_min_supervision_boundary/summary.json
```

Experiment sketch:

1. Start from the retained seed `2` step `60` aux-zero checkpoint.
2. Continue with `aux_operand_loss_weight=0.0`,
   `adaptive_interface_loss_weight=0.0`, and answer loss only.
3. Compare:
   - interface-only trainable baseline;
   - unfreeze only the calculator hook input projection plus the hook-read
     layer block;
   - optionally unfreeze all upstream encoder layers only if the narrow unfreeze
     is stable.
4. Use dense snapshots and run full diagnostics on any checkpoint that appears
   retained or any checkpoint that drifts.

Success would be modest:

```text
Upstream unfreezing preserves the learned protocol and does not introduce a
shortcut or private-code confound.
```

Failure would also be useful:

```text
Upstream unfreezing causes protocol drift, showing Phase 5 needs a more careful
regularized transfer or less-teacher curriculum before broader discovery.
```

## Reporting Contract

Every Phase 5 task should report:

- the exact claim being tested;
- whether it is testing discovery, retention, transfer, or stability;
- run paths and selected checkpoints;
- final aux/adaptive/anchor weights;
- freeze/unfreeze settings and trainable parameter groups;
- normal, injection-zero, forced-random, and oracle-at-eval exact;
- learned operand exact, pair exact, and calculator-result accuracy;
- private all-pair protocol metrics;
- full-enum learned-minus-true and learned-minus-best gaps;
- comparison to Phase 4 retained and failed boundary checkpoints;
- a go/no-go recommendation for the next task.
