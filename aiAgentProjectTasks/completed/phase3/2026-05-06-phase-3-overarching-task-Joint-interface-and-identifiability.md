# Phase 3 Overarching Task: Joint Interface and Identifiability

## Mission

Phase 2 produced a useful narrowing result, not a dead end for the whole project.

The strict answer-decoder bottleneck, oracle controls, dense checkpoint selection, full-enum action diagnostics, and private-protocol decoding now show a consistent picture:

```text
The downstream answer decoder can use correct calculator actions.
The learned interface can retain a partially causal protocol after supervision.
Answer-NLL-derived action signal exists.
But the current independent A/B input projection does not reliably learn or preserve
a clean true-operand protocol under frozen-upstream, no-aux Phase 2 training.
```

Phase 3 should stop treating "find a better STE replacement" as the main research frame. Other discrete-learning techniques are still welcome, but the evidence now points to a deeper issue:

```text
Addition answer loss identifies useful result-producing action pairs, not necessarily
the unique true operand pair.
The current objective often scores joint (a,b) actions, then collapses that signal
into independent A and B marginal targets.
That marginalization can erase pair structure.
```

The Phase 3 goal is to test whether better action representation, sharper identifiability, or an honest curriculum can move beyond the Phase 2 ceiling.

## Phase 2 Carry-Forward Facts

Preserve these as established guardrails:

- The `answer_decoder` bottleneck is the strongest current anti-bypass mechanism.
- Oracle-at-eval recovery remains a wiring/control metric, not success by itself.
- Injection-zero and forced-random controls are required for calculator-dependence claims.
- Dense checkpointing and checkpoint selection are now part of the standard workflow.
- `scripts/run_causal_calculator_protocol_diagnostics.py`, `scripts/run_action_loss_diagnostic.py`, `scripts/run_full_enum_action_loss_diagnostic.py`, and `scripts/diagnose_private_protocol.py` are the canonical diagnostic stack.
- Full enumeration over `20 x 20` actions is a good teacher-quality diagnostic, but the Phase 2 marginal A/B implementation did not justify upstream unfreezing.

## Phase 3 Research Bet

Old Phase 2 bet:

```text
If we stabilize the independent A/B input projection under a strict bottleneck,
answer-loss-derived targets may eventually improve the retained calculator protocol.
```

Updated Phase 3 bet:

```text
The interface may need to represent calculator actions jointly, or the task signal
may need to make operands identifiable, before a hidden tool protocol can be learned
without permanent true-operand supervision.
```

This is still the same overall project: connecting a non-differentiable calculator directly into a tiny transformer's internal computation. Phase 3 changes the interface and objective, not the north star.

## Primary Phase 3 Tracks

### Track A: Joint Pair-Action Interface

Replace the independent operand heads with a joint action distribution over `(a,b)` pairs.

Core idea:

```text
Score or train over the full 20 x 20 action space directly.
Do not marginalize the full-enum teacher into independent A/B distributions before training.
```

This is the most direct next experiment because it attacks the clearest Phase 2 failure mode.

Candidate implementations:

- A joint linear head from concatenated A/B read vectors to `operand_vocab_size ** 2` logits.
- A low-rank or factorized pair head only after the direct joint head establishes the baseline.
- A joint Gumbel/ST action sampler if hard differentiable-ish action selection is needed.

### Track B: Identifiability Curriculum

If the project wants true operand extraction, addition-only answer loss may be underidentified.

Test tasks where incorrect operand pairs cannot hide behind the same calculator result, for example:

- Multi-operation prompts with addition/subtraction/multiplication-style tool outputs.
- A calculator returning extra structured information such as `(a, b, sum)` during curriculum-only phases.
- Auxiliary tool queries that require operand identity, then decay to ordinary addition.

This track should distinguish "calculator result use" from "true operand protocol learning."

### Track C: Honest Supervision and Retention

True-operand supervision may not be a contaminant. It may be the correct way to teach the hidden tool-call language.

Continue to study:

- How much direct operand supervision is needed.
- Whether the interface retains the protocol after supervision reaches exactly `0.0`.
- Whether retention transfers across seeds, operand ranges, or model sizes.
- Whether upstream unfreezing preserves or destroys a supervised protocol.

Do not sell these as pure answer-loss discovery. Sell them as protocol teaching and retention.

## What Phase 3 Should Stop Doing

Do not spend more primary time on:

- Plain STE answer-only variants in the old independent A/B interface.
- Single-sample REINFORCE variants unless paired with a changed joint action representation.
- More adaptive-interface LR/entropy/soft-result sweeps in the existing Phase 2 form.
- More replay/EMA sampled-candidate work in the `20 x 20` regime now that full enumeration exists.
- Long continuations without dense checkpoint selection.
- Upstream unfreezing before the interface itself has stronger evidence.

These tools can remain as controls or baselines, but they should not define the next phase.

## Standard Phase 3 Evaluation Contract

Every primary Phase 3 checkpoint should report:

- Built-in eval exact.
- Canonical causal normal exact.
- Injection-zero, forced-zero, forced-random, and oracle-at-eval exact.
- Canonical causal classification and bottleneck label.
- Canonical action-loss learned, true, random, and shuffled NLL.
- Learned-minus-true and learned-minus-best gaps when available.
- Tie-aware or result-equivalence-aware action-best metrics where appropriate.
- Private-protocol all-pair answer exact, operand exact, and calculator-result accuracy.
- Group behavior for carry/no-carry, small/large operands, and symmetric pairs.
- Proof that aux weight, anchor weight, upstream freezing, and trainable parameter groups match the intended claim.

For joint-action runs, add:

- Pair exact match.
- Result-equivalent pair accuracy.
- Pair-distribution entropy/effective action count.
- True pair probability or rank, for reporting only.
- Best-pair probability/rank under full-enum answer NLL.

## Success Criteria

A useful Phase 3 positive result does not need to solve the whole project. It should show at least one of:

- Joint pair-action training improves learned action NLL and private-protocol all-pair accuracy beyond the best Phase 2 checkpoints.
- The learned joint action becomes result-equivalent to the true action substantially more often than the independent A/B interface.
- A curriculum creates a protocol that is retained after supervision is exactly `0.0`, remains calculator-dependent, and beats Phase 2 retention.
- A changed identifiability task reveals that operand protocols are learnable when the answer signal actually identifies operands.

A useful Phase 3 negative result should show:

- The bottleneck and oracle controls remain healthy.
- The teacher/action landscape has been measured directly.
- The joint representation or curriculum still fails across multiple seeds.
- Diagnostics distinguish optimization failure from non-identifiability.

## First Task

Start with Track A.

Create and evaluate a joint pair-action interface under the same strict Phase 2 bottleneck. Use full-enum answer-NLL pair targets directly, without independent A/B marginalization. Compare against the best Phase 2 independent-head checkpoints and decide whether the joint representation breaks the current ceiling.

See:

```text
aiAgentProjectTasks/2026-05-06-phase-3-first-task-Joint-pair-action-interface.md
```

