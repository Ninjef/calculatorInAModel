# Phase 4 Overarching Task: Identifiability-First Protocol Teaching

## Mission

Phase 3 should be treated as completed.

Its core diagnosis was correct:

```text
The downstream answer decoder can use correct calculator actions.
The current bottleneck and diagnostics are useful.
Addition-only answer loss underidentifies the true operand pair.
```

But the Phase 3 joint-pair implementation did not break the Phase 2 ceiling. The
best Phase 3 retained joint checkpoints stayed far below the best Phase 2
independent-head retention checkpoints on private all-pair answer accuracy,
operand/pair exact match, calculator-result accuracy, and learned-vs-true
action-loss gap. The joint pair logits also stayed close to uniform over all
`20 x 20` actions.

Phase 4 should therefore stop asking whether another estimator can squeeze a
clean operand protocol out of the same underidentified addition-only task. The
next phase should ask a sharper question:

```text
Can the model learn and retain a calculator-query protocol when the task signal
itself makes operand identity useful or necessary?
```

## Anti-Waste Rule for Future Agents

Do not report oracle calculator success as a discovery. It is a known Phase 1
fact that downstream components can answer correctly when oracle operands or
correct calculator outputs are supplied. Oracle-only runs are allowed only as
minimal wiring checks after a code change that could break the bottleneck or
semantic decoder. They are not evidence that the model learned how to use a
calculator.

For Phase 4, the question is upstream/interface learning. Spend experiment
budget on:

- supervised interface warm starts;
- aux-zero or teacher-zero retention;
- private all-pair protocol decoding;
- learned operand/pair exact match;
- learned calculator-result accuracy;
- learned-vs-true and learned-vs-best action-loss gaps;
- seed/checkpoint replication once a learned-interface result is nontrivial.

If an oracle semantic decoder checkpoint already exists and has passed
injection-zero/forced-random wiring controls, use it as infrastructure and move
on. Do not rerun it unless the relevant wiring changed.

## Phase 3 Carry-Forward Facts

Preserve these as established guardrails:

- The `answer_decoder` bottleneck remains the strongest anti-bypass setup.
- Oracle-at-eval recovery is a wiring/control metric, not success.
- Injection-zero and forced-random controls are mandatory for calculator-dependence claims.
- Dense checkpointing and checkpoint selection are mandatory for retention claims.
- Full-enum action-loss diagnostics are excellent teacher-quality diagnostics in the `20 x 20` regime.
- Private all-pair protocol decoding is the clearest end-to-end protocol metric.
- Supervision is not contamination if the claim is protocol teaching and retention.

Preserve these as negative results:

- Plain STE answer-only variants are not a promising primary direction.
- Single-sample REINFORCE connected gradients but did not discover the intended protocol.
- Adaptive-interface LR/entropy/soft-result ladders mostly diagnosed collapse.
- Replay/EMA sampled-candidate methods are obsolete in the `20 x 20` setting now that full enumeration exists.
- The current direct joint pair head did not learn a sharp action distribution.
- Upstream unfreezing before a stable interface exists tends to create more drift than clarity.

## Phase 4 Research Bet

Old Phase 3 bet:

```text
A joint action representation or a light identity curriculum might move beyond
the Phase 2 independent-head retention ceiling.
```

Updated Phase 4 bet:

```text
The project needs an identifiable task or an explicit protocol-teaching phase.
Once a protocol is learned, the central research question becomes retention:
how much of that protocol survives when direct supervision reaches exactly 0.0?
```

This is still the same project: a non-differentiable calculator wired into a
tiny transformer. Phase 4 changes the learning environment so the intended
hidden tool language is actually learnable.

## Primary Tracks

### Track A: Identifiable Calculator Tasks

Create tasks where the final answer cannot be solved by any operand pair with
the same sum.

Candidate task families:

- Multi-output arithmetic: predict both `a+b` and `a-b`, or `a+b` and `a`.
- Multi-operation prompts: randomly request `sum`, `diff`, `min`, `max`, parity, or carry.
- Structured calculator curriculum: temporarily return or decode `(a, b, sum)` before decaying back to ordinary addition.
- Ambiguity-breaking prompts: include a second query whose answer depends on one operand, not only on `a+b`.

The first task should be minimal: keep `digits=2`, `operand_max=19`, the
`answer_decoder` bottleneck, and the existing small model. Add the smallest data
and evaluation change that makes operand identity matter.

### Track B: Honest Protocol Teaching and Retention

Treat true-operand supervision as teaching the hidden tool-call language.

Study:

- How much operand supervision is needed before retention appears.
- Whether retention survives with `aux_operand_loss_weight == 0.0`.
- Whether retention transfers across seeds and operand groups.
- Whether a supervised protocol can be retained without upstream unfreezing.
- Whether upstream unfreezing helps only after interface-only retention is stable.

This track should not be presented as pure answer-loss discovery. The claim is
cleaner and stronger if it is framed as tool-protocol teaching.

### Track C: Estimators Only After Identifiability

REINFORCE, Gumbel-Softmax, surrogate gradients, zeroth-order estimators, and RL
methods should not define the next phase yet. They become interesting only after
the task signal is identifiable enough that a better estimator has something
unambiguous to optimize.

Recommended order:

1. Identifiable supervised/decayed curriculum.
2. Full-enum action-loss teacher in the identifiable task.
3. Gumbel-Softmax or joint sampling only if full enum is too expensive or too broad.
4. REINFORCE/RL only for larger action spaces where enumeration is impossible.

## What Phase 4 Should Stop Doing

Do not spend primary time on:

- More old independent-head STE answer-only experiments.
- More single-sample REINFORCE on addition-only prompts.
- More LR/entropy sweeps for the existing adaptive-interface objective.
- More sampled-candidate replay/EMA in `20 x 20`.
- More broad joint-pair ladders before improving identifiability.
- Full diagnostics on every weak run.
- Upstream unfreezing as a default early move.

These can remain controls, not the main line.

## Fast Evaluation Gates

Use cheap gates before expensive diagnostics.

Every exploratory checkpoint should report:

- Built-in eval exact.
- Normal calculator trace exact.
- Injection-zero exact.
- Forced-random exact.
- Oracle-at-eval exact.
- Operand or pair exact.
- Calculator-result accuracy.
- Entropy/effective action count for the active interface.
- Final aux weight, anchor weight, upstream-freeze status, and trainable groups.

Only run full canonical/private/action diagnostics when a checkpoint clears at
least one of these gates:

- operand or pair exact match above `0.35`;
- calculator-result accuracy above `0.40`;
- normal exact is calculator-dependent and above `0.35`;
- learned-minus-true action-loss gap beats the Phase 2 selected-start range;
- the run is a necessary negative control.

## Standard Success Criteria

A useful Phase 4 positive result should show all of:

- Aux/supervision weight is exactly `0.0` at the selected checkpoint.
- Injection-zero and forced-random are near chance.
- Oracle-at-eval stays high.
- Private all-pair operand or pair exact beats the best Phase 2 retention range.
- Calculator-result accuracy beats the best Phase 2 retention range.
- Learned-vs-true action-loss gap improves over Phase 2.
- The result holds across at least two seeds or clearly explains seed variance.

A useful Phase 4 negative result should show:

- The bottleneck and oracle controls remain healthy.
- The identifiable task genuinely rewards operand identity.
- The learned interface still fails despite identifiable signal.
- Diagnostics distinguish task-design failure from optimization failure.

## First Task

Start with Track A.

Create the smallest identifiable-task extension to the current data/model stack.
Preferred first version:

```text
Keep the existing addition prompt shape as much as possible, but train/evaluate
an answer format that requires both the sum and one operand-derived value.
```

Examples:

```text
07+12=19,07<eos>
07+12=19,-05<eos>
07+12=19,carry1<eos>
```

Choose the variant that requires the least invasive data/model change while
making true operand identity harder to fake than in plain addition. Run a
three-seed compact ladder with dense checkpoints, then compare retained
aux-zero checkpoints against the best Phase 2 retention baselines and the best
Phase 3 joint checkpoints.

## Reporting Contract

Every completed Phase 4 task should end with:

- The exact claim being tested.
- The run paths and selected checkpoints.
- Whether the selected checkpoint has aux exactly `0.0`.
- Fast-gate metrics.
- Full diagnostics only for selected or control checkpoints.
- A direct comparison to best Phase 2 and Phase 3 results.
- A go/no-go recommendation for the next task.
