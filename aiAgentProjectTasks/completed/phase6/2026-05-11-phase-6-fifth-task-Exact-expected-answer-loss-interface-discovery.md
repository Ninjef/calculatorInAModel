# Phase 6 Fifth Task: Exact Expected Answer-Loss Interface Discovery

## Mission

Test whether the strict identifiable calculator interface can be learned from
answer loss itself, without true-operand labels, oracle operands, or a
hard-best local pseudo-label.

Phase 6 has already shown that the answer-derived hard-best local target is
sharp and can teach the true calculator-query protocol, even in the strict
`semantic_decoder_only` branch. That is strong evidence that answer NLL contains
the right information. The remaining question is whether the model-side
interface can use that information as an objective directly, rather than being
trained toward an argmin target selected outside the model.

This task should implement and test an exact full-enum expected answer-loss
objective:

```text
C_i(a,b) = answer NLL for prompt i when calculator action (a,b) is forced
p_i(a,b) = p_theta(a | prompt i) * p_theta(b | prompt i)
L_i      = sum_{a,b} p_i(a,b) * stopgrad(C_i(a,b))
```

This is the exact low-variance form of policy-gradient credit assignment over
the `20 x 20` action space. It uses only answer labels and counterfactual
calculator calls. It must not use true operands, true sums, or hard-best pairs
to construct the training loss.

## Why This Is The Next Best Step

Helpful results so far:

- The `sum_left_operand` identifiable setup made the full-enum action landscape
  sharp: best pair equals true operands `1.000`, effective pairs about `1.078`,
  and true-pair probability about `0.988`.
- The matched hard-best local target solved Stage 1 and Stage 2 retention in
  the full-model branch.
- The strict `semantic_decoder_only` branch also solved Stage 1 and retained
  exact protocol metrics after the local target was exactly `0.0`; this removed
  dependence on the Stage 0B oracle-trained upstream representation.
- Minimum-handoff work showed that answer-only continuation can improve partial
  protocols, but it needs a near-gated interface: step `25` and `50` improved
  but did not become exact, while step `75` retained exactly.
- Dense checkpointing and full diagnostics are essential; final-only metrics
  have repeatedly hidden transient drift or partial recovery.

Less helpful directions right now:

- Oracle-only reruns. Oracle-at-eval is a wiring gate, not progress.
- More broad no-handoff answer-only sweeps. Phase 5 already made that negative.
- More single-sample REINFORCE. It had too much variance and could solve answer
  accuracy while leaving the calculator protocol wrong.
- More sampled candidate/replay action-loss work in the small `20 x 20` space.
  Full enumeration is available and has already proven sharper.
- Simple linear local-target decay. The strict decay ladder failed through
  `150` decay steps despite exact target sharpness.
- Returning to addition-only full-enum targets. That setting was
  underidentified; the current identifiable setup is the useful one.

The next bottleneck is not whether the answer-loss landscape identifies the
right action. It does. The next bottleneck is whether an interface policy can
minimize expected answer loss directly and end with a hard argmax protocol that
matches the calculator query.

## Fixed Setup

Unless a branch explicitly says otherwise, keep the Phase 6 strict setup:

```text
digits=2
operand_max=19
calculator_operand_vocab_size=20
n_layer=2
n_head=1
n_embd=16
mlp_expansion=1
calculator_hook_after_layer=1
answer_format=sum_left_operand
calculator_output_format=sum_left_operand
calculator_read_position=operand_spans
calculator_read_span_width=2
calculator_bottleneck_mode=answer_decoder
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_semantic_decoder=true
oracle_train=false
oracle_warmup_steps=0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
```

Use the standard Stage 0B semantic decoder checkpoint recorded in the Phase 4
and Phase 6 fact sheets.

## Critical Guardrail

Do not call this objective a local target unless the implementation actually
converts answer losses into CE targets. The primary implementation should not
do that.

Allowed:

- answer NLL from forced calculator actions;
- the model's own action probabilities;
- exact enumeration of all action pairs;
- optional detached per-example cost centering or scaling for optimization;
- optional entropy bonus/schedule over the model action distribution.

Forbidden for training loss construction:

- true operand targets;
- true sum targets;
- hard-best pair CE;
- soft target CE distilled from `softmax(-C/T)`;
- oracle operands during training;
- direct `aux_operand_loss`;
- semantic decoder movement.

True operands may appear only in diagnostics and parity reports.

## Implementation Requirements

Add a narrowly named estimator, for example:

```text
calculator_estimator=full_enum_expected_answer_loss
```

or:

```text
calculator_estimator=expected_answer_loss_full_enum_interface
```

Required behavior:

- Enumerate all `20 x 20` action pairs for each prompt in the batch.
- Force each action pair through the frozen answer decoder and compute answer
  NLL.
- Compute the model action distribution from the current learned operand
  logits. Start with independent operand heads:

```text
p(a,b) = softmax(a_logits / policy_temperature)[a]
       * softmax(b_logits / policy_temperature)[b]
```

- Minimize expected answer NLL under that distribution:

```text
mean_i sum_{a,b} p_i(a,b) * detached_answer_nll_i(a,b)
```

- Keep calculator forward/eval behavior hard-argmax, so success must show up in
  learned operand/pair/calc metrics, not just lower expected loss.
- Record enough metrics to diagnose distribution collapse:
  expected answer NLL, best NLL, true NLL for reporting only, learned NLL,
  expected-minus-best gap, learned-minus-best gap, learned-minus-true gap,
  policy entropy/effective pairs, probability mass on the best pair, probability
  mass on the true pair for reporting only, hard learned-best fraction, and
  hard learned pair exact.

Recommended knobs:

```text
--expected-answer-loss-policy-temperature
--expected-answer-loss-cost-normalization none|center|zscore
--expected-answer-loss-entropy-weight
--expected-answer-loss-entropy-decay-steps
--expected-answer-loss-chunk-size
```

If existing `--adaptive-interface-loss-weight` machinery is reused to avoid CLI
churn, metrics must still label the objective as expected answer loss, not as a
hard-best local target.

## Stage 0: Gradient And Objective Gate

Before running full training, add a compact gate that proves the objective is
wired correctly under `semantic_decoder_only`.

Report on a fixed 128-sample batch:

- oracle-at-eval exact `1.000` as a wiring gate;
- injection-zero and forced-random near chance;
- expected answer-loss value;
- best/true/learned NLLs;
- hard learned operand/pair/calc at initialization;
- policy entropy/effective pairs at initialization;
- one-step parameter delta for `calculator_hook.input_proj`;
- one-step parameter delta for upstream, depending on freeze setting;
- semantic decoder grad and delta exactly `0.0`;
- no aux, anchor, oracle, or hard-best local-target construction.

Do not proceed if the one-step expected-loss objective does not move
`calculator_hook.input_proj`.

## Stage 1: Frozen-Upstream Strict Expected-Loss Training

Primary branch:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=0.0
expected_answer_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
steps=300
snapshot_every=25
checkpoint_every=25
input_proj_lr=0.03
```

Run a tiny optimization ladder only if needed:

| Branch | Policy temp | Entropy | Input LR | Reason |
| --- | ---: | ---: | ---: | --- |
| A | `1.0` | `0.0` | `0.03` | Pure expected answer loss |
| B | `1.0` | small decayed | `0.03` | Prevent early wrong collapse |
| C | `0.5` | `0.0` | `0.03` | Sharper but still expected-loss based |

Stop the ladder as soon as a branch reaches a fast-gate protocol threshold of
at least `0.90`. Do not turn this into a broad sweep.

Selection criteria:

- fast-gate normal/operand/pair/calc;
- policy entropy/effective pairs;
- hard learned pair exact;
- full-enum learned-minus-true and learned-minus-best gaps;
- private all-pair protocol metrics on selected checkpoints.

## Stage 2: Hard-Argmax Answer-Only Retention

If Stage 1 reaches a checkpoint with fast-gate operand/pair/calc at or above
`0.90`, continue from the first qualifying checkpoint and the best qualifying
checkpoint:

```text
calculator_estimator=adaptive_interface
semantic_decoder_checkpoint_load_scope=full_model
freeze_semantic_decoder=true
freeze_upstream_encoder=true
answer_loss_weight=1.0
expected_answer_loss_weight=0.0
adaptive_interface_loss_weight=0.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0
input_proj_lr=0.0003
steps=1000
snapshot_every=50
checkpoint_every=50
```

This stage tests whether a protocol learned by expected answer loss survives
when only the normal hard-argmax answer path remains.

## Stage 3: Upstream-Open Strict Branch

Run this only after Stage 1/2 frozen-upstream succeeds, or if Stage 1 clearly
fails because the frozen random upstream representation is limiting the policy.

Use the same expected-loss objective with:

```text
semantic_decoder_checkpoint_load_scope=semantic_decoder_only
freeze_upstream_encoder=false
input_proj_lr=0.03
upstream_lr=0.00003 or 0.0001
```

Keep dense checkpoints. The important question is whether upstream movement
helps the interface discover a protocol from expected answer loss, not whether
final answer exact alone rises.

## Required Diagnostics

For every selected checkpoint, run:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/diagnose_private_protocol.py
scripts/run_full_enum_action_loss_diagnostic.py
```

Report:

- built-in eval exact;
- normal exact;
- injection-zero exact;
- forced-zero exact;
- forced-random exact;
- oracle-at-eval exact;
- learned operand exact;
- learned pair exact;
- learned calculator-result accuracy;
- private all-pair answer/operand/pair/calc;
- full-enum learned NLL, true NLL, best NLL;
- learned-minus-true and learned-minus-best gaps;
- learned-best and true-best fractions;
- expected-loss policy entropy/effective pairs;
- expected-loss probability mass on selected hard action;
- probability mass on the true pair for reporting only;
- trainable parameter groups;
- input-proj, upstream, and semantic decoder parameter deltas;
- final weights for answer, expected answer loss, local target/adaptive
  interface, aux, and anchor objectives.

## Success Criteria

Useful positive:

```text
The expected answer-loss objective moves a strict semantic-decoder-only
interface from random hard actions to materially better learned operand/pair/calc
metrics without hard-best CE targets or true-operand supervision.
```

Strong positive:

```text
Stage 1 reaches near-exact hard learned protocol metrics, and Stage 2 retains
that protocol after expected answer loss is exactly off.
```

Very strong positive:

```text
The upstream-open strict branch also learns and retains the protocol while the
semantic decoder remains frozen and aux/local/hard-best target weights are
exactly 0.0.
```

## Failure Interpretation

If the objective has no useful gradient:

```text
The answer-loss landscape is sharp, but expected-loss credit assignment is not
reaching the interface under the current implementation. Fix wiring before
interpreting research results.
```

If expected loss decreases but hard argmax protocol stays wrong:

```text
The model is spreading probability mass over useful actions without producing a
usable hard calculator query. Next try entropy annealing, lower policy
temperature, or a Gumbel/Concrete hard-forward soft-backward bridge.
```

If Stage 1 learns but Stage 2 retention fails:

```text
Expected answer loss can teach the interface, but hard answer loss cannot hold
it. Revisit gate-triggered handoff or short expected-loss floor schedules; do
not return to linear decay by default.
```

If frozen-upstream fails but upstream-open improves:

```text
The strict random upstream representation was limiting the readout, and true
model-side representation learning is necessary. Continue upstream-open with
dense diagnostics and conservative LR.
```

## Reporting Contract

When complete, update:

```text
factSheets/PHASE_6_EXPERIMENT_FACT_SHEET.md
aiAgentWorkHistory/phase6/<date>-exact-expected-answer-loss-interface-discovery.md
```

Include:

- code changes;
- exact commands;
- run paths;
- Stage 0 gradient/objective gate;
- Stage 1 and Stage 2 tables;
- Stage 3 table if run;
- selected checkpoints;
- objective-weight proof that no true-operand aux, oracle, hard-best target, or
  anchor was active;
- parameter movement summary;
- comparison to the strict hard-best local-target result and the strict decay
  negative;
- go/no-go recommendation for Gumbel/Concrete or upstream-open continuation.

When the task is complete, move this file to:

```text
aiAgentProjectTasks/completed/phase6/
```

then commit and push.
