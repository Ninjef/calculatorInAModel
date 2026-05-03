# Action-Loss-Aligned Self-Training Under the Lower-LR Retention Window

## Mission

The previous phase-2 task showed that the lower-LR Stage C retention result is real but time-limited:

```text
1000-step Stage C continuations with input_proj_lr=0.0003, upstream frozen,
no anchor, and aux_operand_loss_weight=0.0 replicated across three seeds and
beat the high-LR drift control.

The 3000-step continuation regressed, and canonical diagnostics stayed at
causally_useful_opaque_private_code / strict_bottleneck_unvalidated.
```

The next best question is whether we can turn retention into self-improvement without reintroducing true-operand supervision or unfreezing upstream.

Primary questions:

```text
1. Can action-loss-aware checkpoint selection reliably pick better retained interfaces?
2. Can an action-loss-aligned training signal improve learned actions under the strict bottleneck?
3. Does self-training improve true operand/calculator-result behavior, or only stabilize a private code?
```

Do not unfreeze upstream yet. Do not add true-operand auxiliary supervision. Do not claim a learned true-operand protocol unless canonical diagnostics and private-protocol decoding support it.

## Starting Points

Stage B handoff checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Prior lower-LR checkpoint:

```text
runs/2026-05-01_114850_238336_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Best replication checkpoint:

```text
runs/2026-05-01_122402_250571_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed3/final_weights.pt
```

Long-regressed checkpoint:

```text
runs/2026-05-01_123323_807024_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Semantic decoder root checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Useful diagnostics from the previous task:

```text
scripts/run_causal_calculator_protocol_diagnostics.py
scripts/run_action_loss_diagnostic.py
scripts/diagnose_private_protocol.py
```

## Fixed Regime

Keep the strict bottleneck setup:

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
calculator_bottleneck_mode=answer_decoder
calculator_estimator=adaptive_interface unless explicitly adding a new named estimator
adaptive_interface_target_mode=hard_pair for the baseline
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
aux_operand_loss_weight=0.0 for all primary claims
input_proj_anchor_weight=0.0 for all primary claims
```

Success evidence must come from checkpoints whose true-operand aux weight is exactly `0.0`, upstream is frozen, and anchor weight is exactly `0.0`.

## Part 1: Checkpoint Selection Within the Retention Window

Run lower-LR Stage C with dense snapshots so we can ask whether checkpoint selection beats final-step selection.

Required primary runs:

| Run | Start | Steps | Input LR | Adaptive loss | Aux weight | Seed | Anchor | Snapshots |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| C-low-lr-dense-seed1 | Stage B | 1500 | `0.0003` | `1.0` | `0.0` | `1` | none | every `50` |
| C-low-lr-dense-seed2 | Stage B | 1500 | `0.0003` | `1.0` | `0.0` | `2` | none | every `50` |
| C-low-lr-dense-seed3 | Stage B | 1500 | `0.0003` | `1.0` | `0.0` | `3` | none | every `50` |

If existing snapshot files do not save weights, add a narrow checkpoint-saving option such as:

```text
--checkpoint-every
```

or extend `--snapshot-every` to optionally save weights. Keep the implementation scoped and document exactly what is saved.

For each dense run:

- evaluate final checkpoint;
- evaluate the best snapshot by built-in normal exact;
- evaluate the best snapshot by learned-target agreement;
- evaluate the best snapshot by canonical action-loss learned-minus-true gap;
- compare whether selection would have avoided the 3000-step drift behavior.

Important: if selecting by action-loss requires evaluating many checkpoints, use a smaller sample count first, then re-run canonical diagnostics on the selected checkpoints.

## Part 2: Add a Candidate Action-Loss-Aligned Objective

Add one narrow training variant that uses answer loss itself to reward better calculator actions without true operand labels.

Preferred first candidate:

```text
calculator_estimator=action_loss_weighted_interface
```

Suggested behavior:

- For each prompt, score a small candidate set of calculator actions through the frozen answer decoder.
- Candidate set must include:
  - the current learned action;
  - random actions;
  - optionally top-k actions from the current input projection;
  - optionally local perturbations around learned A/B.
- Convert candidate answer NLLs into a soft target distribution over action pairs.
- Train only `calculator_hook.input_proj` toward that soft target.
- Do not use true operands or true sums to create the target distribution.
- Keep this checkpoint-first and strictly bottlenecked; no upstream gradients.

If this is too large for one pass, implement a diagnostic-only prototype first:

```text
scripts/run_action_loss_candidate_diagnostic.py
```

The diagnostic should estimate whether answer-NLL-ranked candidate actions often contain a better action than the learned action. If the candidate pool almost never contains better actions, report that before adding a training objective.

## Part 3: Required Training Variants

Run these only after the candidate diagnostic says there is usable signal:

| Run | Start | Steps | Input LR | Objective | Aux weight | Seed | Anchor |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| C-actionloss-selftrain-seed1 | Stage B | 1000 | `0.0003` | action-loss weighted | `0.0` | `1` | none |
| C-actionloss-selftrain-seed2 | Stage B | 1000 | `0.0003` | action-loss weighted | `0.0` | `2` | none |
| C-actionloss-selftrain-seed3 | Stage B | 1000 | `0.0003` | action-loss weighted | `0.0` | `3` | none |

Optional ablations if the primary runs are promising:

| Run | Change |
| --- | --- |
| Smaller candidate pool | Verify improvement is not just expensive search. |
| Larger candidate pool | Check whether signal quality is candidate-limited. |
| Mixed objective | Blend baseline adaptive-interface hard-pair loss with action-loss weighted target. |
| Stop-selected continuation | Train to 1500 but select by action-loss gap. |

Do not run anchor fallback unless self-training is unstable and the report clearly labels the result as retention-stabilized rather than primary adaptive-only evidence.

## Part 4: Canonical Diagnostics

Use canonical names in commands and reports:

```bash
PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py ...
PYTHONPATH=. python3 scripts/diagnose_private_protocol.py ...
```

For at least these checkpoints:

- Stage B handoff;
- prior `C-low-lr`;
- best lower-LR replication;
- long-regressed checkpoint;
- each dense-run final checkpoint;
- each selected dense-run checkpoint;
- each action-loss self-training checkpoint if run;

report:

- injection-zero, forced-zero, forced-random, and oracle-at-eval;
- canonical causal classification and bottleneck label;
- forced-result sweep learned-best fraction;
- forced-result sweep true-sum-best fraction;
- canonical action-loss true, learned, random, and shuffled NLL;
- learned-minus-true, random-minus-true, and shuffled-minus-true gaps;
- true-best and learned-best fractions;
- private-protocol all-pair operand exact, calculator-result accuracy, and majority/affine mapping improvement;
- per-group carry/no-carry and large/small operand behavior for the best selected checkpoint.

## Decision Criteria

A useful checkpoint-selection result requires:

- selected checkpoints beat final checkpoints in at least two of three dense runs on canonical action-loss learned-minus-true gap;
- selection does not merely pick high answer exact with worse operand/calculator-result accuracy;
- selected checkpoints preserve injection-zero near zero and oracle-at-eval high;
- selected checkpoints keep `final_aux_operand_loss_weight=0.0`, frozen upstream, and no anchor.

A useful action-loss self-training result requires:

- at least two of three self-training runs beat the prior lower-LR replications on action-loss learned-minus-true gap;
- at least one self-training run improves operand exact or calculator-result accuracy over the best prior replication;
- learned-best action-loss fraction becomes meaningfully nonzero, or the report explains why the objective improved NLL without making learned actions best;
- canonical classification improves, or private-protocol decoding shows a clearer true-operand-like structure.

A useful negative result should distinguish:

- checkpoint selection helps, but training objective does not;
- candidate actions contain better actions, but optimization fails;
- candidate actions rarely contain better actions, making the objective candidate-limited;
- action-loss self-training improves answer behavior but worsens true operand/calculator-result structure;
- the private code is stable enough for retention but not trainable under answer-NLL-derived targets.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes, if any;
- proof that aux weight stayed exactly `0.0`;
- trainable parameter groups for every primary run;
- checkpoint-selection table comparing final vs selected snapshots;
- candidate-action diagnostic table;
- self-training metric table if training is run;
- comparison against Stage B, prior C-low-lr, best replication, drift control, and long-regressed checkpoint;
- canonical causal diagnostic summary;
- canonical action-loss summary;
- private-protocol decoding summary;
- parameter delta table versus both the semantic decoder checkpoint and Stage B handoff;
- explicit go/no-go recommendation for upstream unfreezing.

Also update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

This task succeeds if it tells us whether the next path is checkpoint selection, action-loss self-training, a better candidate-action search, or a different objective entirely. The goal is to find the first no-operand-supervision signal that can improve the retained calculator interface rather than merely preserve it.
