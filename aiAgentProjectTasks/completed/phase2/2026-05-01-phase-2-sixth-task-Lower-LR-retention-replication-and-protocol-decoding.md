# Lower-LR Retention Replication and Protocol Decoding Under Strict Bottleneck

## Mission

The previous phase-2 task found the first stable post-supervision retention result:

```text
Starting from the Stage B supervision-zero handoff, a 1000-step Stage C continuation
with input_proj_lr=0.0003, upstream frozen, no anchor, and aux_operand_loss_weight=0.0
preserved or improved the learned calculator interface.
```

The next task is to check whether this is a robust finding and to understand what kind of interface was retained.

Primary questions:

```text
1. Does lower-LR Stage C retention replicate across seeds/checkpoint samples?
2. Is the retained interface moving toward true operands, or is it a stable private code?
3. If it is private code, what structure does that code have?
```

Do not unfreeze upstream yet. Do not introduce new true-operand supervision. Do not claim a clean learned protocol unless canonical diagnostics support it.

## Starting Points

Primary Stage B handoff checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Best current lower-LR Stage C checkpoint:

```text
runs/2026-05-01_114850_238336_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Semantic decoder root checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Immediate drift/control checkpoint:

```text
runs/2026-05-01_114850_422866_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2/final_weights.pt
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
calculator_estimator=adaptive_interface
adaptive_interface_target_mode=hard_pair
freeze_semantic_decoder=true
freeze_upstream_encoder=true
trainable=calculator_hook.input_proj only
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
input_proj_anchor_weight=0.0 for primary claims
```

Success evidence must come from checkpoints whose true-operand aux weight is exactly `0.0`.

## Part 1: Replicate the Lower-LR Retention Result

Run a small replication ladder from the Stage B checkpoint.

Required primary variants:

| Run | Start | Steps | Input LR | Adaptive loss | Aux weight | Seed | Anchor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C-low-lr-rep-seed1 | Stage B | 1000 | `0.0003` | `1.0` | `0.0` | `1` | none |
| C-low-lr-rep-seed2 | Stage B | 1000 | `0.0003` | `1.0` | `0.0` | `2` | none |
| C-low-lr-rep-seed3 | Stage B | 1000 | `0.0003` | `1.0` | `0.0` | `3` | none |

Important seed note:

- The existing successful run used CLI `--seed 0`, which produced run name `seed2` because the script adds `num_digits`.
- For this task, record both the CLI seed and the resulting run-name seed.

If the three replications are stable, add one longer continuation:

| Run | Start | Steps | Input LR | Adaptive loss | Aux weight | Anchor |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C-low-lr-long | Stage B | 3000 | `0.0003` | `1.0` | `0.0` | none |

If the replications are unstable, do not immediately unfreeze upstream. Instead run the fallback anchor diagnostic:

| Run | Start | Steps | Input LR | Adaptive loss | Aux weight | Anchor |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| C-anchor-rep-fallback | Stage B | 1000 | `0.0003` | `1.0` | `0.0` | Stage B, weight `0.001` |

Report anchor results separately as retention-stabilized, not pure adaptive-only.

## Part 2: Protocol Decoding Diagnostics

The previous result stayed classified as:

```text
causally_useful_opaque_private_code
strict_bottleneck_unvalidated
```

This task should inspect what the private code is doing.

Use existing diagnostics where possible. Add narrow checkpoint-first diagnostics only if needed.

Required analyses for Stage B, the prior successful C-low-lr, and the best replication:

- learned `a_pred`, `b_pred`, and learned result distribution over all `0..19` operand pairs;
- confusion matrices: true `a` vs learned `a`, true `b` vs learned `b`, true sum vs learned result;
- per-operand exact rates, especially whether errors concentrate on carries, large operands, or symmetric operand pairs;
- mutual information already exposed by canonical diagnostics, summarized in the report;
- whether a simple affine/permutation mapping from learned operand classes to true operand values improves operand exact or calculator-result accuracy;
- whether learned result classes form a stable code for true sums even when operand classes are not individually true;
- read-vector intervention sensitivity: corrupt/swap A/B read vectors and compare normal, corrupt, and swap behavior;
- top examples where learned operands are wrong but final answer is right;
- top examples where learned operands are right but final answer is wrong.

If adding code, prefer a new script with a clear checkpoint-first name, for example:

```text
scripts/diagnose_private_protocol.py
```

Keep it read-only with respect to model weights.

## Part 3: Canonical Diagnostics

Use canonical names in commands and reports:

```bash
PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py ...
```

For at least these checkpoints:

- Stage B handoff;
- previous `C-low-lr`;
- each replicated `C-low-lr` run;
- `C-low-lr-long` if run;
- anchor fallback if run;

report:

- injection-zero, forced-zero, forced-random, and oracle-at-eval;
- canonical causal classification and bottleneck label;
- forced-result sweep learned-best fraction;
- forced-result sweep true-sum-best fraction;
- canonical action-loss true, learned, random, and shuffled NLL;
- learned-minus-true, random-minus-true, and shuffled-minus-true gaps;
- true-best and learned-best fractions.

## Decision Criteria

A robust positive retention result requires:

- every primary checkpoint has `final_aux_operand_loss_weight=0.0`;
- no upstream unfreezing;
- no anchor in the primary claim;
- at least two of three replication runs beat the prior Stage C drift checkpoint on operand exact, calculator-result accuracy, learned-target agreement, and learned-minus-true action-loss gap;
- at least one replication is close to the original `C-low-lr` result;
- injection-zero remains near zero and oracle-at-eval remains high;
- true-sum forced-result sweep remains high.

A stronger protocol result requires one of:

- canonical classification improves beyond `causally_useful_opaque_private_code`;
- operand exact and calculator-result accuracy become reliably high across replications;
- a stable, simple learned-code mapping explains most of the gap between learned operands and true operands;
- learned-best action-loss fraction becomes nonzero in a meaningful way.

A useful negative result should distinguish:

- lower LR was a lucky single-run result;
- lower LR retains answer performance but not the operand protocol;
- the private code is stable and causal but not true-operand-like;
- the code is not stable across seeds;
- the answer decoder can use true sums, but learned actions remain far from action-loss optimal.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes, if any;
- starting Stage B checkpoint path;
- proof that aux weight stayed exactly `0.0`;
- trainable parameter groups for every primary run;
- replication metric table;
- comparison table against Stage B, prior Stage C drift, and previous `C-low-lr`;
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

This task succeeds if it tells us whether the lower-LR post-supervision retention result is repeatable, and whether the retained interface is best understood as a true-operand protocol, a stable private code, or an unstable artifact.
