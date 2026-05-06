# Identifiability Curriculum After Joint-Smoke Collapse

## Mission

The first Track A joint pair-action smoke preserved bottleneck controls but failed the gate:

```text
The full-enum answer-NLL teacher strongly preferred true/result-equivalent actions.
The learned joint head stayed nearly uniform by entropy.
Argmax actions collapsed to a small set of result classes.
Pair exact and result-equivalent pair accuracy stayed near zero.
```

This task starts Track B:

```text
Test whether operand identity becomes learnable when the task/tool curriculum makes
operand identity identifiable, rather than relying on addition-only answer loss.
```

## Starting Evidence

Joint smoke run:

```text
runs/2026-05-06_082124_888126_model-c-op0-19-action_loss_full_enum_joint_interface-joint_pair-inlr0.0003-uplr0.0003-fullt1-fullchunk64-answer_decoder/model-c-2digit-seed103
```

Key diagnostic numbers:

```text
canonical injection-zero exact: 0.0000
canonical oracle-at-eval exact: 0.9063
canonical pair exact: 0.0000
private all-pair answer exact: 0.0450
private pair exact: 0.0000
private calculator-result accuracy: 0.0275
full-enum learned-minus-true gap: 5.6210
full-enum learned-best fraction: 0.0000
full-enum teacher effective pairs: 29.2116
pair-logit effective pairs: 399.7591
```

## Hypothesis

Addition-only answer loss identifies many useful result-producing action pairs but does not reliably force a hidden true-operand protocol. A curriculum that makes operand identity visible in the downstream task should create a cleaner interface protocol that can later be tested for retention after identity supervision is removed.

## Fixed Controls

Preserve the strict bottleneck controls unless the task explicitly varies them:

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
freeze_semantic_decoder=true
answer_loss_weight=1.0
input_proj_anchor_weight=0.0 unless explicitly testing retention anchors
```

For primary claims, final checkpoints must still prove:

```text
final_aux_operand_loss_weight exactly 0.0 when claiming post-curriculum retention
final_input_proj_anchor_weight exactly 0.0 unless the claim is explicitly anchor-based
injection-zero near zero
oracle-at-eval high
```

## Candidate Curriculum Designs

Choose one narrow design first.

### Option A: Multi-Operation Identifiability

Use prompts that require operation-specific tool outputs, for example addition plus subtraction-like or ordered-pair-sensitive answers. The same `(sum)` should no longer be enough to solve all prompts.

Success signal:

```text
learned actions improve on pair exact and action-loss gap
retained post-curriculum checkpoint remains calculator-dependent
```

### Option B: Structured Calculator Curriculum

During curriculum only, let the calculator output structured information such as `(a, b, sum)` through a bottlenecked decoder. Then decay structured supervision/output back to ordinary addition.

Success signal:

```text
the interface learns true operands during curriculum
some operand protocol survives after structured outputs are removed or decayed
```

### Option C: Auxiliary Identity Query

Interleave ordinary addition prompts with prompts whose answer requires one operand identity directly, for example "return A", "return B", or "return ordered pair", while using the same calculator interface.

Success signal:

```text
identity-query accuracy and addition accuracy both depend on calculator actions
post-query-retention beats Phase 2 and the joint-smoke baseline
```

## Required Diagnostics

For every primary checkpoint:

- Built-in eval exact by task type.
- Canonical causal diagnostics by task type where applicable.
- Injection-zero, forced-zero, forced-random, and oracle-at-eval exact.
- Pair exact and result-equivalent pair accuracy.
- Full-enum learned, true, and best NLL.
- Learned-minus-true and learned-minus-best gaps.
- Learned-best and tie-aware learned-best fractions.
- Private-protocol all-pair answer exact, operand exact, pair exact, and calculator-result accuracy.
- Group behavior for carry/no-carry, large/small operands, and symmetric pairs.
- Proof of final aux/anchor/upstream-freezing/trainable parameter groups.

## Decision Criteria

A positive result must show a retained checkpoint, after identity-specific curriculum pressure is removed or reaches exactly zero, that beats the joint-smoke baseline on at least two:

```text
pair exact
result-equivalent pair accuracy
learned-minus-true or learned-minus-best action-loss gap
private-protocol calculator-result accuracy
private-protocol all-pair answer exact
```

while preserving:

```text
injection-zero near zero
oracle-at-eval high
strict bottleneck controls
```

A useful negative result should distinguish:

- curriculum did not teach operand identity;
- identity was learned but not retained;
- retained behavior was not calculator-dependent;
- addition-only evaluation still underidentified the protocol.

## Required Reporting

Write a Phase 3 work history and update `factSheets/PHASE_3_EXPERIMENT_FACT_SHEET.md` with:

- exact code changes;
- exact commands and run paths;
- curriculum design chosen;
- final zero-supervision/retention proof;
- canonical causal table;
- action-loss table;
- private-protocol table;
- comparison against the joint-smoke run and best Phase 2 checkpoints;
- go/no-go recommendation for the following Phase 3 task.
