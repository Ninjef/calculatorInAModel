# Post-Supervision Retention Stabilization Under Strict Bottleneck

## Mission

The previous phase-2 task produced the first useful partial positive result:

```text
A clean aux-only warm start made a nontrivial input-proj calculator interface.
The supervision-zero handoff improved further.
The final adaptive-only continuation retained materially better behavior than prior baselines, but degraded.
```

The next task is to stabilize that post-supervision phase.

Do not rerun broad estimator sweeps. Start from the known good supervision-zero handoff and isolate why Stage C drifts after true-operand supervision is gone.

Primary question:

```text
Can we preserve or improve the Stage B zero-supervision interface during a longer Stage C without any true-operand supervision?
```

Success evidence must come from checkpoints whose true-operand aux weight is exactly `0.0`.

## Starting Point

Use the best current supervision-zero handoff as the main starting checkpoint:

```text
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2/final_weights.pt
```

Reference final drift checkpoint:

```text
runs/2026-05-01_112843_524620_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Semantic decoder root checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Keep the strict bottleneck regime:

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
aux_operand_loss_weight=0.0
```

## Why This Is The Best Next Step

The current evidence says the bottleneck and downstream decoder are healthy:

- oracle-at-eval remains high;
- true-sum forced-result sweep remains best on `0.921875` of prompts;
- true actions have much lower action-loss NLL than random or shuffled actions.

The new failure mode is not "no interface can be learned." It is:

```text
Stage B zero handoff: useful interface, operand exact around 0.48 in built-in trace.
Stage C final: partial retention, operand exact around 0.26, learned-target agreement drops.
```

So the next experiment should not ask for another way to create the interface. It should ask how to keep the already-created interface from drifting under adaptive-only optimization.

## Part 1: Add Minimal Stabilization Knobs If Needed

If existing options can express all variants cleanly, do not modify code.

If not, add narrowly scoped support for one or both of these retention stabilizers:

### Lower-LR / Longer Stage C

This likely needs no code change. Use lower `--input-proj-lr` while loading the Stage B checkpoint.

Required variants:

```text
input_proj_lr in {3e-4, 1e-4}
steps in {500, 1000}
adaptive_interface_loss_weight=1.0
answer_loss_weight=1.0
aux_operand_loss_weight=0.0
```

### Checkpoint-Relative Interface Anchor

If lower LR alone is not enough, add a checkpoint-relative L2 anchor on `calculator_hook.input_proj`:

```text
--input-proj-anchor-checkpoint <path>
--input-proj-anchor-weight <float>
--input-proj-anchor-decay-steps <int>
```

Anchor target should be the Stage B zero-supervision handoff checkpoint, not the original semantic decoder checkpoint.

Important interpretation rule:

- An anchor is allowed as a retention stabilizer because it uses no true operand labels.
- But anchor-active runs are not "adaptive objective alone." Report them separately from pure adaptive-only lower-LR runs.

Suggested anchor ladder:

```text
input_proj_lr=3e-4
input_proj_anchor_weight in {0.001, 0.01, 0.03}
input_proj_anchor_decay_steps in {0, 500}
steps=1000
```

Record per-run:

- anchor checkpoint path;
- anchor weight at each logged step;
- anchor loss at each logged step;
- final anchor weight;
- input-proj delta from both the semantic decoder checkpoint and the Stage B anchor checkpoint.

## Part 2: Drift Diagnosis During Stage C

For each Stage C run, save or reconstruct enough per-step diagnostics to identify drift:

- built-in diagnostic snapshots every `100` or `200` steps;
- aux operand CE even though aux weight is `0.0`;
- adaptive target result accuracy;
- learned-target agreement;
- operand exact match;
- calculator result accuracy;
- mean operand entropy/confidence;
- `input_proj` L2/max delta from the Stage B handoff;
- `input_proj` L2/max delta from the semantic decoder checkpoint.

In the report, explicitly distinguish:

- true-operand protocol drift: operand exact and calculator result accuracy fall;
- private-code drift: causal normal stays high but operand exact falls;
- objective drift: adaptive learned-target agreement falls while target-result accuracy stays high;
- confidence drift: entropy collapses or confidence spikes without accuracy improving.

## Part 3: Experiment Ladder

Run this focused ladder first:

| Run | Start | Steps | Input LR | Answer loss | Adaptive loss | Aux weight | Anchor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C-control-repeat | Stage B | 500 | `0.003` | `1.0` | `1.0` | `0.0` | none |
| C-low-lr | Stage B | 1000 | `0.0003` | `1.0` | `1.0` | `0.0` | none |
| C-very-low-lr | Stage B | 1000 | `0.0001` | `1.0` | `1.0` | `0.0` | none |
| C-adaptive-low-weight | Stage B | 1000 | `0.0003` | `1.0` | `0.3` | `0.0` | none |
| C-anchor-light | Stage B | 1000 | `0.0003` | `1.0` | `1.0` | `0.0` | Stage B, weight `0.001` |
| C-anchor-medium | Stage B | 1000 | `0.0003` | `1.0` | `1.0` | `0.0` | Stage B, weight `0.01` |

Stop early only if the first pure adaptive-only lower-LR variants clearly preserve Stage B-level performance. If that happens, prioritize canonical diagnostics on those runs over adding anchor variants.

Do not unfreeze upstream in the first pass. Upstream unfreezing should wait until input-proj-only retention is stable.

## Canonical Diagnostics

Use canonical names in commands and reports:

```bash
PYTHONPATH=. python3 scripts/run_causal_calculator_protocol_diagnostics.py ...
PYTHONPATH=. python3 scripts/run_action_loss_diagnostic.py ...
```

For at least these checkpoints:

- Stage B starting handoff;
- control repeat final;
- best pure adaptive-only lower-LR final;
- best anchor final if anchor variants are run;

report:

- injection-zero, forced-zero, forced-random, and oracle-at-eval;
- canonical causal classification and bottleneck label;
- forced-result sweep learned-best fraction;
- forced-result sweep true-sum-best fraction;
- canonical action-loss true, learned, random, and shuffled NLL;
- learned-minus-true, random-minus-true, and shuffled-minus-true gaps;
- true-best and learned-best fractions.

## Comparison Baselines

Compare against:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
runs/2026-05-01_080805_707417_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder/model-c-2digit-seed2
runs/2026-05-01_085013_079390_model-c-op0-19-adaptive_interface-inlr0.0003-uplr0.0003-answer_decoder-aux0.03-auxdecay500/model-c-2digit-seed2
runs/2026-05-01_112523_133504_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder-aux1-auxdecay500/model-c-2digit-seed2
runs/2026-05-01_112843_524620_model-c-op0-19-adaptive_interface-inlr0.003-uplr0.003-answer_decoder/model-c-2digit-seed2
```

The Stage B handoff is the retention target. The Stage C final drift checkpoint is the immediate baseline to beat.

## Decision Criteria

A strong positive result requires all of:

- final true-operand aux weight exactly `0.0`;
- no upstream unfreezing in the primary positive claim;
- final eval and diagnostics after at least `1000` Stage C steps unless the run is explicitly a 500-step control;
- final operand exact and calculator result accuracy at least as good as the prior Stage C final and preferably close to Stage B;
- learned-target agreement better than the prior Stage C final (`0.1953`) and preferably close to Stage B (`0.3828`);
- action-loss learned-minus-true gap better than prior Stage C (`5.5241`) and preferably close to or better than Stage B (`3.0642`);
- calculator dependence under counterfactuals: injection-zero near zero, oracle-at-eval high;
- true-sum forced-result sweep remains high;
- clear reporting of whether the run is pure adaptive-only or anchor-stabilized.

A useful negative result should distinguish:

- lower LR slows drift but does not prevent it;
- answer/adaptive objective itself pulls away from true operands;
- anchor can preserve behavior but pure adaptive-only cannot;
- learned behavior remains a private code even when operand metrics improve;
- the interface is retained but learned-best action-loss fraction remains zero.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes, if any;
- starting Stage B checkpoint path;
- Stage C configs and trainable parameter groups;
- proof that aux weight stayed exactly `0.0`;
- metric table for every Stage C variant;
- comparison table against prior baselines and the Stage B retention target;
- counterfactual table;
- canonical causal diagnostic summary;
- canonical action-loss summary;
- parameter delta table versus both the semantic decoder checkpoint and Stage B handoff;
- explicit go/no-go recommendation for upstream unfreezing.

Also update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

This task succeeds if it answers whether the Stage B warm-started calculator interface can be stabilized during a longer no-supervision Stage C, and identifies the best next intervention if it cannot.

Do not claim success from Stage B. The claim must be based on post-supervision checkpoints with `aux_operand_loss_weight=0.0`.
