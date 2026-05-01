# Stabilize Adaptive Calculator Interface Objective

## Mission

Determine whether the phase-2 adaptive-interface failure came from the adaptive target signal itself or from unstable optimization of the residual-to-operand interface. The first adaptive run showed that downstream counterfactual loss usually identifies the true result class, but the learned interface collapses to an overconfident constant protocol. This task should isolate and stabilize that failure mode without relaxing the strict `answer_decoder` bottleneck.

The research object remains:

```text
h -> calculator_hook.input_proj(h) -> operand logits -> hard operands -> calculator
```

but this task should test safer ways to train `calculator_hook.input_proj` from counterfactual downstream result targets.

## Required Starting Point

Use the same validated strict regime as the previous phase-2 run:

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
```

Use this oracle-trained strict-bottleneck semantic decoder checkpoint:

```text
runs/2026-04-30_175805_513968_model-c-oracle-op0-19-answer_decoder/model-c-2digit-seed2/final_weights.pt
```

Preserve these defaults unless the experiment explicitly says otherwise:

- Freeze `calculator_hook.output_proj`, `answer_offset_emb`, and `answer_decoder`.
- Keep `calculator_hook.input_proj` trainable.
- Use `calculator_estimator=adaptive_interface` or a clearly named stabilized variant.
- Do not scale beyond `operand_max=19`.

## Experiment Ladder

### 1. Frozen-upstream diagnostic

Run adaptive-interface training with the upstream encoder frozen:

```text
freeze_upstream_encoder=true
```

Purpose: test whether the existing residual representation already contains enough stable operand information for the adaptive CE rule to train only `input_proj`.

Required metrics:

- answer exact match;
- operand exact match;
- calculator result accuracy;
- adaptive target result accuracy;
- learned-target agreement;
- adaptive interface CE curve;
- `input_proj` parameter delta;
- Track 3 classification;
- Track 4 action-loss gaps.

Interpretation:

- If frozen-upstream still collapses, the hard target-to-operand CE rule is likely the central problem.
- If frozen-upstream improves substantially, encoder/interface co-adaptation is destabilizing the target chase.

### 2. Separate optimizer / lower interface LR

Run the default trainable-upstream setup, but use separate optimizer groups:

```text
input_proj_lr <= 3e-4
upstream_lr <= 3e-4
decoder_lr = 0 because frozen
```

If script support does not exist, add minimal CLI flags for separate adaptive-interface optimizer groups. Do not introduce a broad training framework refactor.

Purpose: test whether the collapse is mainly from an overly aggressive interface update.

Compare to the failed phase-2 baseline:

```text
runs/2026-05-01_070903_085855_model-c-op0-19-adaptive_interface-answer_decoder/model-c-2digit-seed2
```

### 3. Soft pair target instead of hard tie-break

Add an adaptive objective variant that trains all operand pairs compatible with the selected result class instead of selecting a single `(a*, b*)` by current interface probability.

For target result `r*`, define valid pairs:

```text
P(r*) = {(a,b): a + b = r*}
```

Train the interface to increase total probability mass assigned to `P(r*)`, for example:

```text
loss = -log sum_{(a,b) in P(r*)} p_a(a) * p_b(b)
```

Name this variant clearly, such as:

```text
calculator_estimator=adaptive_soft_result
```

or add a separate flag:

```text
--adaptive-interface-target-mode soft_result
```

Purpose: remove the self-reinforcing hard pair tie-break that likely contributed to collapse.

### 4. Anti-collapse regularization

For the best of steps 1-3, add a small entropy floor or entropy bonus on operand distributions:

```text
entropy_weight in {0.001, 0.003, 0.01}
```

Do not treat high entropy as success. It is only a stabilizer. The checkpoint must still pass answer accuracy and counterfactual gates.

## Validation Gates

A checkpoint is interesting only if it improves over the failed adaptive baseline on at least one of:

- learned-target agreement;
- operand exact match;
- calculator result accuracy;
- answer exact match under normal calculator use.

Any claimed positive checkpoint must also satisfy:

- `calculator_hook.input_proj` changes during training;
- frozen semantic decoder components remain unchanged;
- injection-zero, forced-zero, and forced-random counterfactuals collapse relative to normal accuracy;
- oracle-at-eval remains high;
- Track 3 does not classify it as `calculator_ignored_or_bypassed`;
- Track 4 learned actions beat random and shuffled actions, or at minimum close a large portion of the learned-minus-true NLL gap.

## Required Reporting

Write a phase-2 work history with:

- exact commands and run paths;
- code changes made for optimizer groups or soft-result targets;
- semantic decoder checkpoint path;
- whether semantic decoder and upstream encoder were frozen;
- answer exact match, operand exact match, and calculator result accuracy;
- adaptive target result accuracy and learned-target agreement;
- adaptive interface CE curves and whether collapse occurred;
- `input_proj` parameter delta and frozen decoder parameter deltas;
- injection-zero, forced-zero, forced-random, and oracle-at-eval counterfactuals;
- Track 3 classification and bottleneck label;
- Track 4 action-loss gaps;
- a go/no-go recommendation for further adaptive-interface variants.

Also update:

```text
factSheets/PHASE_2_EXPERIMENT_FACT_SHEET.md
```

## Success Criteria

A positive result is not merely better answer accuracy. A positive result must show calculator-dependent answer accuracy and improved learned calculator actions under the strict bottleneck.

A useful negative result is also acceptable if it distinguishes the cause of failure:

- frozen-upstream collapse means the adaptive objective or target-to-operand mapping is broken;
- frozen-upstream improvement but trainable-upstream collapse means co-adaptation is unstable;
- soft-result improvement means the hard operand tie-break was the likely culprit;
- lower-LR improvement means the previous failure was partly optimizer-driven.

The main deliverable is a sharper answer to this question:

```text
Can the counterfactual downstream result signal train a stable calculator input interface if we remove the hard-pair and optimization collapse modes?
```
