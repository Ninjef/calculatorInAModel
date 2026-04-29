# Overview

The Step 8 diagnostics matched the expected failure mode: once training has to hop over a discrete calculator program, downstream use of correct calculator results is learnable, but getting the upstream nodes to adjust themselves so they send the right operands is difficult.

Please design and run the next set of experiments to figure out how the upstream side can learn to use the calculator properly.

# Current findings to build on

The output side is not the primary blocker:

- Oracle-trained Model C reached `236/256 = 0.922` exact match on 1-digit addition.
- Oracle diagnostics on that checkpoint reached `59/64 = 0.922`, with perfect operand and calculator-result accuracy.
- Learned operand diagnostics remain poor:
  - failed Model C checkpoint: operand exact match `0.000`, calculator result accuracy `0.000`;
  - oracle-trained checkpoint without oracle operands: operand exact match `0.000`, calculator result accuracy `0.094`.

Interpretation:

The calculator result path is wired in a learnable way. The main research problem is now the upstream latent operand protocol: how does the transformer learn to put the right discrete operands into the calculator interface when gradients must pass through a hard non-differentiable program?

# Step 9 candidate experiments: upstream operand learning

## 1. Tiny-vocabulary learned-operand curriculum

Run learned-operand Model C with restricted operand ranges before full `0-9`:

- `0-1`
- `0-2`
- `0-4`
- then `0-9`

Keep the same model architecture and calculator hook. Use the diagnostic script to report:

- answer exact match,
- operand exact match,
- calculator result accuracy,
- operand entropy/confidence,
- representative trace rows.

Research question:

Can the upstream network learn any hard calculator language when the operand vocabulary is extremely small?

Decision rule:

- If `0-1` or `0-2` works but `0-9` fails, the problem is likely optimization/vocabulary scale.
- If even `0-1` fails, the current hard STE signal is probably too weak or misleading.

## 2. Auxiliary operand-loss diagnostic

Add an optional training-only auxiliary loss on the calculator input projection:

- Cross-entropy from `a_logits` to true operand A.
- Cross-entropy from `b_logits` to true operand B.
- Configurable weight, for example `0.01`, `0.1`, `1.0`.

This should be a diagnostic, not the final claimed method. Keep normal answer loss as the main metric.

Research question:

Can the input projection and upstream residual stream represent the correct operands if given a direct learning signal?

Decision rule:

- If auxiliary loss makes operand accuracy high and answer accuracy high, the representation/output path is fine and the unsupervised downstream loss is the bottleneck.
- If auxiliary loss makes operand accuracy high but answer accuracy stays low, revisit result injection or downstream integration.
- If auxiliary loss cannot make operand accuracy high, inspect residual probe results and read-position choice.

## 3. Probe-informed read-location check

Run residual probes at multiple candidate read locations:

- after layer 1,
- after layer 2,
- after layer 3,
- optionally at the token positions for operand A, operand B, and `=`.

Research question:

Where are operands linearly accessible in the residual stream?

Decision rule:

- If probes succeed at another layer/position but not the current `=` after layer 2, move or augment the calculator read site.
- If probes fail everywhere, the model is not making operands linearly available without extra pressure.

## 4. Straight-through estimator variants

Only after the tiny-vocab and auxiliary-loss diagnostics, compare gradient estimators:

- current hard argmax STE,
- soft expected-sum relaxation during training,
- Gumbel-Softmax or temperature-annealed softmax,
- optional hard-forward/soft-backward variant.

Research question:

Is the current gradient estimator the reason upstream operands fail to align?

Decision rule:

- If a soft estimator learns operands but hard STE does not, the issue is estimator smoothness.
- If all estimators fail without auxiliary loss, the downstream answer loss alone may be too indirect for operand protocol discovery.

## 5. Scheduled oracle-to-learned transition

Try a curriculum where training starts with oracle operands and gradually shifts to learned operands:

- teacher forcing probability starts at `1.0`,
- decays to `0.0` over training,
- diagnostics track whether learned operands take over before oracle support disappears.

Research question:

Can downstream competence stabilize the task enough for the upstream interface to learn?

Decision rule:

- If scheduled transition works, the problem is coordination/bootstrapping.
- If learned operands collapse when oracle probability decays, the input-side estimator still needs improvement.

# Suggested order

1. Tiny-vocabulary learned-operand curriculum.
2. Auxiliary operand-loss diagnostic on the smallest failing range.
3. Probe read-location sweep.
4. Estimator variants.
5. Scheduled oracle-to-learned transition.

Do not jump to larger models or multi-digit tasks until at least one learned-operand setup reliably solves 1-digit or tiny-range addition.

# Acceptance criteria

For each experiment, save:

- config,
- final checkpoint,
- training curve,
- answer metrics,
- diagnostic summary,
- calculator trace rows.

The next useful milestone is a short writeup that answers:

- What is the smallest operand range learned by the current hard STE?
- Does auxiliary operand supervision make the protocol learnable?
- Are true operands linearly present at the current calculator read site?
- Is failure caused by estimator smoothness, bootstrapping, or missing residual information?
