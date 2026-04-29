# Overview

The previous upstream-learning pass confirmed the key failure mode: hard STE plus downstream answer loss does not reliably teach the upstream transformer to emit meaningful calculator operands.

Please implement and run the next strategy: treat calculator operands as stochastic actions sampled from distributions parameterized by the upstream residual stream, then train the input side with policy-gradient / score-function estimators.

# Findings to build on

The calculator output side is learnable:

- Oracle-trained Model C reached `236/256 = 0.922` exact match on 1-digit addition.
- Oracle diagnostics reached `59/64 = 0.922`, with perfect operand and calculator-result accuracy.

The hard-STE input side is not learning the intended protocol from answer loss alone:

- True tiny-vocab `0..1`, `0..2`, and `0..4` Model C runs can reach perfect answer accuracy.
- But answer accuracy is misleading:
  - `0..1`: operand exact match `0.203`, calculator result accuracy `0.203`;
  - `0..2`: operand exact match `0.375`, calculator result accuracy `0.438`;
  - `0..4`: operand exact match `0.000`, calculator result accuracy `0.000`, with near-1.0 confidence.
- Representative `0..4` trace rows show the model answering correctly while sending confident wrong calculator operands, so the ordinary transformer path is solving the tiny task around the calculator.

The residual stream and input projection are capable in principle:

- Layer/position probes often recover operands from the residual stream.
- On `0..4`, auxiliary operand supervision at weight `0.1` moved operand exact match from `0.000` to `0.938` while keeping answer exact match at `1.000`.

Interpretation:

The current bottleneck is not missing operand information or output injection. It is credit assignment through the discrete calculator input interface. Hard STE is biased and appears to provide misleading or insufficient signal.

# Step 10 candidate strategy: stochastic calculator actions

Treat each calculator input as a sampled action:

- upstream residual at the calculator read position produces `a_logits` and `b_logits`;
- sample `a ~ Categorical(a_logits)` and `b ~ Categorical(b_logits)`;
- run the hard calculator on sampled operands;
- inject the calculator result as before;
- compute normal downstream answer loss;
- update the operand distributions with a score-function / policy-gradient estimator.

The basic identity:

```text
grad_theta E[L] = E[L * grad_theta log p_theta(action)]
```

For minimization, use an advantage-style loss:

```text
policy_loss = (answer_loss.detach() - baseline) * (logp_a + logp_b)
total_loss = answer_loss + policy_loss
```

Equivalently, using reward `reward = -answer_loss`:

```text
policy_loss = -(reward - baseline) * (logp_a + logp_b)
```

The calculator remains fully hard and non-differentiable. Unlike STE, this estimator is unbiased for the stochastic objective, but it may have high variance.

# Experiments to run first

## 1. Vanilla REINFORCE with moving baseline

Add a calculator estimator mode such as `reinforce` or `sampled-policy`.

Run Model C with:

- `0..1`, calculator operand vocab size `2`;
- `0..2`, calculator operand vocab size `3`;
- `0..4`, calculator operand vocab size `5`;
- then `0..9`, calculator operand vocab size `10`.

Use:

- moving-average baseline for answer loss;
- entropy bonus on operand distributions, initially nonzero and decayed or configurable;
- multiple seeds;
- optionally multiple operand samples per prompt if a single-sample estimator is too noisy.

Report:

- answer exact match;
- operand exact match;
- calculator result accuracy;
- operand entropy/confidence;
- reward/loss baseline curve;
- representative trace rows.

Decision rule:

- If policy gradient improves calculator result accuracy but not true operand exact match, the model may be learning a valid but non-human latent calculator language.
- If it improves true operand exact match, it is directly learning the intended protocol.
- If it fails even on `0..1`, inspect variance, baseline behavior, and whether downstream loss changes enough across sampled operands.

## 2. REINFORCE plus tiny auxiliary warmup

If vanilla policy gradient is too noisy, add a short auxiliary operand-loss warmup:

- start with small aux weight, for example `0.03` or `0.1`;
- decay aux weight to `0.0`;
- keep policy-gradient learning active throughout.

Research question:

Can a small supervised nudge establish the protocol, and can unbiased answer-loss policy gradient maintain it after the nudge disappears?

Decision rule:

- If operand accuracy survives after aux decay, the protocol can be maintained by answer pressure once discovered.
- If it collapses, downstream answer loss alone may not identify true operands, only useful sums.

## 3. Multi-sample estimator diagnostic

For each prompt, sample `K` operand pairs and evaluate the downstream answer loss for each.

Try small `K`, for example `2`, `4`, or `8`.

Research question:

Does lower-variance action comparison make the calculator protocol discoverable?

Implementation note:

This can be expensive if each sample requires a full downstream pass. Keep it tiny-range first. It is acceptable for this to be diagnostic-only.

## 4. Control-variate variants, only if needed

If vanilla policy gradient shows the right direction but too much variance, consider:

- NVIL-style learned baseline;
- MuProp;
- REBAR;
- RELAX.

Do not jump here until the simple moving-baseline version has been measured.

# Important measurement warning

Do not treat answer accuracy alone as success.

For this project, a run is calculator-protocol-successful only if at least one of these is true:

- true operand exact match is high;
- calculator result accuracy is high and trace rows show a coherent alternate protocol worth studying.

Tiny-range answer success with wrong calculator inputs should be considered "ordinary transformer solved around the calculator," not success for the calculator interface.

# Acceptance criteria

Save for each experiment:

- config;
- final checkpoint;
- training curve;
- answer metrics;
- diagnostic summary;
- calculator trace rows;
- policy-gradient-specific metrics, including baseline, policy loss, entropy, and sampled operand log-probability.

The next useful milestone is a short writeup answering:

- Does unbiased policy gradient learn any calculator protocol where hard STE did not?
- What is the smallest operand range where sampled-action learning works?
- Does it learn true operands or merely useful sums?
- How sensitive is it to baseline choice, entropy regularization, and number of samples?
- Does auxiliary warmup plus policy gradient maintain the protocol after aux decay?
