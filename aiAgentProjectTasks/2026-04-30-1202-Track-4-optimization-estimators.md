# Track 4: Optimization Scientist - Estimators and Action-Loss Signal

## Mission

Determine whether the current hard STE is the main optimization bottleneck after the interface is fixed.

This track should not race ahead into estimator complexity before Track 1's operand-read interface exists. Track 1's first operand-read run suggests the non-bottleneck architecture can bypass the calculator, so estimator comparisons should prioritize a calculator-required bottleneck setting where action choices have to matter.

Track 2 is complete; use its work history before starting estimator work:

```text
aiAgentWorkHistory/2026-04-30-track-2-training-signal-protocol-supervision.md
```

Important handoff: Track 2 implemented `calculator_injection_mode=replace`, but that mode is leaky in the autoregressive setup. It replaces only active `=` residual positions, and the Model B/off replacement control reached `0.889` exact match. Do not treat this replacement mode as a valid calculator-required bottleneck for estimator conclusions; first implement or obtain a stricter bottleneck where answer-token positions cannot recover operand information through the normal residual stream.

Its first job is to measure whether operand choices create enough downstream loss signal to train from in both:

- the existing non-bottleneck residual-injection setup;
- a strict or relaxed bottleneck setup where the calculator result is required for arithmetic success.

## Tactical Work

Start with a multi-sample action-loss diagnostic:

- For a fixed prompt/batch, evaluate multiple sampled or forced operand pairs.
- Measure how much downstream answer loss changes as calculator actions change.
- Estimate whether a policy-gradient or straight-through estimator has a meaningful signal to exploit.
- Compare this diagnostic before training, during training, and after ordinary answer learning.
- In bottleneck mode, verify that true calculator actions reduce loss more than random or shuffled actions before running estimator sweeps.

Then compare estimator families under the same architecture/interface, starting with the bottleneck variant:

- current structured STE;
- straight-through Gumbel-Softmax with temperature schedules;
- soft expected-result relaxation during training, hard result at eval;
- REINFORCE with multi-sample per-prompt baseline;
- leave-one-out or group-relative action baseline if multi-sample variance is high.

## Experiments

Use the same starting regime as Track 1:

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
```

Keep everything fixed except the estimator/training objective.

Suggested order:

1. Action-loss sensitivity diagnostic on Track 1 non-bottleneck checkpoints.
2. Implement a stricter calculator-required bottleneck variant with oracle and supervised-encoder controls; do not use Track 2's `replace` mode as the valid bottleneck.
3. Action-loss sensitivity diagnostic in the bottleneck variant.
4. Current STE baseline in the bottleneck variant.
5. ST Gumbel-Softmax temperature sweep in the bottleneck variant.
6. Soft expected-result relaxation in the bottleneck variant.
7. Multi-sample REINFORCE only if action-loss sensitivity is nontrivial.
8. Return to the non-bottleneck setup only after at least one estimator learns a protocol in the bottleneck setting.

## Required Metrics

Report:

- answer exact match;
- true operand exact match;
- calculator result accuracy;
- action-loss variance per prompt;
- action-loss gap between true, learned, random, and shuffled calculator actions;
- bottleneck mode and bypass/compression setting;
- estimator-specific loss terms;
- operand entropy/confidence;
- injection-zero exact match;
- forced-zero exact match;
- forced-random exact match;
- oracle-at-eval exact match;
- exact run path and seed.

For stochastic estimators, report multiple evaluation seeds or argmax-policy diagnostics before making fine-grained claims.

## Success Criteria

A strong positive result:

- An estimator learns the intended protocol more reliably than current STE.
- Calculator dependence survives counterfactual tests.
- Gains reproduce across at least 3 seeds after the first promising run.
- The estimator succeeds first in the bottleneck setting, then transfers or partially survives when bypass is relaxed.

A strong negative result:

- In a valid bottleneck, action choices meaningfully affect downstream loss, but estimators still fail.
- Or action choices barely affect downstream loss even in the bottleneck, indicating an architecture/downstream-decoder flaw rather than just an estimator flaw.
- In the non-bottleneck setup, action choices barely affect downstream loss after answer learning, explaining why policy-gradient routes are weak there.

Both are useful: the first points to a better optimizer; the second says the training objective is under-informative.

## Guardrails

- Do not change architecture while comparing estimators.
- Do not mix estimator changes with auxiliary supervision unless explicitly labeled.
- Do not report stochastic single-sample evals as stable facts.
- Keep this track subordinate to the fixed interface and shared diagnostics.
- Do not spend large sweeps on the non-bottleneck architecture until the bottleneck capability test has at least one working learned protocol.
