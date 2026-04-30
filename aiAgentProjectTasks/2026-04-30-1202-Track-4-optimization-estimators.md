# Track 4: Optimization Scientist - Estimators and Action-Loss Signal

## Mission

Determine whether the current hard STE is the main optimization bottleneck after the interface and bottleneck are made valid.

This track should not race ahead into estimator complexity before the experiment has a read-site interface and a calculator-required bottleneck where action choices have to matter. Track 3 sharpened this requirement: the additive setup is non-bottleneck/leaky by design, and the current `replace` mode is invalid or leaky for calculator-required claims because answer-token positions can still use ordinary autoregressive context.

Track 2 and Track 3 are complete; use their work histories before starting estimator work:

```text
aiAgentWorkHistory/2026-04-30-track-2-training-signal-protocol-supervision.md
aiAgentWorkHistory/2026-04-30-track-3-causal-diagnostics-codebooks.md
```

Important handoff: Track 2 implemented `calculator_injection_mode=replace`, but Track 3 confirmed that mode is leaky in the autoregressive setup. It replaces only active `=` residual positions, and the Model B/off replacement control stayed high (`0.889` exact match in Track 2, `0.844` exact match in the Track 3 64-sample diagnostic). Do not treat this replacement mode as a valid calculator-required bottleneck for estimator conclusions; first implement or obtain a stricter bottleneck where answer-token positions cannot recover operand information through the normal residual stream.

The strongest positive control is now the replacement oracle Model C diagnostic: exact match, operand exact match, and calculator result accuracy all reached `1.000`, while injection-zero and forced-random collapsed to `0.047` and `0.062`. That means the hook, discrete calculator, and downstream result-use path can work when true operands are supplied. The unsolved problem is learned protocol formation under a non-leaky training objective, not basic calculator wiring.

Its first job is to measure whether operand choices create enough downstream loss signal to train from in both:

- the existing non-bottleneck residual-injection setup, as a bypass baseline only;
- a strict or relaxed bottleneck setup where the calculator result is required for arithmetic success.

## Tactical Work

Start with a multi-sample action-loss diagnostic:

- For a fixed prompt/batch, evaluate multiple sampled or forced operand pairs.
- Measure how much downstream answer loss changes as calculator actions change.
- Estimate whether a policy-gradient or straight-through estimator has a meaningful signal to exploit.
- Compare this diagnostic before training, during training, and after ordinary answer learning.
- In bottleneck mode, verify that true calculator actions reduce loss more than random or shuffled actions before running estimator sweeps.
- Reuse the Track 3 forced-result and counterfactual machinery where practical, but add operand-action sweeps rather than only result-class sweeps.
- Include the Track 3 classifications in the diagnostic output so estimator results cannot be mistaken for true calculator use when they are bypass, private code, harmful calculator, or invalid/leaky bottleneck cases.

Before estimator comparisons, implement or select a stricter bottleneck. The exact design can vary, but it must pass these checks:

- Model B/off cannot solve arithmetic above the ordinary no-calculator baseline for that constrained setting.
- Oracle Model C reaches high exact match and fails under injection-zero or random calculator output.
- Learned answer-token computation cannot attend to or reconstruct operands through normal post-`=` context.
- Track 3 diagnostics label the setting as a valid calculator-required bottleneck, not `non_bottleneck_leaky_by_design` or `invalid_or_leaky_bottleneck`.

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

1. Action-loss sensitivity diagnostic on existing additive and replacement checkpoints as bypass/leakage baselines, using Track 3's six-checkpoint manifest where useful.
2. Implement a stricter calculator-required bottleneck variant with oracle and supervised-encoder controls; do not use Track 2's `replace` mode as the valid bottleneck.
3. Run Track 3 diagnostics on the bottleneck oracle and Model B/off controls before any estimator sweep.
4. Action-loss sensitivity diagnostic in the valid bottleneck variant.
5. Current STE baseline in the bottleneck variant.
6. ST Gumbel-Softmax temperature sweep in the bottleneck variant.
7. Soft expected-result relaxation in the bottleneck variant.
8. Multi-sample REINFORCE only if action-loss sensitivity is nontrivial.
9. Return to the non-bottleneck setup only after at least one estimator learns a protocol in the bottleneck setting.

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
- force-learned-result exact match;
- Track 3 classification and bottleneck/leakage label;
- result and operand codebook summaries;
- exact run path and seed.

For stochastic estimators, report multiple evaluation seeds or argmax-policy diagnostics before making fine-grained claims.

## Success Criteria

A strong positive result:

- An estimator learns the intended protocol more reliably than current STE.
- Calculator dependence survives counterfactual tests.
- Learned A/B match true A/B at the calculator read sites, or the learned protocol is causally decodable and not merely a private result code.
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
- Treat the Track 3 aux-decay transient as a private-code hint, not as evidence that the intended operand protocol has already emerged.
