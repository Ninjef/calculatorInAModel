# Hypothesis Ledger

Tiny claims and outcomes to prevent retesting settled branches.

## Phase 7

DISPROVEN: Vanilla result-space policy gradient is mainly blocked by finite-sample variance.
Conclusion: Exact result-marginal gradients align with sampled PG but both anti-align with the boundary ceiling.
Do not repeat: Longer vanilla PG or learned-baseline runs that estimate the same raw expected-cost gradient.
Next allowed test: A qualitatively different backward channel with a fixed-grid alignment gate.
Source: `aiAgentWorkHistory/phase7/2026-05-14-exact-result-marginal-answer-loss-gradient-gate.md`

DISPROVEN: Decoder calibration alone rescues ordinary expected-cost discovery.
Conclusion: Contrastive-margin decoder passed local sign alignment, then Stage 1 collapsed to wrong low-entropy results.
Do not repeat: Decoder-only sharpening/calibration without a stronger backward channel.
Next allowed test: Synthetic gradients, direct feedback alignment, or learned shadow-gradient modules.
Source: `aiAgentWorkHistory/phase7/2026-05-14-gradient-friendly-result-decoder-alignment-gate.md`

DISPROVEN: Output-projection boundary feedback is sufficient for natural result-space discovery.
Conclusion: Stage 0 aligned with the boundary ceiling, but Stage 1 reached only `0.155` best snapshot calculator-result accuracy and `0.160` final exact match.
Do not repeat: Plain output-projection feedback with the same weight/schedule as a mainline long run.
Next allowed test: A learned shadow-gradient module or stronger feedback training objective that must pass Stage 0 and show early Stage 1 lift.
Source: `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`

DISPROVEN: One fixed-random direct-feedback matrix is enough to clear the Stage 0 result-head gate.
Conclusion: Seed `0` fixed-random feedback had result-head cosine `-0.0036` against the boundary ceiling despite positive upstream cosine.
Do not repeat: Single-seed fixed-random DFA long training without a positive result-head Stage 0 gate.
Next allowed test: Multi-seed random-feedback screening or learned feedback, but only with Stage 0 gating.
Source: `aiAgentWorkHistory/phase7/2026-05-28-boundary-feedback-gradient-gate.md`

DISPROVEN: A fit-once linear shadow map from injection gradients to boundary result-logit gradients is enough for early natural result discovery.
Conclusion: Stage 0 model-update alignment was very high (`0.9983` result-proj, `0.9854` upstream), but 200-step Stage 1 reached only `0.070` best snapshot accuracy and `0.040` final exact match.
Do not repeat: Frozen fit-once linear shadow feedback with the same exact-grid calibration and weight/schedule.
Next allowed test: Heldout-validated or online-trained shadow modules with an early-lift gate, not a fixed linear map fit once at initialization.
Source: `aiAgentWorkHistory/phase7/2026-05-28-linear-shadow-feedback-gate.md`

DISPROVEN: Same-batch fit-once linear shadow alignment is a reliable Stage 0 gate.
Conclusion: With a deterministic `320/80` split, train result-proj cosine was `0.9981` but heldout result-proj cosine fell to `0.2622`, with a `0.7359` train-heldout gap.
Do not repeat: Treating same-batch linear shadow alignment as sufficient for training budget.
Next allowed test: Online MLP shadow feedback that includes result-policy state and must pass heldout warmup before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-heldout-linear-shadow-feedback-gate.md`

DISPROVEN: A simple online MLP shadow module with injection-gradient plus result-logit state cleanly passes the heldout warmup gate.
Conclusion: Hidden size `64` reached heldout result/upstream cosines `0.7167/0.7601`, but train-heldout gaps were `0.2683/0.2202`; hidden size `16` reduced the gap but heldout result cosine fell to `0.6255`.
Do not repeat: Launching Stage 1 from these simple online-MLP warmups, or rerunning the same `h64`/`h16`, `lr=1e-3`, `100`-step gate as novelty.
Next allowed test: Add a genuinely stronger shadow-generalization mechanism, such as validation early stopping, regularization, target normalization, richer policy state, or a different synthetic-gradient objective, and gate it heldout before Stage 1.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-warmup-gate.md`
