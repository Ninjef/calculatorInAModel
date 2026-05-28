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

DISPROVEN: Validation-selected early stopping is enough to rescue the simple online MLP shadow module.
Conclusion: With `h64`, `lr=1e-3`, `100` steps, `0.1` validation and `0.2` heldout test, the selected step `60` reached test result/upstream cosines `0.6449/0.7266` with train-test gaps `0.3201/0.2414`.
Do not repeat: Treating validation-best checkpoints from this same simple MLP as a Stage 1 go signal.
Next allowed test: Change the learned-gradient target or state itself, such as target normalization, regularization, richer policy features, or a different synthetic-gradient objective.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gate.md`

DISPROVEN: Fit-split per-result z-scored shadow targets are sufficient to clear the online MLP heldout warmup gate.
Conclusion: Target normalization improved heldout cosines, but `h64/h32/h16` still missed the train-heldout gap gate; best near miss was `h16` with heldout `0.7259/0.7549` and gaps `0.1723/0.1458`.
Do not repeat: The same per-result z-score target-normalized `h64/h32/h16/h8`, `lr=1e-3`, `100`-step validation-selected Stage 0B sweep as novelty.
Next allowed test: Change the shadow input/state or objective more substantially, e.g. richer policy features, explicit regularization, a different loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-normalization-gate.md`

DISPROVEN: Appending raw result policy-state features rescues target-normalized online MLP shadow feedback.
Conclusion: Adding result probabilities, log-probabilities, and entropy to the shadow input did not clear the heldout gap gate; `h32` reached heldout `0.7037/0.7611` but gaps were `0.2853/0.2131`, and `h16` missed the result threshold.
Do not repeat: Raw `injection_grad_policy_state` features with per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Feature scaling/standardization, explicit regularization, a different synthetic-gradient loss, or a more stable target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-policy-state-gate.md`

DISPROVEN: Fit-split per-feature z-score standardization rescues target-normalized online MLP shadow feedback.
Conclusion: Feature z-scoring hurt the raw policy-state branch (`h16` heldout `0.5942/0.3997`, `h32` `0.4340/0.4023`) and did not rescue the simpler logits branch (`h32` heldout `0.6691/0.7028`, gaps `0.2830/0.2658`).
Do not repeat: Plain `fit_zscore_per_feature` with `injection_grad_logits` or `injection_grad_policy_state`, per-result target z-score, `h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Change objective/regularization or target construction, not just raw feature scale.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-feature-standardization-gate.md`

DISPROVEN: Directional cosine shadow losses alone cleanly clear the online MLP heldout warmup gate.
Conclusion: `cosine` and `mse_plus_cosine` improved heldout cosines for the simple logits state (`h16/h32` around `0.76-0.79` result, `0.80-0.83` upstream), but result train-heldout gaps stayed around `0.20`; h8 missed heldout cosine.
Do not repeat: Plain `cosine` or `mse_plus_cosine` with `injection_grad_logits`, per-result target z-score, `h8/h16/h32`, `lr=1e-3`, `100` steps as novelty.
Next allowed test: Add explicit norm/gap regularization, a more stable target construction, or a qualitatively different learned-gradient state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-directional-loss-gate.md`

DISPROVEN: Gap-penalized validation selection alone resolves directional-loss overfit.
Conclusion: Gap penalties moved `cosine` h16 earlier, but penalty `4` still had result gap `0.1673`, while penalty `5` reduced gap to `0.1511/0.1220` and dropped heldout to `0.6872/0.6979`.
Do not repeat: Gap-penalized selection on the same directional-loss `injection_grad_logits`, target-normalized h16/h32 setup with penalties `1/3/4/5`.
Next allowed test: Use training-time regularization, target stabilization, or a different learned-gradient state, not checkpoint selection alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-gap-penalized-selection-gate.md`

DISPROVEN: Simple dropout regularization rescues directional-loss online MLP shadow overfit.
Conclusion: Dropout `0.1/0.2` with `weight_decay=0.01` preserved heldout cosines on the target-normalized `cosine` branch, but h16/h32 still had result train-heldout gaps near `0.20`; best h32/dropout `0.1` reached heldout `0.7920/0.8248` with gaps `0.2039/0.1564`.
Do not repeat: Dropout-only h16/h32 sweeps on the same `injection_grad_logits`, target-normalized, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: Change target construction or learned-gradient state, or add explicit training-time gap/norm penalties rather than ordinary dropout alone.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-dropout-regularization-gate.md`

DISPROVEN: Per-example unit-norm target stabilization rescues directional-loss online MLP shadow overfit.
Conclusion: Unit-normalizing each target row before fit-split z-scoring preserved the same heldout cosines but kept result gaps near `0.20`; best h32/cosine reached heldout `0.7936/0.8270` with gaps `0.2025/0.1545`.
Do not repeat: `unit_norm_per_example` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: More substantial target stabilization, a different learned-gradient state, or explicit train-time gap/norm penalties.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-transform-gate.md`

DISPROVEN: Fit-split result-prototype target averaging cleanly rescues directional-loss online MLP shadow overfit.
Conclusion: Prototype targets slightly improved the tradeoff but not enough; h32/cosine reached heldout `0.8040/0.8243` with gaps `0.1909/0.1557`, and h16/cosine plus gap selection reached `0.7540/0.7855` with gaps `0.1705/0.1409`.
Do not repeat: `fit_result_prototype` target transform on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100`-step setup, including gap penalties `3/4/5`, as novelty.
Next allowed test: Different learned-gradient state, explicit train-time gap/norm penalties, or a target construction richer than boundary-best class prototypes.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-target-prototype-gate.md`

DISPROVEN: Appending the raw result-projection input rescues directional-loss online MLP shadow overfit.
Conclusion: The result-input state improved upstream heldout alignment, but result gaps remained high; h16/cosine reached heldout `0.7676/0.8372` with gaps `0.1958/0.1269`, and h32/cosine reached `0.7895/0.8294` with gaps `0.2079/0.1533`.
Do not repeat: `injection_grad_logits_result_input` with target z-score, h16/h32, `cosine`/`mse_plus_cosine`, `lr=1e-3`, `100` steps, including h16 gap penalties `3/4/5`, as novelty.
Next allowed test: Explicit train-time gap/norm penalties, Jacobian-conditioned state, or another genuinely different learned-gradient target/state.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-result-input-state-gate.md`

DISPROVEN: Train-time validation prediction-loss regularization rescues directional-loss online MLP shadow overfit.
Conclusion: h32 with validation-loss weight `0.5/1.0` kept heldout cosines high (`0.7953/0.8233`, `0.7915/0.8195`) but result gaps stayed near `0.199`; h16/weight `1.0` reduced gaps to `0.1595/0.1150` but dropped heldout to `0.7274/0.7381` and inflated norms.
Do not repeat: Validation-loss weights `0.5/1.0` on the same `injection_grad_logits`, target-normalized h16/h32, `cosine`, `lr=1e-3`, `100`-step setup as novelty.
Next allowed test: A direct split-gradient gap/norm objective, Jacobian-conditioned state, or a richer target construction.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-loss-gate.md`

PARTIAL: Direct validation model-gradient regularization can clear Stage 0B, but a fixed calibrated module does not produce Stage 1 lift.
Conclusion: h32/validation-gradient `0.5`/norm `0.1` reached heldout `0.8068/0.8083` with gaps `0.1227/0.1343` and norms `1.1276/1.0736`; fixed-module Stage 1 weights `1.0/0.01/0.001` ended at `0.075/0.005/0.035` exact match.
Do not repeat: The same h16/h32 validation-gradient `0.5`, norm `0/0.1` Stage 0B grid or fixed-module Stage 1 weights `1.0/0.01/0.001` as novelty.
Next allowed test: Keep the direct gradient objective, but refresh the shadow module on-policy, add trust-region/norm clamps, or condition on state that remains valid after model movement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-validation-gradient-gate.md`
