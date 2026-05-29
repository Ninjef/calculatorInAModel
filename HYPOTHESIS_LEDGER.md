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

DISPROVEN: Simple fixed-module feedback L2 clamping rescues validation-gradient online shadow Stage 1.
Conclusion: Apply clamps `3.5` and `10` kept feedback norm bounded, but both runs ended at `0.075` final exact match with best snapshot `0.0525`, unchanged from unclamped weight `1.0`.
Do not repeat: The same fixed h32 validation-gradient module with simple apply max-norm clamps `3.5` or `10` as novelty.
Next allowed test: On-policy shadow refresh or a trust region that refreshes gradient agreement, not only output-vector norm.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-apply-norm-clamp-gate.md`

DISPROVEN: Periodic on-policy shadow refresh alone rescues validation-gradient online shadow Stage 1.
Conclusion: Refresh every `50` steps restored excellent current-model heldout gradient agreement (`0.982-0.998` result cosine, ~`1.0` upstream), but Stage 1 ended at `0.025` final exact match with best snapshot `0.0475`.
Do not repeat: Same h32 validation-gradient module with refresh every `50`, `shadow_feedback_weight=1.0`, no apply clamp, and 200-step budget as novelty.
Next allowed test: Add training-dynamics constraints such as step-level trust region, entropy/diversity stabilization, or a target/state that avoids single-result collapse.
Source: `aiAgentWorkHistory/phase7/2026-05-28-online-shadow-feedback-on-policy-refresh-gate.md`

DISPROVEN: Soft result-policy entropy/batch-diversity stabilization rescues refreshed online-shadow Stage 1.
Conclusion: Low diversity weight `1.0` still collapsed to one hard result with final exact `0.015` unbounded and `0.005` clamped; high diversity weight `100` plus clamp `10` kept hard usage broader (`9.14` effective hard results) but reached only `0.070` final and `0.080` best snapshot.
Do not repeat: Same refreshed h32 validation-gradient module with soft result-policy diversity weights `1` or `100`, optional tiny entropy, 200-step budget, and clamp `0/10` as novelty.
Next allowed test: A hard/assignment-style usage constraint, step-level trust region, Jacobian-conditioned state, or richer target that links diverse requests to per-example improvement.
Source: `aiAgentWorkHistory/phase7/2026-05-28-result-policy-soft-diversity-gate.md`

DISPROVEN: Bounding actual optimizer parameter-step L2 norm rescues refreshed online-shadow Stage 1.
Conclusion: Trust caps `0.05` and `0.10` scaled proposed AdamW deltas from about `0.17-0.20`, stabilized shadow norms and refresh agreement, but ended at only `0.075`/`0.040` final exact with best snapshots `0.060`/`0.045`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, optimizer step max deltas `0.05` or `0.10`, and 200-step budget as novelty.
Next allowed test: Trust region that validates per-step improvement, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.
Source: `aiAgentWorkHistory/phase7/2026-05-28-optimizer-step-trust-region-gate.md`

DISPROVEN: Hard-path answer-loss step acceptance rescues refreshed online-shadow Stage 1.
Conclusion: Accept/reject gating with tolerances `0.0` and `0.1` accepted only `6/200` proposed steps (`3%`) and ended at `0.050` final exact with best snapshot `0.070`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss acceptance tolerance `0.0` or `0.1`, and 200-step budget as novelty.
Next allowed test: A mechanism that repairs/constructs useful directions rather than simply rejecting most shadow steps, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-step-acceptance-gate.md`

DISPROVEN: Hard-path answer-loss line search over proposed shadow step scales rescues refreshed online-shadow Stage 1.
Conclusion: Scales `1,0.5,0.25,0.1,0` accepted only `5/200` steps (`2.5%`); best snapshot improved to `0.0925`, but final exact was only `0.060`.
Do not repeat: Same refreshed h32 validation-gradient module with feedback clamp `10`, answer-loss line-search scales `1,0.5,0.25,0.1,0`, and 200-step budget as novelty.
Next allowed test: Construct better directions, hard assignment-style usage constraints, Jacobian-conditioned state, or richer targets rather than selecting among mostly harmful proposed shadow steps.
Source: `aiAgentWorkHistory/phase7/2026-05-28-answer-loss-line-search-gate.md`

PARTIAL: Output-Jacobian-conditioned shadow features improve Stage 0B but do not rescue refreshed online-shadow Stage 1.
Conclusion: `injection_grad_logits_output_jacobian` with feature z-scoring cleared Stage 0B at `0.9073/0.9011` heldout result/upstream cosines, but refreshed clamp-`10` Stage 1 ended at `0.055` final exact with best snapshot `0.065`.
Do not repeat: h16/h32 raw output-Jacobian features or h32 fit-split feature z-scoring with validation-gradient `0.5`, norm `0.1`, refresh every `50`, clamp `10`, and 200-step budget as novelty.
Next allowed test: Hard assignment-style usage constraints, richer targets, or a more substantial learned-gradient update path; do not treat this state-only Jacobian feature as enough.
Source: `aiAgentWorkHistory/phase7/2026-05-28-output-jacobian-shadow-feature-gate.md`

PARTIAL: Hard answer-loss improvement assignments can produce Stage 1 lift.
Conclusion: Assignment weight `10` reached `0.170` final exact with refreshed shadow and `0.400` final exact without shadow at 200 steps; weight `1` collapsed at `0.0475`.
Do not repeat: Same seed-2/seed-4 exact-grid 200-step assignment weights `1` or `10` as novelty.
Next allowed test: Longer convergence, target-off retention, seed replication, or cheaper/scalable assignment construction; do not claim final success because this still scores forced result classes during training.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-gate.md`

DISPROVEN: Plain linear target-off decay lets answer loss retain the hard improvement-assignment interface.
Conclusion: Assignment weight `10 -> 0` over 200 steps with `answer_loss_weight=1` peaked at `0.370` before shutoff, was `0.3475` at step `200`, collapsed to `0.105` by step `250`, and ended at `0.1075`.
Do not repeat: Same seed-2/seed-4 exact-grid no-shadow assignment decay over `200` steps with 400-step budget as novelty.
Next allowed test: Longer always-on convergence, seed replication, a stronger handoff bridge, or lower-cost assignment approximation.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-retention-gate.md`

PARTIAL: Always-on hard improvement assignment can train a natural result-space calculator interface across seeds.
Conclusion: With assignment weight `10` kept on, three 1600-step exact-grid seeds ended at `0.915`, `0.860`, and `0.820` final exact; best snapshots reached `0.9475`, `0.870`, and `0.920`.
Do not repeat: The same no-shadow 800/1600-step always-on assignment runs on CLI seeds `2/4/5` as novelty.
Next allowed test: Cheaper/scalable assignment construction, stronger target-off handoff, stability/selection to avoid late drift, or a non-bottleneck version of the gate.
Source: `aiAgentWorkHistory/phase7/2026-05-28-hard-improvement-assignment-convergence-gate.md`

DISPROVEN: The bottleneck hard-assignment signal transfers directly to an additive non-bottleneck model.
Conclusion: Non-bottleneck answer+assignment reached `0.700` final exact, but calculator-result accuracy stayed near chance (`0.0275` final) and assignment target accuracy fell to `0.0033`; the answer-only baseline also solved substantially with high injection-zero.
Do not repeat: Same additive result-space `ste`, answer-loss `1`, assignment weight `10`, 800-step seed-2 exact-grid gate as novelty.
Next allowed test: Add causal calculator-use pressure, staged bottleneck-to-additive handoff, or a target that remains tied to true calculator utility when the neuron path can bypass.
Source: `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-hard-assignment-gate.md`

DISPROVEN: A zero-injection causal-gap hinge is enough to make non-bottleneck hard assignment learn calculator use.
Conclusion: Gap weights `10/50` with margin `0.5` produced final causal gaps `1.27/0.84`, but final calculator-result accuracy stayed `0.000/0.0425` and final exact fell to `0.560/0.4225`.
Do not repeat: Same additive assignment-weight `10`, causal-gap margin `0.5`, weights `10/50`, 800-step seed-2 exact-grid gate as novelty.
Next allowed test: A staged bottleneck-to-additive handoff or causal target that rewards correct result-level utility, not merely making zero-injection worse.
Source: `aiAgentWorkHistory/phase7/2026-05-28-non-bottleneck-causal-gap-gate.md`

PARTIAL: A frozen bottleneck-trained result policy can be handed to an additive non-bottleneck model.
Conclusion: Compatible checkpoint loading preserved a `0.9125` bottleneck result policy; without freezing it collapsed to `0.0300` by step `50`, but freezing embeddings/pre-hook block/result head kept final calculator-result accuracy `0.9200` and produced `0.9475` normal versus `0.0175` injection-zero.
Do not repeat: Same seed/checkpoint compatible transfer without freezing, or same frozen-policy 800-step handoff as novelty.
Next allowed test: Seed/checkpoint replication, staged unfreezing, or a scalable/non-prescriptive way to acquire and preserve the policy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-gate.md`

PARTIAL: Frozen handoff replicates for a strong source checkpoint but is sensitive to source checkpoint quality.
Conclusion: Strong source `src2` transferred to additive seeds `2/4` with final eval `0.9400/0.9525` and learned calc `0.9200/0.9150`; weaker sources `src4/src5` preserved learned calc around `0.80-0.87` but reached only `0.3025-0.5550` final eval by 800 steps.
Do not repeat: The same frozen 800-step matrix cells `src2_add2`, `src2_add4`, `src4_add2`, `src4_add4`, or `src5_add5` as novelty.
Next allowed test: Source checkpoint selection/quality metrics, longer or stronger downstream readout adaptation, staged unfreezing, or a less prescriptive source-policy training method.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-transfer-replication.md`

PARTIAL: Longer downstream adaptation helps weak frozen handoffs but does not erase source sensitivity.
Conclusion: Continuing weak cells for another 800 steps improved `src5_add5` final eval `0.5550 -> 0.8175` and `src4_add2` `0.3025 -> 0.6050`, while injection-zero stayed near chance and learned calc stayed `0.8000/0.8725`.
Do not repeat: The same `src4_add2` or `src5_add5` one-extra-800-step continuation as novelty.
Next allowed test: Better source checkpoint selection, stronger readout adaptation, controlled unfreezing, or source-policy training that produces more handoff-friendly representations.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-downstream-adaptation.md`

DISPROVEN: Low-LR full-policy unfreezing preserves adapted non-bottleneck calculator use.
Conclusion: From adapted weak-source checkpoints, unfreezing all policy parameters at LR `3e-4` for 400 steps collapsed learned calc from `0.8725 -> 0.3000` and `0.8000 -> 0.2525`; answer accuracy did not improve.
Do not repeat: The same `src4_add2` or `src5_add5` adapted-checkpoint low-LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Selective unfreezing, explicit policy-retention regularization, or unfreeze schedules gated by calculator-result accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-low-lr-unfreeze.md`

PARTIAL: Result-policy anchoring can make full-policy unfreeze useful after staged handoff.
Conclusion: KL anchor weight `10` at LR `3e-4` preserved learned calc (`0.8075/0.7950`) and improved final eval over frozen adapted baselines (`src4_add2 0.6050 -> 0.7475`, `src5_add5 0.8175 -> 0.9525`).
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, LR `3e-4`, 400-step KL-anchor full unfreeze as novelty.
Next allowed test: Anchor decay/off-ramp, selective unfreeze, source checkpoint selection, or less prescriptive source-policy acquisition.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-policy-anchor-unfreeze.md`

DISPROVEN: A fast KL-anchor off-ramp makes the adapted non-bottleneck policy self-sustaining.
Conclusion: Decaying anchor weight `10 -> 0` over the first `200/400` unfreeze steps preserved calc accuracy at shutoff (`0.8300/0.8225`) but final calc fell to `0.5950/0.3850`, with final eval `0.5925/0.6750`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `10`, decay `200`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Slower or floored anchor schedules, calculator-accuracy-gated unfreezing, selective unfreeze, or a source policy that is robust without anchoring.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-decay-offramp.md`

PARTIAL: Reduced constant KL anchors can preserve non-bottleneck calculator use.
Conclusion: Anchor weights `1.0` and `0.1` at LR `3e-4` kept final calc near `0.77-0.81`, final eval `0.7775/0.9925` for weight `1` and `0.8325/0.9750` for weight `0.1`, with injection-zero near chance.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weights `1.0` or `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Even weaker/floored/gated anchors, selective unfreeze, or source-policy training that reduces the need for an anchor.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-reduced-anchor-strength.md`

MIXED: Constant KL anchor `0.01` is below the clean policy-retention region.
Conclusion: Anchor `0.01` kept injection-zero near chance and final eval `0.7850/0.9375`, but final calc fell to `0.7625/0.6425` and anchor agreement to `0.8825/0.7050`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor weight `0.01`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Floored or gated schedules around the `0.1` region, selective unfreezing, or policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-threshold.md`

PARTIAL: A nonzero anchor floor rescues the failed zero-off-ramp pattern.
Conclusion: Anchor `1.0 -> 0.1` over 200 steps kept final calc `0.8175/0.7800`, final eval `0.7925/0.9775`, and injection-zero `0.0250/0.0075`, but did not beat constant anchor `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, anchor `1.0`, decay `200`, floor `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Calculator-accuracy-gated retention, adaptive floors, selective unfreeze, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-anchor-floor-schedule.md`

DISPROVEN: Freezing only the calculator action head preserves transferred policy during unfreeze.
Conclusion: With `result_proj` frozen and only upstream trainable, adapted `src4_add2/src5_add5` still collapsed to final calc `0.3000/0.2525` and final eval `0.5200/0.8100`, matching the earlier plain unfreeze failure.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-action-head`, no anchor, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Behavior-level anchoring/gating, freezing the upstream policy path, or a more targeted selective parameter set that prevents upstream representation drift.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-freeze-action-head.md`

PARTIAL: Behavior-gated anchoring can reduce average anchor weight but does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with agreement gate `<0.9 -> 0.1` ended with final calc `0.7700/0.7700` and final eval `0.8050/0.9675`; it improved over constant `0.01` but was roughly comparable to, not better than, constant `0.1`.
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate threshold `0.9`, gate weight `0.1`, argmax-agreement gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Better gate metric/threshold, adaptive continuous weights, calculator-accuracy-gated retention, or source-policy acquisition that reduces active anchoring needs.
Source: `aiAgentWorkHistory/phase7/2026-05-28-bottleneck-to-additive-behavior-gated-anchor.md`

MIXED: Simple calculator-accuracy-gated anchoring is threshold-sensitive and does not beat fixed `0.1`.
Conclusion: Base anchor `0.01` with `current_argmax_accuracy` gates `<0.80` or `<0.82 -> 0.1` reached `src5` final eval `0.9825` at both thresholds, but `src4` reached only `0.7725/0.7900`, below fixed anchor `0.1` (`0.8325`).
Do not repeat: Same adapted `src4_add2/src5_add5`, base anchor `0.01`, gate thresholds `0.80` or `0.82`, gate weight `0.1`, calculator-accuracy gate, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Continuous/adaptive anchor control, selective policy-path unfreezing, stronger source acquisition, or a retention signal that combines calculator accuracy with answer utility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-accuracy-gated-anchor.md`

PARTIAL: Continuous calculator-accuracy anchor control can lower average retention weight but is not a clean fixed-`0.1` replacement.
Conclusion: Linear gate with base `0.01`, threshold `0.85`, band `0.10`, and max `0.1` reached `src4` final eval `0.8375` with mean weight `0.0385`, but `src5` ended `0.9725`, slightly below fixed `0.1`/discrete accuracy gates.
Do not repeat: Same adapted `src4_add2/src5_add5`, base `0.01`, linear `current_argmax_accuracy` gate threshold `0.85`, band `0.10`, gate weight `0.1`, LR `3e-4`, 400-step full unfreeze as novelty.
Next allowed test: Source-policy acquisition, selective policy-path unfreezing, or a retention controller that combines calculator accuracy with answer utility instead of metric-only anchor scaling.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-continuous-anchor-gate.md`

PARTIAL: Freezing the policy backbone prevents no-anchor policy collapse, but action-head/readout adaptation alone is weaker than anchored unfreezing.
Conclusion: `--freeze-calculator-policy-backbone` with no anchor preserved final learned calc `0.8200/0.8025` and improved adapted weak handoffs to final eval `0.7250/0.8700`, but stayed below lightweight anchor results.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Combine policy-backbone freezing with lightweight/utility-aware retention, improve source-policy acquisition, or test whether a different movable parameter set improves readout without policy drift.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-freeze.md`

NO-GAIN: Tiny anchoring is redundant when the policy backbone is frozen.
Conclusion: `--freeze-calculator-policy-backbone` plus KL anchor `0.01` kept anchor agreement `1.0000/0.9975` and learned calc `0.8200/0.8000`, but final eval `0.7125/0.8600` was slightly below no-anchor backbone freeze.
Do not repeat: Same adapted `src4_add2/src5_add5`, `--freeze-calculator-policy-backbone`, result-policy anchor `0.01`, LR `3e-4`, 400-step unfreeze as novelty.
Next allowed test: Improve downstream/readout adaptation under stable policy, use answer-utility-aware retention, or improve source-policy acquisition; tiny action-policy anchoring is not the missing ingredient here.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-tiny-anchor.md`

MIXED: Longer stable-policy readout adaptation helps good sources but does not erase weak-source sensitivity.
Conclusion: `--freeze-calculator-policy-backbone`, no anchor, 1600 steps lifted `src5_add5` to final eval `0.9500` with learned calc `0.8325`, but `src4_add2` reached only `0.7550` despite learned calc `0.8550`.
Do not repeat: Same adapted `src4_add2/src5_add5`, no anchor, `--freeze-calculator-policy-backbone`, LR `3e-4`, 1600-step adaptation as novelty.
Next allowed test: Source-policy acquisition/selection, stronger downstream adaptation targeted at weak sources, or a utility-aware readout objective under stable policy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-bottleneck-to-additive-policy-backbone-long-adaptation.md`

PARTIAL: Source checkpoint selection improves weak-source handoff but source action accuracy is not sufficient.
Conclusion: Reproducing `src5` with checkpoint snapshots and transferring the source step-1500 checkpoint (`0.9200` source normal/calc) improved immediate frozen-policy additive handoff from the old final-checkpoint baseline `0.5550` to `0.6975`.
Do not repeat: Same `src5` step-1500 selected-source checkpoint into additive seed `5`, frozen-policy, 800-step handoff as novelty.
Next allowed test: Source-selection metrics beyond normal/calc accuracy, source acquisition for handoff-friendly geometry, stronger selected-source replication, or utility-aware stable-policy readout adaptation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-checkpoint-selection-gate.md`
