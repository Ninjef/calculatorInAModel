# Hypothesis Ledger

Tiny claims and outcomes to prevent retesting settled branches.

Use this file tactically. The strategic synthesis lives in
`RESEARCH_STATE.md`; this ledger prevents local reruns and records which
families are paused. Direction-level synthesis lives in `researchMemory/`.

Maintenance rule: individual entries may accumulate here, but when several
entries point to the same lesson, consolidate that lesson into
`researchMemory/` and update the family status index rather than relying on
future agents to infer it from chronology.

## Family Status Index

| Family | Status | Strategic implication |
| --- | --- | --- |
| Oracle calculator wiring | Paused | Use only as a wiring/control check; not progress toward the thesis. |
| Generic retention-after-teaching | Paused | Already proven in identifiable/scaffolded settings; only run for a new interface or stability question. |
| Vanilla score-function / expected answer-loss discovery | Paused | Sampling variance and decoder calibration were not the main blockers. |
| Simple decoder calibration | Paused | Local gradient sign improvements did not produce Stage 1 discovery. |
| Simple direct feedback / fixed shadow gradients | Paused | Stage 0 alignment alone was not enough; do not repeat without a new dynamics mechanism. |
| Simple online shadow-gradient variants | Paused | Normalization, validation selection, dropout, and directional losses did not clear the useful-training gate. |
| Hard improvement assignment | Active but constrained | Strong bottleneck source ceiling, but scalability and prescriptiveness remain unresolved. |
| Bottleneck-to-additive staged handoff | Active but constrained | Proves non-bottleneck viability, but source quality and policy protection remain bottlenecks. |
| Cheap source-checkpoint selectors | Paused | Frozen-state, geometry, short-slope, ridge, and embedded 500-step probes are not reliable replacements for actual handoff gates. |
| Source acquisition for transfer geometry | Active | Current best strategic direction if it directly targets handoff/readout behavior. |
| Online hard result-boundary memory + numeric prior | Active benchmark, constrained | Full-memory numeric-prior replay fixes the heldout prompt source gate and clears trusted handoff. Every-2 plus sustained train-memory convergence cuts prior updates to `1889`; first-hit convergence, every-10 cadence, and random half-memory fits underfit. Continue with validation-aware or structured coreset fitting. |
| Target propagation / local targets | Active candidate, constrained | Exact/full-enum local-target gates are positive, but simple proposal approximations are paused after sparse/adaptive, replay, corrected, online learned, and pretrained learned variants failed scalability stress; continue only with a different estimator, different target construction, or explicitly streaming/generalizing learned proposal. |

Rule: if a proposed experiment belongs to a paused family, it needs a new
mechanism and should be reconciled with `RESEARCH_STATE.md` before running.

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

DISPROVEN: Highest source normal/calculator accuracy is a reliable source checkpoint selector.
Conclusion: Reproduced `src2` with checkpoints; source step `1300` had higher source normal/calc (`0.9475`) than final (`0.9150`) but transferred worse into additive seed `4` (`0.8675` vs final-control `0.9525`).
Do not repeat: Same `src2` step-1300 versus final additive seed-4 frozen-policy 800-step transfer as novelty.
Next allowed test: Source-quality probes for handoff geometry, source acquisition optimized for transfer/readout learnability, or selected-source replication with a selector beyond source accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-selection-metric-replication.md`

PARTIAL: Short additive handoff progress predicts final handoff better than source accuracy.
Conclusion: Across six non-continued frozen-policy transfer cells, normal accuracy at step `400` correlated with final eval at `0.9374`, and step `600` at `0.9935`; step `200` was noisy (`-0.0959`).
Do not repeat: Same trace audit over the current frozen-policy transfer cells as novelty.
Next allowed test: Use a 400/600-step handoff probe for checkpoint selection, build a cheaper readout/linear proxy for that probe, or optimize source acquisition for early additive handoff slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-short-handoff-probe-audit.md`

DISPROVEN: The initial frozen-state readout probe was a reliable handoff proxy.
Conclusion: Reusable script validation exposed that the scratch probe used the wrong `EQ_ID`/leaky position. Correct safe features over five checkpoints had weak correlation with final handoff (`read_pair 0.2118`, `layer2_pair 0.2865`).
Do not repeat: Same five-checkpoint safe frozen-state readout probe as novelty, or the leaky/wrong-position answer-token probe.
Next allowed test: Build a better non-leaky geometry proxy, validate source selectors on unseen checkpoints, or use 400/600-step handoff probes until a cheaper proxy is proven.
Source: `aiAgentWorkHistory/phase7/2026-05-29-frozen-state-readout-probe.md`

POSITIVE: A 600-step handoff probe can select a better source checkpoint than source accuracy.
Conclusion: On `src5`, the 600-step probe selected step `1100` (source normal `0.8400`) over the source-accuracy-selected step `1500` (`0.9200`); full additive handoff improved from `0.6975` to `0.7950`.
Do not repeat: Same `src5` step `1100/1400/1500/final`, additive seed `5`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use the 600-step handoff probe on newly acquired source checkpoints, reduce/approximate its cost, or optimize source acquisition for probe score.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-validation.md`

POSITIVE: The 600-step handoff probe rescues weak `src4` source selection.
Conclusion: Reproduced `src4` with snapshots; 600-step probe selected step `1200` (source normal `0.7550`) over final (`0.8700`), and full frozen handoff improved from old final-source `0.3025` to `0.7800`.
Do not repeat: Same `src4` step `1000/1200/final`, additive seed `2`, frozen-policy handoff-probe comparison as novelty.
Next allowed test: Use probe score during source acquisition, reduce probe cost, or test whether probe-selected sources reduce later anchor/long-adaptation needs.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-probe-selector-src4.md`

MIXED-POSITIVE: Probe-selected sources help later stable-policy adaptation for weak handoffs, but are not a universal long-adaptation selector.
Conclusion: 1600-step no-anchor `--freeze-calculator-policy-backbone` adaptation from probe-selected checkpoints lifted `src4` from old final-source long adaptation `0.7550` to `0.8900`, but `src5` reached `0.9250`, below the old final-source long adaptation `0.9500`.
Do not repeat: Same probe-selected `src4` step-1200/add2 or `src5` step-1100/add5 frozen handoff checkpoint into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Add a second-stage long-adaptation/readout-compatibility selector, optimize source acquisition for both 600-step handoff slope and later readout adaptability, or reduce the handoff-probe cost.
Source: `aiAgentWorkHistory/phase7/2026-05-29-probe-selected-policy-backbone-adaptation.md`

DISPROVEN: The `src5` source-accuracy-selected step-1500 checkpoint is the better long-adaptation candidate.
Conclusion: Starting from the existing step-1500 800-step frozen handoff and adapting for 1600 steps with policy backbone frozen reached final eval `0.9100`, below the handoff-probe-selected step-1100 result `0.9250`, despite higher final calc accuracy (`0.9325` vs `0.8275`).
Do not repeat: Same `src5` step-1500 800-step frozen handoff into no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Compare against the exact old final-source checkpoint lineage, inspect reproduced-versus-old final-source differences, or optimize source acquisition for downstream readout compatibility.
Source: `aiAgentWorkHistory/phase7/2026-05-29-long-adaptation-selector-probe.md`

POSITIVE: The old `src5` final-source long-adaptation advantage was mostly a continuation-depth fairness issue.
Conclusion: Giving the handoff-probe-selected `src5` step-1100 lineage the same extra 800-step frozen-policy continuation lifted it from `0.7950` to `0.8800`, then 1600-step stable-policy adaptation reached `0.9425`, nearly matching the old final-source `0.9500`.
Do not repeat: Same `src5` step-1100 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Apply the fair continuation recipe to `src4` step-1200, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-selected-source-continuation-fairness.md`

POSITIVE: Fair continuation also improves the weak `src4` selected-source lineage.
Conclusion: `src4` step-1200 selected handoff improved from `0.7800` to `0.8150` with another 800 frozen-policy steps, then reached `0.9125` after policy-backbone-frozen long adaptation, beating direct selected long (`0.8900`) and old final-source long (`0.7550`).
Do not repeat: Same `src4` step-1200 selected handoff plus extra 800-step frozen-policy continuation plus no-anchor policy-backbone-frozen 1600-step adaptation as novelty.
Next allowed test: Reduce handoff-probe/continuation cost, or optimize source acquisition for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src4-selected-source-continuation-fairness.md`

MIXED-POSITIVE: Stable readout adaptation can be cut to 600 steps, but 200 steps is source-sensitive.
Conclusion: From continued selected checkpoints, 200-step readout worked for `src5` (`0.9275`) but not `src4` (`0.8775`); 600-step readout passed both (`src4 0.9025`, `src5 0.9325`) with injection-zero near zero and forced-random near chance.
Do not repeat: Same 200/600-step no-anchor policy-backbone-frozen readout adaptation from continued selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Reduce the 600-step handoff probe or 800-step frozen-policy continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-readout-budget-validation.md`

MIXED-POSITIVE: The source-selection handoff probe can be cut to 500 steps on current candidates, but not to 400.
Conclusion: Existing traces show 400-step probe would pick `src5` step-1500, but 500-step probe picks the same checkpoints as 600 for both audited families (`src5` step-1100, `src4` step-1200).
Do not repeat: Same trace audit over existing `src5` 1100/1400/1500/final and `src4` 1000/1200/final handoff probes as novelty.
Next allowed test: Validate 500-step selection on new source checkpoints, reduce the 800-step continuation cost, or optimize source acquisition for early handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-shorter-handoff-probe-trace-audit.md`

MIXED-NEGATIVE: Reducing the frozen-policy continuation from 800 to 600 steps is source-sensitive.
Conclusion: With 600-step readout after reduced continuation, `src5` still passed (`0.9275` vs `0.9325` reference), but weak `src4` fell below gate (`0.8750` vs `0.9025` reference) despite retained calculator dependence.
Do not repeat: Same 600-step continuation plus 600-step policy-backbone-frozen readout from selected `src4` step-1200/add2 and `src5` step-1100/add5 checkpoints as novelty.
Next allowed test: Keep 800 continuation for weak sources, test 700-step continuation only for fine-grained tuning, or optimize source acquisition for continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-reduced-continuation-budget-validation.md`

POSITIVE: The 500-step handoff selector validates on the `src2` source-accuracy counterexample.
Conclusion: For `src2` additive seed `4`, 500-step handoff progress picks final/source step-1600 (`0.6900`) over source-accuracy-favored step-1300 (`0.5875`), matching final handoff (`0.9525` vs `0.8675`).
Do not repeat: Same `src2` step-1300 versus final additive seed-4 400/500/600-step trace audit as novelty.
Next allowed test: Validate 500-step selection on new source checkpoints, or optimize source acquisition for early handoff/continuation slope directly.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src2-500-step-selector-validation.md`

SYNTHESIS: The selected-source non-bottleneck recipe is progressing, not looping.
Conclusion: Recent work moved from disproving source-accuracy selection to validating a 500-step handoff selector on `src2/src4/src5`, keeping 800 continuation for weak sources, and reducing stable readout to 600 steps (`src4 0.9025`, `src5 0.9325`).
Do not repeat: Existing source-accuracy selector tests, existing `src4/src5` 200/600 readout cuts, existing 600-step continuation cut, or existing `src2` 500-step selector trace as novelty.
Next allowed test: Validate the 500-step selector on newly acquired source checkpoints, or train source acquisition directly for early handoff and continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-periodic-review-selected-source-recipe.md`

MIXED-NEGATIVE: The 500-step selector does not generalize cleanly to fresh `src6`.
Conclusion: Fresh `src6` 500-step handoff scores pick step `1500` (`0.7200`) over final (`0.6850`), but full 800-step handoff is better from final (`0.8975` vs `0.8875`); 600-step scores would pick final (`0.8050` vs `0.7800`).
Do not repeat: Same `src6` step-1200/step-1500/final additive seed-6 frozen-policy 800-step comparison as novelty.
Next allowed test: Use 600-step selection for fresh sources, run continuation/readout from fresh `src6` final, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-new-source-500-selector-validation.md`

POSITIVE: The selected-source continuation/readout recipe clears the gate on fresh `src6`.
Conclusion: Fresh `src6` final-source handoff was near-gate (`0.8975`); 800 frozen-policy continuation reached `0.9625`, and 600 policy-backbone-frozen readout reached `0.9850` with controls far below normal.
Do not repeat: Same `src6` final-source continuation plus 600-step readout as novelty.
Next allowed test: Replicate on another fresh source with 600-step selection, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src6-selected-continuation-readout.md`

MIXED: The 600-step selector replicated on fresh `src7`, but the reduced recipe did not clear the gate.
Conclusion: On weak fresh `src7`, 600-step handoff selection picked step `1400` (`0.5025`) over step `1000` (`0.4850`) and final (`0.4150`), matching the full-handoff winner; continuation/readout improved `0.7325 -> 0.8125 -> 0.8825` but missed the `0.90` gate.
Do not repeat: Same `src7` step-1000/step-1400/final handoff comparison or step-1400 continuation/readout chain as novelty.
Next allowed test: Optimize source acquisition for 600-step handoff/continuation slope, or use fresh-source replications only as planned acquisition gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-src7-600-selector-replication.md`

DISPROVEN: Decaying small entropy/batch-diversity source stabilization to zero will rescue weak fresh-source acquisition.
Conclusion: Fresh seed-9 source acquisition with entropy `0.05`, batch diversity `0.1`, improvement assignment `10`, and decay-to-zero over 1200 steps peaked at `0.7050` around steps `700/900`, then collapsed to final `0.1825`.
Do not repeat: Same decay-to-zero source-only recipe with answer loss off as novelty.
Next allowed test: Keep a nonzero source-objective floor, add policy anchoring, or optimize source acquisition for 600-step handoff/continuation slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-variant.md`

MIXED-POSITIVE: Keeping entropy/diversity source stabilization active fixes collapse; initial handoff is weak, but continuation/readout can unlock the source.
Conclusion: No-decay entropy `0.05` + batch diversity `0.1` + improvement assignment `10` reached source step `1400` normal `0.9100` and final eval `0.8575`; final-source handoff started weak (`0.6500`) but 800-step continuation reached `0.9050` and 600-step readout reached `0.9575`.
Do not repeat: Same no-decay source recipe plus step `1400`/final additive seed-9 handoff, direct readout, continuation, and readout chain as novelty.
Next allowed test: Replicate no-decay stabilized continuation/readout on another seed, reduce continuation cost, or build a cheaper proxy for continuation/readout slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-acquisition-stabilization-floor.md`

POSITIVE: The no-decay stabilized final-source lineage clears the non-bottleneck gate after continuation/readout.
Conclusion: Starting from final-source handoff `0.6500`, direct 600-step readout reached `0.8000`, 800-step frozen-policy continuation reached `0.9050`, and 600-step post-continuation readout reached `0.9575` with controls far below normal.
Do not repeat: Same no-decay final-source handoff into direct readout, continuation, and post-continuation readout as novelty.
Next allowed test: Replicate on another fresh stabilized source, reduce continuation cost, or identify a cheap continuation/readout-slope proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-readout.md`

POSITIVE: The no-decay stabilized source can use 600-step continuation and still clear the non-bottleneck gate.
Conclusion: Reading out from the step-600 continuation checkpoint reached final eval `0.9425` with injection-zero `0.0078`, forced-random `0.0781`, and learned calc `0.8750`; this is only `0.0150` below the 800-continuation readout.
Do not repeat: Same no-decay stabilized step-600 continuation checkpoint into 600-step readout as novelty.
Next allowed test: Replicate the 600-continuation recipe on another stabilized source, test 500 continuation, or build a proxy for deciding continuation budget.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-reduced-continuation.md`

MIXED-POSITIVE: The no-decay stabilized source continuation budget can be cut to 400 steps, but not 300 by final eval.
Conclusion: 600-step readout from continuation checkpoints reached final eval `0.9400` at step `500`, `0.9175` at step `400`, and `0.8850` at step `300`; controls stayed far below normal.
Do not repeat: Same no-decay stabilized `600/500/400/300` continuation-checkpoint readout ladder as novelty.
Next allowed test: Replicate the 400/500 boundary on another no-decay stabilized source, validate a readout-snapshot selector, or build a cheap continuation-budget proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-continuation-boundary.md`

MIXED-NEGATIVE: No-decay stabilized source acquisition reliably implies transferable non-bottleneck geometry.
Conclusion: Fresh CLI seed `10` reached source final `0.9000` and learned calc `0.8984`, but final-source handoff reached only `0.3275`, direct readout `0.4275`, and 800-step continuation `0.4350`.
Do not repeat: Same seed-10 no-decay source, final-source handoff, direct readout, or 800-step continuation as novelty.
Next allowed test: Compare seed-9 positive vs seed-10 negative geometry, build a transfer/readout proxy, or optimize source acquisition for continuation/readout geometry.
Source: `aiAgentWorkHistory/phase7/2026-05-29-stabilized-source-seed10-replication.md`

MIXED-NEGATIVE: The seed-10 transfer failure is only a bad final-checkpoint selection artifact.
Conclusion: Earlier seed-10 checkpoints improved 600-step handoff over final (`0.4475/0.4325/0.4225` vs `0.3375`), but all stayed below seed-9 final reference (`0.5250` at 600 and `0.6500` final eval); frozen-state linear probing was not a valid selector because it ranked seed-10 final highest (`0.4500`) despite worst handoff.
Do not repeat: Same seed-10 step `1000`/`1300`/`1400` 600-step handoff sweep or frozen-state linear probe over these checkpoints as novelty.
Next allowed test: Build an additive learning-slope or injection-to-answer geometry proxy, or optimize source acquisition for early handoff slope.
Source: `aiAgentWorkHistory/phase7/2026-05-29-seed10-source-checkpoint-geometry-sweep.md`

MIXED-NEGATIVE: A direct additive handoff geometry probe can replace the 400/600-step handoff selector.
Conclusion: Forced-result geometry flags seed-10 as hostile (`true_best=0.0`, true top-3 `0.03-0.045`, true-best gap `0.0058-0.0063`) versus seed-9 positive (`true_best=0.0625`, top-3 `0.2125`, gap `0.0034`), but it does not cleanly separate `src6` positive from `src7` boundary-negative and 100-step loss slope is not a reliable selector.
Do not repeat: Same geometry probe over seed-9 final, seed-10 `1000/1300/1400/final`, `src6` final, or `src7` step `1400` as novelty.
Next allowed test: Add forced-result geometry as a source-training snapshot metric, optimize it during source acquisition, or keep using actual handoff probes as selection gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-handoff-geometry-probe.md`

DISPROVEN: Additive forced-result geometry can select source checkpoints inside known source families.
Conclusion: On known `src2/src4/src5` handoff comparisons, geometry only partially identified `src5` step `1100`; it tied or favored non-winners for `src4` and tied `src2` step `1300` versus final, while true-best gap selected wrong checkpoints.
Do not repeat: Same geometry scan over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Use geometry only as logging/warning, optimize source acquisition for actual early handoff slope, or design a stronger one/few-update proxy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-geometry-selector-validation.md`

DISPROVEN: A 25/50/100-step downstream loss-slope proxy can replace the additive handoff selector.
Conclusion: On known `src2/src4/src5` handoff comparisons, 100-step loss/loss-drop selects the wrong checkpoint for `src5` and `src4`, though it selects `src2` final correctly; existing exact-match traces show `src5` still needs about 500 steps to select its known winner.
Do not repeat: Same 0/25/50/100-step loss-slope probe over `src2` step `1300`/final, `src4` step `1000/1200`/final, or `src5` step `1100/1400/1500`/final as novelty.
Next allowed test: Keep 500/600-step handoff gates, optimize source acquisition against actual early handoff exact, or train a learned proxy on accumulated handoff traces.
Source: `aiAgentWorkHistory/phase7/2026-05-29-short-slope-selector-validation.md`

DISPROVEN: Lowering the hard improvement-assignment source weight from `10` to `5` rescues transfer-hostile seed-10 geometry.
Conclusion: The weight-5 seed-10 source weakened to final eval `0.6750`; its best source snapshots were around `0.78`, and 600-step additive handoffs from step `1200`/final reached only `0.3425`/`0.2475` snapshots and `0.3000`/`0.2325` final eval.
Do not repeat: Same seed-10 no-decay entropy `0.05`, diversity `0.1`, improvement weight `5`, 1600-step source run or step-1200/final 600-step frozen handoffs as novelty.
Next allowed test: Optimize source acquisition against actual 500/600-step handoff behavior, add a direct handoff/readout geometry term, or train a learned selector validated against the handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-29-source-assignment-weight5-transfer-probe.md`

DISPROVEN: A simple learned ridge selector over early handoff trace features can replace the 500/600-step handoff gate.
Conclusion: Leave-family-out ridge over 21 deduped candidates and 8 source families reached `3/8`, `4/8`, `3/8`, and `5/8` winner accuracy at prediction steps `200/300/400/500`; raw early exact matched or beat it at every step and reached `6/8` at step `500`.
Do not repeat: Same ridge selector over normal/zero/oracle/forced-random/calc early trace features on the current Phase 7 handoff trace dataset as novelty.
Next allowed test: Add logging-only in-training additive handoff probes, collect more labeled families, or test a richer learned selector only if it beats raw early exact under leave-family-out validation.
Source: `aiAgentWorkHistory/phase7/2026-05-29-handoff-trace-learned-selector-audit.md`

TOOLING: Logging-only in-training additive handoff probes are implemented.
Conclusion: `overfit_one_batch.py` can now clone current source state into additive non-bottleneck mode, freeze the calculator policy, train a bounded downstream probe, and log probe rows/metrics without feeding probe gradients back into source training.
Do not repeat: One-step smoke under `runs/2026-05-29_phase7_additive_handoff_probe_logging_smoke` as novelty.
Next allowed test: Run a real source-acquisition lineage with meaningful 500-step probe logging and verify selected checkpoints with the established handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-29-in-training-additive-handoff-probe-logging.md`

MIXED-NEGATIVE: A 500-step in-training additive handoff probe is not yet a reliable source-checkpoint selector.
Conclusion: On fresh no-decay source seed `11`, in-training probe normal @500 chose source step `400` (`0.5625`) over step `800` (`0.5525`), but standalone 600-step frozen-policy handoff favored step `800` by a wide margin (`0.6925` snapshot, `0.7075` final eval) over step `400` (`0.5975` snapshot, `0.6050` final eval).
Do not repeat: Same source step `400` vs step `800`, 500-step embedded probe plus standalone 600-step verification as novelty.
Next allowed test: Treat embedded 500-step probes as logging/triage only, verify with standalone 600-step handoffs, or run embedded probes with 600 steps / richer trend metrics before using them for selection.
Source: `aiAgentWorkHistory/phase7/2026-05-29-intraining-probe-source-selection-validation.md`

PARTIAL: Local-target propagation style result-boundary targets can pass Stage 0, but sharp settings mostly recover the full-enum boundary teacher.
Conclusion: A new exact-grid Stage 0 diagnostic found that current-policy-reweighted forced-loss targets align with the hard boundary ceiling when sharp (`t=0.25` result/upstream cosine `~1.0/~1.0`) and remain strongly aligned at `t=1.0` (`0.9355/0.8766`), while ordinary expected answer loss stayed anti-aligned (`-0.1045/-0.0034`). Local logit-descent targets also aligned when weakly proximal (`p=0.01` `~1.0/~1.0`, `p=0.1` `0.9998/0.9997`), but a stronger proximity setting collapsed toward the failed expected-loss direction (`p=1.0` `-0.0895/-0.0028`).
Do not repeat: The same seed-2 exact-grid Stage 0 sweep over policy-reweighted temperatures `0.25/0.5/1/2` and logit-descent proximity `0.01/0.1/1` as novelty.
Next allowed test: If continuing this family, run a short Stage 1 lift gate for the softer aligned settings (`policy_reweighted_t1` or `logit_descent_p0.1`) against the hard-boundary ceiling, then design an approximation that avoids full result-class enumeration before calling it scalable.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-propagation-gate.md`

PARTIAL-POSITIVE: The softer policy-reweighted local target produces Stage 1 lift above the failed expected-loss baseline.
Conclusion: In a 200-step exact-grid Stage 1 gate, `policy_reweighted_t1` reached `0.5600` exact-grid calculator-result accuracy and `0.5391` sampled normal accuracy with controls low (`injection_zero=0.0234`, `forced_random=0.0156`), slightly above the hard-boundary ceiling run at the same budget (`0.5500` calc, `0.4844` normal). Ordinary expected loss collapsed to near chance (`0.0025` calc, `0.0000` normal), while `logit_descent_p0.1` improved but lagged (`0.2950` calc, `0.1953` normal).
Do not repeat: The same seed-2, 200-step Stage 1 comparison of `hard_boundary`, `expected_loss`, `policy_reweighted_t1`, and `logit_descent_p0.1` as novelty.
Next allowed test: Replicate or extend `policy_reweighted_t1` to a longer convergence/retention gate, then design a sampled/top-k/learned approximation that avoids full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-stage1-lift-gate.md`

PARTIAL-POSITIVE: `policy_reweighted_t1` local targets can survive and improve during answer-only retention, but target training is nonmonotonic.
Conclusion: In an 800-step target-training plus 200-step answer-only retention gate, `policy_reweighted_t1` trailed hard-boundary at target step 800 (`0.7050` vs `0.8200` exact-grid calc) after peaking at step 600 (`0.8925`), but finished retention at `0.8925` exact-grid calc and `0.8750` sampled normal versus hard-boundary `0.8050`/`0.8281`. Controls remained causal (`injection_zero=0.0234`, `forced_random=0.0156`, oracle `1.0000`).
Do not repeat: The same seed-2, 800-target-step plus 200-retention-step comparison of `hard_boundary` and `policy_reweighted_t1` as novelty.
Next allowed test: Seed-replicate only if stability is the explicit question; otherwise approximate `policy_reweighted_t1` with sampled/top-k/learned targets that avoid full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-local-target-convergence-retention-gate.md`

MIXED-NEGATIVE: Naive sparse sampled `policy_reweighted_t1` targets do not preserve the full-enum learning signal unless candidate coverage is near-full.
Conclusion: In a 200-step no-replacement sparse approximation gate, exact `policy_reweighted_t1` reached `0.5600` exact-grid calc and `0.5391` sampled normal. Sparse uniform branches improved with coverage but lagged badly: `u16` reached `0.1975` calc, `u24` `0.2800`, `u32` `0.3350`, and near-full `u36` `0.4100`; only full-vocabulary `u39` recovered/exceeded the signal at `0.6250`. Top-k plus uniform was worse (`k8_u8` `0.0925` calc).
Do not repeat: The same seed-2 200-step sparse candidate ladder over `k8_u8/k0_u16/k0_u24/k0_u32/k0_u36/k0_u39` as novelty.
Next allowed test: Use a smarter proposal/learned candidate generator or importance-corrected target that improves true-result coverage without near-full forced-result enumeration.
Source: `aiAgentWorkHistory/phase7/2026-05-29-sampled-local-target-approximation-gate.md`

DISPROVEN: Simple loss-ranked neighborhood expansion rescues sparse `policy_reweighted_t1` local targets.
Conclusion: In a 200-step adaptive proposal gate, raw uniform `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` exact-grid calc and `0.3438` sampled normal. Adaptive low-loss-neighborhood branches underperformed at similar raw scoring budgets: `u8_b4_r2` `0.2025` calc, `u8_b4_r3` `0.2600`, and `u12_b4_r2` `0.2700`; the adaptive branches had lower unique coverage (`18.42-22.08` unique results) and lower true-result coverage (`0.6350-0.7700`) than raw `u32` (`32` unique, `0.8450` coverage).
Do not repeat: The same seed-2 200-step adaptive neighborhood gate over `u8_b4_r2/u8_b4_r3/u12_b4_r2` as novelty.
Next allowed test: Use a learned proposal or importance/bias-corrected sampled target; otherwise pivot to source-acquisition-for-handoff geometry instead of more local sampled-candidate variants.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-local-target-proposal-gate.md`

MIXED-POSITIVE: A forced-true additive readout auxiliary can shape transfer geometry during bottleneck source acquisition.
Conclusion: In a reduced `operand_max=9`, 100-step seed-13 source gate, adding `--additive-forced-true-loss-weight 0.5` made the true result the best forced additive result on `0.5900` of the grid (`top3=0.6900`) versus baseline `0.0000`/`0.0000`, and lowered 50-step additive slope final loss (`0.7367` vs `1.5305`). It also weakened source policy acquisition at the same budget (`0.2800` source calc and `0.2800` final eval vs baseline `0.3500` calc and `0.3800` final eval).
Do not repeat: The same small `operand_max=9`, 100-step seed-13 baseline vs aux-weight `0.5` source/geometry gate as novelty.
Next allowed test: Use a scheduled/gated auxiliary or retention anchor to avoid competing with source policy acquisition, then verify on `operand_max=19` with targeted standalone 600-step additive handoff gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-source-aux-gate.md`

POSITIVE: Delaying the forced-true additive auxiliary can preserve source acquisition while shaping additive geometry.
Conclusion: On the same reduced `operand_max=9`, 100-step seed-13 gate, turning on `--additive-forced-true-loss-weight 0.5` only after step `50` beat both baseline and always-on aux on source acquisition (`0.3900` source calc and `0.4000` final eval, vs baseline `0.3500`/`0.3800` and always-on `0.2800`/`0.2800`) while keeping a large additive geometry gain (`forced_best_true=0.5100`, `top3=0.5600`, 50-step slope final loss `0.7979` vs baseline `0.0000`/`0.0000`/`1.5305`).
Do not repeat: The same small `operand_max=9`, seed-13, 100-step, start-step-50 schedule gate as novelty.
Next allowed test: Scale to `operand_max=19` with source-only checkpointing first, then verify promising scheduled-aux checkpoints with targeted standalone 600-step additive handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-schedule-gate.md`

POSITIVE: The scheduled forced-true additive source objective improves full-grid standalone handoff.
Conclusion: On `operand_max=19`, seed-13, 200-step source acquisition, scheduled aux (`weight=0.5`, start step `50`) approximately tied baseline source policy (`0.2800` vs `0.2875` train calc; `0.2750` vs `0.2825` final eval) but strongly improved additive geometry (`forced_best_true=0.2125` vs `0.0000`, 50-step slope loss `1.0360` vs `1.8058`) and the trusted 600-step frozen-policy handoff (`0.4150` final eval / `0.3925` step-600 snapshot vs baseline `0.2525` / `0.2625`).
Do not repeat: The same seed-13, `operand_max=19`, 200-step baseline vs scheduled step-50 source gate plus 600-step handoff as novelty.
Next allowed test: Extend scheduled source acquisition to longer horizons (`400/600/800`) and verify selected checkpoints with standalone 600-step additive handoff; add a policy-retention anchor if source accuracy drifts.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-op19-gate.md`

POSITIVE: Longer scheduled forced-true source acquisition compounds handoff geometry through step 600.
Conclusion: Extending the seed-13 scheduled source to `800` steps improved forced-result geometry from step `200` to `600` (`forced_best_true 0.2125 -> 0.9800`, 50-step slope loss `1.0360 -> 0.4719`) and the step-600 source checkpoint reached `0.7725` final eval under the trusted 600-step frozen-policy additive handoff. Step `800` had perfect forced-result geometry but worse handoff (`0.6750` final), so final source checkpoint is not automatically best.
Do not repeat: The same seed-13 scheduled source `200/400/600/800` geometry ladder or step-600 vs step-800 handoff comparison as novelty.
Next allowed test: Run continuation/readout from the step-600 handoff lineage to test whether the scheduled source can clear the high non-bottleneck gate; replicate on a fresh seed only if stability is the explicit question.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-true-long-source-gate.md`

MIXED-POSITIVE: The scheduled step-600 source lineage remains calculator-dependent but misses the high continuation/readout gate.
Conclusion: Starting from scheduled step-600 handoff final `0.7725`, 800-step frozen-policy continuation reached only `0.7775`, 600-step readout after continuation reached `0.8175`, and an extra 1000 stable-policy readout steps reached `0.8475`; controls stayed low (`injection_zero <=0.0547`, forced-random <=`0.0391`), but learned calc stayed around `0.5391`.
Do not repeat: The same scheduled step-600 handoff -> 800 continuation -> 600 readout -> extra 1000 readout chain as novelty.
Next allowed test: Improve source policy accuracy while preserving scheduled geometry, or run continuation/readout only after a scheduled source checkpoint shows both strong handoff geometry and materially higher learned calculator accuracy.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-continuation-readout.md`

POSITIVE: Gentle low-LR recovery from a scheduled source checkpoint restores learned calc and clears readout.
Conclusion: Continuing the scheduled step-600 source for 30 CPU steps with LR `3e-4` and lower forced-true weight `0.1` raised source calc from `0.5800` to `0.7950` while keeping forced-true loss low; the resulting frozen-policy 600-step handoff reached `0.8425`, 800-step continuation reached `0.8900` final / `0.9175` best snapshot, and 600-step readout reached `0.9320` final eval with final diagnostic controls normal `0.9225`, injection-zero `0.0300`, forced-random `0.0325`, learned calc `0.7925`.
Do not repeat: The same seed-13 scheduled step-600 -> 30-step low-LR `aux=0.1` recovery -> 600 handoff -> 800 continuation -> 600 readout chain as novelty.
Next allowed test: Replicate on a fresh scheduled source seed or integrate the low-LR/lower-aux recovery as an automatic late-source phase, then verify with the trusted 600-step handoff and continuation/readout gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-scheduled-source-low-lr-recovery.md`

POSITIVE: The gentle scheduled-source recovery recipe replicates on a fresh seed and clears handoff directly.
Conclusion: Fresh seed-14 scheduled source training reached step-600 source eval `0.6675`; the same 30-step CPU recovery (`lr=3e-4`, forced-true weight `0.1`) raised source eval to `0.8850`, and the trusted 600-step frozen-policy additive handoff reached `0.9600` final eval / `0.9650` step-600 snapshot with learned calc `0.8700`, injection-zero `0.0850`, and forced-random `0.0875`.
Do not repeat: The same seed-14 scheduled source -> 30-step low-LR recovery -> 600-step frozen-policy handoff as novelty.
Next allowed test: Automate the late-source transition or test a third seed only if the explicit question is stability; keep the 600-step handoff/readout gates as arbiter and monitor the somewhat higher seed-14 zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-scheduled-source-recovery-replication.md`

POSITIVE: The late scheduled-source recovery phase can be automated in one source run.
Conclusion: Adding an in-run late-source recovery switch at step `600` (LR multiplier `0.1`, forced-true weight override `0.1`) preserved the seed-14 recovery effect without manual checkpoint relaunch: final source eval reached `0.8775`, and the trusted 600-step frozen-policy handoff reached `0.9400` final eval / `0.9475` step-600 snapshot with learned calc `0.8725`, injection-zero `0.0800`, and forced-random `0.0775`.
Do not repeat: The same seed-14 automated fixed-step-600 recovery plus 600-step handoff as novelty.
Next allowed test: Replace the fixed recovery step with adaptive transition criteria, or move the source branch toward less prescriptive/scalable assignment while preserving the handoff/readout gates.
Source: `aiAgentWorkHistory/phase7/2026-05-29-automated-scheduled-source-recovery.md`

MIXED-POSITIVE: A simple source-accuracy trigger can replace the fixed late-source recovery step on seed 14, but controls worsen.
Conclusion: After wiring adaptive recovery to switch both LR and forced-true weight, `result_policy_argmax_result_accuracy >= 0.65` with min step `500` fired at step `528`; source final eval was only `0.8250`, but the trusted 600-step frozen-policy handoff reached `0.9850` final eval / `0.9775` step-600 snapshot with learned calc `0.8425`, injection-zero `0.1325`, and forced-random `0.1325`. This beats the fixed step-600 handoff final eval (`0.9400`) but has higher controls than the fixed-step run (`0.0800`/`0.0775`).
Do not repeat: The same seed-14 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Replicate the adaptive trigger on a fresh scheduled source, or use a smoothed/conjunctive trigger that preserves the handoff lift while reducing zero/random controls.
Source: `aiAgentWorkHistory/phase7/2026-05-29-adaptive-source-recovery-trigger.md`

MIXED-NEGATIVE: The simple source-accuracy adaptive recovery trigger does not replicate on a fresh seed.
Conclusion: On a fresh seed-17 source run, the same `result_policy_argmax_result_accuracy >= 0.65` min-step-500 trigger never fired; source final eval reached only `0.6100`, and trusted 600-step frozen-policy handoff reached `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`. A matched fixed step-600 control did better but still missed the high gate: source final `0.7450`, handoff `0.7675` final / `0.7850` snapshot, learned calc `0.7350`, injection-zero `0.0500`, forced-random `0.0375`.
Do not repeat: The same fresh seed-17 `argmax_result_accuracy >= 0.65`, min-step-500 adaptive run or matched fixed step-600 control as novelty.
Next allowed test: Use a smoothed/conjunctive recovery trigger or a different transition metric; do not treat raw source argmax accuracy thresholding as validated.
Source: `aiAgentWorkHistory/phase7/2026-05-29-fresh-adaptive-recovery-trigger-replication.md`

MIXED: Forced-true loss is a better adaptive recovery trigger than raw source accuracy on seed 17, but it does not clear the gate.
Conclusion: On seed 17, `additive_forced_true_loss <= 0.05` with min step `500` triggered at step `500`, reduced the late forced-true weight to `0.1`, and improved source final eval to `0.7225` versus `0.6100` for the no-trigger source. The trusted 600-step frozen-policy handoff reached `0.7625` final eval / `0.7825` step-600 snapshot with learned calc `0.7350`, injection-zero `0.0450`, and forced-random `0.0325`, close to the fixed step-600 control (`0.7675`) and above the raw source-accuracy trigger (`0.6825`), but still below the high gate.
Do not repeat: The same seed-17 `additive_forced_true_loss <= 0.05`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Use a smoothed/conjunctive transition criterion or move back toward scalable assignment; one raw metric can recover fixed-step timing but has not produced robust high-gate clears.
Source: `aiAgentWorkHistory/phase7/2026-05-29-forced-loss-adaptive-recovery-trigger.md`

MIXED-POSITIVE: EMA plus patience forced-loss recovery improves the hard seed-17 handoff but still misses the high gate.
Conclusion: Adding trigger EMA/patience support and running `additive_forced_true_loss <= 0.05` with EMA beta `0.8`, patience `10`, and min step `500` fired at step `509`. Source final eval reached `0.7625`, and trusted 600-step frozen-policy handoff reached `0.8025` final eval / `0.7975` step-600 snapshot with learned calc `0.7425`, injection-zero `0.0625`, and forced-random `0.0325`. This beats raw forced-loss trigger (`0.7625` handoff), fixed step-600 (`0.7675`), and raw source-accuracy trigger (`0.6825`), but remains below the high gate.
Do not repeat: The same seed-17 forced-loss `threshold=0.05`, EMA beta `0.8`, patience `10`, min-step-500 adaptive recovery plus 600-step handoff as novelty.
Next allowed test: Try a conjunctive source-plus-geometry trigger or return to scalable assignment; smoothing/patience helps timing but is not sufficient by itself.
Source: `aiAgentWorkHistory/phase7/2026-05-29-smoothed-forced-loss-recovery-trigger.md`

MIXED-NEGATIVE: A hard source-accuracy conjunction is too conservative for the hard seed-17 recovery trigger.
Conclusion: Adding an optional secondary adaptive trigger and requiring forced-loss readiness (`additive_forced_true_loss <= 0.05`, EMA beta `0.8`, patience `10`, min step `500`) plus `result_policy_argmax_result_accuracy >= 0.70` never activated recovery on seed 17. The primary forced-loss condition was ready for `132` consecutive logged steps and ended with EMA `0.0055`, but the secondary source-accuracy metric ended at `0.6325`; source final stayed `0.6100`, and trusted 600-step frozen-policy handoff reached only `0.6825` final eval / `0.6925` step-600 snapshot with learned calc `0.6075`, injection-zero `0.0400`, and forced-random `0.0500`.
Do not repeat: The same seed-17 forced-loss EMA/patience plus secondary source-accuracy `>=0.70` conjunctive recovery trigger and 600-step handoff as novelty.
Next allowed test: Do not keep tuning hard source-accuracy gates. Return to scalable assignment or source objectives that improve handoff/readout geometry directly; if another adaptive transition is tried, it needs a new signal family or a predeclared reason it should avoid the seed-17 no-fire failure.
Source: `aiAgentWorkHistory/phase7/2026-05-29-conjunctive-recovery-trigger.md`

POSITIVE: Replay-memory local targets can approximate `policy_reweighted_t1` with fewer fresh forced-result scores per step.
Conclusion: Adding `memory_policy_reweighted_t1_u8_m24` to the Stage 1 local-target runner lets each prompt cache observed forced-result losses and train on 8 fresh uniform candidates plus 24 low-loss cached candidates. At 200 steps it beat the raw uniform `u32` baseline while scoring one quarter as many fresh results per step: exact-grid calc `0.5900` and sampled normal `0.5391` versus `0.3350`/`0.3438`; target true-candidate coverage reached `1.0000`, target argmax accuracy `0.9850`, and controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, the memory branch reached target `0.9600` exact calc / `0.9766` sampled normal and retained `0.8600` calc / `0.8750` sampled normal under answer-only training.
Do not repeat: The same seed-2 replay-memory `u8_m24` versus raw uniform `u32` 200-step gate or the same single-branch 800+200 retention gate as novelty.
Next allowed test: Stress scalability rather than rerun the positive: reduce fresh scoring (`u4` or lower), add aging/rescoring to handle stale losses, or test whether a learned/generalized memory proposal works beyond the fixed exhaustive grid.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-local-target-gate.md`

POSITIVE: Replay-memory local targets still work with only 2 fresh forced-result scores per step, but retention weakens.
Conclusion: A lower fresh-score budget sweep compared raw uniform `u32` with replay-memory `u8_m24`, `u4_m28`, `u2_m30`, and `u1_m31`. At 200 steps, `u2_m30` was best: exact-grid calc `0.6025` and sampled normal `0.6016` with only 2 fresh forced-result scores per step, versus `u8_m24` `0.5900`/`0.5391`, `u4_m28` `0.5100`/`0.4844`, `u1_m31` `0.4075`/`0.4219`, and raw `u32` `0.3350`/`0.3438`; controls stayed low (`injection_zero=0.0234`, `forced_random=0.0156`). In an 800+200 retention gate, `u2_m30` reached target `0.9000` calc / `0.8750` normal and retained `0.7850` calc / `0.7656` normal, below the prior `u8_m24` retention (`0.8600`/`0.8750`) but still far above sparse uniform baselines.
Do not repeat: The same seed-2 lower-budget sweep over `u8_m24/u4_m28/u2_m30/u1_m31` or the same `u2_m30` 800+200 retention gate as novelty.
Next allowed test: Move from budget sweeps to scalability stressors: stale-cache aging/rescoring, memory reset, streaming/non-exhaustive prompts, or learned/generalized candidate memory. Treat `u2_m30` as the best current low-fresh-score point and `u1_m31` as below the useful budget floor at 200 steps.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-lower-budget-gate.md`

MIXED-NEGATIVE: Simple top-cached-candidate rescoring does not improve replay-memory local-target retention.
Conclusion: Adding optional `_rN` cached-candidate rescoring to replay-memory branches showed no benefit for the best low-fresh branch. At 200 steps, `memory_policy_reweighted_t1_u2_m30_r2` exactly tied no-rescore `u2_m30` (`0.6025` exact calc / `0.6016` sampled normal) at double the forced-score cost (`4` vs `2` scores per step), while heavier rescoring was worse: `r4` reached `0.5300` calc / `0.5781` normal and `r8` reached `0.4675` / `0.4609`. The 800+200 `r2` retention gate also exactly tied no-rescore `u2_m30`: target `0.9000` calc / `0.8750` normal and retention `0.7850` calc / `0.7656` normal.
Do not repeat: The same seed-2 `u2_m30` rescore sweep over `r2/r4/r8` or the same `u2_m30_r2` 800+200 retention gate as novelty.
Next allowed test: Stop simple rescore-count tweaking. Attack transduction directly with finite/reset memory, streaming/non-exhaustive prompts, or learned/generalized candidate memory.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-rescore-gate.md`

MIXED-NEGATIVE: Finite reset windows expose replay-memory's fixed-grid transductive dependence.
Conclusion: Adding optional `_resetN` replay-memory syntax showed that persistent per-prompt caches are doing important work. In the 200-step stress gate, no-reset `u2_m30` reached `0.6025` exact calc / `0.6016` sampled normal, while `reset50` fell to `0.2500` / `0.2578`, `reset25` to `0.1650` / `0.2188`, and `reset10` to `0.0950` / `0.1406`. A 199-step boundary check removed the final-reset snapshot caveat: no-reset was `0.5925` / `0.5938`, `reset100` was only `0.4575` / `0.4453`, and `reset50` was `0.2575` / `0.2812` despite mostly restored target coverage (`0.9925` and `0.9525` true-candidate coverage respectively).
Do not repeat: The same seed-2 replay-memory reset sweep over `reset10/reset25/reset50` or the 199-step `reset50/reset100` boundary check as novelty.
Next allowed test: Do not tune reset intervals as a local fix. Move replay-memory work to streaming/non-exhaustive prompt stress or learned/generalized proposal memory, where the method cannot rely on persistent prompt-identity caches.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md`

MIXED-NEGATIVE: Streaming minibatches remove replay-memory's strong fixed-grid local-target lift.
Conclusion: Adding `--streaming-train-batch-size` and prompt-keyed replay caches showed that replay memory does not preserve the fixed-grid advantage under sampled minibatch training. At 200 steps with batch `16`, exact `policy_reweighted_t1` reached only `0.1100` exact calc, raw uniform `u32` `0.0700`, `u2_m30` `0.0475`, and `u8_m24` `0.0950`. Extending to 800 batch-16 steps raised exact and raw `u32` to `0.2450`, `u8_m24` to a comparable `0.2650`, and `u2_m30` to only `0.1850`; batch-64 for 200 steps also stayed weak (`0.1650` exact, `0.1475` u8, `0.0975` u2). Prompt memory reached all 400 prompts and `u8_m24` often had full current-batch target coverage, so the missing lift is not just absent prompt keys.
Do not repeat: The same seed-2 streaming minibatch gates at batch `16` for `200/800` steps or batch `64` for `200` steps over exact, raw `u32`, `u2_m30`, and `u8_m24` as novelty.
Next allowed test: Do not treat prompt-keyed replay memory as the scalable answer. Continue local targets only with a learned/generalized proposal, estimator correction, or a different target construction; otherwise return mainline compute to source objectives aimed at handoff/readout geometry.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md`

PAUSED: Fixed replay-memory local-target proposals are not the scalable path.
Conclusion: The replay-memory branch produced a real fixed-grid positive, but the follow-up stress tests identify the mechanism as prompt-identity transduction rather than a scalable candidate proposal. Lower fresh budgets worked on the fixed grid, but rescoring did not improve retention, reset windows damaged learning, and streaming minibatches removed the strong lift. This pauses fixed per-prompt replay caches as a family, including fresh-count, rescore-count, reset-interval, batch-size, and longer-run variants.
Do not repeat: Do not run more fixed replay-memory budget ladders, rescore ladders, reset intervals, streaming batch-size/length checks, or seed replications as novelty.
Next allowed test: Local-target work needs a genuinely new mechanism: learned/generalized candidate proposal, estimator/bias correction, or a target construction that does not require the useful result to already be in a hand-coded candidate set. Otherwise prioritize source objectives that improve additive handoff/readout geometry.
Source: `researchReviews/2026-05-29-replay-memory-branch-review.md`

MIXED-POSITIVE: A scheduled additive forced-margin auxiliary can shape source handoff geometry without hurting small-gate source policy acquisition.
Conclusion: Adding `--additive-forced-margin-loss-weight` trains the additive path contrastively: the true forced result should have lower answer loss than sampled wrong forced results. On the matched `operand_max=9`, seed-13, 100-step scheduled small gate (`start_step=50`, weight `0.5`, margin `0.05`, 4 negatives), source result-policy accuracy reached `0.4100` and final eval `0.3800`, comparable to the earlier scheduled forced-true small gate (`0.3900`/`0.4000`) and better than always-on forced-true source accuracy (`0.2800`). Geometry improved versus baseline and partly versus scheduled forced-true: `forced_best_true=0.6200`, `top3=0.7500`, and `true-best gap=0.0082`, but 50-step slope final loss `1.0238` was worse than scheduled forced-true (`0.7979`) while still better than baseline (`1.5305`).
Do not repeat: Do not rerun the same `operand_max=9`, seed-13, 100-step scheduled forced-margin small gate as novelty.
Next allowed test: If pursuing this branch, run a full-grid `operand_max=19` scheduled forced-margin source gate with targeted geometry/handoff validation against the existing scheduled forced-true source objective; otherwise keep source objectives focused on actual 600-step handoff/readout behavior.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-source-aux-gate.md`

POSITIVE: A budgeted one-negative forced-margin source objective improves the matched full-grid early handoff gate.
Conclusion: The full-grid 4-negative forced-margin branch was too costly locally and was stopped after it only wrote config. Reducing to one sampled negative per prompt made the contrastive objective practical and positive on the matched `operand_max=19`, seed-13, 200-step source gate. The source reached `0.3225` train calc / `0.3600` final eval, above the earlier scheduled forced-true 200-step source (`0.2800`/`0.2750`). Geometry was mixed: forced-result ranking was strong (`forced_best_true=0.6725`) but 50-step slope final loss was `1.4660`, worse than scheduled forced-true (`1.0360`). The trusted 600-step frozen-policy additive handoff resolved the conflict positively: final eval `0.6600`, step-600 normal `0.7050`, injection-zero `0.0000`, forced-random `0.0250`, learned calc `0.3375`, beating the matched scheduled forced-true handoff (`0.4150`) and baseline (`0.2525`).
Do not repeat: Do not rerun the same seed-13, `operand_max=19`, 200-step one-negative forced-margin source plus 600-step handoff as novelty, and do not run the 4-negative full-grid branch without a compute-reduction plan.
Next allowed test: Extend the one-negative forced-margin source to longer horizons (`400/600`) and verify with trusted 600-step handoff, or replicate on a fresh seed if the explicit question is stability. Keep slope/geometry as diagnostics only; actual handoff remains arbiter.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-op19-gate.md`

MIXED-POSITIVE: Longer one-negative forced-margin source training improves handoff but does not clearly beat scheduled forced-true step-600.
Conclusion: Extending the one-negative forced-margin source branch on `operand_max=19`, seed-13 improved source accuracy and handoff, but exposed checkpoint/RNG sensitivity. A fresh 600-step source run reached `0.5225` train calc / `0.4800` final source eval with near-perfect geometry (`forced_best_true=0.9925`), and its step-600 checkpoint reached `0.7330` final eval / `0.7500` step-600 normal under trusted frozen-policy handoff with injection-zero `0.0000`, forced-random `0.0225`, learned calc `0.4975`. Continuing the exact prior positive step-200 checkpoint gave a better intermediate source checkpoint after 200 continuation steps (`0.4725` calc, `forced_best_true=0.9675`) whose handoff reached `0.7400` final eval / `0.7850` step-600 normal with injection-zero `0.0000`, forced-random `0.0300`, learned calc `0.4175`; continuing to 400 steps degraded source final eval back to `0.3600`. This improves over the one-negative 200-step handoff (`0.6600`) but does not clearly beat scheduled forced-true step-600 final handoff (`0.7725`).
Do not repeat: Do not rerun the same seed-13 one-negative forced-margin 600-step source ladder, the same continuation from step-200, or handoffs from the tested step-400/step-600/continued-step-200 checkpoints as novelty.
Next allowed test: Try a late source-recovery/retention phase for one-negative margin only if explicitly testing the source-policy bottleneck; otherwise compare on a fresh seed or move to less prescriptive/scalable assignment. Keep actual 600-step handoff as arbiter because slope/geometry stayed imperfect selectors.
Source: `aiAgentWorkHistory/phase7/2026-05-29-additive-forced-margin-long-source-gate.md`

CONSTRAINED: One-negative forced-margin is a useful auxiliary, not a standalone new mainline.
Conclusion: Reviewing the forced-margin branch shows a real but bounded result. The one-negative objective is the scalable/practical variant and improves early full-grid handoff, but many-negative full-grid margin is too costly, longer same-seed one-negative training is checkpoint-sensitive, and the best longer handoff (`0.7400` final / `0.7850` snapshot) does not clearly surpass scheduled forced-true step-600 (`0.7725` final). The branch still relies on hard assignment and true-result forcing, so it does not solve non-prescriptive scalable credit assignment.
Do not repeat: Do not continue with negative-count tweaks, same-seed longer ladders, start-step tweaks, slope-proxy selection, or geometry-only checkpoint fishing as novelty.
Next allowed test: Stay in forced-margin only for a predeclared source recovery/retention test or a fresh-seed stability replication with trusted 600-step handoff. Otherwise move effort toward learned/generalized proposals, estimator correction, or a less prescriptive target construction.
Source: `researchReviews/2026-05-29-forced-margin-branch-review.md`

DISPROVEN: Preserving unscored policy mass does not rescue sparse policy-reweighted local targets.
Conclusion: Added `corrected_policy_reweighted_t<T>_u<U>_b<mean|current|max>`, which scores uniform candidates but imputes a baseline loss for unscored result classes instead of forcing their target mass to zero. In the 200-step full-grid gate, exact `policy_reweighted_t1` reached `0.5600` exact calc / `0.5391` sampled normal and raw `sampled_policy_reweighted_t1_k0_u32` reached `0.3350` / `0.3438`; corrected branches were worse: `u8_bmean` `0.1150` / `0.0938`, `u8_bcurrent` `0.1100` / `0.0938`, `u8_bmax` `0.0675` / `0.0625`, `u16_bmean` `0.2100` / `0.2500`, and `u16_bcurrent` `0.2500` / `0.2500`. The correction diluted pressure and did not overcome low true-candidate coverage (`0.1850` for `u8`, `0.4050` for `u16`).
Do not repeat: Do not rerun corrected/imputed sparse targets with the same mean/current/max baselines or simply tune `u8/u16` sample counts as novelty.
Next allowed test: Local-target approximation still needs a learned/generalized proposal, a stronger estimator correction with an explicit bias/variance argument, or a target construction that creates useful pressure without requiring true-result coverage.
Source: `aiAgentWorkHistory/phase7/2026-05-29-corrected-sparse-local-target-gate.md`

PARTIAL: A simple online learned loss proposal improves fixed-grid sparse local targets but does not solve streaming scalability.
Conclusion: Added `learned_policy_reweighted_t<T>_u<U>_p<P>_h<H>_e<E>`, which trains a small parametric forced-loss predictor on observed candidate scores and proposes low predicted-loss result classes alongside uniform exploration. In the 200-step full-grid gate at the same 32 forced scores per step, raw `u32` reached `0.3350` exact calc / `0.3438` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1` reached `0.5850` / `0.5703`, with proposal true-candidate coverage `1.0000`, target argmax `0.9175`, injection-zero `0.0234`, and forced-random `0.0156`; other learned 32-score branches reached `0.4850-0.5050` calc. But streaming minibatches removed the lift: at batch `16`, 200 steps gave exact `0.1100`, raw `u32` `0.0700`, and learned `0.0925`; at 800 steps, raw `u32` and learned tied at `0.2350` exact calc, with sampled normal `0.2734` vs `0.2656`.
Do not repeat: Do not rerun the same polynomial-feature online MLP proposal branches (`u4_p28_h32_e1`, `u8_p24_h32_e1`, `u16_p16_h32_e1`, `u8_p24_h64_e3`) on the fixed grid as novelty, and do not claim fixed-grid proposal coverage is scalability evidence without streaming/generalization lift.
Next allowed test: Learned proposal work needs an explicit streaming/generalization mechanism or validation objective, such as proposal training across heldout prompt ranges, replayed/off-policy proposal data that is not prompt-keyed, or a target construction that uses proposal uncertainty. Otherwise pivot local-target work away from proposal knobs.
Source: `aiAgentWorkHistory/phase7/2026-05-29-learned-proposal-local-target-gate.md`

MIXED-NEGATIVE: Random-prompt learned-proposal pretraining does not cleanly rescue streaming local targets.
Conclusion: Added optional `_wN` proposal pretraining for learned local-target branches, using random prompt/result forced-loss observations before model training. In a 200-step streaming batch-16 screen, raw `u32` reached `0.0700` exact calc / `0.0703` sampled normal, the online learned branch reached `0.0925` / `0.0938`, `_w20` reached `0.0975` / `0.0625`, and `_w50` reached `0.0950` / `0.0547`. In the 800-step streaming stress, raw `u32` reached `0.2350` exact calc / `0.2734` sampled normal, while `learned_policy_reweighted_t1_u4_p28_h32_e1_w20` reached `0.2625` exact calc but only `0.1797` sampled normal. The pretraining can slightly raise policy accuracy, but it did not produce a clean functional streaming lift.
Do not repeat: Do not keep tuning `_w20/_w50` warmup counts, pretrain batch size, or the same polynomial-feature MLP as novelty.
Next allowed test: If continuing learned proposals, change the generalization mechanism itself, such as heldout-range validation, a proposal state tied to evolving model features, uncertainty-aware candidate sets, or a different target construction. Otherwise pivot away from learned-proposal warmups.
Source: `aiAgentWorkHistory/phase7/2026-05-29-pretrained-learned-proposal-gate.md`

PAUSED: Simple local-target proposal approximation is not the scalable path.
Conclusion: Reviewing the local-target approximation cluster shows a consistent failure mode. Exact `policy_reweighted_t1` remains a useful ceiling and proof of principle, and replay/learned proposal variants produced fixed-grid positives, but simple proposal mechanisms did not survive scalability stress: raw/top-k/adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, unscored-mass imputation diluted pressure, the online learned proposal tied raw `u32` under 800-step streaming, and random-prompt proposal pretraining hurt sampled normal despite a small exact-calc nudge.
Do not repeat: Do not run more raw count ladders, top-k/neighborhood variants, fixed replay cache variants, mean/current/max imputation branches, or the same polynomial-feature learned proposal with or without `_wN` warmup as novelty.
Next allowed test: Local targets need a different estimator, a different target construction, or a learned proposal whose validation objective explicitly targets streaming/full-grid generalization. Otherwise prioritize source objectives aimed at actual 600-step additive handoff/readout behavior.
Source: `researchReviews/2026-05-29-local-target-proposal-branch-review.md`

POSITIVE: Low-LR recovery shows one-negative forced-margin was source-policy-maturity limited.
Conclusion: Continuing the longer one-negative forced-margin step-600 source checkpoint for `30` low-LR CPU steps (`lr=0.0003`, margin weight reduced from `0.5` to `0.1`, source stabilization retained) raised source calculator accuracy from `0.5225` to `0.7725` and final source eval to `0.7825`. The trusted frozen-policy 600-step non-bottleneck handoff from recovered step `30` reached `0.8700` final eval / `0.9050` step-600 normal, with injection-zero `0.0000`, forced-random `0.0313`, and learned calculator accuracy `0.8594`. This beats the unrecovered forced-margin handoffs (`0.7330-0.7400` final) and the old scheduled forced-true step-600 handoff (`0.7725` final), but remains below automated scheduled-source recovery (`0.9400` final) and still depends on hard assignment plus true-result contrastive forcing.
Do not repeat: Do not rerun the same seed-15 step-600 forced-margin checkpoint recovery with `lr=3e-4`, margin weight `0.1`, `30` source steps, and the same 600-step frozen-policy handoff as novelty.
Next allowed test: If staying in forced-margin, test fresh-seed stability or fold the recovery into an automated source run. Otherwise use it as evidence that source objectives need late gentle recovery while moving toward less-prescriptive target construction or estimator work.
Source: `aiAgentWorkHistory/phase7/2026-05-29-forced-margin-low-lr-source-recovery.md`

POSITIVE: Automated forced-margin recovery replicates strongly on a fresh seed.
Conclusion: Adding `--late-source-recovery-additive-forced-margin-loss-weight` folded the manual one-negative forced-margin recovery into a single source run. On fresh seed `16`, the step-600 late phase (`lr` multiplier `0.1`, forced-margin weight override `0.1`) improved source calc from `0.5825` at step `600` to `0.8825` at step `630`; final source eval was `0.8700`. The trusted frozen-policy 600-step non-bottleneck handoff from source step `630` reached `0.9875` final eval / `0.9800` step-600 normal, with injection-zero `0.0156-0.0250`, forced-random `0.0938`, and learned calc `0.8906`. This is the strongest forced-margin handoff so far and replicates the recovery mechanism beyond the manual checkpoint continuation, but it remains prescriptive because source training still uses hard assignment and true-result contrastive forcing.
Do not repeat: Do not rerun the same seed-16 630-step automated forced-margin recovery with late step `600`, LR multiplier `0.1`, margin weight override `0.1`, and the same 600-step frozen-policy handoff as novelty.
Next allowed test: If staying in forced-margin, test broader stability/scale or use it as a stepping stone toward less-prescriptive target construction; otherwise pivot back to scalable credit assignment, because this does not solve answer-loss-only discovery.
Source: `aiAgentWorkHistory/phase7/2026-05-30-automated-forced-margin-source-recovery.md`

REVIEW: Forced-margin recovery is now a staged-transfer benchmark, not a local knob branch.
Conclusion: The forced-margin branch answered its post-review questions. Manual low-LR recovery raised one-negative forced-margin handoff to `0.8700` final / `0.9050` step-600 normal, and automated fresh-seed recovery raised source calc `0.5825 -> 0.8825` during the late window and reached `0.9875` trusted frozen-policy handoff final eval / `0.9800` step-600 normal. This makes automated one-negative forced-margin recovery the current best staged-transfer source recipe and a benchmark for future objectives, but not the final solution because it still depends on hard improvement assignment, true-result forced-margin pressure, and frozen-policy staged transfer.
Do not repeat: Do not rerun seed-15 manual recovery, seed-16 automated recovery plus handoff, or same-setup forced-margin start-step/margin/negative-count/late recovery length tweaks as novelty.
Next allowed test: Use the recipe as a benchmark; future compute should either stress scale/stability or remove prescriptiveness by replacing hard assignment or true-result forcing with a new target construction or estimator.
Source: `researchReviews/2026-05-30-forced-margin-recovery-review.md`

POSITIVE: Answer-derived result-boundary source transfers but is not scalable.
Conclusion: The older full-grid result-boundary source checkpoint, trained with `result_boundary_target_loss_weight=1` and `hard_best_result`, transfers into the trusted frozen-policy additive non-bottleneck gate. The 600-step handoff reached `0.8825` final eval / `0.8425` step-600 normal, with injection-zero `0.0000`, forced-random `0.0391`, and learned calc `0.9922`. This shows that true-result forced-margin pressure is not strictly required for causal staged transfer; an answer-derived best-result target can create transferable result-level calculator use. It remains weaker than automated forced-margin recovery (`0.9875` final) and still depends on full forced-result enumeration plus frozen-policy staged transfer.
Do not repeat: Do not rerun the same May 13 stage-1 result-boundary step-800 checkpoint through the same 600-step frozen-policy additive handoff as novelty.
Next allowed test: Use this as a bridge toward less-prescriptive target construction or estimator work: approximate or replace full forced-result enumeration, test fresh-source stability only if predeclared, or compare a new answer-derived source objective against the forced-margin recovery benchmark.
Source: `aiAgentWorkHistory/phase7/2026-05-30-answer-derived-boundary-handoff.md`

MIXED-NEGATIVE: Naive hidden-state amortized boundary critics do not recover full-enum result-boundary targets.
Conclusion: A new diagnostic trained a shared MLP critic on sparse forced-result scores using prompt hidden-state features plus candidate calculator output vectors, then evaluated whether predicted losses recover the full-enum result-boundary argmin on heldout prompts. The full-enum boundary target was valid at all checked checkpoints (`1.0000` best=true-sum), but sparse critic recovery was poor: with `8` scores per train prompt, heldout argmin recovery was `0.0800` at step `0`, `0.0800` at step `100`, and `0.1700` at step `800`; with `24` scores per train prompt, it was still only `0.2600` at step `0` and `0.1900` at step `800`. Do not treat this pointwise hidden/output critic as a scalable replacement for full result-boundary enumeration.
Do not repeat: Do not rerun the same hidden-state plus candidate-output-vector pointwise MLP critic on the May 13 result-boundary source checkpoints with `k=8` or `k=24` sparse scores per prompt as novelty.
Next allowed test: Continue result-boundary approximation only with a stronger mechanism: rank/contrastive or uncertainty-aware critic objectives, feature validation tied to evolving model states, or a different target construction that does not require pointwise loss prediction to identify the exact argmin.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-amortized-critic-diagnostic.md`

REVIEW: Hidden-output boundary critic family is not the scalable result-boundary bridge.
Conclusion: The answer-derived result-boundary source transfer remains strategically useful, but the simple amortized critic family should pause. Pointwise hidden/output critics recovered heldout full-enum argmins only `0.0800-0.1700` at `k=8` and `0.1900-0.2600` at `k=24`. Pairwise ranking improved the trained step-800 checkpoint to `0.4000` argmin recovery at `k=24`, but step `0` stayed `0.2600`, hybrid was worse, and `k=24` already uses most of the 39-class result vocabulary. This is not a practical replacement for full enumeration.
Do not repeat: Do not continue pointwise, pairwise, hybrid, hidden-size, epoch-count, or learning-rate variants of the same hidden-state plus candidate-output-vector critic as novelty.
Next allowed test: Continue answer-derived result-boundary work only with a different target construction, uncertainty-aware compute allocation, or a generalization mechanism validated across evolving model states or prompt ranges.
Source: `researchReviews/2026-05-30-result-boundary-approximation-review.md`

DISPROVEN: Sparse sampled pairwise-preference targets do not train result-space calculator use.
Conclusion: Added `sampled_pairwise_preference_uN[_gG]`, which scores sparse forced-result candidates and trains the policy to rank lower answer-loss candidates above higher-loss candidates. In the 200-step fixed-grid Stage 1 gate, pairwise `u8` and `u16` stayed at `0.0050` final exact-grid calc / `0.0078` sampled normal, and pairwise `u32` reached only `0.0425` calc / `0.0234` normal despite `0.8450` true-candidate coverage. The same-budget `sampled_policy_reweighted_t1_k0_u32` comparator reached `0.3350` calc / `0.3438` normal. Simple sparse pairwise preference is therefore not a useful target construction here.
Do not repeat: Do not rerun `sampled_pairwise_preference_u8/u16/u32` on the same 200-step fixed-grid gate, and do not run simple candidate-count or `_gG` loss-gap sweeps as novelty.
Next allowed test: Pairwise-style work needs a materially different mechanism: policy-aware weighting, uncertainty-aware active sampling, accumulated preferences, or a different target construction. Otherwise return to broader target construction or source-geometry questions.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-pairwise-preference-target-gate.md`

REVIEW: Local-target approximation is a ceiling, not the current scalable path.
Conclusion: Exact `policy_reweighted_t1` remains a useful proof of principle and diagnostic ceiling, but the tested scalable approximation families have now failed from enough angles to pause the branch as a mainline. Sparse uniform/top-k and adaptive proposals need near-full coverage, fixed replay memory is prompt-transductive, imputed sparse targets dilute pressure, simple learned proposals do not retain lift under streaming, random-prompt warmup is mixed-negative, and sparse pairwise preference failed even when `u32` covered the true result in `0.8450` of prompts (`0.0425` exact calc / `0.0234` sampled normal versus same-budget policy-reweighted `u32` at `0.3350` / `0.3438`).
Do not repeat: Do not run more sparse count ladders, top-k/neighborhood proposal tweaks, replay-cache tuning, imputed-loss variants, polynomial learned-proposal hidden-size/epoch/warmup sweeps, or sparse pairwise count/gap sweeps as novelty.
Next allowed test: Local targets only with a materially different estimator or target construction, or with predeclared streaming/heldout generalization validation. Otherwise pivot compute to source-geometry objectives or less-prescriptive answer-derived boundary methods that reduce full forced-result enumeration.
Source: `researchReviews/2026-05-30-local-target-approximation-direction-review.md`

MIXED-POSITIVE: Automated forced-margin recovery clears a second fresh seed but with variance.
Conclusion: Repeating the automated one-negative forced-margin recovery recipe on CLI seed `19` / effective model seed `21` replicated the late recovery mechanism and cleared the trusted handoff gate, but below the prior very strong seed. Source calc rose from `0.5625` at step `600` to `0.8325` at step `630`, and final source eval was `0.8600`. The trusted 600-step frozen-policy additive handoff reached `0.8975` final eval / `0.9050` step-600 normal with injection-zero `0.0000`, forced-random `0.0350`, and learned calc `0.8425` at step `600`. This confirms the recipe is a useful staged-transfer benchmark with real seed variance, not a solved final method.
Do not repeat: Do not rerun the same CLI seed-19/effective-seed-21 automated forced-margin recovery source plus 600-step handoff as novelty.
Next allowed test: Forced-margin work should either test broader stability/scale or remove prescriptiveness by replacing hard assignment or true-result forcing; do not tune start step, margin, negative count, or recovery length on this setup as novelty.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-second-fresh-seed-stability.md`

POSITIVE: Automated forced-margin recovery survives a wider model scale stress.
Conclusion: Using an existing wider semantic decoder (`n_embd=32`, `n_head=2`, non-product answer decoder), the automated one-negative forced-margin recovery recipe remained viable and transferred strongly. The wider source reached `0.9125` final eval and improved source calc from `0.7825` at step `600` to `0.8825` at step `630`. The trusted frozen-policy additive handoff from step `630` reached `1.0000` final eval and `1.0000` step-600 normal, with zero-injection `0.0625`, forced-random `0.0325`, and learned calc `0.8850` at step `600`. This supports scale/stability of the staged benchmark, but it is still prescriptive and has a non-product decoder caveat.
Do not repeat: Do not rerun the same `n_embd=32`, `n_head=2`, effective-seed-25 wider forced-margin source plus 600-step handoff as novelty.
Next allowed test: Further scale work should use a matching product semantic decoder, larger operand range, larger architecture family, or remove hard assignment / true-result forcing rather than tuning local forced-margin knobs.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-wider-model-scale-stress.md`

REVIEW: Automated forced-margin recovery is the staged-transfer benchmark, not the next knob branch.
Conclusion: The post-recovery forced-margin branch now has enough evidence to stop local expansion. Manual recovery reached `0.8700` final / `0.9050` step-600 handoff; first automated fresh seed reached `0.9875` / `0.9800`; second fresh seed reached `0.8975` / `0.9050`, exposing variance; and a wider `n_embd=32`, `n_head=2` non-product decoder stress reached `1.0000` final / `1.0000` step-600 handoff with low controls. This makes automated one-negative forced-margin recovery the benchmark to beat for staged transfer, but it still depends on hard improvement assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.
Do not repeat: Do not run more local forced-margin start-step, margin, negative-count, LR, recovery-length, same-scale seed-only, cheap-selector, or same wider non-product stress variants as novelty.
Next allowed test: Forced-margin compute should stress a new thesis-relevant axis such as product-decoder parity, larger operand range, larger architecture, or many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.
Source: `researchReviews/2026-05-30-forced-margin-benchmark-direction-review.md`

POSITIVE: Automated forced-margin recovery survives wider product-decoder parity.
Conclusion: Training a matching `n_embd=32`, `n_head=2`, `answer_decoder_interaction=product` oracle semantic decoder produced a clean scaffold (`1.0000` oracle eval). Using that checkpoint, the automated one-negative forced-margin recovery source improved sharply during the late window (`0.6375 -> 0.9475` source calc from step `600` to `630`) and reached `0.9475` final source eval. The trusted 600-step frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with injection-zero `0.0000`, forced-random `0.0225` in the step-600 snapshot, and learned calc `0.9700` at step `600` (`0.9297` in the 128-sample summary). This removes the prior wider non-product decoder caveat for the staged benchmark, but remains prescriptive because it still uses hard assignment, true-result forced-margin pressure, a pretrained semantic decoder, and frozen-policy transfer.
Do not repeat: Do not rerun the same wider product-decoder oracle scaffold plus effective-seed-26 automated forced-margin source/handoff as novelty.
Next allowed test: Further forced-margin work should stress a genuinely new axis such as larger operand range, larger architecture family, many-calculator cost, or remove hard assignment / true-result forcing with a new target construction or estimator.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-product-decoder-parity.md`

MIXED-NEGATIVE: Automated forced-margin recovery does not clear the op29 range stress.
Conclusion: A matching wider product oracle decoder for `operand_max=29` reached full-grid `1.0000`, so decoder wiring was not the bottleneck. With the same automated one-negative forced-margin recovery recipe, the op29 source improved during late recovery from `0.3533` at step `600` to `0.6889` at step `630`, with final source eval `0.7133`. The trusted 600-step frozen-policy additive handoff reached `0.8533` final eval / `0.8278` step-600 normal, with low controls (`0.0344` injection-zero, `0.0189` forced-random at step `600`) but learned calc only `0.6522` at step `600`. This keeps the calculator path causal but fails the high non-bottleneck gate, showing that the op19 full-grid hard-assignment forced-margin recipe is not yet a scalable range solution.
Do not repeat: Do not rerun the same `operand_max=29`, product-decoder, effective-seed-29 source-plus-handoff, and do not jump to op49 with the identical full-grid hard-assignment recipe as novelty.
Next allowed test: Further range work should change source acquisition, reduce assignment cost with a predeclared exact-grid ceiling comparison, or test materially more source capacity/recovery only if the goal is to diagnose the op29 range failure mode.
Source: `aiAgentWorkHistory/phase7/2026-05-30-forced-margin-op29-range-stress.md`

REVIEW: The staged forced-margin benchmark is strong at op19 but not range-scalable yet.
Conclusion: Product-decoder parity removed the wider-decoder caveat at `operand_max=19`, but the first larger-range stress at op29 did not clear. The op29 oracle product decoder reached `1.0000`, while source acquisition plateaued far lower and the trusted handoff reached only `0.8533` final despite low ablation controls. This makes range scaling an unresolved source-acquisition/assignment-cost problem, not a decoder/readout wiring problem.
Do not repeat: Do not respond to the op29 miss with local forced-margin knob tuning, same-seed reruns, or op49 full-grid repetition as novelty.
Next allowed test: Prioritize changed source objectives, scalable assignment approximations against the exact-grid ceiling, or a diagnostic source-capacity/recovery test that explains the op29 failure before larger-range runs.
Source: `researchReviews/2026-05-30-forced-margin-range-stress-review.md`

MIXED-POSITIVE: op29 low-LR source recovery partly rescues the range stress but adds prescriptive compute.
Conclusion: Continuing the failed op29 product source step-630 checkpoint for `90` low-LR recovery steps (`lr=0.0003`, one-negative forced-margin weight `0.1`, source stabilization retained) raised source calc from `0.6889` to `0.8211` and final source eval to `0.8233`. The trusted 600-step frozen-policy additive handoff from recovered step `90` reached `0.9067` final eval / `0.8978` step-600 normal, with low controls (`0.0122` injection-zero, `0.0111` forced-random at step `600`) and learned calc `0.8233`. This shows the op29 miss was partly source-maturity limited, but the rescue adds another prescriptive full-grid source continuation and still does not make the method scalable.
Do not repeat: Do not rerun the same op29 step-630 to low-LR step-90 recovery and handoff as novelty, and do not extend the same continuation ladder unless explicitly diagnosing a new capacity/recovery hypothesis.
Next allowed test: Further range work should change source acquisition, reduce assignment cost against an exact-grid ceiling, or test a materially different source-capacity/recovery mechanism rather than more of the same low-LR recovery.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-low-lr-source-recovery-diagnostic.md`

POSITIVE: A hidden result head rescues op29 forced-margin range transfer.
Conclusion: Adding `--calculator-result-head-hidden-size 64` to the op29 product forced-margin source changed the source-capacity picture. With the same op29 oracle decoder and automated one-negative forced-margin schedule, source calc reached `0.9978` at step `630` and final source eval was `0.9978`, versus `0.7133` for the shallow op29 source and `0.8233` after shallow low-LR recovery. The trusted 600-step frozen-policy additive handoff from the `rhead64` step-630 checkpoint reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0244` injection-zero, `0.0156` forced-random at step `600`) and learned calc `0.9967`. This shows the op29 range failure was strongly source-capacity sensitive, but the method remains prescriptive and full-grid.
Do not repeat: Do not rerun the same op29 `rhead64`, effective-seed-29 source-plus-handoff as novelty.
Next allowed test: Test whether this capacity fix survives fresh seeds/larger ranges or can be paired with cheaper assignment; otherwise prioritize removing hard assignment / true-result forcing.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-hidden-result-head-capacity-diagnostic.md`

POSITIVE: op29 hidden result-head capacity fix survives a fresh seed.
Conclusion: Repeating the op29 `rhead64` source-plus-handoff on a new CLI seed `31` / effective model seed `33` replicated the capacity fix. The source recovered from `0.7122` source calc at step `600` to `0.9967` at step `630`, with source final eval `897/900 = 0.9967`; controls stayed low at step `630` (`0.0200` injection-zero, `0.0133` forced-random). The trusted frozen-policy additive handoff reached `1.0000` final eval / `1.0000` step-600 normal, with low controls (`0.0344` injection-zero, `0.0111` forced-random) and learned calc `1.0000` at step `600`. This upgrades the hidden-head op29 result from a one-seed capacity rescue to a replicated staged range-capacity finding, while preserving the caveat that the method is still full-grid, prescriptive, and per-calculator-head-costly.
Do not repeat: Do not rerun the completed op29 `rhead64` effective-seed-29 or effective-seed-33 source-plus-handoff pairs as novelty.
Next allowed test: Further work should either stress a new axis such as larger operand ranges or many-calculator parameter/training cost, or reduce/remove full-grid hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op29-rhead64-fresh-seed-replication.md`

MIXED-POSITIVE: op39 rhead64 larger-range stress is causal but costly and below the perfect gate.
Conclusion: A new op39 product oracle decoder cleared full-grid eval (`1600/1600 = 1.0000`), so decoder wiring was not the blocker. The op39 `rhead64` source run was interrupted after about `33` minutes of local CPU time with checkpoints through step `540`; a zero-step eval of step `540` was only `0.543` exact / `0.547` snapshot normal. A bounded 90-step continuation from that checkpoint, with the late-recovery switch at continuation step `60`, lifted source final eval to `1504/1600 = 0.940` and source step `90` normal/calc to `0.9431`, with low controls (`0.0213` injection-zero, `0.0113` forced-random). The trusted frozen-policy handoff from continuation step `90` reached `1516/1600 = 0.9475` final eval / `0.9419` step-600 normal, with low controls (`0.0000` injection-zero, `0.0138` forced-random) and learned calc `0.9375`. This is causal larger-range transfer, but it is not op29-style perfect, and the full-grid source cost/continuation requirement strengthens the scalability warning.
Do not repeat: Do not rerun the same op39 effective-seed-39 full-grid `rhead64` source, step-540 continuation, and 600-step handoff as novelty, and do not jump to op49 full-grid without a declared assignment-cost or capacity-scaling change.
Next allowed test: Use op39 as evidence to prioritize cheaper assignment, many-calculator cost accounting, or a materially different source-capacity/credit-assignment mechanism; further full-grid range tests need an explicit scalability hypothesis.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op39-rhead64-range-stress.md`

DISPROVEN: Uniform sampled hard-assignment does not preserve the op19 exact assignment ceiling.
Conclusion: Added `--result-policy-improvement-assignment-sample-count` to approximate hard improvement assignment by scoring the learned result plus uniform random result classes, then ran an op19 `rhead64` 200-step source gate against an exact full-result ceiling. The exact branch scored all `39` result classes, reached best snapshot normal `0.8625` at step `150`, final eval `294/400 = 0.7350`, step-200 true-result coverage `1.0000`, and assignment target accuracy `0.9900`. Sample16 scored `16/39` classes but reached only best snapshot `0.3650`, final `141/400 = 0.3525`, true coverage `0.6125`, and target accuracy `0.4581`. Sample32 scored `32/39` classes but reached only best snapshot `0.4050`, final `152/400 = 0.3800`, true coverage `0.7400`, and target accuracy `0.6773`. Wall-clock savings were modest at this local op19 gate (about `115s` exact, `88s` sample16, `106s` sample32), so the accuracy loss is not a good trade.
Do not repeat: Do not run more uniform sample-count ladders on the same op19 `rhead64` 200-step forced-margin source gate as novelty, and do not expect duplicate-prone uniform result sampling to solve hard-assignment cost without a proposal or coverage mechanism.
Next allowed test: Assignment-cost reduction needs a smarter candidate mechanism, such as coverage-aware/active/structured proposals, an accumulated candidate state validated beyond prompt transduction, or a different non-enumerative credit signal. Compare any such method to an exact-grid assignment ceiling.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-hard-assignment-cost-gate.md`

MIXED-NEGATIVE: Fixed-cadence exact assignment refresh weakens the op19 rhead64 source signal.
Conclusion: Added `--result-policy-improvement-assignment-refresh-interval` to refresh exact full-result hard-assignment targets every N steps on fixed exhaustive-grid batches and reuse cached targets between refreshes. Against the same op19 `rhead64` exact ceiling, refresh2 should cut assignment scoring calls from `201` refreshes to `101`, and refresh5 to `41`. But source quality fell well below exact: exact reached best snapshot `0.8625` and final `0.7350`; refresh2 reached best snapshot `0.5875` and final `237/400 = 0.5925`; refresh5 reached best snapshot/final `0.4950`. Target accuracy at step `200` remained decent for refresh2 (`0.9603`) but the stale-target cadence slowed source acquisition, and local wall time barely improved in the full diagnostic gate (`115.5s` exact, `106.4s` refresh2, `105.1s` refresh5) because snapshots/checkpoints/other objectives dominate.
Do not repeat: Do not run more fixed refresh-interval ladders on the same op19 `rhead64` 200-step gate as novelty. Fixed stale exact targets are not a good assignment-cost reduction path at this budget.
Next allowed test: Temporal amortization needs an adaptive freshness/trust criterion, predictive target update, or other mechanism that preserves exact-ceiling source acquisition while proving real wall-clock or many-calculator savings. Otherwise prioritize different credit-assignment mechanisms.
Source: `aiAgentWorkHistory/phase7/2026-05-30-exact-assignment-refresh-cadence-gate.md`

MIXED-POSITIVE: Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling.
Conclusion: Added `--result-policy-improvement-assignment-unique-sampling` so sampled hard-assignment candidates include the learned result plus per-prompt random result classes without replacement. This directly tested whether duplicate waste caused the prior sampled-assignment failure. Unique16 improved step-200 true coverage only slightly (`0.6525` vs duplicate sample16 `0.6125`) and reached final `162/400 = 0.4050`, still weak. Unique32 was meaningfully better than duplicate sample32: true coverage `0.9275` vs `0.7400`, target accuracy `0.8156` vs `0.6773`, best snapshot `0.6250` vs `0.4050`, and final `244/400 = 0.6100` vs `0.3800`. But it still missed the exact assignment ceiling (`0.8625` best snapshot, `0.7350` final, target accuracy `0.9900`), while scoring most of the `39` result classes. Duplicate removal matters, but sparse unique coverage is not enough.
Do not repeat: Do not run more unique-uniform sample-count ladders on the op19 `rhead64` 200-step gate as novelty. Unique32 is the useful diagnostic point; lower counts are too coverage-limited, and higher counts approach exact enumeration.
Next allowed test: Candidate-cost reduction needs a smarter non-uniform proposal, active/uncertainty allocation, or target construction that closes the remaining exact-ceiling gap at materially lower scoring cost. Validate against exact assignment, not just duplicate sampled baselines.
Source: `aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md`

MIXED-POSITIVE: Policy-topk plus unique sampled assignment recovers much of the exact op19 source signal.
Conclusion: Added `--result-policy-improvement-assignment-policy-topk-count` so sampled hard-assignment candidates reserve slots for the model's current result-policy top-k classes, then fill the rest with unique random candidates. On the op19 `rhead64` 200-step source gate, topk8+unique16 scored only `16/39` result classes but reached step-200 true coverage `1.0000`, target accuracy `0.9333`, best snapshot `0.6850`, and final `269/400 = 0.6725`, far above unique16 (`0.4050` final). Topk8+unique24 reached true coverage/target accuracy `1.0000`, best snapshot `0.7725`, and final `300/400 = 0.7500`, slightly above the exact branch final `0.7350` while scoring `24/39` classes. Topk8+unique32 reached final `344/400 = 0.8600`, above exact final and near exact best snapshot (`0.8625`), while scoring `32/39`. This is the first assignment-cost proposal that preserves most of the source signal at materially lower scorer count, but it remains a source gate only and needs validation beyond op19/seed43 before being treated as scalable.
Do not repeat: Do not run more topk8 unique count ladders on the same op19 `rhead64` 200-step gate as novelty; the useful threshold is already mapped at 16/24/32.
Next allowed test: Validate policy-aware proposals where it matters: longer source plus trusted handoff, fresh seed, larger range, or many-calculator cost accounting. Compare against exact assignment and keep coverage/target-quality diagnostics.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-assignment-proposal-gate.md`

POSITIVE: Policy-topk unique24 survives longer op19 source training and trusted additive handoff.
Conclusion: Extended the promising `topk8+unique24` sampled hard-assignment proposal from the 200-step source screen to the staged op19 `rhead64` source recipe with late recovery (`630` steps, `24/39` result classes scored per assignment). The source reached `400/400 = 1.0000` final eval and step-630 normal/calc `1.0000`, with low controls (`0.0275` injection-zero, `0.0300` forced-random). The trusted frozen-policy additive handoff from the step-630 checkpoint reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with injection-zero `0.0200` and forced-random `0.0325`. This upgrades policy-topk assignment from a short source-screen positive to a real op19 staged-transfer positive at lower assignment scoring cost, but it remains one seed/range and still uses hard assignment plus frozen transfer.
Do not repeat: Do not rerun the same effective-seed-43 op19 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty.
Next allowed test: Validate the policy-aware proposal on a fresh seed, larger operand range, many-calculator cost accounting, or with reduced prescriptiveness. Keep exact-assignment comparators and coverage/target-quality diagnostics.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-source-handoff-validation.md`

POSITIVE: Policy-topk unique24 op19 source/handoff survives a fresh seed.
Conclusion: Repeated the `topk8+unique24` op19 `rhead64` staged recipe on CLI seed `45` / effective seed `47`, again scoring only `24/39` result classes for hard improvement assignment. The source reached `400/400 = 1.0000` final eval and step-630 normal/source-calc `1.0000`, with injection-zero `0.0325`, oracle `1.0000`, and forced-random `0.0250`. The trusted frozen-policy additive handoff from step `630` reached `400/400 = 1.0000` final eval and step-600 normal/learned-calc `1.0000`, with injection-zero `0.0475`, oracle `1.0000`, and forced-random `0.0250` at the snapshot (`0.03125` final metrics). This makes the lower-cost policy-aware assignment proposal a replicated op19 staged-transfer positive, not a single-seed artifact, while still relying on hard assignment, forced-margin source shaping, pretrained product decoder, and frozen transfer.
Do not repeat: Do not run more op19 `rhead64` topk8+unique24 source630 plus trusted handoff fresh-seed replications as novelty; two seeds have cleared this validation axis.
Next allowed test: Move the policy-aware proposal to a thesis-relevant new axis: operand-range stress, many-calculator cost/accounting, reduced hard-assignment prescriptiveness, or a non-enumerative proposal that beats this replicated baseline.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-fresh-seed-validation.md`

POSITIVE: Policy-topk unique24 preserves the op29 rhead64 range source and handoff ceiling on one seed.
Conclusion: Tested the replicated `topk8+unique24` sparse hard-assignment proposal on the op29 `rhead64` staged recipe, using the exact full-grid effective-seed-29 ceiling as the comparator. The sparse source scored only `24/59` result classes per assignment instead of exact `59/59`, yet reached `900/900 = 1.0000` final eval and step-630 normal/source-calc `1.0000`, with injection-zero `0.0233`, oracle `1.0000`, and forced-random `0.0144`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal/learned-calc `1.0000`, with injection-zero `0.0356`, oracle `1.0000`, and forced-random `0.0189`. This is the first operand-range validation that policy-aware sparse assignment can preserve an exact-grid source/handoff ceiling at much lower result-class scoring cost, but it remains one op29 seed and still uses hard assignment, forced-margin source shaping, a pretrained product decoder, hidden result-head capacity, and frozen transfer.
Do not repeat: Do not rerun the same effective-seed-29 op29 `rhead64` topk8+unique24 source630 plus handoff600 path as novelty; it has already been compared to the exact ceiling on that seed.
Next allowed test: Validate this range result on a fresh op29 seed, stress op39/many-calculator cost with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-range-validation.md`

POSITIVE: Policy-topk unique24 op29 range source/handoff replicates on a fresh seed.
Conclusion: Repeated the `topk8+unique24` op29 `rhead64` staged recipe on CLI seed `31` / effective seed `33`, matching the exact full-grid fresh-range comparator seed while still scoring only `24/59` result classes per assignment. The source reached `899/900 = 0.9989` final eval and step-630 normal/source-calc `0.9989`, with injection-zero `0.0200`, oracle `1.0000`, and forced-random `0.0133`. The trusted frozen-policy additive handoff from step `630` reached `900/900 = 1.0000` final eval and step-600 normal `1.0000` / learned-calc `0.9989`, with injection-zero `0.0333`, oracle `1.0000`, and forced-random `0.0111`; final 128-sample metrics reported learned calc `1.0000`. This upgrades policy-aware sparse assignment from one-seed op29 validation to replicated op29 range evidence, while preserving the caveat that the method still uses hard assignment, forced-margin source shaping, pretrained product decoder, hidden result-head capacity, and frozen transfer.
Do not repeat: Do not run more op29 `rhead64` topk8+unique24 source630 plus trusted handoff seed replications as novelty; effective seeds `29` and `33` have now cleared this range axis.
Next allowed test: Move to many-calculator cost/accounting, op39 range with an explicit compute hypothesis, or reduce/remove hard assignment and true-result forced-margin pressure.
Source: `aiAgentWorkHistory/phase7/2026-05-30-policy-topk-op29-fresh-seed-validation.md`

REVIEW: Policy-topk sparse assignment reduces candidate-scoring slope but does not solve many-calculator scaling.
Conclusion: Added `scripts/analyze_assignment_scaling.py` to make the scorer and result-head accounting reproducible. For op29 over 630 assignment steps and the full `900`-prompt grid, exact hard assignment costs `33,453,000` forced result evaluations per calculator, while topk8+unique24 costs `13,608,000`; at 16 independent calculators this is `535,248,000` versus `217,728,000`. At op39, the same accounting is `79,632,000` exact versus `24,192,000` sampled per calculator, and `1,274,112,000` versus `387,072,000` at 16 calculators. Result-head parameters are linear too if calculators have independent `rhead64` heads (`12,091` each at op29, `13,391` each at op39). This was an accounting/review result: topk changes the per-calculator result-class slope, but true many-calculator scalability still needs routing/multi-hook validation or a non-enumerative credit signal.
Do not repeat: Do not treat further op19/op29 policy-topk seed replications as many-calculator evidence, and do not claim topk solves scaling without an actual multi-calculator/routing implementation or active-calculator accounting.
Next allowed test: Implement a true multi-calculator/routed bottleneck diagnostic, stress op39 with a declared compute hypothesis, or replace hard assignment with a less-prescriptive/non-enumerative target construction.
Source: `researchReviews/2026-05-30-many-calculator-scaling-accounting.md`

PARTIAL: Same-layer multi-hook forward support is a prerequisite for many-calculator diagnostics, not proof of scalable training.
Conclusion: Added `GPTConfig.calculator_hook_count`, independent extra calculator hooks, combined same-layer injections, diagnostics for `calculator_active_hook_count` and per-hook injections, `--calculator-hook-count` in `scripts/overfit_one_batch.py`, and optimizer/freezing support so extra hook policy heads are not silently treated as upstream. Tests verify injection summation and multi-hook freezing/grouping. A zero-step smoke with `--calculator-hook-count 3` wrote `calculator_hook_count=3` in config and metrics and grouped hook input projections separately from upstream.
Do not repeat: Do not claim many-calculator success from this alone. It does not provide routed/scattered hook placement, per-hook task specialization, or evidence that independent calculator policies train under assignment pressure.
Next allowed test: Build a routed or task-partitioned multi-hook diagnostic that measures active hooks, scorer calls, per-hook policy quality, and leakage/interference; alternatively move to a non-enumerative credit signal that makes hook count less central.
Source: `aiAgentWorkHistory/phase7/2026-05-30-same-layer-multi-hook-forward-support.md`

PARTIAL: Left-operand routed multi-hook support enables task-partition diagnostics but does not prove multi-hook training.
Conclusion: Added `calculator_hook_routing='left_operand_mod'`, which routes each fixed-width prompt to one active same-layer hook by final left-operand digit modulo `calculator_hook_count`. Diagnostics now report `calculator_hook_route` and `calculator_hook_route_counts`, and per-hook applied injections are zeroed for non-routed examples. `scripts/overfit_one_batch.py` exposes `--calculator-hook-routing left_operand_mod`; a zero-step smoke with `--calculator-hook-count 3 --calculator-hook-routing left_operand_mod` wrote matching routing/count fields in `config.json` and `metrics.json`.
Do not repeat: Do not treat routing support or zero-step smoke as evidence that independent hooks can learn specialized calculator policies. It only makes a routed diagnostic possible.
Next allowed test: Run a small task-partitioned training diagnostic that reports per-hook route counts, per-hook calculator-result accuracy, scorer calls under topk/exact assignment, and whether routed hooks interfere or specialize.
Source: `aiAgentWorkHistory/phase7/2026-05-30-left-operand-routed-multi-hook-support.md`

PARTIAL: Routed snapshot metrics expose per-hook quality but do not prove routed training.
Conclusion: Updated diagnostic snapshots so routed runs select each example's active hook trace instead of always reading the primary hook. Snapshot rows now include `calculator_hook_route_distribution`, `calculator_hook_active_count`, and per-hook fields such as `hook_0_route_count`, `hook_0_normal_exact_match`, `hook_0_operand_exact_match`, `hook_0_calculator_result_accuracy`, and `hook_0_mean_sampled_logp`. A zero-step smoke with `--calculator-hook-count 2 --calculator-hook-routing left_operand_mod --snapshot-every 1` wrote balanced route counts (`{"0": 4, "1": 4}`) and per-hook accuracy columns to `diagnostic_snapshots.csv`; regression tests passed (`141 passed`).
Do not repeat: Do not treat routed snapshot fields as training evidence. They are instrumentation for measuring specialization/interference in the next routed training diagnostic.
Next allowed test: Run the small task-partitioned training diagnostic promised by the routing support: compare exact/topk scorer calls, route balance, per-hook calculator-result accuracy, and normal/injection-zero controls over training.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-multi-hook-snapshot-metrics.md`

MIXED-POSITIVE: Cloned output projection makes a routed two-hook topk source train both hooks.
Conclusion: Made result-policy reads route-aware, added per-route assignment metrics and forced-eval counts, and tested op19 `rhead64` two-hook `left_operand_mod` routing. Without cloning the primary calculator output projection into extra hooks, routed exact/topk 200-step source runs mostly trained hook 0 only: exact reached final eval `0.4825` with hook calc `0.8767/0.0387`, and topk8+unique24 reached `0.5250` with hook calc `0.9315/0.0110`. A 50-step exact diagnostic showed why: hook 1 received targets, but target accuracy was only `0.0839`, because the extra hook's frozen random output projection made forced-result scoring semantically invalid. Adding `--clone-primary-calculator-output-proj` fixed the semantic interface: exact50 route target accuracy became `0.8831/0.9333`, oracle eval became `1.0000`, and the cloned topk8+unique24 source200 reached final eval `361/400 = 0.9025`, step-200 normal `0.9250`, hook calc `0.9315/0.9171`, target accuracy `1.0000`, and scored only `24/39` result classes (`9,600` forced evals per full-grid step versus `15,600` exact). Correction: the reported `0.4325` injection-zero control was later found invalid because the temporary injection-scale helper zeroed only the primary hook; a corrected matched rerun reached `0.9225` normal with `0.0200` injection-zero and hook calc `0.9406/0.9006`.
Do not repeat: Do not run routed multi-hook source diagnostics with frozen random extra-hook output projections and interpret hook 1 collapse as a routing/assignment failure. The output semantic interface must be cloned/shared or trained before assignment targets are meaningful.
Next allowed test: Run trusted additive handoff from the cloned routed topk checkpoint, validate on a fresh seed, or replace cloned initialization with a genuinely shared/tied output projection that reduces many-calculator parameters.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-cloned-output-source-gate.md`

SUPERSEDED-MEASUREMENT-BUG: Routed cloned-output source/handoff controls were only zeroing the primary hook.
Conclusion: Tightened `--freeze-semantic-decoder` so it also freezes calculator output projections for `ste` handoff runs, then tested cloned-output routed handoff/source controls. A strict frozen-policy handoff from the 200-step cloned routed source reached high additive accuracy (`0.9075` final, `0.9175` step-600 normal) but failed the causal control (`0.4925` injection-zero, `0.0175` forced-random). The matched `embd32` routed source630, using the same product-decoder parity checkpoint and architecture as the single-hook positive, reached `400/400 = 1.0000` final and step-630 normal `0.9975` with both hooks trained (`1.0000/0.9944` hook calc), but still had high injection-zero (`0.4600`, 128-sample final counterfactual `0.53125`). A frozen-upstream routed source200 reduced leakage (`0.1875` injection-zero) but learned much more slowly (`0.4150` normal, hook calc `0.4384/0.3867`). Therefore routed multi-hook sparse assignment can train active hooks, but open-upstream source acquisition creates a direct residual path, and freezing upstream trades leakage for undertraining at this budget.
Correction: the apparent `0.46-0.53` leakage was a multi-hook control bug. `temporary_calculator_injection_scale` only changed `model.calculator_hook`, leaving `extra_calculator_hooks` active. After fixing it to scale all hook modules, the same open-upstream source630 checkpoint re-evaluated at `1.0000` final / `0.9950` snapshot normal with `0.0250` injection-zero, and the strict source200 handoff checkpoint re-evaluated at `0.9075` final / `0.9250` snapshot normal with `0.0000` injection-zero. The source/handoff were calculator-causal; the old leakage interpretation is superseded.
Do not repeat: Do not cite the old routed multi-hook injection-zero values unless explicitly labeling them invalid. Multi-hook counterfactuals must scale every calculator hook, not only the primary hook.
Next allowed test: With corrected controls, validate the stronger `embd32` routed source630 through a trusted additive handoff, replicate routed training on a fresh seed/more hooks, or replace cloned output projections with shared/tied output projections for parameter scalability.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-source-leakage-gate.md`

SUPERSEDED-MEASUREMENT-BUG: Longer frozen-upstream routed source did not prove leakage returns.
Conclusion: Extended the fair `embd32` two-hook routed source with cloned output projections and frozen upstream from `200` to `630` steps. The longer run recovered source learning: final eval reached `379/400 = 0.9475`, the last step-630 snapshot reached `0.9750` normal, and both active hooks trained (`0.9955/0.9494` hook calc on the 400-sample snapshot; `0.9286/0.9444` in the final 128-sample routed summary). But the causal control rose with learning: step-630 injection-zero was `0.4400`, and final 128-sample counterfactual injection-zero was `0.5000`, close to the open-upstream routed source leak (`0.4600` snapshot, `0.53125` final). The earlier source200 frozen-upstream run was low-leak mainly because it was undertrained, not because this recipe had solved routed causal acquisition.
Correction: this conclusion used the same invalid primary-hook-only zeroing helper. It should not be used as evidence that frozen-upstream learning leaks. The broader lesson is instrumentation discipline: routed multi-hook controls need all hooks ablated before interpreting source leakage.
Do not repeat: Do not rerun frozen-upstream source630 as an anti-leak test unless the question is specifically about frozen-upstream learning speed; the leakage claim was invalidated by the counterfactual-control bug.
Next allowed test: Re-evaluate or rerun routed handoff/source claims only with corrected all-hook counterfactuals, then move to fresh-seed/more-hook/shared-output validation.
Source: `aiAgentWorkHistory/phase7/2026-05-30-frozen-upstream-routed-source630.md`

POSITIVE-CORRECTION: Multi-hook injection-zero controls now ablate every routed calculator hook.
Conclusion: Fixed `temporary_calculator_injection_scale` so evaluation, causal-gap, and zero-injection contexts set `injection_scale` on every module returned by `model.calculator_hook_modules()`, not just `model.calculator_hook`. Added a regression test that two routed hooks are both scaled inside the context and restored afterward. Corrected evidence changed the routed interpretation: a matched source200 rerun reached `0.9400` final / `0.9225` snapshot normal with low controls (`0.0200` injection-zero, `0.0325` forced-random) and both hooks trained (`0.9406/0.9006` hook calc). Re-evaluating the previous open-upstream source630 checkpoint gave `1.0000` final / `0.9950` snapshot normal, `0.0250` injection-zero, and hook calc `1.0000/0.9893`. Re-evaluating the strict source200 handoff checkpoint gave `0.9075` final / `0.9250` snapshot normal, `0.0000` injection-zero, `0.0300` forced-random, and hook calc `0.9108/0.9198`. The routed multi-hook source and handoff were calculator-causal under corrected controls.
Do not repeat: Do not interpret old routed multi-hook `injection_zero_exact_match` numbers from before this fix as causal evidence. Any multi-hook ablation or causal-gap objective must verify all hook scales are changed.
Next allowed test: Use corrected controls to validate the stronger `embd32` routed source630 in additive handoff, test fresh seeds or more routed hooks, and replace cloned output projections with shared/tied projections for real many-calculator parameter scaling.
Source: `aiAgentWorkHistory/phase7/2026-05-30-multihook-injection-zero-fix.md`

POSITIVE: Corrected-control routed embd32 source630 clears trusted additive handoff.
Conclusion: Ran the trusted 600-step frozen-policy additive handoff from the corrected-control fair routed `embd32` source630 checkpoint. The source was the two-hook `left_operand_mod` topk8+unique24 run with cloned output projections and product decoder parity. The handoff reached `400/400 = 1.0000` final eval with final loss effectively zero; the step-600 snapshot had normal `1.0000`, injection-zero `0.0550`, forced-random `0.0300`, oracle `1.0000`, and active-hook calculator-result accuracy `1.0000/0.9955`. Final 128-sample counterfactuals also stayed causal (`0.078125` injection-zero, `0.0234375` forced-random). This is the first corrected-control two-hook routed non-bottleneck staged-transfer positive, so routed sparse assignment is no longer source-only. It remains one seed/op19 and still depends on hard assignment, frozen transfer, cloned per-hook output projections, and a pretrained product decoder.
Do not repeat: Do not rerun this same effective-seed-43 op19 routed `embd32` source630-to-handoff600 path as novelty. The corrected-control gate is positive.
Next allowed test: Move to a thesis-relevant scaling axis: fresh routed seed, more hooks/routes with active-hook cost accounting, or a shared/tied output projection that removes cloned per-hook semantic-output parameter growth.
Source: `aiAgentWorkHistory/phase7/2026-05-30-routed-embd32-source630-handoff.md`

POSITIVE: Four routed calculator hooks train and transfer under corrected controls.
Conclusion: Stress-tested the corrected-control routed recipe from two hooks to four hooks using `left_operand_mod` routing, cloned output projections, `embd32`, and topk8+unique24 source assignment. The four-hook source630 reached `398/400 = 0.9950` final eval and step-630 normal/source-calc `0.9975`, with low controls (`0.0275` injection-zero, `0.0225` forced-random) and all hooks trained on the 400-sample snapshot (`0.9928/1.0000/1.0000/1.0000` hook calc). The trusted 600-step frozen-policy additive handoff from that source reached `400/400 = 1.0000` final eval and step-600 normal/calc `1.0000`, with corrected controls still low (`0.0400` injection-zero, `0.0200` forced-random) and all four hooks perfect on the final snapshot. This is the first more-than-two-hook routed non-bottleneck positive, directly advancing the many-calculator axis. Caveat: the current implementation still executes every hook before route masking and uses cloned per-hook output projections, so it proves trainability/transfer under route partitioning, not efficient active-only execution or parameter scaling.
Do not repeat: Do not rerun the same effective-seed-43 op19 four-hook source630/handoff600 path as novelty. The 4-hook route-partition gate is positive.
Next allowed test: Implement active-only routed hook execution and/or shared/tied output projection, then validate the same 4-hook gate with compute/parameter accounting; alternatively run a fresh-seed 4-hook replication only if needed for stability after the efficiency change.
Source: `aiAgentWorkHistory/phase7/2026-05-30-four-hook-routed-source-handoff.md`

POSITIVE-IMPLEMENTATION: Routed calculator execution is now active-only for present routes.
Conclusion: Updated the model forward path so `calculator_hook_routing='left_operand_mod'` invokes only hooks with examples in the current batch, scatters their traces/injections back into full-batch diagnostics, and reports both configured hooks (`calculator_active_hook_count`) and actually invoked hooks (`calculator_invoked_hook_count`). Updated the routed result-logit helper used by source training so it applies each hook's `result_proj` only to routed examples instead of stacking every hook's logits over the full batch. Regression coverage verifies a four-hook batch routed only to hooks `0` and `2` calls only those hooks, leaves non-routed hook injections zero, and reads result logits only from present routes. This removes the known all-hooks-forward waste from routed batches, but it is an implementation/scaling improvement rather than a new credit-assignment method.
Do not repeat: Do not describe the four-hook routed result as still requiring all hooks to execute before route masking after this patch. Also do not claim this solves parameter scaling: cloned/independent output projections still grow with hook count.
Next allowed test: Add shared/tied output projections or explicit compute accounting in a routed training run; then return to reduced prescriptiveness or non-enumerative credit assignment rather than more same-seed routed smoke tests.
Source: `aiAgentWorkHistory/phase7/2026-05-30-active-only-routed-hook-execution.md`

POSITIVE-IMPLEMENTATION: Routed calculator hooks can share one output projection.
Conclusion: Added `calculator_share_output_proj` / `--share-calculator-output-proj` so extra calculator hooks tie their result-to-residual `output_proj` module to the primary hook instead of cloning independent parameters. A three-hook shared model removes two extra output-projection parameter matrices while preserving a single shared semantic interface; tests verify object identity, parameter-count reduction, CLI config/metrics recording, and compatibility when loading older untied checkpoints by canonicalizing extra-hook output keys to the primary output projection. A zero-step three-hook routed CLI smoke wrote `share_calculator_output_proj=True` in both config and metrics. This fixes the routed many-calculator parameter-slope issue at the semantic output interface, but it has not yet re-run the source/handoff training gate with tied outputs.
Do not repeat: Do not keep using cloned per-hook output projections as the only fair routed semantic-interface option. Use tied output projections when testing many-calculator parameter scaling.
Next allowed test: Run a small tied-output routed source gate, preferably matching the known 4-hook active-only setup with topk8+unique24, and compare per-hook calculator accuracy/controls against the cloned-output result before moving back to less-prescriptive credit assignment.
Source: `aiAgentWorkHistory/phase7/2026-05-30-shared-routed-output-projection.md`

MIXED: Shared-output four-hook routed source trains, but trusted handoff misses the cloned-output gate.
Conclusion: Replaced the cloned per-hook output projections in the known four-hook op19 `embd32` topk8+unique24 recipe with `--share-calculator-output-proj`. The first source trained cleanly: final eval `400/400 = 1.0000`, step-630 normal/calc `1.0000`, injection-zero `0.0275`, forced-random `0.0300`, and all four hooks reached calculator-result accuracy `1.0000`. Its trusted 600-step frozen-policy additive handoff reached only `0.7625` final eval / `0.7800` step-600 normal, with step-600 calculator-result accuracy `0.9950`, injection-zero `0.0875`, and forced-random `0.0725`; a 600-step continuation improved only to `0.7925` final / `0.8050` snapshot normal. A later audit found this first A/B was confounded because the cloned positive used `--additive-forced-margin-start-step 50` while the shared-output miss used the default `0`. A matched shared-output rerun with delayed margin still trained the source (`399/400 = 0.9975`, diagnostic calculator-result accuracy `0.9922`, low controls) but its matched-head trusted handoff still missed (`299/400 = 0.7475`, step-600 normal `0.7225`, injection-zero `0.0875`, forced-random `0.0725`, calculator-result accuracy `0.9900`). A regression/audit verifies that a tied-output checkpoint loaded into tied and independent-hook models gives identical logits, injections, routes, and hook result predictions, so this is not explained by state-dict tying/loading behavior. Shared output projection preserves routed source training and removes parameter growth, but it is not a drop-in replacement for cloned output projections in the trusted non-bottleneck handoff geometry.
Do not repeat: Do not claim tied output projections have preserved the four-hook non-bottleneck result until a new source/handoff geometry mechanism clears the trusted handoff gate. Do not rerun the same tied-output source630 plus handoff600/continuation600 as novelty.
Next allowed test: Diagnose or redesign the transfer geometry for shared-output sources, or move back to less-prescriptive credit assignment. If continuing this branch, require a new mechanism such as handoff-aware source shaping, route-aware downstream readout, or a predeclared tied-output handoff geometry objective.
Source: `aiAgentWorkHistory/phase7/2026-05-30-shared-output-routed-source-handoff.md`

REVIEW: Post-shared-output steering favors less-prescriptive credit over more same-recipe scaling audits.
Conclusion: After active-only routing, shared output projection, four-hook routed positives, and the matched hard-assignment shared-output handoff miss, the many-calculator scaling branch reached a clear boundary: routed calculators can train and transfer with cloned outputs, tied outputs remove parameter growth, but the source objective must create a handoff-friendly shared semantic interface. A later online-hard-memory plus additive-semantic-distillation source is the first new mechanism to clear the four-hook shared-output trusted handoff, so the old shared-output miss was a transfer-geometry/objective problem rather than architectural inevitability. The central unsolved requirement remains scalable, less-prescriptive credit assignment into the calculator-query policy across seeds, ranges, and prompt regimes.
Do not repeat: Do not spend mainline turns on same shared-output source630/handoff600 variants, same forced-margin recovery knobs, op19/op29 topk replications, selector proxies, or the same semantic-distilled four-hook shared-output seed. Do not treat many-calculator implementation work as solving the training method while the source remains fixed-grid/frozen-transfer based.
Next allowed test: Fresh routed/shared seed replication, streaming/fresh-prompt online memory, larger-range routed/shared stress, or a less-prescriptive answer-derived target/estimator that reduces or replaces forced-result enumeration with a predeclared Stage 0/Stage 1 gate.
Source: `researchReviews/2026-05-30-post-shared-output-steering-review.md`

MIXED: Result-boundary critic proposals help only when followed by broad candidate rescoring.
Conclusion: Extended the result-boundary amortized critic diagnostic with mean/LCB candidate proposals and optional critic ensembles. A pairwise critic trained on only `8` forced scores per train prompt still had weak direct heldout argmin recovery at step `800` (`0.20` single critic, `0.24` four-member ensemble). If the critic only proposes candidates and those candidates are then actually scored, recovery improves: single-critic top-8 proposal recovers the full-enum best on `0.79` of heldout prompts and top-16 reaches `0.96`; four-member top-8 reaches `0.84` and top-16 reaches `1.00` by mean proposal. But top-16 already scores `16/39 = 41%` of result classes, and the four-member ensemble uses `32` sparse scores per train prompt. LCB uncertainty did not beat the mean proposal (`0.79` vs `0.84` at top-8 step-800 for the ensemble; `0.98` vs `1.00` at top-16). This is a useful candidate-rescoring diagnostic, not a solved scalable/non-prescriptive training method.
Do not repeat: Do not claim hidden/output critic argmin is solved, and do not run more beta/ensemble/count tweaks as novelty. The useful finding is that broad proposal-plus-rescoring can approach the full-enum target only at substantial candidate cost; uncertainty LCB did not provide the hoped adaptive-compute advantage.
Next allowed test: Change the mechanism before spending training budget: adaptive stopping/calibration that expands only uncertain prompts, soft/set targets that tolerate missing the exact argmin, or a streaming/evolving-checkpoint proposal gate that beats this static fixed-grid diagnostic at materially lower scoring cost.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-uncertainty-proposal-diagnostic.md`

MIXED-NEGATIVE: Adaptive result-boundary proposal expansion has useful margin signal but not enough leverage.
Conclusion: Added adaptive top-8-to-top-16 expansion metrics to the result-boundary critic diagnostic. On the trained step-800 result-boundary checkpoint, cutoff-margin expansion beat random expansion at matched average candidate count: single critic reached `0.85` vs random `0.82` at mean `10/39` candidates and `0.92` vs `0.88` at mean `12/39`; the four-critic ensemble reached `0.91` vs `0.88` at mean `10/39` and `0.97` vs `0.91` at mean `12/39`. However, fixed top-16 still reached `0.96` single-critic and `1.00` ensemble, the ensemble uses `32` train scores per prompt, and explicit std/LCB uncertainty scores were weaker than the simple margin heuristic. Adaptive expansion is better than random, but the current critic does not give enough adaptive-compute advantage to be the scalable result-boundary bridge.
Do not repeat: Do not spend more mainline turns on threshold, beta, or expand-fraction sweeps over this same static fixed-grid diagnostic. The useful result is the margin-vs-random comparison and the remaining gap to fixed top-16.
Next allowed test: Change the mechanism: a calibrated proposal model validated across evolving checkpoints, a soft/set target that tolerates missing the exact argmin, or a source-training gate that uses a materially different target construction rather than static top-k expansion.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-adaptive-proposal-diagnostic.md`

DISPROVEN: Static soft result-boundary targets improve source acquisition over hard-best targets.
Conclusion: Tested existing `soft_result` result-boundary target construction on the matched full-grid upstream-open 200-step source gate. Temperature probe showed `t=1` is meaningfully soft (`0.8003` true-result target mass, `2.72` effective results), while `t=4` is broad (`0.1336` true mass, `28.35` effective results). Training was worse than the matched hard-best comparator: hard-best step-200 learned calc `0.5450` / final eval `0.5475`; soft `t=1` learned calc `0.2900` / final eval `0.2775`; soft `t=4` learned calc `0.1350` / final eval `0.1275`. Simple temperature-softened full-enum targets tolerate uncertainty by diluting the signal, not by improving scalable source learning.
Do not repeat: Do not run static `soft_result` temperature ladders on the same full-grid result-boundary source gate as novelty.
Next allowed test: If using result-boundary targets, change the mechanism materially: uncertainty/regret-based set targets, evolving-checkpoint proposal validation, or a proposal model that reduces enumeration without merely softening the full-enum teacher.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-soft-target-training-gate.md`

REVIEW: Static result-boundary approximation is paused after critic, proposal, adaptive, and soft-target gates.
Conclusion: The answer-derived result-boundary source still transfers causally, but static approximations have hit a local boundary. Direct hidden/output critics miss heldout argmins, proposal rescoring needs broad candidate sets, adaptive expansion has only modest margin signal, and static soft-result targets train worse than hard-best. This cluster should not keep consuming mainline compute through small static variants.
Do not repeat: Do not continue pointwise/rank critic variants, top-k/beta/ensemble/threshold/fraction sweeps, or static soft-target temperature ladders as novelty.
Next allowed test: Evolving-state/generalization validation, genuinely different uncertainty/regret set targets, calibrated proposal learning, or a different less-prescriptive credit-assignment family.
Source: `researchReviews/2026-05-30-result-boundary-static-approximation-steering-review.md`

DISPROVEN: Static full-enum regret-set result-boundary targets improve source acquisition.
Conclusion: Added `regret_set` target mode, a uniform target over forced result classes within a fixed NLL margin of the best forced result. Margins `0.05`, `0.25`, and `1.0` collapsed to hard-best (`1.0` effective results); margin `2.0` was still nearly hard (`1.06` effective results); margin `4.0` was genuinely set-valued (`5.6975` effective results, true result always in set, true-result target mass `0.2413`). But the margin-4 200-step source gate learned much worse than the matched hard-best comparator: regret-set step-200 learned calc / final eval `0.0900` / `0.0900` versus hard-best `0.4625` / `0.4225`. Simple static set-valued targets dilute the useful answer-derived signal instead of improving less-prescriptive source learning.
Do not repeat: Do not run fixed full-enum `regret_set` margin ladders or simple top-N-low-regret static target variants on this same gate as novelty.
Next allowed test: If staying with result-boundary, change the mechanism to adaptive/evolving validation or calibrated proposal learning; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-regret-set-training-gate.md`

REVIEW: Static result-boundary set targets are paused after soft and regret-set negatives.
Conclusion: The target-construction escape hatch from the previous review has now been tested in its simplest static form. Soft-result targets and fixed-margin regret-set targets both weaken source acquisition relative to hard-best. Result-boundary remains a useful answer-derived bridge, but static broad target construction is now a local rut.
Do not repeat: Do not tune static soft temperatures, fixed regret margins, or top-N low-regret static targets over the same full-loss table as mainline work.
Next allowed test: Evolving-checkpoint/streaming validation, calibrated proposal learning that preserves target quality while reducing scoring, adaptive uncertainty/regret selection, or a different credit-assignment family.
Source: `researchReviews/2026-05-30-result-boundary-set-target-steering-review.md`

DISPROVEN: Frozen result-boundary proposal critics generalize across evolving checkpoints.
Conclusion: Added a cross-checkpoint diagnostic that trains a sparse pairwise result-boundary critic on one checkpoint and evaluates top-8 proposal-plus-rescoring on other checkpoints from the same May 13 source lineage. Same-state top-8 recovery improved with maturity (step100 `0.48`, step400 `0.74`, step800 `0.79`), but forward transfer collapsed: train step100 to eval step400/800 recovered only `0.11/0.12`, and train step400 to eval step800 recovered `0.23`. Backward transfer from step800 was partial (`0.42` to step100, `0.58` to step400) but not strong enough. Static sparse critics are state-local, not a bridge into evolving training.
Do not repeat: Do not use a frozen/static result-boundary critic trained at one checkpoint as a scalable source-training proposal mechanism, and do not run same-state critic count/seed tweaks as evidence of evolving training viability.
Next allowed test: Result-boundary proposals need online refresh, state calibration, or explicit evolving validation; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-cross-checkpoint-critic-gate.md`

PARTIAL-NEGATIVE: Simple online-calibrated result-boundary proposal critics do not restore same-state quality.
Conclusion: Extended the cross-checkpoint critic diagnostic with warm-start online calibration: retarget critic normalization at the eval checkpoint and fine-tune on fresh sparse forced scores before proposing top-8 candidates. Calibration repaired part of the forward-transfer collapse but did not match same-state quality. Step400-to-step800 top-8 recovery improved from frozen `0.23` to adapted `0.59` with `2` fresh scores per train prompt, `0.54` with `4`, and `0.62` with `8`; the same-state step800 critic was `0.79`. Step100-to-step400/800 with `2` fresh scores improved only to `0.36`/`0.41`. Simple warm-start calibration is helpful, but not enough to be a scalable source-training proposal mechanism.
Do not repeat: Do not wire this warm-start calibrated critic into source training or spend mainline compute on small adapt-lr/epoch/sample-count tweaks as if the proposal mechanism were solved.
Next allowed test: Only continue result-boundary proposals with a stronger online learner, active proposal/training co-design, or a materially different state-calibrated objective; otherwise move to a different less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-result-boundary-online-calibrated-critic-gate.md`

DISPROVEN: Rank-normalized expected answer loss rescues result-space expected-cost discovery.
Conclusion: Added `expected_answer_loss_cost_normalization=rank`, which replaces each prompt's forced-result NLLs with within-prompt ranks before the exact policy expectation. The full-grid Stage 0 diagnostic did not clear the gate: exact-vs-boundary result-proj cosine was only `0.049551` and upstream cosine was `0.002584`, weaker than an earlier contrastive-margin decoder sign flip that still failed Stage 1. Sampled PG remained aligned with the rank objective (`0.723444`/`0.719965`) but only barely aligned with the boundary target (`0.027483`/`0.016271`).
Do not repeat: Do not run rank-normalized expected answer-loss Stage 1, or rank/scale transforms of the same full-enum expected-cost objective, as novelty.
Next allowed test: Expected-loss work needs a stronger structural estimator/objective, not another per-prompt monotonic cost normalization.
Source: `aiAgentWorkHistory/phase7/2026-05-30-rank-normalized-expected-loss-gate.md`

MIXED-NEGATIVE: Policy-topk+unique24 sampled result-boundary source targets do not preserve the full-enum boundary source signal.
Conclusion: Added candidate-scored result-boundary target training, where each prompt scores the current policy top-8 results plus unique sampled candidates for `24/39` total result classes. In the 200-step upstream-open source gate, true-candidate coverage rose from `0.6025` to `0.9600`, but learned-best/source calculator accuracy reached only `0.3425` in the training curve, snapshot calculator accuracy `0.3675`, and final eval `141/400 = 0.3525`. This is materially below the matched full-enum hard-best result-boundary comparators (`0.5450`/`0.5475` in the soft-target gate, `0.4625`/`0.4225` in the regret-set gate).
Do not repeat: Do not ladder `result-boundary-target-sample-count`, top-k count, or unique-sampling variants around this same policy-topk sampled target as novelty. The failure is not mainly candidate coverage; the sparse/candidate hard-best target gives a weaker source signal.
Next allowed test: Result-boundary source work needs active proposal/training co-design, a stronger online/state-calibrated proposal, or a different target construction; otherwise move to another less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-sampled-result-boundary-source-gate.md`

PARTIAL-POSITIVE: Zero-injection improvement result-boundary targets train a less-prescriptive full-enum source but do not yet solve sparse scoring.
Conclusion: Added `result_boundary_target_mode=zero_improvement`, which weights forced result classes by answer-loss improvement over the zero-injection baseline instead of selecting the true sum or argmin directly. In the 200-step upstream-open source gate, full enumeration reached step-200 snapshot calc `0.5700`, learned-best/source calc `0.5475`, and final eval `217/400 = 0.5425`, matching nearby full-enum hard-best comparators while assigning true-result target mass `0.9541` and effective results `1.2692`. The paired topk8+unique24 sparse gate improved over sampled hard-best (`0.4300` final vs `0.3525`) but still missed full-enum zero-improvement (`0.5425`) despite `0.9725` true-candidate coverage.
Do not repeat: Do not treat zero-improvement as solved scalability, and do not run blind sample-count ladders. The valuable finding is that a no-calculator utility baseline is a viable less-prescriptive full-enum target and a better sparse target than sampled hard-best, but sparse scoring still needs a stronger proposal/training mechanism.
Next allowed test: Continue only with a high-leverage scaling step: longer source/handoff validation for full-enum zero-improvement, or an active proposal/streaming mechanism that closes the sampled gap at materially lower scoring cost.
Source: `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-source-gate.md`

MIXED-POSITIVE: Mature zero-improvement sources transfer causally but miss the trusted handoff gate.
Conclusion: Continued the full-enum zero-improvement source from the 800-step checkpoint to a 1600-equivalent source. The source improved from final `0.9150` / step-800 calc `0.8975` to final `0.9850` / continuation step-800 calc `0.9725`, with low injection-zero controls. The trusted 600-step frozen-policy additive handoff from the weaker source reached only `0.3650` final / `0.3900` step-600 normal. The handoff from the mature source improved to `0.6775` final / `0.7150` step-600 normal, with causal controls low (`0.0100` injection-zero, `0.0525` forced-random) and frozen calculator accuracy `0.9725`, but it still missed the old hard-best result-boundary handoff (`0.8825` final / `0.8425` step-600 normal).
Do not repeat: Do not rerun the same source800/source1600 plus 600-step handoff path as novelty, and do not claim zero-improvement has solved non-bottleneck transfer. Source maturity helps, but handoff/readout geometry remains weaker than the old hard-best boundary source.
Next allowed test: If continuing zero-improvement, add a new mechanism: handoff-aware source shaping, additive/readout-geometry auxiliary, or a scalable proposal that preserves source quality. Otherwise compare against another less-prescriptive credit-assignment family.
Source: `aiAgentWorkHistory/phase7/2026-05-30-zero-improvement-boundary-handoff.md`

DISPROVEN: Naive additive-path zero-improvement targets provide handoff-aware source shaping.
Conclusion: Added `result_boundary_target_mode=additive_zero_improvement`, which builds the zero-improvement target from forced-result answer-loss gains through the non-bottleneck additive path. In the 200-step full-enum source gate it learned the additive-path target (`learned_best=0.6025`) but the target itself was non-arithmetic (`hard_best_equals_true_sum=0.0325`, true-result target probability `0.0225`) and calculator-result accuracy stayed near chance (`0.0200` final / snapshot). The untrained additive/readout path creates arbitrary answer-derived result preferences, so this is not a viable handoff-aware shaping signal by itself.
Do not repeat: Do not run longer source/handoff jobs with plain `additive_zero_improvement` from an untrained additive readout as novelty; it first needs a mechanism that makes the additive forced-result loss table meaningful without true-result forcing.
Next allowed test: If using additive-path targets, add a real readout-preconditioning or co-training mechanism and predeclare how it avoids simply reintroducing prescriptive true-result supervision.
Source: `aiAgentWorkHistory/phase7/2026-05-30-additive-zero-improvement-source-gate.md`

MIXED-NEGATIVE: Semantic readout distillation repairs additive target quality but not source-policy uptake.
Conclusion: Added `--additive-semantic-distill-*`, which forces arbitrary result classes and trains the additive non-bottleneck path to match the frozen answer-decoder bottleneck logits. Co-training with additive zero-improvement improved the target slightly (`hard_best_equals_true_sum 0.0325 -> 0.1775`) but final calc stayed weak (`0.0825` snapshot / `0.0450` eval). A 300-step distill-only preconditioner raised teacher/student token agreement to `0.7694`; starting source training from that checkpoint repaired additive target quality (`best=true 0.5225` at step 0 and `0.8200` at step 200 with ongoing distill), but learned-best/source calc stayed low (`0.1400`/`0.0675`). Turning distill off let the policy learn the now-drifting non-arithmetic target (`learned_best=0.6950`, best=true fell to `0.1575`, calc `0.0900`). Distillation teaches readout semantics, but source-policy uptake still needs a stronger mechanism.
Do not repeat: Do not run more plain semantic-distill weight/sample-count/length tweaks as novelty; the next variant must address policy uptake or target drift explicitly.
Next allowed test: Couple readout distillation to a policy-learning mechanism such as staged frozen-readout target construction, policy-target anchoring to the repaired table, or an estimator that preserves target quality while increasing learned-best/true-result uptake.
Source: `aiAgentWorkHistory/phase7/2026-05-30-semantic-distilled-additive-zero-improvement.md`

MIXED-NEGATIVE: Frozen-teacher additive target anchoring preserves target quality but leaves policy uptake weak.
Conclusion: Added `--result-boundary-target-teacher-checkpoint`, which constructs result-boundary targets with a separate frozen teacher while training the live model's result policy. Supporting probes showed the failure modes: freezing the whole encoder/readout preserved the repaired additive table (`best_true=0.5225`) but head-only policy uptake stalled (`learned_best=0.188`, final `0.0225`), while freezing only the post-calculator decoder let the pre-hook residual drift the target back down (`best_true=0.1575`, learned_best `0.6575`, final `0.0900`). The full frozen-teacher anchor preserved `best_true=0.5225` through 800 steps and improved learned-best to `0.4125`, but calculator-result accuracy/final eval reached only `0.1700`/`0.1750`, far below a useful source.
Do not repeat: Do not run more same-checkpoint frozen-teacher additive-anchor length/LR/freezing sweeps as novelty. Target anchoring helps diagnose drift, but it does not solve policy uptake.
Next allowed test: Move to a different policy-uptake mechanism, such as a target that is easier for the policy class to represent, direct optimization of source logits against teacher target tables without additive forced-loss rescoring every step, or a new estimator that can raise true-result uptake while preserving the target table.
Source: `aiAgentWorkHistory/phase7/2026-05-30-frozen-teacher-additive-target-anchor.md`

MIXED: Cached teacher tables separate policy imitation from teacher-target quality.
Conclusion: Added `--result-boundary-target-cache` with `target_weights` and `hard_best` modes. Caching the frozen additive teacher's soft zero-improvement weights reproduced the online-anchor ceiling (`0.4000` learned-best / `0.1650` final at 800), showing repeated forced-result rescoring was not the source of weak uptake. Hard cached teacher-best made the policy imitate the teacher table much better (`0.668` learned-best / `0.338` final at 800; `0.710` learned-best / `0.3725` final at 1600), but the teacher best itself is true only `0.5225` of prompts, so this improves uptake while exposing target-quality ceiling rather than solving calculator learning.
Do not repeat: Do not run more same-teacher cached soft/hard length/LR sweeps as novelty. Cached hard-best is a diagnostic/ceiling tool, not a scalable or sufficiently correct training method by itself.
Next allowed test: Improve the teacher target quality or change the answer-derived target source before optimizing imitation further; alternatively use cached tables only as a cheap diagnostic for new target constructions.
Source: `aiAgentWorkHistory/phase7/2026-05-30-cached-teacher-target-table.md`

PARTIAL-POSITIVE: Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates.
Conclusion: Reused the semantic-distilled preconditioned+ongoing-distill additive checkpoint as the frozen cache teacher. Its additive hard-best table is much better (`best_true=0.8200`) than the preconditioner-only teacher (`0.5225`). Cached soft target weights from this teacher still learned poorly (`0.393` learned-best / `0.298` calc / `0.273` final at 800), but cached hard-best imitation reached `0.728` learned-best / `0.595` calc / `0.562` final at 800 and `0.765` learned-best / `0.618` calc / `0.583` final at 1600. Better target quality plus hardening materially helps, but this still trails the teacher ceiling and the mature bottleneck zero-improvement source.
Do not repeat: Do not run more high-quality-teacher cached hard-best length/LR sweeps as novelty; the curve is useful as a ceiling diagnostic, not a recipe.
Next allowed test: Improve the answer-derived target source itself or return to bottleneck zero-improvement/handoff-aware target construction. Cached hard-best can be used as a cheap diagnostic for candidate target tables before expensive source/handoff runs.
Source: `aiAgentWorkHistory/phase7/2026-05-30-high-quality-cached-teacher-table.md`

MIXED-POSITIVE: Online hard result-boundary memory turns sparse zero-improvement discovery into a strong source but misses handoff.
Conclusion: Added `--result-boundary-target-online-hard-memory`, which scores sparse result-boundary candidates online, keeps each prompt's best discovered answer-improving result as a hard target, and can freeze rescoring when every prompt has a target. On the op19 full-grid zero-improvement source gate with topk8+unique24 candidates, the 200-step branch only matched the old soft sparse target (`0.455` calc / `0.435` final versus old `0.4275` / `0.4300`), but the 800-step branch matured to `0.9675` learned calc and `0.9725` final. The freeze-when-full variant reached the same source result while stopping forced-result scoring after `86,400` cumulative forced evals instead of about `7,689,600`; the memory was full and `best_true=1.000` by step 50. However, the trusted frozen additive handoff from that source reached only `0.465` final / `0.485` step-600 normal, with calculator accuracy still `0.9575` and injection-zero `0.0100`. This is a strong sparse fixed-grid source mechanism, but it does not yet solve non-bottleneck transfer.
Do not repeat: Do not run same-seed op19 online-hard-memory source length/LR repeats as novelty. The useful result is hard online target discovery plus stop-when-full rescoring; the failure mode is handoff/readout geometry.
Next allowed test: Add a handoff-aware geometry mechanism to online hard memory, run a fresh-seed source plus trusted handoff, or test streaming/fresh-prompt memory. A source-only repeat is not enough.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-result-boundary.md`

POSITIVE: Additive semantic distillation repairs online-hard-memory handoff geometry on the op19 gate.
Conclusion: Combined sparse zero-improvement online hard memory with `--additive-semantic-distill-weight 1 --additive-semantic-distill-sample-count 8` during source training. The source still filled/froze memory after `86,400` forced evals, reached final `400/400 = 1.000`, diagnostic calculator-result accuracy `1.000`, and final additive semantic token agreement `0.7459`. The trusted frozen-policy additive handoff from this source reached final `400/400 = 1.000` and step-600 normal `1.000`, with causal controls low (`0.0525` injection-zero, `0.0050` forced-zero, `0.0175` forced-random) and frozen calculator-result accuracy `1.000`. This directly fixes the previous online-hard-memory handoff miss on the same gate without telling the policy which result to request for each prompt.
Do not repeat: Do not tune semantic-distill weight/sample/length on the same op19 seed as novelty. The durable finding is that a non-prescriptive readout-semantics auxiliary can convert the strong sparse source into a handoff-compatible source.
Next allowed test: Fresh-seed replication, streaming/fresh-prompt memory, larger-range stress, or routed/many-calculator validation. Also compare whether the semantic-distill auxiliary remains helpful under less fixed-grid memory.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-handoff.md`

MIXED-POSITIVE: Semantic-distilled online-hard-memory source replicates on a fresh seed, but trusted handoff is seed-sensitive.
Conclusion: Repeated the combined sparse zero-improvement online hard memory plus additive semantic distillation source on CLI seed `7` / effective seed `9`. Source acquisition replicated: final `400/400 = 1.000`, step-800 source calc `1.000`, additive semantic token agreement `0.7403`, and memory froze after `76,800` forced evals with low controls. The trusted 600-step frozen-policy additive handoff preserved calculator accuracy (`1.000`) and low controls (`0.0250` injection-zero, `0.0325` forced-zero, `0.0225` forced-random) but reached only `0.6475` final / `0.6625` step-600 normal. A 600-step continuation improved to `0.823` final / `0.850` step-600 normal with low controls. An alternate downstream handoff seed from the same source also missed (`0.6325` final / step-600 normal), while the original good source passed with that failed handoff seed (`1.000` final / step-600 normal). This points to source/readout geometry rather than downstream seed luck.
Do not repeat: Do not run more same-op19 source-only seeds or semantic-distill weight tweaks as novelty. The source mechanism replicated; the unresolved issue is robust trusted handoff/readout behavior across seeds.
Next allowed test: Diagnose or improve source/readout geometry for handoff robustness, move to streaming/fresh-prompt memory, or validate a many-calculator/routed version if it directly tests scalability.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-fresh-seed.md`

POSITIVE: Semantic-distilled online-hard-memory clears a four-hook shared-output routed handoff on the original seed.
Conclusion: With four `left_operand_mod` routed hooks and `--share-calculator-output-proj`, sparse zero-improvement online hard memory plus additive semantic distillation reached source final/calc `1.0000`, froze memory after `96,000` forced evals, and trained all routed hooks to calculator-result accuracy `1.0000`. The trusted 600-step frozen-policy additive handoff also reached `1.0000` final / step-600 normal, calculator-result accuracy `1.0000`, and low controls (`0.0325` step-600 injection-zero, `0.0050` forced-zero, `0.0175` forced-random; final 128-sample controls `0.0391/0.0000/0.0391`). This is the first shared-output routed handoff pass and suggests semantic distillation supplies transfer geometry that hard-assignment shared-output runs lacked.
Do not repeat: The same four-hook shared-output op19 seed/source/handoff as novelty, or same-seed semantic-distill weight/sample/length tuning.
Next allowed test: Fresh routed/shared seed replication, streaming/fresh-prompt online memory, or larger-range routed/shared stress.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output.md`

POSITIVE: Semantic-distilled online-hard-memory four-hook shared-output handoff replicates on the fresh handoff-sensitive seed.
Conclusion: Repeated the four-hook `left_operand_mod` routed shared-output online-hard-memory plus additive-semantic-distillation source on CLI seed `7` / effective seed `9`, the seed lineage where the single-hook semantic-distilled source previously missed trusted handoff. The fresh routed/shared source reached final/calc `1.0000`, froze memory after `86,400` forced evals, and all four hooks reached calculator-result accuracy `1.0000`. The trusted 600-step frozen-policy additive handoff reached `1.0000` final / step-600 normal with calculator-result accuracy `1.0000`; step-600 controls were low (`0.0525` injection-zero, `0.0075` forced-zero, `0.0125` forced-random), while final 128-sample controls were `0.1094/0.0078/0.0156`. This means routed/shared semantic-distilled transfer is not just the original handoff-friendly seed.
Do not repeat: More same-op19 four-hook shared-output source/handoff seed repeats as novelty; two seeds now clear. Also do not tune same-seed semantic-distill weight/sample/length.
Next allowed test: Streaming/fresh-prompt online memory or larger-range routed/shared stress.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-fresh-seed.md`

POSITIVE: Semantic-distilled online-hard-memory four-hook shared-output handoff clears the op29 range stress.
Conclusion: Ran the four-hook `left_operand_mod` routed shared-output online-hard-memory plus additive-semantic-distillation recipe at `operand_max=29` with a wider product decoder (`n_embd=32`, `n_head=2`), `operand_spans` readout, and shallow result heads. The sparse source reached final/calc `1.0000` on the `900`-prompt grid; online memory filled/froze by step `50`, with cumulative forced-result evals capped at `367,200`. The trusted 600-step frozen-policy additive handoff reached `900/900 = 1.0000` final / step-600 normal, calculator-result accuracy `1.0000`, and low controls (`0.0133` injection-zero, `0.0022` forced-zero, `0.0156` forced-random at step 600). All four routed hooks reached calculator-result accuracy `1.0000`.
Do not repeat: More fixed-grid op19/op29 four-hook shared-output seed/range repeats as novelty. Also do not convert this into local semantic-distill weight/sample tuning.
Next allowed test: Streaming/fresh-prompt online memory, or a materially different many-calculator scaling gate where per-prompt memory cannot simply store the fixed grid.
Source: `aiAgentWorkHistory/phase7/2026-05-31-online-hard-memory-semantic-distill-routed-shared-output-op29.md`

POSITIVE: Prompt-keyed online-hard-memory trains routed shared-output calculators under streaming minibatches when exposure is matched.
Conclusion: Added `--streaming-train-batch-size` and `--result-boundary-target-online-memory-key-mode prompt`, so sparse zero-improvement online hard memory can train on fresh minibatches instead of requiring the fixed exhaustive-grid batch. On the four-hook `left_operand_mod` routed shared-output op19 gate, batch64 for 800 steps filled/froze all `400` prompt entries with true targets but undertrained the policy (`0.6325` final, diagnostic calculator-result accuracy `0.5781`). The predeclared exposure-matched batch64 source for 5000 steps reached final/calc `1.0000`, filled/froze memory after `173,568` forced evals, and trained all four hooks to calculator-result accuracy `1.0000`. The trusted 600-step frozen-policy additive handoff from that streaming source reached `400/400 = 1.0000` final / step-600 normal, with low controls (`0.0781` final injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and all hooks at calculator-result accuracy `1.0000`.
Do not repeat: Do not treat the 800-step batch64 miss as a mechanism failure or rerun same-exposure op19 streaming source/handoff as novelty. Do not return to fixed-grid routed/shared op19/op29 repeats.
Next allowed test: Fresh/heldout prompt generalization for prompt-keyed memory, or a cheaper streaming uptake mechanism that preserves the matched-exposure source/handoff result with fewer optimizer updates and forced evaluations.
Source: `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-streaming.md`

MIXED-NEGATIVE: Prompt-keyed online-hard-memory does not generalize to heldout prompts.
Conclusion: Added a deterministic streaming heldout split and split-specific train/heldout evaluations. On the four-hook shared-output op19 gate, the model trained only on `320` prompts for 5000 batch64 steps, filled/froze exactly those `320` memory entries after `87,552` forced evals, and reached train prompt exact/calc `0.996875` (`319/320`). The `80` heldout prompts, absent from both minibatches and prompt memory, reached only `0.0875` exact/calc (`7/80`), with low controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). This is a crisp transductive-memory boundary rather than an optimization failure on seen prompts.
Do not repeat: Do not launch a trusted handoff or same-exposure repeat from this heldout-failed source as novelty, and do not claim prompt-keyed memory is a fresh-prompt generalization method.
Next allowed test: Add a genuinely non-transductive mechanism: amortized target discovery, fresh-prompt candidate scoring/proposal, a learned memory initializer, or another answer-derived credit signal that can produce calculator targets for prompts not already stored.
Source: `aiAgentWorkHistory/phase7/2026-05-31-prompt-keyed-online-hard-memory-heldout.md`

PARTIAL-POSITIVE: A numeric amortized prior can generalize discovered prompt-memory targets.
Conclusion: Added an operand-conditioned amortized prior trained only from prompt hard-memory entries, plus heldout replay hooks and trace/replay diagnostics. On the prior heldout-failed op19 source, the arbitrary embedding prior fit train memory (`1.000` memory fit, `0.9969` train-vs-true) but got `0.0000` heldout-vs-true, confirming it memorizes prompt keys. Switching the prior to normalized numeric operand features kept train fit (`1.000`) and reached `0.9125` heldout-vs-true on the same `80` heldout prompts. A post-hoc result-head replay gate then transferred those numeric pseudo-targets into the source model: heldout calc/exact moved from `0.0875` to `0.9125` while train stayed `0.990625`. The matched embedding-prior replay control ended at only `0.0125` heldout and `0.959375` train.
Do not repeat: Do not treat embedding-feature amortized priors as a fresh-prompt solution, and do not claim the numeric prior has solved from-scratch/end-to-end training yet. The positive is target-prior generalization plus post-hoc result-head uptake.
Next allowed test: Run the integrated numeric-prior replay streaming source gate, then test whether the source learns seen and heldout prompts without post-hoc replay; only after that consider trusted handoff.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-heldout-diagnostic.md`

POSITIVE-WITH-CAVEAT: Integrated numeric-prior replay fixes the heldout prompt source gate when the prior is fit on full memory.
Conclusion: The first integrated 5000-step source with numeric prior replay but minibatch prior fit improved heldout prompts from `0.0875` to `0.7125`, while train stayed `0.990625`; an offline full-batch prior fit from that final train trace recovered `0.9125` heldout target accuracy, identifying online prior fit quality as the blocker. Adding `--result-boundary-target-amortized-prior-fit-batch-size 0` and rerunning the same op19 four-hook shared-output heldout source reached `398/400 = 0.9950` overall, train `320/320 = 1.0000`, heldout `73/80 = 0.9125`, low heldout controls (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random), and online prior heldout accuracy `0.9250` with `86,016` forced evals. The trusted frozen-policy additive handoff from this source reached `400/400 = 1.0000` with low controls (`0.0234` injection-zero, `0.0078` forced-zero, `0.0156` forced-random) and diagnostic calc `0.984375`.
Do not repeat: Do not rerun same-seed op19 full-memory prior fit as novelty, and do not treat full-memory prior fitting as already scalable.
Next allowed test: Reduce prior-fit cost while preserving heldout source accuracy and trusted handoff causality, using cached/periodic full-memory fits, multiple updates only when memory changes, or a coreset/reservoir fit batch before fresh-seed replication.
Source: `aiAgentWorkHistory/phase7/2026-05-31-integrated-amortized-prior-source-gate.md`

POSITIVE-WITH-CAVEAT: Every-other-step full-memory prior fitting preserves the integrated numeric-prior source and handoff result, but every-10 underfits.
Conclusion: Added `--result-boundary-target-amortized-prior-fit-every` to decouple prior-fit cadence from model replay. On the same op19 four-hook shared-output heldout source, fitting every `10` steps cut prior updates to `501` but underfit the prior (train `0.953125`, heldout prior `0.7875`) and degraded the source to overall `0.9475`, train `0.978125`, heldout `0.7625`. Fitting every `2` steps cut prior updates from `5001` to `2501` while preserving the benchmark source: overall `0.9950`, train `1.0000`, heldout `0.9125`, heldout controls `0.0500/0.0000/0.0125`, prior train/heldout `1.0000/0.9125`, and forced evals still `86,016`. The trusted frozen-policy additive handoff from the every-2 source reached final `395/400 = 0.9875`, diagnostic calc `0.984375`, and low 128-sample controls (`0.015625` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).
Do not repeat: Do not run a cadence ladder (`3/4/5/8/10`) as novelty. Every-10 already identifies prior update starvation; every-2 is the safe benchmark.
Next allowed test: Replace cadence-only thinning with convergence-gated fitting, stop/refresh after memory/prior convergence, or coreset/reservoir prior batches that target fewer than `2501` full-memory updates while preserving the every-2 heldout/handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-fit-cadence-gate.md`

POSITIVE-WITH-CAVEAT: Sustained train-memory convergence gating reduces full-memory prior updates below every-2 while preserving heldout source and handoff quality.
Conclusion: Added `--result-boundary-target-amortized-prior-stop-train-accuracy` and `--result-boundary-target-amortized-prior-stop-patience` so fitting can stop after prompt memory is full and the prior has stayed converged. Stopping at the first train-memory `1.0` cut updates to `1029` but hurt heldout source (`0.875` heldout, `0.9825` overall), showing train fit alone is too optimistic. Requiring `100` converged fit updates preserved the every-2 source gate with fewer updates: overall `398/400 = 0.9950`, train `1.0000`, heldout `0.9125`, low heldout controls (`0.0500/0.0000/0.0125`), prior train/heldout `1.0000/0.9125`, forced evals `86,016`, and prior updates `1889` instead of `2501`/`5001`. The trusted frozen-policy additive handoff reached `397/400 = 0.9925`, diagnostic calc `0.984375`, and low 128-sample controls (`0.0546875` injection-zero, `0.0078125` forced-zero, `0.0078125` forced-random).
Do not repeat: Do not run patience ladders as novelty. First-hit convergence is disproven; patience-100 is the new safe train-convergence benchmark.
Next allowed test: Use a validation/heldout-prior signal or coreset/reservoir prior batches to reduce below `1889` updates while preserving the same heldout source and trusted handoff gate.
Source: `aiAgentWorkHistory/phase7/2026-05-31-amortized-prior-convergence-stop-gate.md`

MIXED-NEGATIVE: Random half-memory prior fit batches do not preserve the integrated numeric-prior heldout source gate.
Conclusion: Tested the coreset-style cost lever by setting `--result-boundary-target-amortized-prior-fit-batch-size 160` with the every-2, stop-accuracy-1.0, patience-100 source recipe. This halves examples per prior fit, but the prior never converged: final prior train/heldout accuracy was only `0.909375` / `0.7750`, stop never activated, and updates remained `2501`. Source train stayed high (`0.996875`), but heldout exact/calc fell to `65/80 = 0.8125` and overall to `387/400 = 0.9675`, with heldout controls still low (`0.0500` injection-zero, `0.0000` forced-zero, `0.0125` forced-random). No trusted handoff was run because the source gate missed.
Do not repeat: Do not run random prior fit-batch-size ladders as novelty. Batch `64` already left heldout at `0.7125`, and random batch `160` still underfits at `0.8125`.
Next allowed test: Use a structured/coverage-aware coreset, reservoir with balanced operand coverage, or validation-aware stopping signal rather than uniform random prior-fit minibatches.
Source: `aiAgentWorkHistory/phase7/2026-05-31-random-half-memory-prior-fit-gate.md`
