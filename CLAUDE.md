# Overview

This repo is intended as a research sandbox. The thesis being researched is related to connecting non-differentiable tools into a neural network such that both the inputs and outputs to the tool are connected directly to the internal workings of the network. We are trying to see if we can get a simple calculator tool to sit within a neural network and whether the network can learn to use it.

# Progress

So far, we've found that if we use an "oracle" approach to give the calculator the correct output for whatever math question comes in, the downstream nodes can absolutely learn to answer the question (no surprise). However, because tools are typically non-differentiable, the upstream neurons have not yet shown an ability to learn how to provide inputs into the calculator such that the network's ability to do math succeeds. We have only heavily tried STE, but there are many other possible approaches available.

We have a lot of other ideas to try here SOLUTION_IDEAS.md

# Critical Research Guardrail

Do not rediscover or re-present oracle calculator success as progress. Since
Phase 1, the project has known that downstream answer components can solve the
task when given correct calculator outputs or oracle operands. Oracle runs,
oracle-at-eval recovery, injection-zero controls, and forced-random controls are
wiring checks only.

The only central research question is whether the upstream/model-side interface
can learn to provide useful calculator inputs, and whether that learned
calculator-query protocol is retained when direct operand supervision or other
teacher signals are removed. Future work should prioritize learned-interface
metrics: operand/pair exact match, calculator-result accuracy from learned
actions, private all-pair protocol decoding, learned-vs-true action-loss gaps,
aux/supervision weight exactly `0.0`, and retention across checkpoints/seeds.

Before running any oracle-only experiment, ask whether it is strictly needed to
validate new wiring. If the wiring has already been validated for the current
configuration, skip oracle-only reruns and move directly to learned-interface
teaching/retention.

Do not rediscover target-off retention as a novel research result. The project
has already tested the general pattern many times: teach or scaffold a
calculator interface, remove the scaffold, and ask whether answer loss retains
or completes the protocol. Phase 4 established seed-robust aux-zero retention
for the identifiable `sum_left_operand` true-operand protocol. Phase 5 showed
upstream-open answer-only continuations can preserve or complete partially
taught identifiable protocols, while no-handoff answer-only discovery still
fails. Phase 6 established relaxation/local-target-off retention for
answer-derived identifiable bridges, including deterministic Concrete across
seeds. Future target-off/retention runs should only be done when they test a
genuinely new interface, objective, action parameterization, or stability
question; do not spend tasks merely re-proving that retention-after-teaching is
possible.

# Current Phase 7 Finding

Phase 7 has a supervised natural `0..19` result-level ceiling, but not yet a
robust answer-loss discovery result.

Exact full-grid upstream-open result-boundary teaching can learn hard natural
result requests. Seed `2` produced a single-seed retained positive
(`0.9675` Stage 1 hard result accuracy; `0.8800` best post-start target-off
retention), and CLI seeds `4` and `5` relearned Stage 1 requests near exact
(`1.0000` and `0.9975`).

However, the strict retention replication gate failed. Seeds `4` and `5`
retained only `87.0%` and `88.2%` of their selected Stage 1 hard result
accuracy at the best post-start target-off checkpoints, below the required
`90%` threshold. This is `exact_grid_seed_replication_negative`, not a robust
retained-positive replication.

The next estimator-family test also produced a useful negative:
`multisample_result_space_policy_gradient_stage0_alignment_negative`.
Result-space REINFORCE is now implemented and wired: `K=16` exact-grid
multi-sample policy gradient produced nonzero result-proj/upstream gradients,
semantic decoder gradient stayed exactly `0.0`, and per-prompt/leave-one-out
baselines reduced advantage variance versus the old global EMA baseline. But
the fixed-grid policy-gradient estimate was anti-aligned with the known
boundary-target ceiling (`result-proj cosine=-0.0945`, upstream
`cosine=-0.1108`), so Stage 1 long training was intentionally skipped.

Do not rerun these as next steps unless debugging new code:

- oracle/readout checks for natural `0..19`;
- random-resampled upstream-open boundary-target repeats;
- frozen linear or frozen MLP result-head boundary teaching;
- the MLP rescue from the full-grid task;
- more target-off retention reruns that do not introduce a genuinely new
  mechanism or diagnose the observed seed fragility.
- vanilla multi-sample result-space policy-gradient long runs without first
  fixing the Stage 0 gradient-alignment problem.

Next best step: run the exact result-marginal answer-loss gradient gate before
spending more long-run budget. The selected task is
`aiAgentProjectTasks/2026-05-14-phase-7-ninth-task-Exact-result-marginal-answer-loss-gradient-gate.md`.
It should enumerate the small `0..38` result action space, compare the exact
expected answer-loss gradient against both sampled PG and the boundary-target
ceiling, and decide whether the PG negative was variance/control-variate
weakness or objective misalignment. Only after that fork should the project
choose actor-critic/NVIL, RELAX/REBAR, surrogate/shadow-calculator gradients,
synthetic gradients/direct feedback alignment, or stricter decoder-phase
bottlenecks. Do not move directly to canonical-query/protocol stabilization as
if Phase 7 retention had robustly replicated.

For details, see `factSheets/PHASE_7_EXPERIMENT_FACT_SHEET.md`.

# Navigation
You can find a valuable set of fact sheets in factSheets/, which keeps track of all the learnings of past experiments by experiment phase
Under aiAgentWorkHistory, we have all the work performed in the past.
Under aiAgentProjectTasks, we have all the intended work to be done by the researchers, completed ones in the completed folder.
You can find the overarching experiment's purpose here: OVERARCHING_EXPERIMENT_PURPOSE.md
You can find a heavy set of ideas we want to try here: SOLUTION_IDEAS.md

# After contributing
- Whenever doing experiments and learning new information, fill out information in the associated phase's fact sheet
- Fill out any work history in aiAgentWorkHisotry that you've accomplished
- Move your task file to the completed folder (if it's in fact fully completed)
- Commit and push
