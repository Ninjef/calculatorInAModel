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
