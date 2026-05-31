# Additive Target Anchoring Steering Review

Date: 2026-05-30

## What Changed

Semantic readout distillation showed that the additive forced-result table can
be made meaningful, but policy uptake stayed weak and the table drifted without
ongoing distillation.

The frozen-teacher anchor separated those two failures. A separate frozen
teacher preserved additive target quality (`best=true=0.5225`) while the live
policy trained. That improved learned-best to `0.4125` after 800 steps, but
source calculator accuracy and final eval stayed low (`0.1700`/`0.1750`).

## What Should Stop

- Plain semantic-distill weight/sample/length tweaks.
- Same-checkpoint frozen-teacher anchor length/LR/freezing sweeps.
- Additive-path scoring from an untrained or drifting additive table.
- Claims that target-table repair alone solves non-bottleneck source
  acquisition.

## What Deserves Compute

- A policy-uptake mechanism that makes the repaired teacher table easier for
  the source policy to represent or imitate.
- Cached/streamed teacher tables as a diagnostic for policy learning, only if
  paired with a new uptake objective rather than repeated full rescoring.
- A different less-prescriptive estimator that preserves target quality while
  moving true-result uptake above the current `0.17` source-calc ceiling.

## Relation To The Goal

This branch moved the diagnosis forward but did not solve the thesis. The
project is closer to knowing the failure boundary: additive readout semantics
and target stability can be supplied, yet the policy still does not learn the
useful calculator result strongly enough. The next mainline step should change
policy learning itself, not the additive target table.
