# Streaming minibatches remove replay-memory's strong fixed-grid local-target lift.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md

Summary:

- Adding `--streaming-train-batch-size` and prompt-keyed replay caches showed that replay memory does not preserve the fixed-grid advantage under sampled minibatch training. At 200 steps with batch `16`, exact `policy_reweighted_t1` reached only `0.1100` exact calc, raw uniform `u32` `0.0700`, `u2_m30` `0.0475`, and `u8_m24` `0.0950`. Extending to 800 batch-16 steps raised exact and raw `u32` to `0.2450`, `u8_m24` to a comparable `0.2650`, and `u2_m30` to only `0.1850`; batch-64 for 200 steps also stayed weak (`0.1650` exact, `0.1475` u8, `0.0975` u2). Prompt memory reached all 400 prompts and `u8_m24` often had full current-batch target coverage, so the missing lift is not just absent prompt keys.

Questions:

- What did we learn about Streaming minibatches remove replay-memory's strong fixed-grid local-target lift?
- Has Streaming minibatches remove replay-memory's strong fixed-grid local-target lift been tested?
- Should we repeat Streaming minibatches remove replay-memory's strong fixed-grid local-target lift?
- What is the status of Streaming minibatches remove replay-memory's strong fixed-grid local-target lift?
- Why did Streaming minibatches remove replay-memory's strong fixed-grid local-target lift fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 streaming minibatch gates at batch `16` for `200/800` steps or batch `64` for `200` steps over exact, raw `u32`, `u2_m30`, and `u8_m24` as novelty.

Next Allowed:

- Do not treat prompt-keyed replay memory as the scalable answer. Continue local targets only with a learned/generalized proposal, estimator correction, or a different target construction; otherwise return mainline compute to source objectives aimed at handoff/readout geometry.

Full Text:

```text
MIXED-NEGATIVE: Streaming minibatches remove replay-memory's strong fixed-grid local-target lift.
Conclusion: Adding `--streaming-train-batch-size` and prompt-keyed replay caches showed that replay memory does not preserve the fixed-grid advantage under sampled minibatch training. At 200 steps with batch `16`, exact `policy_reweighted_t1` reached only `0.1100` exact calc, raw uniform `u32` `0.0700`, `u2_m30` `0.0475`, and `u8_m24` `0.0950`. Extending to 800 batch-16 steps raised exact and raw `u32` to `0.2450`, `u8_m24` to a comparable `0.2650`, and `u2_m30` to only `0.1850`; batch-64 for 200 steps also stayed weak (`0.1650` exact, `0.1475` u8, `0.0975` u2). Prompt memory reached all 400 prompts and `u8_m24` often had full current-batch target coverage, so the missing lift is not just absent prompt keys.
Do not repeat: The same seed-2 streaming minibatch gates at batch `16` for `200/800` steps or batch `64` for `200` steps over exact, raw `u32`, `u2_m30`, and `u8_m24` as novelty.
Next allowed test: Do not treat prompt-keyed replay memory as the scalable answer. Continue local targets only with a learned/generalized proposal, estimator correction, or a different target construction; otherwise return mainline compute to source objectives aimed at handoff/readout geometry.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-streaming-prompt-gate.md`
```
