# Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-03-prior-bootstrap-route-excluded-source.md

Summary:

- Added and then tested high-confidence prior-bootstrap prompt-memory entries for route 1, gated by prior train accuracy `>=0.75`, confidence `>=0.30`, and cap `8` per step. The full op19 route-excluded source wrote `77` prior-bootstrap entries and preserved causal controls, but final eval fell to `308/400 = 0.7700`, best snapshot was only `0.7825`, and final snapshot controls were `0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random. Train prompts reached `0.8125`, heldout stayed `0.5625`, and prior train/heldout stayed `0.7781`/`0.5625`. The excluded route 1 was not rescued: train route 1 `0.6392`, heldout route 1 `0.7391`, diagnostic route 1 `0.7714`, best snapshot route 1 `0.6667`. Bootstrap opened late after the prior train gate and did not change the shared-prior bottleneck. No handoff was run.

Questions:

- What did we learn about Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate?
- Has Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate been tested?
- Should we repeat Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate?
- What is the status of Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate?
- Why did Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-03-prior-bootstrap-route-excluded-source.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run bootstrap confidence, train-accuracy, cap, or same op19 route-excluded variants as novelty. The failure is not lack of route-target writes after a weak prior has already formed.

Next Allowed:

- Move away from post-hoc prompt-memory target bootstrapping toward a genuinely different shared/global target mechanism: train the shared prior on candidate evidence before route memory freezes, learn shared targets jointly across routes, or replace answer-derived candidate scoring with a less-prescriptive credit signal.

Full Text:

```text
MIXED-NEGATIVE: Prior-bootstrap prompt memory writes excluded-route targets but does not improve the op19 route-excluded source gate.
Conclusion: Added and then tested high-confidence prior-bootstrap prompt-memory entries for route 1, gated by prior train accuracy `>=0.75`, confidence `>=0.30`, and cap `8` per step. The full op19 route-excluded source wrote `77` prior-bootstrap entries and preserved causal controls, but final eval fell to `308/400 = 0.7700`, best snapshot was only `0.7825`, and final snapshot controls were `0.0475` injection-zero, `0.0025` forced-zero, `0.0025` forced-random. Train prompts reached `0.8125`, heldout stayed `0.5625`, and prior train/heldout stayed `0.7781`/`0.5625`. The excluded route 1 was not rescued: train route 1 `0.6392`, heldout route 1 `0.7391`, diagnostic route 1 `0.7714`, best snapshot route 1 `0.6667`. Bootstrap opened late after the prior train gate and did not change the shared-prior bottleneck. No handoff was run.
Do not repeat: Do not run bootstrap confidence, train-accuracy, cap, or same op19 route-excluded variants as novelty. The failure is not lack of route-target writes after a weak prior has already formed.
Next allowed test: Move away from post-hoc prompt-memory target bootstrapping toward a genuinely different shared/global target mechanism: train the shared prior on candidate evidence before route memory freezes, learn shared targets jointly across routes, or replace answer-derived candidate scoring with a less-prescriptive credit signal.
Source: `aiAgentWorkHistory/phase7/2026-06-03-prior-bootstrap-route-excluded-source.md`
```
