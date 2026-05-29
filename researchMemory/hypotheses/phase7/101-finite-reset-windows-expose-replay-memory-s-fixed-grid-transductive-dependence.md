# Finite reset windows expose replay-memory's fixed-grid transductive dependence.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md

Summary:

- Adding optional `_resetN` replay-memory syntax showed that persistent per-prompt caches are doing important work. In the 200-step stress gate, no-reset `u2_m30` reached `0.6025` exact calc / `0.6016` sampled normal, while `reset50` fell to `0.2500` / `0.2578`, `reset25` to `0.1650` / `0.2188`, and `reset10` to `0.0950` / `0.1406`. A 199-step boundary check removed the final-reset snapshot caveat: no-reset was `0.5925` / `0.5938`, `reset100` was only `0.4575` / `0.4453`, and `reset50` was `0.2575` / `0.2812` despite mostly restored target coverage (`0.9925` and `0.9525` true-candidate coverage respectively).

Questions:

- What did we learn about Finite reset windows expose replay-memory's fixed-grid transductive dependence?
- Has Finite reset windows expose replay-memory's fixed-grid transductive dependence been tested?
- Should we repeat Finite reset windows expose replay-memory's fixed-grid transductive dependence?
- What is the status of Finite reset windows expose replay-memory's fixed-grid transductive dependence?
- Why did Finite reset windows expose replay-memory's fixed-grid transductive dependence fail?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- The same seed-2 replay-memory reset sweep over `reset10/reset25/reset50` or the 199-step `reset50/reset100` boundary check as novelty.

Next Allowed:

- Do not tune reset intervals as a local fix. Move replay-memory work to streaming/non-exhaustive prompt stress or learned/generalized proposal memory, where the method cannot rely on persistent prompt-identity caches.

Full Text:

```text
MIXED-NEGATIVE: Finite reset windows expose replay-memory's fixed-grid transductive dependence.
Conclusion: Adding optional `_resetN` replay-memory syntax showed that persistent per-prompt caches are doing important work. In the 200-step stress gate, no-reset `u2_m30` reached `0.6025` exact calc / `0.6016` sampled normal, while `reset50` fell to `0.2500` / `0.2578`, `reset25` to `0.1650` / `0.2188`, and `reset10` to `0.0950` / `0.1406`. A 199-step boundary check removed the final-reset snapshot caveat: no-reset was `0.5925` / `0.5938`, `reset100` was only `0.4575` / `0.4453`, and `reset50` was `0.2575` / `0.2812` despite mostly restored target coverage (`0.9925` and `0.9525` true-candidate coverage respectively).
Do not repeat: The same seed-2 replay-memory reset sweep over `reset10/reset25/reset50` or the 199-step `reset50/reset100` boundary check as novelty.
Next allowed test: Do not tune reset intervals as a local fix. Move replay-memory work to streaming/non-exhaustive prompt stress or learned/generalized proposal memory, where the method cannot rely on persistent prompt-identity caches.
Source: `aiAgentWorkHistory/phase7/2026-05-29-replay-memory-reset-stress-gate.md`
```
