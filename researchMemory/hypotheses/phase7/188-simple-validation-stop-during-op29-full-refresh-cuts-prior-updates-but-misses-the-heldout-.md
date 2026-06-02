# Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate.

Kind: hypothesis_memory
Status: MIXED-NEGATIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-06-02-op29-full-refresh-stop-during-refresh-gate.md

Summary:

- Added `--result-boundary-target-amortized-prior-full-refresh-allow-stop`, allowing the existing validation stop rule to end the post-memory-fill full-refresh window. On the exact-matched op29 h128 source, validation `>=0.9` with patience `100` stopped during refresh at `1140` prior updates instead of the positive run's `2755`, but heldout exact/calc fell to `0.8167` and heldout prior to `0.8167` despite train exact/calc `0.9889`, train prior `0.9514`, overall exact `0.9533`, and low heldout controls (`0.0278` injection-zero, `0.0000` forced-zero, `0.0111` forced-random). No handoff was run because the source missed the heldout gate.

Questions:

- What did we learn about Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate?
- Has Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate been tested?
- Should we repeat Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate?
- What is the status of Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate?
- Why did Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate fail?
- What follow-up is allowed for Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-06-02-op29-full-refresh-stop-during-refresh-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not treat the same full-refresh allow-stop run, validation threshold/patience ladders, or earlier-stop-only variants as novelty.

Next Allowed:

- A materially stronger refresh-cost mechanism: staged full refresh followed by coreset replay, dual train+validation/high-confidence stopping, coverage-aware/proportional refresh, or a predeclared stop/freeze transition that preserves heldout before handoff.

Full Text:

```text
MIXED-NEGATIVE: Simple validation-stop during op29 full refresh cuts prior updates but misses the heldout source gate.
Conclusion: Added `--result-boundary-target-amortized-prior-full-refresh-allow-stop`, allowing the existing validation stop rule to end the post-memory-fill full-refresh window. On the exact-matched op29 h128 source, validation `>=0.9` with patience `100` stopped during refresh at `1140` prior updates instead of the positive run's `2755`, but heldout exact/calc fell to `0.8167` and heldout prior to `0.8167` despite train exact/calc `0.9889`, train prior `0.9514`, overall exact `0.9533`, and low heldout controls (`0.0278` injection-zero, `0.0000` forced-zero, `0.0111` forced-random). No handoff was run because the source missed the heldout gate.
Do not repeat: Do not treat the same full-refresh allow-stop run, validation threshold/patience ladders, or earlier-stop-only variants as novelty.
Next allowed test: A materially stronger refresh-cost mechanism: staged full refresh followed by coreset replay, dual train+validation/high-confidence stopping, coverage-aware/proportional refresh, or a predeclared stop/freeze transition that preserves heldout before handoff.
Source: `aiAgentWorkHistory/phase7/2026-06-02-op29-full-refresh-stop-during-refresh-gate.md`
```
