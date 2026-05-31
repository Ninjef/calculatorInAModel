# Cached teacher tables separate policy imitation from teacher-target quality.

Kind: hypothesis_memory
Status: MIXED
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-cached-teacher-target-table.md

Summary:

- Added `--result-boundary-target-cache` with `target_weights` and `hard_best` modes. Caching the frozen additive teacher's soft zero-improvement weights reproduced the online-anchor ceiling (`0.4000` learned-best / `0.1650` final at 800), showing repeated forced-result rescoring was not the source of weak uptake. Hard cached teacher-best made the policy imitate the teacher table much better (`0.668` learned-best / `0.338` final at 800; `0.710` learned-best / `0.3725` final at 1600), but the teacher best itself is true only `0.5225` of prompts, so this improves uptake while exposing target-quality ceiling rather than solving calculator learning.

Questions:

- What did we learn about Cached teacher tables separate policy imitation from teacher-target quality?
- Has Cached teacher tables separate policy imitation from teacher-target quality been tested?
- Should we repeat Cached teacher tables separate policy imitation from teacher-target quality?
- What is the status of Cached teacher tables separate policy imitation from teacher-target quality?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-cached-teacher-target-table.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more same-teacher cached soft/hard length/LR sweeps as novelty. Cached hard-best is a diagnostic/ceiling tool, not a scalable or sufficiently correct training method by itself.

Next Allowed:

- Improve the teacher target quality or change the answer-derived target source before optimizing imitation further; alternatively use cached tables only as a cheap diagnostic for new target constructions.

Full Text:

```text
MIXED: Cached teacher tables separate policy imitation from teacher-target quality.
Conclusion: Added `--result-boundary-target-cache` with `target_weights` and `hard_best` modes. Caching the frozen additive teacher's soft zero-improvement weights reproduced the online-anchor ceiling (`0.4000` learned-best / `0.1650` final at 800), showing repeated forced-result rescoring was not the source of weak uptake. Hard cached teacher-best made the policy imitate the teacher table much better (`0.668` learned-best / `0.338` final at 800; `0.710` learned-best / `0.3725` final at 1600), but the teacher best itself is true only `0.5225` of prompts, so this improves uptake while exposing target-quality ceiling rather than solving calculator learning.
Do not repeat: Do not run more same-teacher cached soft/hard length/LR sweeps as novelty. Cached hard-best is a diagnostic/ceiling tool, not a scalable or sufficiently correct training method by itself.
Next allowed test: Improve the teacher target quality or change the answer-derived target source before optimizing imitation further; alternatively use cached tables only as a cheap diagnostic for new target constructions.
Source: `aiAgentWorkHistory/phase7/2026-05-30-cached-teacher-target-table.md`
```
