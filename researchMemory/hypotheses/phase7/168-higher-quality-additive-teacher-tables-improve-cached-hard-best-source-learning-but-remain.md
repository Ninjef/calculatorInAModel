# Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates.

Kind: hypothesis_memory
Status: PARTIAL-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-high-quality-cached-teacher-table.md

Summary:

- Reused the semantic-distilled preconditioned+ongoing-distill additive checkpoint as the frozen cache teacher. Its additive hard-best table is much better (`best_true=0.8200`) than the preconditioner-only teacher (`0.5225`). Cached soft target weights from this teacher still learned poorly (`0.393` learned-best / `0.298` calc / `0.273` final at 800), but cached hard-best imitation reached `0.728` learned-best / `0.595` calc / `0.562` final at 800 and `0.765` learned-best / `0.618` calc / `0.583` final at 1600. Better target quality plus hardening materially helps, but this still trails the teacher ceiling and the mature bottleneck zero-improvement source.

Questions:

- What did we learn about Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates?
- Has Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates been tested?
- Should we repeat Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates?
- What is the status of Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates?
- What follow-up is allowed for Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-high-quality-cached-teacher-table.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more high-quality-teacher cached hard-best length/LR sweeps as novelty; the curve is useful as a ceiling diagnostic, not a recipe.

Next Allowed:

- Improve the answer-derived target source itself or return to bottleneck zero-improvement/handoff-aware target construction. Cached hard-best can be used as a cheap diagnostic for candidate target tables before expensive source/handoff runs.

Full Text:

```text
PARTIAL-POSITIVE: Higher-quality additive teacher tables improve cached hard-best source learning but remain below source gates.
Conclusion: Reused the semantic-distilled preconditioned+ongoing-distill additive checkpoint as the frozen cache teacher. Its additive hard-best table is much better (`best_true=0.8200`) than the preconditioner-only teacher (`0.5225`). Cached soft target weights from this teacher still learned poorly (`0.393` learned-best / `0.298` calc / `0.273` final at 800), but cached hard-best imitation reached `0.728` learned-best / `0.595` calc / `0.562` final at 800 and `0.765` learned-best / `0.618` calc / `0.583` final at 1600. Better target quality plus hardening materially helps, but this still trails the teacher ceiling and the mature bottleneck zero-improvement source.
Do not repeat: Do not run more high-quality-teacher cached hard-best length/LR sweeps as novelty; the curve is useful as a ceiling diagnostic, not a recipe.
Next allowed test: Improve the answer-derived target source itself or return to bottleneck zero-improvement/handoff-aware target construction. Cached hard-best can be used as a cheap diagnostic for candidate target tables before expensive source/handoff runs.
Source: `aiAgentWorkHistory/phase7/2026-05-30-high-quality-cached-teacher-table.md`
```
