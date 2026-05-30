# Routed calculator hooks can share one output projection.

Kind: hypothesis_memory
Status: POSITIVE-IMPLEMENTATION
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-shared-routed-output-projection.md

Summary:

- Added `calculator_share_output_proj` / `--share-calculator-output-proj` so extra calculator hooks tie their result-to-residual `output_proj` module to the primary hook instead of cloning independent parameters. A three-hook shared model removes two extra output-projection parameter matrices while preserving a single shared semantic interface; tests verify object identity, parameter-count reduction, CLI config/metrics recording, and compatibility when loading older untied checkpoints by canonicalizing extra-hook output keys to the primary output projection. A zero-step three-hook routed CLI smoke wrote `share_calculator_output_proj=True` in both config and metrics. This fixes the routed many-calculator parameter-slope issue at the semantic output interface, but it has not yet re-run the source/handoff training gate with tied outputs.

Questions:

- What did we learn about Routed calculator hooks can share one output projection?
- Has Routed calculator hooks can share one output projection been tested?
- Should we repeat Routed calculator hooks can share one output projection?
- What is the status of Routed calculator hooks can share one output projection?
- What follow-up is allowed for Routed calculator hooks can share one output projection?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-shared-routed-output-projection.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not keep using cloned per-hook output projections as the only fair routed semantic-interface option. Use tied output projections when testing many-calculator parameter scaling.

Next Allowed:

- Run a small tied-output routed source gate, preferably matching the known 4-hook active-only setup with topk8+unique24, and compare per-hook calculator accuracy/controls against the cloned-output result before moving back to less-prescriptive credit assignment.

Full Text:

```text
POSITIVE-IMPLEMENTATION: Routed calculator hooks can share one output projection.
Conclusion: Added `calculator_share_output_proj` / `--share-calculator-output-proj` so extra calculator hooks tie their result-to-residual `output_proj` module to the primary hook instead of cloning independent parameters. A three-hook shared model removes two extra output-projection parameter matrices while preserving a single shared semantic interface; tests verify object identity, parameter-count reduction, CLI config/metrics recording, and compatibility when loading older untied checkpoints by canonicalizing extra-hook output keys to the primary output projection. A zero-step three-hook routed CLI smoke wrote `share_calculator_output_proj=True` in both config and metrics. This fixes the routed many-calculator parameter-slope issue at the semantic output interface, but it has not yet re-run the source/handoff training gate with tied outputs.
Do not repeat: Do not keep using cloned per-hook output projections as the only fair routed semantic-interface option. Use tied output projections when testing many-calculator parameter scaling.
Next allowed test: Run a small tied-output routed source gate, preferably matching the known 4-hook active-only setup with topk8+unique24, and compare per-hook calculator accuracy/controls against the cloned-output result before moving back to less-prescriptive credit assignment.
Source: `aiAgentWorkHistory/phase7/2026-05-30-shared-routed-output-projection.md`
```
