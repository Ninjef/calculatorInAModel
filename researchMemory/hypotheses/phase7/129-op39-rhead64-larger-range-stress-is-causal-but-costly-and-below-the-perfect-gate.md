# op39 rhead64 larger-range stress is causal but costly and below the perfect gate.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-op39-rhead64-range-stress.md

Summary:

- A new op39 product oracle decoder cleared full-grid eval (`1600/1600 = 1.0000`), so decoder wiring was not the blocker. The op39 `rhead64` source run was interrupted after about `33` minutes of local CPU time with checkpoints through step `540`; a zero-step eval of step `540` was only `0.543` exact / `0.547` snapshot normal. A bounded 90-step continuation from that checkpoint, with the late-recovery switch at continuation step `60`, lifted source final eval to `1504/1600 = 0.940` and source step `90` normal/calc to `0.9431`, with low controls (`0.0213` injection-zero, `0.0113` forced-random). The trusted frozen-policy handoff from continuation step `90` reached `1516/1600 = 0.9475` final eval / `0.9419` step-600 normal, with low controls (`0.0000` injection-zero, `0.0138` forced-random) and learned calc `0.9375`. This is causal larger-range transfer, but it is not op29-style perfect, and the full-grid source cost/continuation requirement strengthens the scalability warning.

Questions:

- What did we learn about op39 rhead64 larger-range stress is causal but costly and below the perfect gate?
- Has op39 rhead64 larger-range stress is causal but costly and below the perfect gate been tested?
- Should we repeat op39 rhead64 larger-range stress is causal but costly and below the perfect gate?
- What is the status of op39 rhead64 larger-range stress is causal but costly and below the perfect gate?
- What follow-up is allowed for op39 rhead64 larger-range stress is causal but costly and below the perfect gate?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-op39-rhead64-range-stress.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not rerun the same op39 effective-seed-39 full-grid `rhead64` source, step-540 continuation, and 600-step handoff as novelty, and do not jump to op49 full-grid without a declared assignment-cost or capacity-scaling change.

Next Allowed:

- Use op39 as evidence to prioritize cheaper assignment, many-calculator cost accounting, or a materially different source-capacity/credit-assignment mechanism; further full-grid range tests need an explicit scalability hypothesis.

Full Text:

```text
MIXED-POSITIVE: op39 rhead64 larger-range stress is causal but costly and below the perfect gate.
Conclusion: A new op39 product oracle decoder cleared full-grid eval (`1600/1600 = 1.0000`), so decoder wiring was not the blocker. The op39 `rhead64` source run was interrupted after about `33` minutes of local CPU time with checkpoints through step `540`; a zero-step eval of step `540` was only `0.543` exact / `0.547` snapshot normal. A bounded 90-step continuation from that checkpoint, with the late-recovery switch at continuation step `60`, lifted source final eval to `1504/1600 = 0.940` and source step `90` normal/calc to `0.9431`, with low controls (`0.0213` injection-zero, `0.0113` forced-random). The trusted frozen-policy handoff from continuation step `90` reached `1516/1600 = 0.9475` final eval / `0.9419` step-600 normal, with low controls (`0.0000` injection-zero, `0.0138` forced-random) and learned calc `0.9375`. This is causal larger-range transfer, but it is not op29-style perfect, and the full-grid source cost/continuation requirement strengthens the scalability warning.
Do not repeat: Do not rerun the same op39 effective-seed-39 full-grid `rhead64` source, step-540 continuation, and 600-step handoff as novelty, and do not jump to op49 full-grid without a declared assignment-cost or capacity-scaling change.
Next allowed test: Use op39 as evidence to prioritize cheaper assignment, many-calculator cost accounting, or a materially different source-capacity/credit-assignment mechanism; further full-grid range tests need an explicit scalability hypothesis.
Source: `aiAgentWorkHistory/phase7/2026-05-30-op39-rhead64-range-stress.md`
```
