# Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling.

Kind: hypothesis_memory
Status: MIXED-POSITIVE
Phase: Phase 7
Source: aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md

Summary:

- Added `--result-policy-improvement-assignment-unique-sampling` so sampled hard-assignment candidates include the learned result plus per-prompt random result classes without replacement. This directly tested whether duplicate waste caused the prior sampled-assignment failure. Unique16 improved step-200 true coverage only slightly (`0.6525` vs duplicate sample16 `0.6125`) and reached final `162/400 = 0.4050`, still weak. Unique32 was meaningfully better than duplicate sample32: true coverage `0.9275` vs `0.7400`, target accuracy `0.8156` vs `0.6773`, best snapshot `0.6250` vs `0.4050`, and final `244/400 = 0.6100` vs `0.3800`. But it still missed the exact assignment ceiling (`0.8625` best snapshot, `0.7350` final, target accuracy `0.9900`), while scoring most of the `39` result classes. Duplicate removal matters, but sparse unique coverage is not enough.

Questions:

- What did we learn about Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling?
- Has Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling been tested?
- Should we repeat Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling?
- What is the status of Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling?
- What follow-up is allowed for Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling?

Representative evidence:

- `aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md`
- `HYPOTHESIS_LEDGER.md`

Do Not Repeat:

- Do not run more unique-uniform sample-count ladders on the op19 `rhead64` 200-step gate as novelty. Unique32 is the useful diagnostic point; lower counts are too coverage-limited, and higher counts approach exact enumeration.

Next Allowed:

- Candidate-cost reduction needs a smarter non-uniform proposal, active/uncertainty allocation, or target construction that closes the remaining exact-ceiling gap at materially lower scoring cost. Validate against exact assignment, not just duplicate sampled baselines.

Full Text:

```text
MIXED-POSITIVE: Unique sampled hard-assignment improves coverage but still misses the op19 exact ceiling.
Conclusion: Added `--result-policy-improvement-assignment-unique-sampling` so sampled hard-assignment candidates include the learned result plus per-prompt random result classes without replacement. This directly tested whether duplicate waste caused the prior sampled-assignment failure. Unique16 improved step-200 true coverage only slightly (`0.6525` vs duplicate sample16 `0.6125`) and reached final `162/400 = 0.4050`, still weak. Unique32 was meaningfully better than duplicate sample32: true coverage `0.9275` vs `0.7400`, target accuracy `0.8156` vs `0.6773`, best snapshot `0.6250` vs `0.4050`, and final `244/400 = 0.6100` vs `0.3800`. But it still missed the exact assignment ceiling (`0.8625` best snapshot, `0.7350` final, target accuracy `0.9900`), while scoring most of the `39` result classes. Duplicate removal matters, but sparse unique coverage is not enough.
Do not repeat: Do not run more unique-uniform sample-count ladders on the op19 `rhead64` 200-step gate as novelty. Unique32 is the useful diagnostic point; lower counts are too coverage-limited, and higher counts approach exact enumeration.
Next allowed test: Candidate-cost reduction needs a smarter non-uniform proposal, active/uncertainty allocation, or target construction that closes the remaining exact-ceiling gap at materially lower scoring cost. Validate against exact assignment, not just duplicate sampled baselines.
Source: `aiAgentWorkHistory/phase7/2026-05-30-unique-sampled-assignment-coverage-gate.md`
```
