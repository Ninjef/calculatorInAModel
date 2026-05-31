# 2026-05-31 - Prompt-Keyed Online Hard Memory Heldout Prompt Gate

## Question

Does prompt-keyed online hard memory generalize to prompts that were never seen
in the streaming minibatches and never stored in memory?

## Implementation

- Added `--streaming-train-heldout-fraction` and
  `--streaming-train-heldout-seed`.
- In heldout mode, the script builds a deterministic exhaustive-grid split,
  samples streaming training batches only from the train pool, and preserves
  the heldout pool for final deterministic diagnostics.
- Added split-specific final evaluations and trace CSVs:
  `train_prompt_trace_rows.csv` and `heldout_prompt_trace_rows.csv`.
- Added regression coverage for the split helper and train-only sampling.

## Run

```text
runs/ohm_semdist_hooks4_shareout_streamb64_heldout20_src5000/2026-05-30_214753_463974_model-c-op0-19-fullgrid-streamb64-heldout0.2-gumbel_concrete_interface-result_space-inlr0.01-uplr0.0003-rbt1-zero_improvement-rbtt1-rbtchunk64-rbts24-rbtuniq-rbttopk8-rb-7006aed4f3/model-c-2digit-seed9
```

Settings:

- CLI seed `7`, effective seed `9`, op19 `400`-prompt grid.
- Four `left_operand_mod` routed hooks with shared output projection.
- `operand_spans` readout, span width `2`, product decoder, `n_embd=16`.
- Sparse zero-improvement online hard memory, topk8+unique24, freeze when full.
- Additive semantic distillation weight `1`, sample count `8`.
- Batch64 streaming source for `5000` steps.
- Heldout fraction `0.2`, yielding `320` train prompts and `80` heldout
  prompts.

## Results

- Overall final random eval: `325/400 = 0.8125`.
- Prompt memory entries: `320/320`, exactly the train pool.
- Cumulative forced-result evals: `87,552`.
- Train prompts: exact/calc `0.996875`, `319/320` correct.
- Train controls: injection-zero `0.046875`, forced-zero `0.003125`,
  forced-random `0.01875`.
- Heldout prompts: exact/calc `0.0875`, `7/80` correct.
- Heldout controls: injection-zero `0.0500`, forced-zero `0.0000`,
  forced-random `0.0125`.
- Heldout routed calculator accuracy: hook0 `0/23`, hook1 `5/23`, hook2
  `2/21`, hook3 `0/13`.

## Interpretation

This is a clean transductive-memory boundary. Prompt-keyed online hard memory
can solve prompts that get a stored target, but it does not learn a reusable
fresh-prompt calculator-query rule in this setup. The train split nearly
solved, the heldout split stayed near chance, and controls remained low.

Do not run the trusted handoff from this source as a thesis gate. The source
already fails the fresh-prompt requirement. The next high-leverage work needs a
non-transductive mechanism: amortized target discovery, a learned memory
initializer, fresh-prompt candidate scoring/proposal, or another answer-derived
credit signal that can supply targets for prompts not already in memory.

## Verification

```text
PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/overfit_one_batch.py
PYTHONPATH=. PYTHONPYCACHEPREFIX=/tmp/codex_pycache pytest tests/test_model.py -q -k "online_hard_result_boundary_memory or prompt_keyed_online_hard_memory or streaming_heldout_split"
```

The focused test run passed: `3 passed, 150 deselected`.
