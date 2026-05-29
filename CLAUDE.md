# Agent Entry Point

This repo is a research sandbox for one question:

Can a model learn to use a non-differentiable calculator embedded directly
inside its neural computation?

Everything else is subordinate to that question.

## Read Order

Before proposing or running an experiment, read these in order:

1. `RESEARCH_STATE.md` - current strategic synthesis and active bet.
2. `HYPOTHESIS_LEDGER.md` - compact anti-rerun ledger and paused families.
3. `researchMemory/` - direction-level memories for retrieval by topic.
4. The relevant phase fact sheet in `factSheets/` - archival detail only.
5. Recent matching work logs in `aiAgentWorkHistory/` - exact commands/results.

Do not derive strategy from chronology alone. Chronological logs answer what
happened; `RESEARCH_STATE.md` answers what it means now.

## File Roles

- `RESEARCH_STATE.md`: the current principal-investigator memo. It should stay
  short, current, and decision-oriented.
- `HYPOTHESIS_LEDGER.md`: tiny claims, outcomes, anti-rerun notes, and family
  status. It prevents local retesting.
- `researchMemory/`: consolidated memories by research direction. These are
  higher-level than the ledger and easier to retrieve by topic.
- `factSheets/`: durable experimental record by phase. These are archive and
  evidence, not the strategic compass.
- `aiAgentWorkHistory/`: lab notebook entries for completed work.
- `researchReviews/`: periodic zoom-out memos that decide whether the project
  is still pursuing the right class of experiments.
- `SOLUTION_IDEAS.md`: idea bank. Treat it as possibilities, not priorities.
- `OVERARCHING_EXPERIMENT_PURPOSE.md`: the high-level motivation.

## Strategic Guardrails

```bash
python3 researchMemory/scripts/serve_memory.py
python3 researchMemory/scripts/search_memory.py "<experiment idea or question>"
python3 researchMemory/scripts/build_memory_index.py
```

Before proposing an experiment, search the vector/graph memory index. Prefer
the warm local server during active work; use the one-shot search CLI when it is
not running. Rebuild with the default BGE local semantic backend when the index
is stale. The `hash backend` is only an offline fallback for tests/no-key
environments. Read top memories and evidence; paused directions require a new
mechanism and strategic updates.

Do not rediscover or present oracle calculator success as progress. Oracle
calculator outputs, oracle-at-eval recovery, injection-zero controls, and
forced-random controls are wiring checks only.

Do not rediscover generic retention-after-teaching as progress. Earlier phases
already showed that scaffolded or identifiable calculator protocols can often
be retained after the scaffold is removed. A retention experiment is only worth
running if it tests a new interface, objective, action parameterization, or
stability question.

Do not treat "next allowed test" as "next strategically valuable test." The
ledger may permit a local variant while `RESEARCH_STATE.md` says the whole
family is paused.

Do not continue a paused family without a new mechanism. A new seed, slightly
different weight, longer run, or cheaper proxy is not a new mechanism unless
`RESEARCH_STATE.md` explicitly says that family is active again.

If an experiment mainly reduces the cost of a known proxy or selector, first
ask whether selector-cost reduction is still the bottleneck. If not, write a
strategy review instead of running it.

## Document Growth Discipline

`CLAUDE.md` and `RESEARCH_STATE.md` must not become append-only logs.

- Keep `CLAUDE.md` under about `120` lines. It is an entrypoint and rule file.
- Keep `RESEARCH_STATE.md` under about `200` lines. It is a current synthesis.
- Update these files by replacing stale text, not by adding another historical
  layer.
- Move old strategic context into `researchReviews/`, topic synthesis into
  `researchMemory/`, and raw detail into `factSheets/` or
  `aiAgentWorkHistory/`.
- If updating `RESEARCH_STATE.md` would push it over the line budget, first
  delete or compress lower-value material.

## Current Bottom Line

The architecture can use a calculator, and scaffolded calculator-use policies
can be learned, retained, and transferred into a non-bottleneck model. The
unsolved problem is scalable, non-prescriptive credit assignment into the
calculator-query policy. Read `RESEARCH_STATE.md` before acting.

## Contribution Rules

When doing research work:

- Update `RESEARCH_STATE.md` only when the strategic picture changes.
- Update `HYPOTHESIS_LEDGER.md` for each durable hypothesis outcome.
- Update `researchMemory/` when a cluster of hypotheses should be remembered
  as a direction-level lesson.
- For every tested hypothesis, add/update a hypothesis memory document and
  rebuild the semantic index.
- Update the relevant phase fact sheet with durable experimental evidence.
- Add a concise work-history entry under `aiAgentWorkHistory/`.
- Add or update a `researchReviews/` memo after a major branch outcome, after
  about 5-10 experiments, or whenever the next step is unclear.
- Move completed task files to the completed folder when applicable.
- Commit and push completed work.

If your work cannot be summarized as a change to either the strategic state or
the hypothesis ledger, it may be too local to run.
