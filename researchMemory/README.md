# Research Memory

This directory stores direction-level memories.

Use it to make tested research directions easy to retrieve without rereading
every fact sheet, work-history entry, or hypothesis-ledger item.

## What Belongs Here

- One file per major research direction or phase-level cluster.
- Higher-level lessons that summarize many hypothesis-ledger entries.
- Pointers to representative evidence, not full experiment logs.
- Clear status labels: `active`, `active but constrained`, `paused`,
  `candidate`, or `archived`.

## What Does Not Belong Here

- Raw command output.
- Long chronological experiment logs.
- Every tiny hypothesis outcome.
- Replacement for `RESEARCH_STATE.md`.

## Update Rule

When a new experiment changes a direction-level conclusion:

1. Update the relevant memory by replacing stale summary text.
2. Add only the smallest source pointer needed.
3. Keep each memory file compact enough to skim quickly.
4. Keep ordinary memory files under about `250` lines. If a topic needs more,
   split by direction rather than appending indefinitely.
5. If a memory starts becoming chronological, move the details to
   `factSheets/` or `aiAgentWorkHistory/` and compress the memory back down.

Agents should read the relevant memory before proposing local variants in that
direction.

## Vector And Graph Index

The local index is rebuildable from these markdown memories:

```bash
python3 researchMemory/scripts/build_memory_index.py
```

Search it before proposing experiments:

```bash
python3 researchMemory/scripts/search_memory.py "Should we try more REINFORCE?"
```

The generated files live under `researchMemory/index/`:

- `memories.jsonl`: memory records with title, summary, generated questions,
  source pointer, status, and relations.
- `embeddings.npz`: deterministic local vector embeddings for titles,
  summaries, generated questions, and compact document text.
- `graph.json`: graph-like nodes and relation edges.

The embeddings are offline hashing embeddings by default so the guardrail is
testable without network access. They are a retrieval aid, not the source of
truth.

To scaffold a new memory file:

```bash
python3 researchMemory/scripts/add_memory.py "New Direction" --status candidate --date YYYY-MM-DD
```
