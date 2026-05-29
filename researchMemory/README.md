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
