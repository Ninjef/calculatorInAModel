"""Generate full-text memory docs from HYPOTHESIS_LEDGER.md."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from researchMemory.scripts.memory_index import slugify


ENTRY_RE = re.compile(r"^([A-Z][A-Z-]*):\s+(.+)$")


@dataclass
class HypothesisEntry:
    phase: str
    label: str
    title: str
    lines: list[str]
    index: int


def read_entries(ledger_path: Path) -> list[HypothesisEntry]:
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    entries: list[HypothesisEntry] = []
    phase = "Unknown"
    current: HypothesisEntry | None = None
    index = 0
    for line in lines:
        if line.startswith("## "):
            phase = line[3:].strip()
            continue
        match = ENTRY_RE.match(line)
        if match:
            if current is not None:
                entries.append(current)
            index += 1
            current = HypothesisEntry(
                phase=phase,
                label=match.group(1),
                title=match.group(2).strip(),
                lines=[line],
                index=index,
            )
            continue
        if current is not None:
            current.lines.append(line)
    if current is not None:
        entries.append(current)
    return entries


def first_line_with_prefix(lines: list[str], prefix: str) -> str:
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def questions_for(entry: HypothesisEntry, conclusion: str) -> list[str]:
    title = entry.title.rstrip(".")
    questions = [
        f"What did we learn about {title}?",
        f"Has {title} been tested?",
        f"Should we repeat {title}?",
        f"What is the status of {title}?",
    ]
    lowered = f"{entry.label} {conclusion}".lower()
    if "disproven" in lowered or "negative" in lowered or "failed" in lowered:
        questions.append(f"Why did {title} fail?")
    if "positive" in lowered or "active" in lowered:
        questions.append(f"What follow-up is allowed for {title}?")
    return questions


def render_entry(entry: HypothesisEntry, ledger_path: Path) -> str:
    conclusion = first_line_with_prefix(entry.lines, "Conclusion")
    source = first_line_with_prefix(entry.lines, "Source")
    no_repeat = first_line_with_prefix(entry.lines, "Do not repeat")
    next_allowed = first_line_with_prefix(entry.lines, "Next allowed test")
    summary = conclusion or "See full entry."
    source_pointer = source.strip("`") if source else ledger_path.as_posix()
    questions = questions_for(entry, conclusion)
    full_text = "\n".join(entry.lines).strip()
    question_block = "\n".join(f"- {question}" for question in questions)
    return f"""# {entry.title}

Kind: hypothesis_memory
Status: {entry.label}
Phase: {entry.phase}
Source: {source_pointer}

Summary:

- {summary}

Questions:

{question_block}

Representative evidence:

- `{source_pointer}`
- `{ledger_path.as_posix()}`

Do Not Repeat:

- {no_repeat or "See full entry."}

Next Allowed:

- {next_allowed or "See full entry."}

Full Text:

```text
{full_text}
```
"""


def generate_docs(ledger_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.md"):
        old_file.unlink()
    written: list[Path] = []
    for entry in read_entries(ledger_path):
        filename = f"{entry.index:03d}-{slugify(entry.title)[:90]}.md"
        path = output_dir / filename
        path.write_text(render_entry(entry, ledger_path), encoding="utf-8")
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate hypothesis memory docs.")
    parser.add_argument("--ledger", type=Path, default=Path("HYPOTHESIS_LEDGER.md"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("researchMemory/hypotheses/phase7"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = generate_docs(args.ledger, args.output_dir)
    print(f"wrote {len(written)} hypothesis memory docs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
