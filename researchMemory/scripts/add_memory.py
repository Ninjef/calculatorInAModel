"""Scaffold a compact research-memory markdown file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from researchMemory.scripts.memory_index import slugify


TEMPLATE = """# {title}

Status: {status}
Last updated: {date}

This file consolidates lessons for one research direction. Keep it compact and
replace stale synthesis instead of appending chronology.

## Central Lesson

- TODO: State the direction-level lesson in one or two bullets.

## Direction: {title}

Status: {status}

Memory:

- TODO: Summarize what is known.
- TODO: State what should not be repeated.
- TODO: State what would make this direction active or paused.

Representative evidence:

- TODO: Add the smallest useful source pointer.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scaffold a research-memory file.")
    parser.add_argument("title", help="memory title")
    parser.add_argument("--status", default="candidate")
    parser.add_argument("--date", default="YYYY-MM-DD")
    parser.add_argument("--memory-root", type=Path, default=Path("researchMemory"))
    parser.add_argument("--filename", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    memory_root = args.memory_root
    memory_root.mkdir(parents=True, exist_ok=True)
    filename = args.filename or f"{slugify(args.title)}.md"
    path = memory_root / filename
    if path.exists():
        raise SystemExit(f"memory file already exists: {path}")
    path.write_text(
        TEMPLATE.format(title=args.title, status=args.status, date=args.date),
        encoding="utf-8",
    )
    print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
