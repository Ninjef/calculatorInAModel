"""CLI wrapper for rebuilding the research-memory index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from researchMemory.scripts.memory_index import main


if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))
