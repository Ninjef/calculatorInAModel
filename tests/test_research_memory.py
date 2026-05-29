import json
import os
import subprocess
import sys

import numpy as np

from researchMemory.scripts.add_memory import main as add_memory_main
from researchMemory.scripts.memory_index import (
    build_index,
    build_records,
    search_index,
)


def write_memory(root, name="test-direction-memory.md"):
    memory_root = root / "researchMemory"
    memory_root.mkdir()
    path = memory_root / name
    path.write_text(
        """# Test Direction Memory

Status: active synthesis
Last updated: 2026-05-29

## Central Lesson

The test memory checks that retrieval works.

## Direction: Plain Answer-Loss Discovery

Status: paused without a new mechanism

Memory:

- REINFORCE failed because its gradient aligned with raw expected cost.
- Exact result-marginal answer loss did not fix calculator discovery.
- Decoder calibration improved local signs but collapsed in Stage 1.

Representative evidence:

- `aiAgentWorkHistory/phase7/example.md`

## Direction: Target Propagation

Status: active candidate

Memory:

- Target propagation changes the credit-assignment family.
- A first test should use a small feasibility gate before long training.

Representative evidence:

- `SOLUTION_IDEAS.md`
""",
        encoding="utf-8",
    )
    return memory_root


def test_build_records_extracts_direction_memories(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    memory_root = write_memory(tmp_path)

    records = build_records(tmp_path, memory_root)
    ids = {record.id for record in records}

    assert "doc/test-direction-memory" in ids
    assert "direction/plain-answer-loss-discovery" in ids
    assert "direction/target-propagation" in ids
    answer_loss = next(
        record for record in records if record.id == "direction/plain-answer-loss-discovery"
    )
    assert answer_loss.status == "paused without a new mechanism"
    assert "REINFORCE failed" in answer_loss.summary
    assert any(question.startswith("Why is") for question in answer_loss.questions)
    assert any(
        relation["type"] == "supported_by"
        and relation["target"] == "aiAgentWorkHistory/phase7/example.md"
        for relation in answer_loss.relations
    )


def test_build_index_writes_vectors_and_graph_then_searches(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    memory_root = write_memory(tmp_path)
    output_dir = memory_root / "index"

    build_index(tmp_path, memory_root, output_dir, dim=64)

    memories = [
        json.loads(line)
        for line in (output_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    embeddings = np.load(output_dir / "embeddings.npz", allow_pickle=True)
    graph = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))

    assert {memory["id"] for memory in memories} >= {
        "direction/plain-answer-loss-discovery",
        "direction/target-propagation",
    }
    assert embeddings["embeddings"].shape[1] == 64
    assert len(embeddings["memory_ids"]) >= len(memories)
    assert any(node["id"] == "direction/target-propagation" for node in graph["nodes"])
    node_ids = {node["id"] for node in graph["nodes"]}
    assert all(edge["source"] in node_ids for edge in graph["edges"])
    defined_in_edges = [edge for edge in graph["edges"] if edge["type"] == "defined_in"]
    assert len(defined_in_edges) == len(node_ids)
    assert all((tmp_path / edge["target"]).exists() for edge in defined_in_edges)
    assert any(
        edge["source"] == "direction/plain-answer-loss-discovery"
        and edge["type"] == "supported_by"
        for edge in graph["edges"]
    )

    results = search_index(
        output_dir,
        "Did REINFORCE fail because of sampling variance and expected answer loss?",
        top_k=2,
    )
    assert results[0]["record"]["id"] == "direction/plain-answer-loss-discovery"
    assert results[0]["score"] > 0.0


def test_search_cli_returns_relevant_memory(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    memory_root = write_memory(tmp_path)
    output_dir = memory_root / "index"
    build_index(tmp_path, memory_root, output_dir, dim=64)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "researchMemory.scripts.search_memory",
            "--index-dir",
            str(output_dir),
            "--top-k",
            "1",
            "Should we try target propagation local targets?",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Target Propagation" in completed.stdout
    assert "source:" in completed.stdout


def test_add_memory_scaffolds_compact_file(tmp_path):
    memory_root = tmp_path / "researchMemory"

    add_memory_main(
        [
            "New Direction",
            "--status",
            "candidate",
            "--date",
            "2026-05-29",
            "--memory-root",
            str(memory_root),
        ]
    )

    path = memory_root / "new-direction.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "# New Direction" in text
    assert "Status: candidate" in text
    assert "Representative evidence:" in text


def test_agent_docs_require_memory_search_and_size_discipline():
    claude = open("CLAUDE.md", encoding="utf-8").read()
    research_state = open("RESEARCH_STATE.md", encoding="utf-8").read()

    assert "python3 researchMemory/scripts/search_memory.py" in claude
    assert "must not become append-only logs" in claude
    assert "Keep `CLAUDE.md` under about `120` lines" in claude
    assert "Keep `RESEARCH_STATE.md` under about `200` lines" in claude
    assert len(claude.splitlines()) <= 120
    assert len(research_state.splitlines()) <= 200
