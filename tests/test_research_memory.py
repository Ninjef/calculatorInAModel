import json
import os
import subprocess
import sys

import numpy as np

from researchMemory.scripts.add_memory import main as add_memory_main
from researchMemory.scripts.generate_hypothesis_memories import generate_docs
from researchMemory.scripts.memory_index import (
    build_index,
    build_records,
    embed_query,
    local_texts_for_backend,
    search_index,
)
from researchMemory.scripts.serve_memory import search_payload


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

    build_index(tmp_path, memory_root, output_dir, dim=64, backend="hash")

    memories = [
        json.loads(line)
        for line in (output_dir / "memories.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    embeddings = np.load(output_dir / "embeddings.npz", allow_pickle=True)
    graph = json.loads((output_dir / "graph.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert {memory["id"] for memory in memories} >= {
        "direction/plain-answer-loss-discovery",
        "direction/target-propagation",
    }
    assert embeddings["embeddings"].shape[1] == 64
    assert metadata["backend"] == "hash"
    assert metadata["model"] == "hashing-vectorizer"
    assert metadata["view_count"] == len(embeddings["memory_ids"])
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
    build_index(tmp_path, memory_root, output_dir, dim=64, backend="hash")

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


def test_memory_search_server_payload_uses_shared_search_path(tmp_path, monkeypatch):
    output_dir = tmp_path / "index"
    output_dir.mkdir()
    (output_dir / "embeddings.npz").write_text("placeholder", encoding="utf-8")
    calls = []

    def fake_search(index_dir, query, top_k=5, allow_download=False, **kwargs):
        calls.append((index_dir, query, top_k, allow_download, kwargs))
        return [
            {
                "score": 0.9,
                "matched_view": "summary",
                "record": {
                    "id": "doc/test",
                    "title": "Test Memory",
                    "status": "active",
                    "source_path": "researchMemory/test.md",
                    "source_anchor": "#test-memory",
                    "summary": "A warm server can answer search requests.",
                },
            }
        ]

    monkeypatch.setattr("researchMemory.scripts.serve_memory.search_index", fake_search)
    payload = search_payload(output_dir, "warm memory", 1, allow_download=False)

    assert payload["ok"] is True
    assert payload["results"][0]["record"]["title"] == "Test Memory"
    assert payload["duration_ms"] >= 0
    assert calls == [(output_dir, "warm memory", 1, False, {})]


def test_search_uses_semantic_backend_from_metadata(tmp_path, monkeypatch):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    memory_root = write_memory(tmp_path)
    output_dir = memory_root / "index"

    calls = []

    def fake_openai(texts, model, dimensions=None, batch_size=128):
        calls.append((list(texts), model, dimensions, batch_size))
        rows = []
        for text in texts:
            lowered = text.lower()
            if lowered.startswith("# test direction memory"):
                rows.append([0.0, 0.0, 1.0])
            elif "target propagation" in lowered or "local targets" in lowered:
                rows.append([1.0, 0.0, 0.0])
            elif "reinforce" in lowered or "answer loss" in lowered:
                rows.append([0.0, 1.0, 0.0])
            else:
                rows.append([0.0, 0.0, 1.0])
        return np.array(rows, dtype=np.float32)

    monkeypatch.setattr(
        "researchMemory.scripts.memory_index.embed_texts_openai",
        fake_openai,
    )

    build_index(
        tmp_path,
        memory_root,
        output_dir,
        backend="openai",
        model="text-embedding-3-small",
        openai_dimensions=3,
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "openai"
    assert metadata["model"] == "text-embedding-3-small"
    assert metadata["openai_dimensions"] == 3

    results = search_index(
        output_dir,
        "Could local targets help credit assignment?",
        top_k=1,
    )

    assert results[0]["record"]["id"] == "direction/target-propagation"
    assert len(calls) >= 2


def test_sentence_transformers_backend_uses_bge_query_prefix(tmp_path, monkeypatch):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    memory_root = write_memory(tmp_path)
    output_dir = memory_root / "index"
    calls = []

    def fake_local(texts, model, query=False, local_files_only=False, batch_size=32):
        calls.append((list(texts), model, query, local_files_only, batch_size))
        rows = []
        for text in texts:
            lowered = text.lower()
            if lowered.startswith("# test direction memory"):
                rows.append([0.0, 1.0])
            elif "target propagation" in lowered or "local targets" in lowered:
                rows.append([1.0, 0.0])
            else:
                rows.append([0.0, 1.0])
        return np.array(rows, dtype=np.float32)

    monkeypatch.setattr(
        "researchMemory.scripts.memory_index.embed_texts_sentence_transformers",
        fake_local,
    )

    build_index(
        tmp_path,
        memory_root,
        output_dir,
        backend="sentence-transformers",
        model="BAAI/bge-small-en-v1.5",
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backend"] == "sentence-transformers"
    assert metadata["model"] == "BAAI/bge-small-en-v1.5"

    results = search_index(output_dir, "Could local targets help?", top_k=1)

    assert results[0]["record"]["id"] == "direction/target-propagation"
    assert any(call[2] is False for call in calls)
    assert any(call[2] is True for call in calls)
    assert any(call[3] is True for call in calls)
    assert local_texts_for_backend(
        ["Could local targets help?"],
        model="BAAI/bge-small-en-v1.5",
        query=True,
    )[0].startswith("Represent this sentence for searching")


def test_openai_backend_requires_api_key_without_mock(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        embed_query(
            "semantic query",
            backend="openai",
            dim=3,
            model="text-embedding-3-small",
            openai_dimensions=3,
        )
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("OpenAI backend should require OPENAI_API_KEY")


def test_generate_hypothesis_memory_docs_and_index(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("test", encoding="utf-8")
    ledger = tmp_path / "HYPOTHESIS_LEDGER.md"
    ledger.write_text(
        """# Hypothesis Ledger

## Phase 7

DISPROVEN: Decoder calibration alone rescues ordinary expected-cost discovery.
Conclusion: Contrastive-margin decoder passed local sign alignment, then Stage 1 collapsed.
Do not repeat: Decoder-only sharpening/calibration.
Next allowed test: Try a stronger backward channel.
Source: `aiAgentWorkHistory/phase7/decoder.md`

POSITIVE: A 600-step handoff probe can select a better source checkpoint than source accuracy.
Conclusion: The handoff probe selected the better checkpoint.
Do not repeat: Same source handoff comparison.
Next allowed test: Use 600-step handoff probes on new source checkpoints.
Source: `aiAgentWorkHistory/phase7/handoff.md`
""",
        encoding="utf-8",
    )
    memory_root = tmp_path / "researchMemory"
    output_dir = memory_root / "hypotheses" / "phase7"

    written = generate_docs(ledger, output_dir)

    assert len(written) == 2
    text = written[0].read_text(encoding="utf-8")
    assert "Kind: hypothesis_memory" in text
    assert "Questions:" in text
    assert "Full Text:" in text

    records = build_records(tmp_path, memory_root)
    hypothesis_records = [record for record in records if record.kind == "hypothesis_memory"]
    assert len(hypothesis_records) == 2
    assert any("Decoder calibration" in record.title for record in hypothesis_records)
    decoder = next(record for record in hypothesis_records if "Decoder calibration" in record.title)
    assert any("Why did" in question for question in decoder.questions)


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
    assert "python3 researchMemory/scripts/serve_memory.py" in claude
    assert "BGE local semantic backend" in claude
    assert "hash backend" in claude
    assert "must not become append-only logs" in claude
    assert "Keep `CLAUDE.md` under about `120` lines" in claude
    assert "Keep `RESEARCH_STATE.md` under about `200` lines" in claude
    assert len(claude.splitlines()) <= 120
    assert len(research_state.splitlines()) <= 200
