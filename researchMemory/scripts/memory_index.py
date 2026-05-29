"""Build and query the local research-memory index.

The index is intentionally rebuildable from markdown source files. The default
backend is deterministic and offline for tests, but real semantic retrieval is
available with the OpenAI embeddings backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_DIM = 256
DEFAULT_BACKEND = "hash"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
INDEX_DIRNAME = "index"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-/]*")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class MemoryRecord:
    id: str
    title: str
    kind: str
    status: str
    source_path: str
    source_anchor: str
    summary: str
    questions: list[str] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    text: str = ""

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "source_path": self.source_path,
            "source_anchor": self.source_anchor,
            "summary": self.summary,
            "questions": self.questions,
            "relations": self.relations,
            "tags": self.tags,
            "text": self.text,
        }


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "memory"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed_text_hash(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Return a deterministic signed hashing embedding."""

    vector = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    if not tokens:
        return vector
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def normalize_vector(vector: Iterable[float]) -> np.ndarray:
    array = np.array(list(vector), dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm > 0.0:
        array /= norm
    return array


def embed_texts_hash(texts: list[str], dim: int = DEFAULT_DIM) -> np.ndarray:
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([embed_text_hash(text, dim=dim) for text in texts]).astype(np.float32)


def openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for semantic OpenAI embeddings. "
            "Use --backend hash only for offline tests/fallback."
        )
    return api_key


def embed_texts_openai(
    texts: list[str],
    model: str = DEFAULT_OPENAI_MODEL,
    dimensions: int | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    """Embed text with the OpenAI embeddings API.

    Uses only the standard library so the repo does not need a new dependency.
    """

    if not texts:
        return np.zeros((0, dimensions or 0), dtype=np.float32)
    api_key = openai_api_key()
    endpoint = "https://api.openai.com/v1/embeddings"
    embeddings: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload: dict[str, object] = {"model": model, "input": batch}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI embeddings request failed: {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc
        batch_data = sorted(response_data["data"], key=lambda item: item["index"])
        embeddings.extend(normalize_vector(item["embedding"]) for item in batch_data)
    return np.vstack(embeddings).astype(np.float32)


def embed_texts(
    texts: list[str],
    *,
    backend: str = DEFAULT_BACKEND,
    dim: int = DEFAULT_DIM,
    model: str = DEFAULT_OPENAI_MODEL,
    openai_dimensions: int | None = None,
) -> np.ndarray:
    if backend == "hash":
        return embed_texts_hash(texts, dim=dim)
    if backend == "openai":
        return embed_texts_openai(texts, model=model, dimensions=openai_dimensions)
    raise ValueError(f"unknown embedding backend: {backend}")


def embed_query(
    query: str,
    *,
    backend: str,
    dim: int,
    model: str,
    openai_dimensions: int | None,
) -> np.ndarray:
    return embed_texts(
        [query],
        backend=backend,
        dim=dim,
        model=model,
        openai_dimensions=openai_dimensions,
    )[0]


def cosine_scores(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return np.array([], dtype=np.float32)
    return embeddings @ query_embedding


def markdown_anchor(title: str) -> str:
    return "#" + slugify(title)


def split_sections(markdown: str) -> list[tuple[int, str, list[str]]]:
    """Split markdown into heading sections.

    Returns `(level, title, lines)` for each heading and its body.
    """

    sections: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None
    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if current is not None:
                sections.append(current)
            current = (len(match.group(1)), match.group(2).strip(), [])
        elif current is not None:
            current[2].append(line)
    if current is not None:
        sections.append(current)
    return sections


def extract_status(lines: Iterable[str]) -> str:
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("status:"):
            return stripped.split(":", 1)[1].strip() or "unknown"
    return "unknown"


def extract_named_block(lines: list[str], heading: str) -> list[str]:
    start = None
    target = heading.lower()
    for idx, line in enumerate(lines):
        if line.strip().lower() == target:
            start = idx + 1
            break
    if start is None:
        return []
    block: list[str] = []
    for line in lines[start:]:
        if line.startswith("## ") or line.strip().endswith(":") and block:
            break
        block.append(line)
    return [line for line in block if line.strip()]


def compact_summary(lines: list[str], fallback_lines: list[str], max_chars: int = 700) -> str:
    source = lines or fallback_lines
    parts: list[str] = []
    in_code = False
    for line in source:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if stripped.lower().startswith(("status:", "representative evidence:")):
            continue
        parts.append(stripped)
        if sum(len(part) for part in parts) > max_chars:
            break
    summary = " ".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def extract_relations(lines: list[str], source_path: str) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    in_evidence = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "representative evidence:":
            in_evidence = True
            continue
        if in_evidence and stripped.startswith("## "):
            break
        if in_evidence and stripped.startswith("- "):
            target = stripped[2:].strip().strip("`")
            code_match = re.search(r"`([^`]+)`", stripped)
            if code_match:
                target = code_match.group(1).strip()
            if target:
                relations.append({"type": "supported_by", "target": target})
    relations.append({"type": "defined_in", "target": source_path})
    return relations


def generate_questions(title: str, status: str, summary: str) -> list[str]:
    clean_title = title.replace("Direction:", "").strip()
    questions = [
        f"What have we learned about {clean_title}?",
        f"Should we continue work on {clean_title}?",
        f"What is the status of {clean_title}?",
    ]
    if "paused" in status.lower():
        questions.append(f"Why is {clean_title} paused?")
    if "active" in status.lower():
        questions.append(f"What is the next useful work for {clean_title}?")
    if "not" in summary.lower() or "failed" in summary.lower():
        questions.append(f"What failed in {clean_title}?")
    deduped: list[str] = []
    for question in questions:
        if question not in deduped:
            deduped.append(question)
    return deduped


def parse_memory_markdown(path: Path, repo_root: Path) -> list[MemoryRecord]:
    relative_path = path.relative_to(repo_root).as_posix()
    markdown = path.read_text(encoding="utf-8")
    sections = split_sections(markdown)
    records: list[MemoryRecord] = []

    doc_title = sections[0][1] if sections else path.stem
    doc_status = extract_status(markdown.splitlines())
    doc_summary = compact_summary([], markdown.splitlines())
    records.append(
        MemoryRecord(
            id=f"doc/{slugify(path.stem)}",
            title=doc_title,
            kind="memory_document",
            status=doc_status,
            source_path=relative_path,
            source_anchor=markdown_anchor(doc_title),
            summary=doc_summary,
            questions=generate_questions(doc_title, doc_status, doc_summary),
            relations=[{"type": "defined_in", "target": relative_path}],
            tags=[slugify(path.stem)],
            text=markdown,
        )
    )

    for level, title, lines in sections:
        if level != 2 or not title.lower().startswith("direction:"):
            continue
        clean_title = title.split(":", 1)[1].strip()
        status = extract_status(lines)
        memory_lines = extract_named_block(lines, "memory:")
        summary = compact_summary(memory_lines, lines)
        tags = sorted(set(tokenize(clean_title)[:8]))
        records.append(
            MemoryRecord(
                id=f"direction/{slugify(clean_title)}",
                title=clean_title,
                kind="direction_memory",
                status=status,
                source_path=relative_path,
                source_anchor=markdown_anchor(title),
                summary=summary,
                questions=generate_questions(clean_title, status, summary),
                relations=extract_relations(lines, relative_path),
                tags=tags,
                text="\n".join(lines).strip(),
            )
        )
    return records


def discover_memory_files(memory_root: Path) -> list[Path]:
    return sorted(
        path
        for path in memory_root.glob("*.md")
        if path.name.lower() != "readme.md" and path.is_file()
    )


def build_records(repo_root: Path, memory_root: Path) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for path in discover_memory_files(memory_root):
        records.extend(parse_memory_markdown(path, repo_root))
    seen: set[str] = set()
    unique: list[MemoryRecord] = []
    for record in records:
        if record.id in seen:
            raise ValueError(f"duplicate memory id: {record.id}")
        seen.add(record.id)
        unique.append(record)
    return unique


def view_texts(record: MemoryRecord) -> list[tuple[str, str]]:
    views = [
        ("title", record.title),
        ("summary", record.summary),
    ]
    if record.text and len(record.text) < 6000:
        views.append(("document", record.text))
    for idx, question in enumerate(record.questions):
        views.append((f"question:{idx}", question))
    return [(kind, text) for kind, text in views if text.strip()]


def build_index(
    repo_root: Path,
    memory_root: Path,
    output_dir: Path,
    dim: int = DEFAULT_DIM,
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_OPENAI_MODEL,
    openai_dimensions: int | None = None,
) -> None:
    records = build_records(repo_root, memory_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    memories_path = output_dir / "memories.jsonl"
    with memories_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")

    view_memory_ids: list[str] = []
    view_kinds: list[str] = []
    view_strings: list[str] = []
    for record in records:
        for kind, text in view_texts(record):
            view_memory_ids.append(record.id)
            view_kinds.append(kind)
            view_strings.append(text)
    embedding_matrix = embed_texts(
        view_strings,
        backend=backend,
        dim=dim,
        model=model,
        openai_dimensions=openai_dimensions,
    )
    actual_dim = int(embedding_matrix.shape[1]) if embedding_matrix.size else dim
    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=embedding_matrix,
        memory_ids=np.array(view_memory_ids, dtype=object),
        view_kinds=np.array(view_kinds, dtype=object),
        dim=np.array([actual_dim], dtype=np.int32),
    )
    metadata = {
        "backend": backend,
        "model": model if backend == "openai" else "hashing-vectorizer",
        "dim": actual_dim,
        "openai_dimensions": openai_dimensions,
        "record_count": len(records),
        "view_count": len(view_strings),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    graph = {
        "nodes": [
            {
                "id": record.id,
                "title": record.title,
                "kind": record.kind,
                "status": record.status,
                "source_path": record.source_path,
            }
            for record in records
        ],
        "edges": [
            {"source": record.id, **relation}
            for record in records
            for relation in record.relations
        ],
    }
    (output_dir / "graph.json").write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_records(index_dir: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    with (index_dir / "memories.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record["id"])] = record
    return records


def load_metadata(index_dir: Path) -> dict[str, object]:
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        return {
            "backend": DEFAULT_BACKEND,
            "model": "hashing-vectorizer",
            "dim": DEFAULT_DIM,
            "openai_dimensions": None,
        }
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def search_index(
    index_dir: Path,
    query: str,
    top_k: int = 5,
    backend: str | None = None,
    model: str | None = None,
) -> list[dict[str, object]]:
    records = load_records(index_dir)
    metadata = load_metadata(index_dir)
    data = np.load(index_dir / "embeddings.npz", allow_pickle=True)
    embeddings = data["embeddings"]
    memory_ids = data["memory_ids"].tolist()
    view_kinds = data["view_kinds"].tolist()
    dim = int(metadata.get("dim") or (data["dim"][0] if "dim" in data else DEFAULT_DIM))
    selected_backend = backend or str(metadata.get("backend", DEFAULT_BACKEND))
    selected_model = model or str(metadata.get("model", DEFAULT_OPENAI_MODEL))
    if selected_model == "hashing-vectorizer":
        selected_model = DEFAULT_OPENAI_MODEL
    openai_dimensions = metadata.get("openai_dimensions")
    openai_dimensions = int(openai_dimensions) if openai_dimensions is not None else None

    query_embedding = embed_query(
        query,
        backend=selected_backend,
        dim=dim,
        model=selected_model,
        openai_dimensions=openai_dimensions,
    )
    scores = cosine_scores(query_embedding, embeddings)
    best_by_memory: dict[str, dict[str, object]] = {}
    for score, memory_id, view_kind in zip(scores.tolist(), memory_ids, view_kinds):
        existing = best_by_memory.get(memory_id)
        if existing is None or float(score) > float(existing["score"]):
            record = records[memory_id]
            best_by_memory[memory_id] = {
                "score": float(score),
                "matched_view": view_kind,
                "record": record,
            }
    results = sorted(best_by_memory.values(), key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def format_search_results(results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        record = result["record"]
        assert isinstance(record, dict)
        lines.append(
            f"{idx}. {record['title']} [{record['status']}] "
            f"score={float(result['score']):.3f} view={result['matched_view']}"
        )
        lines.append(f"   id: {record['id']}")
        lines.append(f"   source: {record['source_path']}{record['source_anchor']}")
        summary = str(record.get("summary", ""))
        if summary:
            lines.append(f"   summary: {summary}")
    return "\n".join(lines)


def repo_root_from(path: Path | None = None) -> Path:
    current = (path or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "CLAUDE.md").exists() and (candidate / "researchMemory").exists():
            return candidate
    return current


def default_paths(repo_root: Path) -> tuple[Path, Path]:
    memory_root = repo_root / "researchMemory"
    output_dir = memory_root / INDEX_DIRNAME
    return memory_root, output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/search local research memory.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="rebuild the memory index")
    build.add_argument("--repo-root", type=Path, default=None)
    build.add_argument("--memory-root", type=Path, default=None)
    build.add_argument("--output-dir", type=Path, default=None)
    build.add_argument("--dim", type=int, default=DEFAULT_DIM)
    build.add_argument("--backend", choices=["hash", "openai"], default=DEFAULT_BACKEND)
    build.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    build.add_argument("--openai-dimensions", type=int, default=None)

    search = subparsers.add_parser("search", help="search the memory index")
    search.add_argument("query")
    search.add_argument("--repo-root", type=Path, default=None)
    search.add_argument("--index-dir", type=Path, default=None)
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--backend", choices=["hash", "openai"], default=None)
    search.add_argument("--model", default=None)
    search.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = repo_root_from(args.repo_root)
    memory_root, output_dir = default_paths(repo_root)

    if args.command == "build":
        build_index(
            repo_root=repo_root,
            memory_root=(args.memory_root or memory_root).resolve(),
            output_dir=(args.output_dir or output_dir).resolve(),
            dim=args.dim,
            backend=args.backend,
            model=args.model,
            openai_dimensions=args.openai_dimensions,
        )
        return 0
    if args.command == "search":
        index_dir = (args.index_dir or output_dir).resolve()
        results = search_index(
            index_dir=index_dir,
            query=args.query,
            top_k=args.top_k,
            backend=args.backend,
            model=args.model,
        )
        if args.json:
            print(json.dumps(results, indent=2, sort_keys=True))
        else:
            print(format_search_results(results))
        return 0
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
