"""Serve warm research-memory semantic search over localhost."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from researchMemory.scripts.memory_index import default_paths, repo_root_from, search_index


LOGGER = logging.getLogger("research-memory-server")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_top_k(values: list[str] | None) -> int:
    if not values:
        return 5
    try:
        return max(1, min(50, int(values[0])))
    except ValueError:
        return 5


def search_payload(index_dir: Path, query: str, top_k: int, allow_download: bool) -> dict[str, Any]:
    started = time.perf_counter()
    LOGGER.info("search start top_k=%s query=%r", top_k, query)
    results = search_index(
        index_dir,
        query,
        top_k=top_k,
        allow_download=allow_download,
    )
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    LOGGER.info("search done results=%s duration_ms=%s", len(results), duration_ms)
    return {
        "ok": True,
        "duration_ms": duration_ms,
        "query": query,
        "top_k": top_k,
        "results": results,
    }


class MemorySearchHandler(BaseHTTPRequestHandler):
    server_version = "ResearchMemoryHTTP/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.client_address[0], format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        started = time.perf_counter()
        if parsed.path == "/health":
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "index_dir": str(self.server.index_dir),
                    "allow_download": bool(self.server.allow_download),
                },
            )
            return
        if parsed.path != "/search":
            json_response(self, 404, {"ok": False, "error": "unknown endpoint"})
            return

        params = parse_qs(parsed.query)
        query = (params.get("q") or [""])[0].strip()
        if not query:
            json_response(self, 400, {"ok": False, "error": "missing q parameter"})
            return

        top_k = parse_top_k(params.get("top_k"))
        try:
            payload = search_payload(
                self.server.index_dir,
                query,
                top_k,
                self.server.allow_download,
            )
        except Exception as exc:
            LOGGER.exception("search failed query=%r", query)
            json_response(self, 500, {"ok": False, "error": str(exc)})
            return

        payload["http_duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        json_response(self, 200, payload)


class MemorySearchServer(ThreadingHTTPServer):
    allow_reuse_address = True
    index_dir: Path
    allow_download: bool


def warm_search(index_dir: Path, allow_download: bool) -> None:
    LOGGER.info("warming model and index index_dir=%s", index_dir)
    started = time.perf_counter()
    search_index(index_dir, "warm research memory query", top_k=1, allow_download=allow_download)
    duration = time.perf_counter() - started
    LOGGER.info("warmup complete duration_s=%.2f", duration)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve warm research-memory search.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    repo_root = repo_root_from(args.repo_root)
    _, output_dir = default_paths(repo_root)
    index_dir = (args.index_dir or output_dir).resolve()

    if not (index_dir / "embeddings.npz").exists():
        LOGGER.error("missing index at %s; run build_memory_index.py first", index_dir)
        return 2
    server = MemorySearchServer((args.host, args.port), MemorySearchHandler)
    server.index_dir = index_dir
    server.allow_download = args.allow_download
    if not args.no_warm:
        try:
            warm_search(index_dir, allow_download=args.allow_download)
        except Exception:
            server.server_close()
            raise
    LOGGER.info("serving research memory search url=http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("shutdown requested")
    finally:
        server.server_close()
        LOGGER.info("server stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
