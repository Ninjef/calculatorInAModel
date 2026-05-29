"""Search research memory through the warm localhost server, with fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from researchMemory.scripts.memory_index import (
    default_paths,
    format_search_results,
    repo_root_from,
    search_index,
)
from researchMemory.scripts.serve_memory import DEFAULT_HOST, DEFAULT_PORT


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast research-memory search via warm server.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    default_server_url = os.environ.get(
        "RESEARCH_MEMORY_SERVER_URL",
        f"http://{DEFAULT_HOST}:{DEFAULT_PORT}",
    )
    parser.add_argument(
        "--server-url",
        default=default_server_url,
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def server_search(server_url: str, query: str, top_k: int, timeout: float) -> dict[str, object]:
    params = urllib.parse.urlencode({"q": query, "top_k": top_k})
    request_url = server_url.rstrip("/") + "/search?" + params
    with urllib.request.urlopen(request_url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fallback_search(args: argparse.Namespace) -> list[dict[str, object]]:
    repo_root = repo_root_from(args.repo_root)
    _, output_dir = default_paths(repo_root)
    index_dir = (args.index_dir or output_dir).resolve()
    return search_index(index_dir, args.query, top_k=args.top_k)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        payload = server_search(args.server_url, args.query, args.top_k, args.timeout)
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "server returned ok=false")))
        results = payload["results"]
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(format_search_results(results))
            print(f"\nserved_by: {args.server_url} duration_ms={payload.get('duration_ms')}")
        return 0
    except (OSError, RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        if args.no_fallback:
            print(f"research memory server unavailable: {exc}", file=sys.stderr)
            return 2
        print(
            f"research memory server unavailable ({exc}); falling back to one-shot search",
            file=sys.stderr,
        )
        results = fallback_search(args)
        if args.json:
            print(
                json.dumps(
                    {"ok": True, "fallback": True, "results": results},
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(format_search_results(results))
            print("\nserved_by: one-shot fallback")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
