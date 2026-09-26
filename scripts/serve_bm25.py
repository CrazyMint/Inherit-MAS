"""Serve the pinned BM25 retriever to external baseline environments."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Handler(BaseHTTPRequestHandler):
    examples: dict
    retriever: object

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=True, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.reply(404, {"error": "not_found"})
            return
        self.reply(200, {"ok": True, "retriever_digest": self.retriever.fingerprint_digest,
                         "n_tasks": len(self.examples)})

    def do_POST(self):
        if self.path != "/search":
            self.reply(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid request size")
            request = json.loads(self.rfile.read(length))
            example = self.examples[request["task_id"]]
            result = dict(self.retriever.search(str(request["query"]), example))
            result.pop("wall_s", None)
            result.pop("cache_hit", None)
            self.reply(200, result)
        except Exception as exc:
            self.reply(400, {"error": type(exc).__name__, "detail": str(exc)[:300]})

    def log_message(self, *args):
        pass


def main():
    from public_runner.common import read_manifest
    from inherit_mas.release_config import ensure_hf_home
    from hotpot_fullwiki.loader import load_examples, open_snapshot
    from hotpot_fullwiki.retrieval import BM25Retriever
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/hotpotqa.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/bm25-cache")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ensure_hf_home()
    manifest = read_manifest(args.manifest, "hotpotqa")
    snapshot = open_snapshot()
    if snapshot.digest != manifest["dataset_revision"]:
        parser.error("dataset/manifest mismatch")
    Handler.examples = {item.id: item.runtime() for item in load_examples(manifest["rows"], snapshot)}
    Handler.retriever = BM25Retriever(cache_dir=args.cache_dir)
    first = next(iter(Handler.examples.values()))
    Handler.retriever.preflight(first.question, first)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps({"status": "ready", "port": args.port,
                      "retriever_digest": Handler.retriever.fingerprint_digest}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
