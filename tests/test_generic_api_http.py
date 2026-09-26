"""Exercise the real SDK against a loopback server, never a model provider."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def compatible_api(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            response = {"id": "local-test", "object": "chat.completion", "created": 0,
                "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    document = {"api": {"base_url": base + "/v1", "models": {
        "gpt-4o-mini": {"model": "worker-name", "api_key_env": "LOCAL_WORKER_KEY", "base_url": base + "/worker/v1"},
        "gpt-5.4-mini": {"model": "controller-name", "api_key_env": "LOCAL_META_KEY", "base_url": base + "/meta/v1"}}}}
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv("INHERIT_MAS_CREDENTIALS", str(path))
    monkeypatch.setenv("LOCAL_WORKER_KEY", "worker-test-value")
    monkeypatch.setenv("LOCAL_META_KEY", "meta-test-value")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    try:
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class Ledger:
    def reserve(self, *args):
        return 0

    def append(self, *args, **kwargs):
        pass


@pytest.mark.parametrize("runtime", ["workbench", "hotpotqa", "baselines"])
def test_generic_credentials_make_correct_sdk_requests(compatible_api, runtime):
    if runtime == "workbench":
        from workbench.task_evolution.models import APIModels
        models = APIModels()
    elif runtime == "hotpotqa":
        from hotpot_fullwiki.models import Models
        models = Models.api(ledger=None)
    else:
        from public_runner.models import make_models
        models = make_models(Ledger())
    try:
        for model in ("gpt-4o-mini", "gpt-5.4-mini"):
            result = models.chat(model=model, role="configuration-test", system="test", user="test", max_tokens=8)
            assert result.text == "ok"
        worker, meta = compatible_api
        assert worker["path"] == "/worker/v1/chat/completions"
        assert worker["auth"] == "Bearer worker-test-value"
        assert worker["body"]["model"] == "worker-name"
        assert worker["body"]["max_tokens"] == 8
        assert meta["path"] == "/meta/v1/chat/completions"
        assert meta["auth"] == "Bearer meta-test-value"
        assert meta["body"]["model"] == "controller-name"
        assert meta["body"]["max_completion_tokens"] == 8
        assert all(request["body"]["temperature"] == 0 for request in compatible_api)
    finally:
        if runtime == "baselines":
            for clients in models._clients.values():
                for client in clients:
                    client.close()
        else:
            models.worker.close()
            models.meta.close()
