"""A fake llama-server that sleeps: streaming responses, with timings on the last data chunk.

Fault injection: assign `fault` a function `(request body, index of this request) -> dict | None`; the return value is one of:
  {"kind": "http", "status": 503}           ordinary HTTP error
  {"kind": "ctx_exceeded"}                  400 exceed_context_size_error (a single request exceeds the slot capacity)
  {"kind": "pool_exhausted"}                500 "Context size has been exceeded." (shared pool exhausted)
  {"kind": "slow", "ttft_s": 2.0}           sleep before the first token (simulates a long prefill; client disconnects are not detected meanwhile)
  {"kind": "disconnect", "after_chunks": 3} disconnect right after sending some data chunks, without sending [DONE]
  {"kind": "stream_pool_exhausted", "after_chunks": 3}  pool exhausted after chunks were already sent: HTTP 200 + SSE error event
  {"kind": "garbage", "after_chunks": 3}    send a line of bad JSON after some data chunks
  {"kind": "no_key"}                        completes normally, but the answer does not contain the key (simulates the model failing to repeat it)
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_tokenizer import FakeTokenizer

NEEDLE_RE = re.compile(r"KEY-[a-z0-9]{8}")
ERRORS = {
    "http": lambda f: (f["status"], {"error": {"code": f["status"], "message": "injected", "type": "server_error"}}),
    "ctx_exceeded": lambda f: (400, {"error": {"code": 400, "type": "exceed_context_size_error",
                                               "message": "request exceeds the available context size",
                                               "n_prompt_tokens": 5000, "n_ctx": 4096}}),
    "pool_exhausted": lambda f: (500, {"error": {"code": 500, "type": "server_error",
                                                  "message": "Context size has been exceeded."}}),
}


class FakeServer:
    def __init__(self, ttft_s: float = 0.05, chunk_interval_s: float = 0.001,
                 health_delay_s: float = 0.0, n_slots: int = 4):
        self.ttft_s, self.chunk_interval_s, self.n_slots = ttft_s, chunk_interval_s, n_slots
        self.fault = None
        self.max_in_flight = 0
        self.deferred = 0  # requests_deferred reported in /metrics; tests assign it directly
        self.aborted = 0  # number of requests where the client disconnected midway, noticed only on write-back
        self._in_flight = 0
        self._lock = threading.Lock()
        self.ready_at = time.monotonic() + health_delay_s
        self.requests: list[dict] = []  # chat request bodies received, for test assertions
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, status, obj):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    ready = time.monotonic() >= server.ready_at
                    self._json(200 if ready else 503, {"status": "ok" if ready else "loading model"})
                elif self.path == "/slots":
                    busy = server._in_flight
                    self._json(200, [{"id": i, "is_processing": i < busy} for i in range(server.n_slots)])
                elif self.path == "/metrics":
                    body = (f"# TYPE llamacpp:requests_processing gauge\n"
                            f"llamacpp:requests_processing {server._in_flight}\n"
                            f"# TYPE llamacpp:requests_deferred gauge\n"
                            f"llamacpp:requests_deferred {server.deferred}\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with server._lock:
                    server.requests.append(body)
                    n = len(server.requests)
                    server._in_flight += 1
                    server.max_in_flight = max(server.max_in_flight, server._in_flight)
                try:
                    fault = server.fault(body, n) if server.fault else None
                    if fault and fault["kind"] in ("http", "ctx_exceeded", "pool_exhausted"):
                        self._json(*ERRORS[fault["kind"]](fault))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self._stream(body, fault or {})
                except (BrokenPipeError, ConnectionResetError):
                    server.aborted += 1
                finally:
                    with server._lock:
                        server._in_flight -= 1

            def _send(self, obj):
                self.wfile.write(b"data: " + (json.dumps(obj) if obj != "[DONE]" else obj).encode() + b"\n\n")
                self.wfile.flush()

            def _stream(self, body, fault):
                messages, max_tokens = body["messages"], body["max_tokens"]
                text = "\n".join(m["content"] for m in messages)
                found = NEEDLE_RE.search(text)  # the answer = the first key appearing in the conversation
                answer = found.group(0) if found and fault.get("kind") != "no_key" else "unknown"
                pieces = [answer] + ["."] * (max_tokens - 1)
                time.sleep(fault.get("ttft_s", server.ttft_s))
                self._send({"choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}]})
                for i, p in enumerate(pieces):
                    if fault.get("kind") == "disconnect" and i >= fault["after_chunks"]:
                        return
                    if fault.get("kind") == "stream_pool_exhausted" and i >= fault["after_chunks"]:
                        self._send(ERRORS["pool_exhausted"](fault)[1])
                        return
                    if fault.get("kind") == "garbage" and i >= fault["after_chunks"]:
                        self.wfile.write(b"data: {not json\n\n")
                        self.wfile.flush()
                        return
                    self._send({"choices": [{"finish_reason": None, "index": 0, "delta": {"content": p}}]})
                    time.sleep(server.chunk_interval_s)
                timings = {"cache_n": 0, "prompt_n": FakeTokenizer().prompt_tokens(messages),
                           "predicted_n": len(pieces)}
                self._send({"choices": [{"finish_reason": "length", "index": 0, "delta": {}}], "timings": timings})
                self._send("[DONE]")

        return Handler
