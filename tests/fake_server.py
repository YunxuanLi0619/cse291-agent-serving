"""会 sleep 的假 llama-server:流式响应,最后一个数据 chunk 带 timings。

故障注入(失败 / 慢响应 / 中途断连 / ctx 超限)留给后续票按需扩展。
"""
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fake_tokenizer import FakeTokenizer

NEEDLE_RE = re.compile(r"KEY-[a-z0-9]{8}")


class FakeServer:
    def __init__(self, ttft_s: float = 0.05, chunk_interval_s: float = 0.001,
                 health_delay_s: float = 0.0):
        self.ttft_s, self.chunk_interval_s = ttft_s, chunk_interval_s
        self.ready_at = time.monotonic() + health_delay_s
        self.requests: list[dict] = []  # 收到的 chat 请求体,供测试断言
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
                else:
                    self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self._stream(body)

            def _send(self, obj):
                self.wfile.write(b"data: " + (json.dumps(obj) if obj != "[DONE]" else obj).encode() + b"\n\n")
                self.wfile.flush()

            def _stream(self, body):
                messages, max_tokens = body["messages"], body["max_tokens"]
                text = "\n".join(m["content"] for m in messages)
                found = NEEDLE_RE.search(text)  # 回答 = 对话里出现的第一个 key
                pieces = [found.group(0) if found else "unknown"] + ["."] * (max_tokens - 1)
                time.sleep(server.ttft_s)
                self._send({"choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}]})
                for p in pieces:
                    self._send({"choices": [{"finish_reason": None, "index": 0, "delta": {"content": p}}]})
                    time.sleep(server.chunk_interval_s)
                timings = {"cache_n": 0, "prompt_n": FakeTokenizer().prompt_tokens(messages),
                           "predicted_n": len(pieces)}
                self._send({"choices": [{"finish_reason": "length", "index": 0, "delta": {}}], "timings": timings})
                self._send("[DONE]")

        return Handler
