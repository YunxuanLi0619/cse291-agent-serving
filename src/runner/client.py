"""Request client: calls llama-server in streaming mode, records the first token and end times, and takes `timings` from the last data chunk.

Failures are uniformly raised as RequestFailed(status, error_type); timeouts are controlled by the caller via asyncio.timeout,
and on cancellation httpx closes the connection, which is the "active disconnect".
"""
import json
from dataclasses import dataclass
from typing import Callable

import httpx


class RequestFailed(Exception):
    def __init__(self, status: str, error_type: str):
        super().__init__(f"{status}/{error_type}")
        self.status, self.error_type = status, error_type


@dataclass
class StreamResult:
    """Filled in place: on failure or cancellation the caller can still read what already happened (e.g. t_first_token)."""
    text: str = ""
    t_first_token: float | None = None
    t_end: float | None = None
    n_pieces: int = 0  # number of non-empty content chunks received, ≈ number of generated tokens
    timings: dict | None = None


def _classify_error(status_code: int, payload: object) -> RequestFailed:
    """payload is the {"error": {...}} returned by the server (an HTTP error body, or an SSE error event in the stream)."""
    err = payload.get("error") if isinstance(payload, dict) else None
    err = err if isinstance(err, dict) else {}
    if err.get("type") == "exceed_context_size_error":  # 400: a single request exceeds the slot capacity
        return RequestFailed("error_ctx_exceeded", "ctx_exceeded")
    if "Context size has been exceeded" in str(err.get("message", "")):
        return RequestFailed("error_http", "pool_exhausted")  # shared pool exhausted, classified separately
    return RequestFailed("error_http", f"http_{status_code}")


async def stream_chat(client: httpx.AsyncClient, messages: list[dict], max_tokens: int,
                      now: Callable[[], float], out: StreamResult, cache_prompt: bool = True) -> StreamResult:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "cache_prompt": cache_prompt}
    pieces, done = [], False
    try:
        async with client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code >= 400:
                try:
                    payload = json.loads(await r.aread())
                except ValueError:
                    payload = None
                raise _classify_error(r.status_code, payload)
            async for line in r.aiter_lines():
                if line == "data: [DONE]":
                    done = True
                    continue
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[len("data: "):])
                except ValueError as e:
                    raise RequestFailed("error_other", "malformed_stream") from e
                if "error" in chunk:  # error after chunks were already sent: HTTP 200 + SSE error event
                    raise _classify_error(r.status_code, chunk)
                choices = chunk.get("choices") or []
                piece = choices[0].get("delta", {}).get("content") if choices else None
                if piece:
                    if out.t_first_token is None:
                        out.t_first_token = now()
                    pieces.append(piece)
                    out.n_pieces += 1
                if "timings" in chunk:  # on the chunk carrying finish_reason, before [DONE]
                    out.timings = chunk["timings"]
    except httpx.HTTPError as e:  # connection failure, disconnect mid-read, etc.
        raise RequestFailed("error_other", type(e).__name__) from e
    out.text = "".join(pieces)
    out.t_end = now()
    if not done:  # stream ended before [DONE] = disconnect mid-stream
        raise RequestFailed("error_other", "incomplete_stream")
    return out
