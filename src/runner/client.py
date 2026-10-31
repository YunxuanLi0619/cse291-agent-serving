"""Request 客户端:流式调用 llama-server,记录首个 token 与结束时刻,取最后一个数据 chunk 的 timings。

失败统一抛 RequestFailed(status, error_type);超时由调用方用 asyncio.timeout 控制,
取消时 httpx 会关闭连接,即「主动断开」。
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
    """就地填充:失败或被取消时,调用方仍能读到已经发生的部分(如 t_first_token)。"""
    text: str = ""
    t_first_token: float | None = None
    t_end: float | None = None
    n_pieces: int = 0  # 已收到的非空内容 chunk 数,≈ 已生成 token 数
    timings: dict | None = None


def _classify_error(status_code: int, payload: object) -> RequestFailed:
    """payload 是 server 返回的 {"error": {...}}(HTTP 错误体,或流中 SSE error 事件)。"""
    err = payload.get("error") if isinstance(payload, dict) else None
    err = err if isinstance(err, dict) else {}
    if err.get("type") == "exceed_context_size_error":  # 400:单个请求超过 slot 容量
        return RequestFailed("error_ctx_exceeded", "ctx_exceeded")
    if "Context size has been exceeded" in str(err.get("message", "")):
        return RequestFailed("error_http", "pool_exhausted")  # 共享池耗尽,单独分类
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
                if "error" in chunk:  # 已发过 chunk 后的错误:HTTP 200 + SSE error 事件
                    raise _classify_error(r.status_code, chunk)
                choices = chunk.get("choices") or []
                piece = choices[0].get("delta", {}).get("content") if choices else None
                if piece:
                    if out.t_first_token is None:
                        out.t_first_token = now()
                    pieces.append(piece)
                    out.n_pieces += 1
                if "timings" in chunk:  # 在 finish_reason 那个 chunk 上,[DONE] 之前
                    out.timings = chunk["timings"]
    except httpx.HTTPError as e:  # 连接失败、读到一半被断开等
        raise RequestFailed("error_other", type(e).__name__) from e
    out.text = "".join(pieces)
    out.t_end = now()
    if not done:  # 流在 [DONE] 之前就结束 = 中途断连
        raise RequestFailed("error_other", "incomplete_stream")
    return out
