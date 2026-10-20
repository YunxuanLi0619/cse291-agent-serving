"""Request 客户端:流式调用 llama-server,记录首个 token 与结束时刻,取最后一个数据 chunk 的 timings。"""
import json
from dataclasses import dataclass
from typing import Callable

import httpx

REQUEST_TIMEOUT_S = 120


@dataclass
class StreamResult:
    text: str
    t_first_token: float | None
    t_end: float
    timings: dict | None


def stream_chat(client: httpx.Client, messages: list[dict], max_tokens: int,
                now: Callable[[], float]) -> StreamResult:
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True}
    text, t_first, timings = [], None, None
    with client.stream("POST", "/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            choices = chunk.get("choices") or []
            piece = choices[0].get("delta", {}).get("content") if choices else None
            if piece:
                if t_first is None:
                    t_first = now()
                text.append(piece)
            if "timings" in chunk:  # 在 finish_reason 那个 chunk 上,[DONE] 之前
                timings = chunk["timings"]
    return StreamResult("".join(text), t_first, now(), timings)
