"""Tokenizer 接口:给定 chat 消息,返回 llama-server 会报告的 prompt_tokens(含 chat template)。

真实实现调 llama-server;测试用假实现(tests/fake_tokenizer.py)。
"""
from typing import Protocol

import httpx


class Tokenizer(Protocol):
    def prompt_tokens(self, messages: list[dict]) -> int: ...


class LlamaServerTokenizer:
    """/apply-template 渲染 chat template,再用 /tokenize 计数(模板已含 BOS,所以不再加特殊 token)。"""

    def __init__(self, base_url: str):
        self._client = httpx.Client(base_url=base_url, timeout=60)

    def prompt_tokens(self, messages: list[dict]) -> int:
        prompt = self._client.post("/apply-template", json={"messages": messages}).raise_for_status().json()["prompt"]
        body = {"content": prompt, "add_special": False, "parse_special": True}
        return len(self._client.post("/tokenize", json=body).raise_for_status().json()["tokens"])
