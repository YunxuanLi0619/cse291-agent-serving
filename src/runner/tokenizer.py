"""Tokenizer interface: given chat messages, return the prompt_tokens that llama-server would report (including the chat template).

The real implementation calls llama-server; tests use a fake (tests/fake_tokenizer.py).
"""
from typing import Protocol

import httpx


class Tokenizer(Protocol):
    def prompt_tokens(self, messages: list[dict]) -> int: ...


class LlamaServerTokenizer:
    """/apply-template renders the chat template, then /tokenize counts (the template already contains BOS, so no extra special token is added)."""

    def __init__(self, base_url: str):
        self._client = httpx.Client(base_url=base_url, timeout=60)

    def prompt_tokens(self, messages: list[dict]) -> int:
        prompt = self._client.post("/apply-template", json={"messages": messages}).raise_for_status().json()["prompt"]
        body = {"content": prompt, "add_special": False, "parse_special": True}
        return len(self._client.post("/tokenize", json=body).raise_for_status().json()["tokens"])
