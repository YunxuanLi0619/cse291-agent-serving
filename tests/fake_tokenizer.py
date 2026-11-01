"""Fake implementation of the Tokenizer interface: 1 token per word, plus a fixed template overhead per message."""
TEMPLATE_OVERHEAD = 32


class FakeTokenizer:
    def prompt_tokens(self, messages: list[dict]) -> int:
        return TEMPLATE_OVERHEAD + sum(len(m["content"].split()) for m in messages)
