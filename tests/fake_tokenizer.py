"""Tokenizer 接口的假实现:每个词 1 token,每条消息 + 固定模板开销。"""
TEMPLATE_OVERHEAD = 32


class FakeTokenizer:
    def prompt_tokens(self, messages: list[dict]) -> int:
        return TEMPLATE_OVERHEAD + sum(len(m["content"].split()) for m in messages)
