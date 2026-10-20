import pytest

from fake_server import FakeServer
from fake_tokenizer import FakeTokenizer


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_server():
    with FakeServer() as server:
        yield server
