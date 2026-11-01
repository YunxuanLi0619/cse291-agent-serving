import os
import subprocess
from pathlib import Path

from fake_server import FakeServer

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "scripts" / "start_server.sh"
WAIT = ROOT / "scripts" / "wait_health.sh"


def test_wait_health_prints_cold_start_seconds_once_server_is_ready():
    with FakeServer(health_delay_s=2.0) as server:  # returns 503 first, ready after 2 s
        r = subprocess.run([str(WAIT), "30"], env={**os.environ, "LLAMA_URL": server.url},
                           capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    assert 1 <= int(r.stdout.strip()) <= 4


def test_wait_health_fails_when_server_never_becomes_ready():
    with FakeServer(health_delay_s=3600) as server:
        r = subprocess.run([str(WAIT), "2"], env={**os.environ, "LLAMA_URL": server.url},
                           capture_output=True, text=True, timeout=60)
    assert r.returncode != 0
    assert r.stdout.strip() == ""


def test_start_server_passes_np_ctx_and_fixed_flags_then_extra_flags(tmp_path):
    stub = tmp_path / "llama-server"
    stub.write_text('#!/usr/bin/env bash\necho "$@"\n')
    stub.chmod(0o755)
    r = subprocess.run([str(START), "4", "16384", "--no-kv-unified"],
                       env={**os.environ, "LLAMA_SERVER_BIN": str(stub), "MODEL": "m.gguf",
                            "SERVER_CONFIG": str(tmp_path / "absent.json")},  # no frozen config → default 8 threads
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.split() == ["-m", "m.gguf", "--host", "127.0.0.1", "--port", "8080",
                                "-np", "4", "-c", "16384", "-t", "8", "--cache-ram", "0", "--metrics",
                                "--no-kv-unified"]


def test_start_server_requires_np_and_ctx():
    r = subprocess.run([str(START)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "usage" in r.stderr.lower()
