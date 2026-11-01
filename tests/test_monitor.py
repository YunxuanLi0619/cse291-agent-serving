import json
import os
import sys
import threading
import time

import httpx
import pytest

from src.runner.logs import JsonlWriter
from src.runner.monitor import MIN_INTERVAL_S, MacOSProbe, MemReading, SystemSampler


class FakeProbe:
    def __init__(self, platform="fake-os", readings=()):
        self.platform = platform
        self.readings = list(readings)  # MemReading or an exception to raise

    def read(self, pid):
        r = self.readings.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_sampler(tmp_path, server_url, probe, live_tokens=lambda: 0, interval_s=MIN_INTERVAL_S):
    out = tmp_path / "system.jsonl"
    return SystemSampler(probe, pid=1234, base_url=server_url, live_tokens=live_tokens, log=JsonlWriter(out),
                         now=lambda: 42.0, interval_s=interval_s), out


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tick_writes_row_with_memory_server_state_and_platform(tmp_path, fake_server):
    fake_server.deferred = 3
    sampler, out = make_sampler(tmp_path, fake_server.url,
                                FakeProbe("fake-os", [MemReading(footprint_mb=3789.0, rss_mb=700.0)]),
                                live_tokens=lambda: 1500)
    sampler.tick()
    assert rows(out) == [{
        "t": 42.0, "platform": "fake-os", "footprint_mb": 3789.0, "rss_mb": 700.0,
        "requests_processing": 0, "requests_deferred": 3, "live_tokens": 1500,
        "slots": [{"is_processing": False}] * 4, "error": None,
    }]


def test_failed_memory_probe_still_writes_a_row_with_error(tmp_path, fake_server):
    probe = FakeProbe("fake-os", [RuntimeError("footprint timed out")])
    sampler, out = make_sampler(tmp_path, fake_server.url, probe, live_tokens=lambda: 7)
    sampler.tick()
    [row] = rows(out)
    assert row["footprint_mb"] is None and row["rss_mb"] is None
    assert "footprint timed out" in row["error"]
    assert row["platform"] == "fake-os"  # the platform label does not depend on this reading succeeding
    assert row["requests_processing"] == 0 and row["live_tokens"] == 7  # the other fields are sampled as usual


def test_unreachable_server_still_writes_a_row_with_error(tmp_path):
    probe = FakeProbe("fake-os", [MemReading(100.0, 90.0)])
    sampler, out = make_sampler(tmp_path, "http://127.0.0.1:9", probe)  # nobody listens on port 9
    sampler.tick()
    [row] = rows(out)
    assert row["footprint_mb"] == 100.0
    assert row["requests_processing"] is None and row["requests_deferred"] is None and row["slots"] is None
    assert row["error"]


def test_slots_report_is_processing_per_slot(tmp_path, fake_server):
    gate = threading.Event()
    fake_server.ttft_s = 0.3
    probe = FakeProbe("fake-os", [MemReading(1.0, 1.0)])
    sampler, out = make_sampler(tmp_path, fake_server.url, probe)
    t = threading.Thread(target=lambda: httpx.post(fake_server.url + "/v1/chat/completions", timeout=10,
                                                   json={"messages": [{"role": "user", "content": "x"}],
                                                         "max_tokens": 2, "stream": True}))
    t.start()
    while fake_server.max_in_flight == 0:
        gate.wait(0.01)
    sampler.tick()
    t.join()
    [row] = rows(out)
    assert [s["is_processing"] for s in row["slots"]] == [True, False, False, False]
    assert row["requests_processing"] == 1


def test_missing_probe_is_reported_in_error_not_silent(tmp_path, fake_server):
    sampler, out = make_sampler(tmp_path, fake_server.url, None)
    sampler.tick()
    [row] = rows(out)
    assert row["footprint_mb"] is None and row["platform"] is None and "memory" in row["error"]


def test_interval_below_half_a_second_is_rejected(tmp_path, fake_server):
    with pytest.raises(ValueError, match="0.5"):
        make_sampler(tmp_path, fake_server.url, FakeProbe(), interval_s=0.2)


def test_background_thread_samples_periodically_and_stop_joins(tmp_path, fake_server):
    probe = FakeProbe("fake-os", [MemReading(1.0, 1.0)] * 10)
    sampler, out = make_sampler(tmp_path, fake_server.url, probe)
    sampler.start()
    time.sleep(1.3)
    sampler.stop()
    n = len(rows(out))
    assert 2 <= n <= 3  # t=0, 0.5, 1.0 (one fewer is possible when the machine is busy)
    time.sleep(0.7)
    assert len(rows(out)) == n  # no more rows are written once stopped


@pytest.mark.skipif(sys.platform != "darwin", reason="footprint is macOS-only")
def test_macos_probe_reads_footprint_and_rss_of_a_live_process():
    mem = MacOSProbe().read(os.getpid())
    assert mem.footprint_mb > 0 and mem.rss_mb > 0
