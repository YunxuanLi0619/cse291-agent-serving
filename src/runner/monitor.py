"""System sampler: a separate thread samples memory, /metrics, /slots and live_tokens every 0.5 s and writes system.jsonl.

Memory reading is a replaceable MemoryProbe (ADR 0004): macOS uses `footprint` (primary) + RSS (secondary), and Linux is left as an interface placeholder.
If one sampling source fails the row is not skipped: the failed field is null and the reason is written to `error`.
"""
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import httpx

from .logs import JsonlWriter

MIN_INTERVAL_S = 0.5  # a single footprint call takes about 0.2–0.5 s, so the interval cannot be shorter (ADR 0004)
HTTP_TIMEOUT_S = 2.0
PROBE_TIMEOUT_S = 5.0
_UNIT_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}
_FOOTPRINT_RE = re.compile(r"Footprint:\s+([\d.]+)\s*(KB|MB|GB)")


@dataclass
class MemReading:
    footprint_mb: float  # primary measure
    rss_mb: float  # secondary measure


class MemoryProbe(Protocol):
    platform: str  # platform label, recorded with every row; data from different platforms is not plotted on one figure

    def read(self, pid: int) -> MemReading:
        """Raise an exception on read failure; the sampler records it in the error field."""
        ...


class MacOSProbe:
    platform = "macos"

    def read(self, pid: int) -> MemReading:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True,
                             timeout=PROBE_TIMEOUT_S, check=True).stdout
        m = _FOOTPRINT_RE.search(out)
        if not m:
            raise RuntimeError("footprint output has no Footprint line")
        rss_kb = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True,
                                timeout=PROBE_TIMEOUT_S, check=True).stdout.strip()
        if not rss_kb:
            raise RuntimeError(f"ps reports no rss for pid {pid}")
        return MemReading(float(m.group(1)) * _UNIT_MB[m.group(2)], int(rss_kb) / 1024)


class LinuxProbe:
    """Interface placeholder: read /proc/<pid>/status. Currently only experimented on macOS; not implemented yet."""
    platform = "linux"

    def read(self, pid: int) -> MemReading:
        raise NotImplementedError("Linux memory probe is not implemented")


def default_probe() -> MemoryProbe | None:
    return MacOSProbe() if sys.platform == "darwin" else None


def _metric(text: str, name: str) -> int:
    m = re.search(rf"^llamacpp:{name}\s+([\d.]+)\s*$", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"/metrics has no {name}")
    return int(float(m.group(1)))


class SystemSampler:
    def __init__(self, probe: MemoryProbe | None, pid: int | None, base_url: str,
                 live_tokens: Callable[[], int], log: JsonlWriter, now: Callable[[], float],
                 interval_s: float = MIN_INTERVAL_S):
        if interval_s < MIN_INTERVAL_S:
            raise ValueError(f"sampling interval {interval_s}s is below the minimum {MIN_INTERVAL_S}s (ADR 0004)")
        self._probe, self._pid, self._live_tokens = probe, pid, live_tokens
        self._log, self._now, self._interval_s = log, now, interval_s
        self._http = httpx.Client(base_url=base_url, timeout=HTTP_TIMEOUT_S)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="system-sampler", daemon=True)

    def tick(self) -> None:
        errors: list[str] = []
        row = {"t": self._now(), "platform": self._probe.platform if self._probe else None,
               "footprint_mb": None, "rss_mb": None, "requests_processing": None,
               "requests_deferred": None, "live_tokens": None, "slots": None}
        if self._probe is None:
            errors.append("memory: no memory probe for this platform")
        elif self._pid is None:
            errors.append("memory: no server pid given")
        else:
            try:
                mem = self._probe.read(self._pid)
                row["footprint_mb"], row["rss_mb"] = mem.footprint_mb, mem.rss_mb
            except Exception as e:
                errors.append(f"memory: {type(e).__name__}: {e}")
        try:
            text = self._get("/metrics").text
            row["requests_processing"] = _metric(text, "requests_processing")
            row["requests_deferred"] = _metric(text, "requests_deferred")
        except Exception as e:
            errors.append(f"metrics: {type(e).__name__}: {e}")
        try:
            row["slots"] = [{"is_processing": bool(s["is_processing"])} for s in self._get("/slots").json()]
        except Exception as e:
            errors.append(f"slots: {type(e).__name__}: {e}")
        try:
            row["live_tokens"] = self._live_tokens()
        except Exception as e:
            errors.append(f"live_tokens: {type(e).__name__}: {e}")
        row["error"] = "; ".join(errors) or None
        self._log.write(row)

    def _get(self, path: str) -> httpx.Response:
        r = self._http.get(path)
        r.raise_for_status()
        return r

    def _loop(self) -> None:
        t0 = time.monotonic()
        next_at = 0.0
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # e.g. log write to disk failed: report to stderr; the thread must not die silently
                print(f"system sampler: tick failed: {type(e).__name__}: {e}", file=sys.stderr)
            next_at += self._interval_s
            delay = next_at - (time.monotonic() - t0)
            if delay < 0:  # one sample exceeded the interval: catch up to the current time instead of firing make-up samples back to back
                next_at, delay = time.monotonic() - t0, 0.0
            if self._stop.wait(delay):
                break

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._http.close()
