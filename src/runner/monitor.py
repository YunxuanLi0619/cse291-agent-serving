"""系统采样器:独立线程每 0.5 s 采一次内存、/metrics、/slots 和 live_tokens,写 system.jsonl。

内存读取是可替换的 MemoryProbe(ADR 0004):macOS 用 `footprint`(主)+ RSS(次),Linux 留作接口位置。
某一路采样失败不跳过这一行:失败的字段为 null,原因写进 `error`。
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

MIN_INTERVAL_S = 0.5  # footprint 单次约 0.2–0.5 s,间隔不能更短(ADR 0004)
HTTP_TIMEOUT_S = 2.0
PROBE_TIMEOUT_S = 5.0
_UNIT_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}
_FOOTPRINT_RE = re.compile(r"Footprint:\s+([\d.]+)\s*(KB|MB|GB)")


@dataclass
class MemReading:
    footprint_mb: float  # 主口径
    rss_mb: float  # 次口径


class MemoryProbe(Protocol):
    platform: str  # 平台标签,随每行数据记录;不同平台的数据不混画一张图

    def read(self, pid: int) -> MemReading:
        """读失败抛异常,由采样器记进 error 字段。"""
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
    """接口位置:读 /proc/<pid>/status。当前只在 macOS 上实验,尚未实现。"""
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
            except Exception as e:  # 比如日志写盘失败:报到 stderr,线程不能静默死掉
                print(f"system sampler: tick failed: {type(e).__name__}: {e}", file=sys.stderr)
            next_at += self._interval_s
            delay = next_at - (time.monotonic() - t0)
            if delay < 0:  # 一次采样超过间隔:追上当前时刻,不连发补采
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
