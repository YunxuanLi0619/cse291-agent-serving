"""Real side effects of the batch driver (macOS): pmset, the llama-server process, the runner subprocess, the analysis smoke test."""
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from .driver import Hooks
from .plan import RunSpec

REPO_DIR = Path(__file__).resolve().parent.parent.parent  # where code and scripts live; the data root can point elsewhere (tests)
HEALTH_TIMEOUT_S = 180
STOP_TIMEOUT_S = 30
SETTLE_S = 3  # after stopping the server, wait a little so the port and memory are released
RUNNER_SLACK_S = 1800  # margin beyond the arrival period: draining the last Task (<=300 s), restart, fluctuation


def port_in_use(host: str = "127.0.0.1", port: int = 8080) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


def _pmset(*args: str) -> str:
    return subprocess.run(["pmset", *args], capture_output=True, text=True, timeout=30, check=True).stdout


class ServerProcess:
    """start_server.sh ends with exec, so the Popen pid is llama-server itself."""

    def __init__(self, proc: subprocess.Popen, log, cold_start_s: float):
        self.proc, self._log, self.pid, self.cold_start_s = proc, log, proc.pid, cold_start_s

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive():
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while port_in_use() and time.monotonic() < deadline:
            time.sleep(0.5)
        time.sleep(SETTLE_S)


class MacHooks:
    def __init__(self, root: Path, base_url: str = "http://127.0.0.1:8080"):
        self.root, self.base_url = Path(root), base_url

    def hooks(self) -> Hooks:
        return Hooks(batt=lambda: _pmset("-g", "batt"), settings=lambda: _pmset("-g"), therm=lambda: _pmset("-g", "therm"),
                     sleep=time.sleep, start_server=self.start_server, run_runner=self.run_runner,
                     smoke_analysis=self.smoke_analysis)

    def start_server(self, spec: RunSpec, log_path: Path) -> ServerProcess:
        if port_in_use():
            raise RuntimeError("port 8080 is already in use (stray llama-server?)")
        log = open(log_path, "w")
        t0 = time.monotonic()
        proc = subprocess.Popen([str(REPO_DIR / "scripts/start_server.sh"), str(spec.np), str(spec.ctx), *spec.server_flags],
                                stdout=log, stderr=subprocess.STDOUT)
        server = ServerProcess(proc, log, 0.0)
        try:
            while True:
                if not server.alive():
                    raise RuntimeError(f"llama-server exited with code {proc.returncode} during startup")
                try:
                    if httpx.get(f"{self.base_url}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() - t0 > HEALTH_TIMEOUT_S:
                    raise RuntimeError(f"/health not ok after {HEALTH_TIMEOUT_S} s")
                time.sleep(0.1)
        except BaseException:
            server.stop()
            raise
        server.cold_start_s = round(time.monotonic() - t0, 2)
        return server

    def run_runner(self, spec: RunSpec, run_dir: Path, handle) -> int:
        cmd = [sys.executable, "-m", "src.runner.run", "--workload", str(self.root / spec.workload), "--out", str(run_dir),
               "--base-url", self.base_url, "--c", str(spec.c), "--lambda", str(spec.lam), "--policy", "fifo",
               "--n-tasks", str(spec.n_tasks), "--server-pid", str(handle.pid),
               "--cache-prompt", "on" if spec.cache_prompt else "off", "--cold-start-s", str(handle.cold_start_s)]
        timeout = spec.n_tasks / spec.lam * 1.6 + RUNNER_SLACK_S  # leave a 60% margin for Poisson fluctuation of the arrival period
        with open(run_dir / "runner.log", "w") as log:
            try:
                return subprocess.run(cmd, cwd=REPO_DIR, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                log.write(f"\nrunner killed after {timeout:.0f} s\n")
                return -9

    def smoke_analysis(self, run_dir: Path) -> list[str]:
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "-m", "src.analysis", "--raw", str(run_dir), "--out", out],
                               cwd=REPO_DIR, capture_output=True, text=True, timeout=300)
            if r.returncode:
                return [f"analysis exited with code {r.returncode}: {r.stderr.strip().splitlines()[-1:]}"]
            if not (Path(out) / "runs.csv").exists() or len((Path(out) / "runs.csv").read_text().splitlines()) < 2:
                return ["analysis produced no run row"]
        return []
