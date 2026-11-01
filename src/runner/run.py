"""Runner: reads a Workload and executes it against llama-server with open-loop Poisson arrivals, a FIFO queue and a concurrency limit c,
writing requests.jsonl, tasks.jsonl, system.jsonl (sampled every 0.5 s in a separate thread) and manifest.json.
Every submitted Task gets exactly one Task log row, whether it succeeds or fails.

Usage: python -m src.runner.run --workload workloads/dev/default_s1.json --base-url http://127.0.0.1:8080 \
        --out results/raw/run_x --c 4 --lambda 0.05 --policy fifo --server-pid <llama-server pid>
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from .client import RequestFailed, StreamResult, stream_chat
from .gen_workload import TOOL_RESULT_TOKENS  # continuation prompt estimate = full context of the previous step + new tool text
from .logs import JsonlWriter
from .manifest import collect_manifest
from .monitor import SystemSampler, default_probe
from .scheduler import POLICIES, ReadyStep, Scheduler

WARMUP_TASKS = 3  # the first 3 Tasks to arrive in each Run are marked warmup
REQUEST_TIMEOUT_S = 120
TASK_TIMEOUT_CAP_S = 300
TASK_TIMEOUT_SOLO_FACTOR = 10  # task-level timeout = min(300, 10 × T_solo), with T_solo = deadline_s / 5
DEADLINE_SOLO_FACTOR = 5
REPO_DIR = Path(__file__).resolve().parent.parent.parent


def task_timeout_s(task: dict) -> float:
    return min(TASK_TIMEOUT_CAP_S, TASK_TIMEOUT_SOLO_FACTOR * task["deadline_s"] / DEADLINE_SOLO_FACTOR)


class Executor:  # note: "Run" in the glossary is a whole experiment; this is just the executor
    def __init__(self, sched: Scheduler, client: httpx.AsyncClient, requests_log: JsonlWriter,
                 tasks_log: JsonlWriter, now: Callable[[], float], request_timeout_s: float,
                 cache_prompt: bool = True):
        self.sched, self.client, self.now = sched, client, now
        self.cache_prompt = cache_prompt
        self.requests_log, self.tasks_log = requests_log, tasks_log
        self.request_timeout_s = request_timeout_s
        self._in_flight: dict[int, tuple[int, StreamResult]] = {}  # in-flight Request → (prompt estimate, stream result)

    def live_tokens(self) -> int:
        """Token estimate held by in-flight Requests = each one's prompt estimate + number of output chunks received so far (the runner has no tokenizer)."""
        return sum(prompt + res.n_pieces for prompt, res in list(self._in_flight.values()))

    async def run_task(self, task: dict, t_arrive: float, warmup: bool) -> None:
        await asyncio.sleep(max(0.0, t_arrive - self.now()))  # open loop: arrival is independent of system state
        status, output = "error_other", ""
        queue_time = idle_time = 0.0
        t_first_dispatch = None
        messages: list[dict] = []
        prompt_est = task["prompt_tokens_measured"]
        try:
            async with asyncio.timeout(t_arrive + task_timeout_s(task) - self.now()):
                for k, step in enumerate(task["steps"], start=1):
                    messages.append({"role": "user", "content": step["user_content"]})
                    entry = self.sched.submit(task["task_id"], k)
                    try:
                        await entry.granted.wait()  # task-level timeout while queued → cancel → finally removes it from the queue
                        t_first_dispatch = t_first_dispatch if t_first_dispatch is not None else entry.started_at
                        res = await self._request(task, k, step, messages, entry, warmup, prompt_est)
                        output = res.text
                        if res.timings:  # next-step prompt ≈ full context of this step + new tool text
                            prompt_est = (res.timings.get("prompt_n", 0) + res.timings.get("cache_n", 0)
                                          + res.timings.get("predicted_n", 0) + TOOL_RESULT_TOKENS)
                    finally:
                        queue_time += entry.queue_time(self.now())
                        self.sched.finish(entry)
                    messages.append({"role": "assistant", "content": output})
                    if step["delay_after_s"] is not None:  # Tool delay does not hold a concurrency slot
                        await asyncio.sleep(step["delay_after_s"])
                        idle_time += step["delay_after_s"]
            status = "ok"
        except TimeoutError:
            status = "timeout"
        except RequestFailed as e:
            status = e.status
        except Exception:  # unexpected errors are also classed as error_other and must not bring down other Tasks
            status = "error_other"
        finally:  # includes external cancellation: each Task still gets exactly one row
            self.tasks_log.write({
                "task_id": task["task_id"], "type": task["type"], "t_arrive": t_arrive,
                "t_first_dispatch": t_first_dispatch, "t_done": self.now(), "queue_time_total": queue_time,
                "idle_time_total": idle_time, "n_steps": len(task["steps"]), "status": status,
                "quality_pass": status == "ok" and task["needle"] in output,
                "deadline_s": task["deadline_s"], "warmup": warmup,
            })

    async def _request(self, task: dict, k: int, step: dict, messages: list[dict], entry: ReadyStep,
                       warmup: bool, prompt_est: int) -> StreamResult:
        res = StreamResult()
        self._in_flight[id(entry)] = (prompt_est, res)
        status, error_type = "ok", ""
        try:
            async with asyncio.timeout(self.request_timeout_s):
                await stream_chat(self.client, messages, step["max_tokens"], self.now, res, self.cache_prompt)
        except TimeoutError:
            status, error_type = "timeout", "request_timeout"
        except RequestFailed as e:
            status, error_type = e.status, e.error_type
        except asyncio.CancelledError:  # the task-level timeout interrupted an in-flight Request
            status, error_type = "timeout", "task_timeout"
            raise
        finally:
            self._in_flight.pop(id(entry), None)
            timings = res.timings or {}
            self.requests_log.write({
                "task_id": task["task_id"], "step": k, "t_submit": entry.ready_at, "t_start": entry.started_at,
                "t_first_token": res.t_first_token, "t_end": res.t_end if res.t_end is not None else self.now(),
                "prompt_tokens": timings.get("prompt_n", 0) + timings.get("cache_n", 0) if timings else None,
                "cached_tokens": timings.get("cache_n"), "output_tokens": timings.get("predicted_n"),
                "status": status, "error_type": error_type or None, "warmup": warmup, "id_slot": None,
            })
        if status != "ok":
            raise RequestFailed(status, error_type)
        return res


def check_slots(base_url: str, c: int) -> None:
    """ADR 0003: c ≤ np, otherwise the extra requests queue inside the server and our FIFO is defeated."""
    try:
        r = httpx.get(f"{base_url}/slots", timeout=10)
        r.raise_for_status()
        n_slots = len(r.json())
    except (httpx.HTTPError, ValueError) as e:
        raise SystemExit(f"cannot read /slots from {base_url} ({e}); refusing to run") from e
    if n_slots < c:
        raise SystemExit(f"/slots reports {n_slots} slots < c={c} (ADR 0003); refusing to run")


def write_manifest(out: Path, manifest: dict) -> None:
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


def stamp(manifest: dict, key: str) -> None:
    t = time.time()
    manifest[f"t_{key}"], manifest[f"t_{key}_iso"] = t, datetime.fromtimestamp(t, timezone.utc).isoformat()


async def run_all(a, workload: dict) -> None:
    tasks = workload["tasks"][:a.n_tasks]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = collect_manifest(repo_dir=REPO_DIR, llama_cpp_dir=Path(a.llama_cpp_dir), server_pid=a.server_pid,
                                workload_path=Path(a.workload), seed=workload.get("seed"), c=a.c, lam=a.lam,
                                timeout_s=a.timeout)
    # Batch variables: E4's cache_prompt, E3's fixed Tool delay (top-level Workload key), cold-start seconds (passed in by the batch script)
    manifest.update(cache_prompt=a.cache_prompt == "on", tool_delay_s=workload.get("tool_delay_s"),
                    cold_start_s=a.cold_start_s)
    probe = default_probe()
    manifest["memory_platform"] = probe.platform if probe else None  # matches the platform label in system.jsonl
    stamp(manifest, "start")
    write_manifest(out, manifest)  # leave start-point info even if the Run crashes midway; overwritten at the end to add t_end
    t0 = time.monotonic()
    now = lambda: time.monotonic() - t0  # noqa: E731
    arrivals, t = [], 0.0
    for task in tasks:  # arrival time = cumulative sum of unit intervals / λ
        t += task["unit_gap_s"] / a.lam
        arrivals.append(t)
    requests_log, tasks_log = JsonlWriter(out / "requests.jsonl"), JsonlWriter(out / "tasks.jsonl")
    system_log = JsonlWriter(out / "system.jsonl")
    sampler = None
    try:
        async with httpx.AsyncClient(base_url=a.base_url, timeout=None) as client:
            executor = Executor(Scheduler(POLICIES[a.policy](), a.c, now), client, requests_log, tasks_log, now, a.timeout,
                                a.cache_prompt == "on")
            sampler = SystemSampler(probe, a.server_pid, a.base_url, executor.live_tokens, system_log, now)
            sampler.start()
            await asyncio.gather(*(executor.run_task(task, arrivals[i], i < WARMUP_TASKS)
                                   for i, task in enumerate(tasks)))
    finally:
        if sampler is not None:
            sampler.stop()
        requests_log.close()
        tasks_log.close()
        system_log.close()
        stamp(manifest, "end")
        write_manifest(out, manifest)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--c", type=int, required=True, help="concurrency limit (number of in-flight Requests)")
    ap.add_argument("--lambda", dest="lam", type=float, required=True, help="Task arrival rate (per second)")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="fifo")
    ap.add_argument("--n-tasks", type=int, help="submit only the first N Tasks of the Workload (default: all)")
    ap.add_argument("--server-pid", type=int, help="llama-server process id; used for memory sampling and the launch command line in the manifest")
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "src/llama.cpp")),
                    help="llama.cpp source directory; the manifest records its commit")
    ap.add_argument("--cache-prompt", choices=["on", "off"], default="on", help="cache_prompt in the request body (E4)")
    ap.add_argument("--cold-start-s", type=float, help="cold-start seconds of the server before this Run, written to the manifest")
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S, help="request-level timeout (seconds)")
    a = ap.parse_args(argv)
    workload = json.loads(Path(a.workload).read_text())
    check_slots(a.base_url, a.c)
    asyncio.run(run_all(a, workload))


if __name__ == "__main__":
    main()
