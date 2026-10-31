"""Runner:读 Workload,按开环 Poisson 到达、FIFO 队列 + 并发上限 c 对 llama-server 执行,
写 requests.jsonl、tasks.jsonl、system.jsonl(独立线程每 0.5 s 采样)与 manifest.json。
每个提交的 Task 恰好一行 Task 日志,不论成败。

用法: python -m src.runner.run --workload workloads/dev/default_s1.json --base-url http://127.0.0.1:8080 \
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
from .gen_workload import TOOL_RESULT_TOKENS  # 续步 prompt 估计 = 上一步完整上下文 + 新 tool 文本
from .logs import JsonlWriter
from .manifest import collect_manifest
from .monitor import SystemSampler, default_probe
from .scheduler import POLICIES, ReadyStep, Scheduler

WARMUP_TASKS = 3  # 每次 Run 最先到达的 3 个 Task 标 warmup
REQUEST_TIMEOUT_S = 120
TASK_TIMEOUT_CAP_S = 300
TASK_TIMEOUT_SOLO_FACTOR = 10  # 任务级超时 = min(300, 10 × T_solo),T_solo = deadline_s / 5
DEADLINE_SOLO_FACTOR = 5
REPO_DIR = Path(__file__).resolve().parent.parent.parent


def task_timeout_s(task: dict) -> float:
    return min(TASK_TIMEOUT_CAP_S, TASK_TIMEOUT_SOLO_FACTOR * task["deadline_s"] / DEADLINE_SOLO_FACTOR)


class Executor:  # 注意:术语表里的 Run 是一次完整实验,这里只是执行器
    def __init__(self, sched: Scheduler, client: httpx.AsyncClient, requests_log: JsonlWriter,
                 tasks_log: JsonlWriter, now: Callable[[], float], request_timeout_s: float,
                 cache_prompt: bool = True):
        self.sched, self.client, self.now = sched, client, now
        self.cache_prompt = cache_prompt
        self.requests_log, self.tasks_log = requests_log, tasks_log
        self.request_timeout_s = request_timeout_s
        self._in_flight: dict[int, tuple[int, StreamResult]] = {}  # 在飞 Request → (prompt 估计, 流结果)

    def live_tokens(self) -> int:
        """在飞 Request 占用的 token 估计 = 各自的 prompt 估计 + 已收到的输出 chunk 数(runner 没有 tokenizer)。"""
        return sum(prompt + res.n_pieces for prompt, res in list(self._in_flight.values()))

    async def run_task(self, task: dict, t_arrive: float, warmup: bool) -> None:
        await asyncio.sleep(max(0.0, t_arrive - self.now()))  # 开环:到达与系统状态无关
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
                        await entry.granted.wait()  # 排队中任务级超时 → 取消 → finally 把它移出队列
                        t_first_dispatch = t_first_dispatch if t_first_dispatch is not None else entry.started_at
                        res = await self._request(task, k, step, messages, entry, warmup, prompt_est)
                        output = res.text
                        if res.timings:  # 下一步 prompt ≈ 本步完整上下文 + 新的 tool 文本
                            prompt_est = (res.timings.get("prompt_n", 0) + res.timings.get("cache_n", 0)
                                          + res.timings.get("predicted_n", 0) + TOOL_RESULT_TOKENS)
                    finally:
                        queue_time += entry.queue_time(self.now())
                        self.sched.finish(entry)
                    messages.append({"role": "assistant", "content": output})
                    if step["delay_after_s"] is not None:  # Tool delay 不占并发名额
                        await asyncio.sleep(step["delay_after_s"])
                        idle_time += step["delay_after_s"]
            status = "ok"
        except TimeoutError:
            status = "timeout"
        except RequestFailed as e:
            status = e.status
        except Exception:  # 意料之外的错误也归 error_other,不能拖垮其它 Task
            status = "error_other"
        finally:  # 含被外部取消:每个 Task 仍恰好一行
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
        except asyncio.CancelledError:  # 任务级超时打断了在飞 Request
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
    """ADR 0003:c ≤ np,否则多出的请求会在 server 内部排队,我们的 FIFO 就失效。"""
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
    # 批跑变量:E4 的 cache_prompt、E3 的固定 Tool delay(Workload 顶层键)、冷启动秒数(批跑脚本传入)
    manifest.update(cache_prompt=a.cache_prompt == "on", tool_delay_s=workload.get("tool_delay_s"),
                    cold_start_s=a.cold_start_s)
    probe = default_probe()
    manifest["memory_platform"] = probe.platform if probe else None  # 与 system.jsonl 的平台标签对得上
    stamp(manifest, "start")
    write_manifest(out, manifest)  # Run 中途崩了也留下起点信息;结束时覆盖补上 t_end
    t0 = time.monotonic()
    now = lambda: time.monotonic() - t0  # noqa: E731
    arrivals, t = [], 0.0
    for task in tasks:  # 到达时刻 = 单位间隔 / λ 的累积和
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
    ap.add_argument("--c", type=int, required=True, help="并发上限(在飞 Request 数)")
    ap.add_argument("--lambda", dest="lam", type=float, required=True, help="Task 到达率 (个/秒)")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="fifo")
    ap.add_argument("--n-tasks", type=int, help="只提交 Workload 的前 N 个 Task(默认全部)")
    ap.add_argument("--server-pid", type=int, help="llama-server 进程号;用于内存采样和 manifest 里的启动命令行")
    ap.add_argument("--llama-cpp-dir", default=os.environ.get("LLAMA_CPP_DIR", str(Path.home() / "src/llama.cpp")),
                    help="llama.cpp 源码目录,manifest 记录其 commit")
    ap.add_argument("--cache-prompt", choices=["on", "off"], default="on", help="请求体里的 cache_prompt(E4)")
    ap.add_argument("--cold-start-s", type=float, help="本次 Run 前 server 的冷启动秒数,写进 manifest")
    ap.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S, help="请求级超时(秒)")
    a = ap.parse_args(argv)
    workload = json.loads(Path(a.workload).read_text())
    check_slots(a.base_url, a.c)
    asyncio.run(run_all(a, workload))


if __name__ == "__main__":
    main()
