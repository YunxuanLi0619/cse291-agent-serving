"""Runner:读 Workload,对 llama-server 执行 Task,写 requests.jsonl 与 tasks.jsonl。

用法: python -m src.runner.run --workload workloads/dev/default_s1.json --base-url http://127.0.0.1:8080 \
        --out results/raw/run_x --task-index 0
"""
import argparse
import json
import time
from pathlib import Path

import httpx

from .client import stream_chat
from .logs import JsonlWriter

WARMUP_TASKS = 3  # 每次 Run 最先到达的 3 个 Task 标 warmup


def run_task(task: dict, warmup: bool, client: httpx.Client, requests_log: JsonlWriter,
             tasks_log: JsonlWriter, now) -> None:
    t_arrive = now()
    messages, queue_time, idle_time = [], 0.0, 0.0
    t_first_dispatch = None
    output = ""
    for k, step in enumerate(task["steps"], start=1):
        messages.append({"role": "user", "content": step["user_content"]})
        t_submit = now()
        t_start = now()
        t_first_dispatch = t_first_dispatch if t_first_dispatch is not None else t_start
        queue_time += t_start - t_submit
        res = stream_chat(client, messages, step["max_tokens"], now)
        output = res.text
        timings = res.timings or {}
        requests_log.write({
            "task_id": task["task_id"], "step": k, "t_submit": t_submit, "t_start": t_start,
            "t_first_token": res.t_first_token, "t_end": res.t_end,
            "prompt_tokens": timings.get("prompt_n", 0) + timings.get("cache_n", 0),
            "cached_tokens": timings.get("cache_n", 0), "output_tokens": timings.get("predicted_n", 0),
            "status": "ok", "error_type": None, "warmup": warmup, "id_slot": None,
        })
        messages.append({"role": "assistant", "content": res.text})
        if step["delay_after_s"] is not None:
            time.sleep(step["delay_after_s"])
            idle_time += step["delay_after_s"]
    tasks_log.write({
        "task_id": task["task_id"], "type": task["type"], "t_arrive": t_arrive,
        "t_first_dispatch": t_first_dispatch, "t_done": now(), "queue_time_total": queue_time,
        "idle_time_total": idle_time, "n_steps": len(task["steps"]), "status": "ok",
        "quality_pass": task["needle"] in output, "deadline_s": task["deadline_s"], "warmup": warmup,
    })


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-index", type=int, default=0, help="执行 Workload 里的第几个 Task")
    a = ap.parse_args(argv)
    task = json.loads(Path(a.workload).read_text())["tasks"][a.task_index]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    requests_log, tasks_log = JsonlWriter(out / "requests.jsonl"), JsonlWriter(out / "tasks.jsonl")
    try:
        with httpx.Client(base_url=a.base_url) as client:
            run_task(task, a.task_index < WARMUP_TASKS, client, requests_log, tasks_log,
                     lambda: time.monotonic() - t0)
    finally:
        requests_log.close()
        tasks_log.close()


if __name__ == "__main__":
    main()
