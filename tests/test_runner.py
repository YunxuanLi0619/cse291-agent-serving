import json
from pathlib import Path

import pytest

from src.runner.gen_workload import generate_workload, write_workload
from src.runner.run import main

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"

REQUEST_KEYS = {"task_id", "step", "t_submit", "t_start", "t_first_token", "t_end", "prompt_tokens",
                "cached_tokens", "output_tokens", "status", "error_type", "warmup", "id_slot"}
TASK_KEYS = {"task_id", "type", "t_arrive", "t_first_dispatch", "t_done", "queue_time_total",
             "idle_time_total", "n_steps", "status", "quality_pass", "deadline_s", "warmup"}


@pytest.fixture
def workload(tmp_path, tokenizer):
    mix = json.loads((CONFIGS / "mix_dev.json").read_text())
    t_solo = json.loads((CONFIGS / "t_solo.json").read_text())
    wl = generate_workload(mix, t_solo, seed=1, n_tasks=10, tokenizer=tokenizer)
    path = tmp_path / "workload.json"
    write_workload(wl, path)
    return path, wl


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(tmp_path, workload_path, server, index):
    out = tmp_path / "run"
    main(["--workload", str(workload_path), "--base-url", server.url, "--out", str(out),
          "--task-index", str(index)])
    return jsonl(out / "requests.jsonl"), jsonl(out / "tasks.jsonl")


def first_index(wl, type_):
    return next(i for i, t in enumerate(wl["tasks"]) if t["type"] == type_ and i >= 3)


def test_single_task_writes_one_request_row_and_one_task_row(tmp_path, workload, fake_server):
    path, wl = workload
    fake_server.ttft_s = 0.2
    i = first_index(wl, "short")
    task = wl["tasks"][i]

    requests, tasks = run(tmp_path, path, fake_server, i)

    assert len(requests) == 1 and len(tasks) == 1
    r, t = requests[0], tasks[0]
    assert set(r) == REQUEST_KEYS and set(t) == TASK_KEYS
    assert r["task_id"] == t["task_id"] == task["task_id"]
    assert r["step"] == 1
    assert r["t_submit"] <= r["t_start"] <= r["t_first_token"] <= r["t_end"]
    assert r["t_first_token"] - r["t_start"] >= 0.15  # server 在首个 token 前 sleep 了 0.2 s
    assert r["prompt_tokens"] == task["prompt_tokens_measured"]  # server 报告的 = 生成器校准的
    assert r["cached_tokens"] == 0
    assert r["output_tokens"] == 64
    assert (r["status"], r["error_type"], r["warmup"], r["id_slot"]) == ("ok", None, False, None)
    assert (t["type"], t["n_steps"], t["status"], t["quality_pass"]) == ("short", 1, "ok", True)
    assert t["deadline_s"] == task["deadline_s"] and t["warmup"] is False
    assert t["t_arrive"] <= t["t_first_dispatch"] <= t["t_done"]
    assert t["queue_time_total"] >= 0 and t["idle_time_total"] == 0


def test_first_three_tasks_of_a_run_are_marked_warmup(tmp_path, workload, fake_server):
    path, _ = workload
    requests, tasks = run(tmp_path, path, fake_server, 2)
    assert requests[0]["warmup"] is True and tasks[0]["warmup"] is True


def test_agent_task_logs_one_request_per_step_and_accumulates_context(tmp_path, workload, fake_server):
    path, wl = workload
    i = first_index(wl, "agent")
    task = wl["tasks"][i]
    for step in task["steps"]:  # 加速:Tool delay 压到 0.05 s
        if step["delay_after_s"] is not None:
            step["delay_after_s"] = 0.05
    path.write_text(json.dumps(wl))

    requests, tasks = run(tmp_path, path, fake_server, i)

    n = len(task["steps"])
    assert [r["step"] for r in requests] == list(range(1, n + 1))
    assert tasks[0]["n_steps"] == n
    assert tasks[0]["idle_time_total"] == pytest.approx(0.05 * (n - 1))
    # 后一个 Request 的 prompt 追加了上一步的真实输出和新的 tool result
    assert [len(b["messages"]) for b in fake_server.requests] == [2 * k - 1 for k in range(1, n + 1)]
    assert fake_server.requests[1]["messages"][1] == {"role": "assistant", "content": task["needle"] + "." * 127}
    assert tasks[0]["quality_pass"] is True
