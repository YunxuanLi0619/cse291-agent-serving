import hashlib
import json
import time
from pathlib import Path

import pytest

from src.runner.gen_workload import generate_workload, write_workload
from src.runner.run import main
from src.runner.scheduler import POLICIES, FifoPolicy, ReadyStep

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
    for task in wl["tasks"]:  # 加速:Tool delay 压到 0.02 s
        for step in task["steps"]:
            if step["delay_after_s"] is not None:
                step["delay_after_s"] = 0.02
    return wl


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(tmp_path, wl, server, *, c=4, lam=1000.0, policy="fifo", extra=(), name="run"):
    """跑一次 Run,返回 (requests, tasks)。lam 很大 = 几乎同时到达。"""
    path, out = tmp_path / f"{name}.json", tmp_path / name
    path.write_text(json.dumps(wl))
    main(["--workload", str(path), "--base-url", server.url, "--out", str(out), "--c", str(c),
          "--lambda", str(lam), "--policy", policy, *extra])
    return jsonl(out / "requests.jsonl"), jsonl(out / "tasks.jsonl")


def with_warmup_prefix(wl, index):
    """前 3 个 Task 作 warmup,第 4 个是被测 Task。"""
    return {**wl, "tasks": wl["tasks"][:3] + [wl["tasks"][index]]}


def first_index(wl, type_):
    return next(i for i, t in enumerate(wl["tasks"]) if t["type"] == type_ and i >= 3)


def test_single_task_writes_one_request_row_and_one_task_row(tmp_path, workload, fake_server):
    fake_server.ttft_s = 0.2
    i = first_index(workload, "short")
    task = workload["tasks"][i]

    requests, tasks = run(tmp_path, with_warmup_prefix(workload, i), fake_server)

    r = next(r for r in requests if r["task_id"] == task["task_id"])
    t = next(t for t in tasks if t["task_id"] == task["task_id"])
    assert set(r) == REQUEST_KEYS and set(t) == TASK_KEYS
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


def test_agent_task_logs_one_request_per_step_and_accumulates_context(tmp_path, workload, fake_server):
    i = first_index(workload, "agent")
    task = workload["tasks"][i]
    wl = {**workload, "tasks": [task]}

    requests, tasks = run(tmp_path, wl, fake_server)

    n = len(task["steps"])
    assert [r["step"] for r in requests] == list(range(1, n + 1))
    assert tasks[0]["n_steps"] == n
    assert tasks[0]["idle_time_total"] == pytest.approx(0.02 * (n - 1))
    # 后一个 Request 的 prompt 追加了上一步的真实输出和新的 tool result
    assert [len(b["messages"]) for b in fake_server.requests] == [2 * k - 1 for k in range(1, n + 1)]
    assert fake_server.requests[1]["messages"][1] == {"role": "assistant", "content": task["needle"] + "." * 127}
    assert tasks[0]["quality_pass"] is True


def test_every_task_gets_exactly_one_task_row_and_in_flight_never_exceeds_c(tmp_path, workload, fake_server):
    requests, tasks = run(tmp_path, workload, fake_server, c=2)

    assert sorted(t["task_id"] for t in tasks) == sorted(t["task_id"] for t in workload["tasks"])
    assert fake_server.max_in_flight == 2  # 打满 c,但从不超过
    spans = sorted((r["t_start"], r["t_end"]) for r in requests)
    for k, (start, _) in enumerate(spans):  # 日志自证:任一 Request 开始时,此前未结束的 < c
        assert sum(1 for _, end in spans[:k] if end > start) < 2


def test_arrival_times_are_unit_gaps_divided_by_lambda(tmp_path, workload, fake_server):
    wl = {**workload, "tasks": workload["tasks"][:3]}
    for task, gap in zip(wl["tasks"], [0.5, 0.25, 0.25]):
        task["unit_gap_s"] = gap
        task["steps"] = task["steps"][:1]

    _, slow = run(tmp_path, wl, fake_server, lam=2.5, name="slow")  # 单位间隔 / λ = 0.2, 0.1, 0.1 s
    _, fast = run(tmp_path, wl, fake_server, lam=5.0, name="fast")  # 同一份 Workload,λ 翻倍

    arrivals = lambda rows: [t["t_arrive"] for t in sorted(rows, key=lambda t: t["task_id"])]  # 日志按完成顺序写
    assert sorted(t["task_id"] for t in slow) == sorted(t["task_id"] for t in fast) == [t["task_id"] for t in wl["tasks"]]
    assert arrivals(slow) == pytest.approx([0.2, 0.3, 0.4])
    assert arrivals(fast) == pytest.approx([0.1, 0.15, 0.2])  # 形状相同,时间轴缩放


class LifoPolicy:
    def pick(self, ready):
        return max(ready, key=lambda s: (s.ready_at, s.seq))


def test_fifo_policy_picks_earliest_ready_step():
    ready = [ReadyStep("a", 1, ready_at=2.0, seq=0), ReadyStep("b", 1, ready_at=1.0, seq=1),
             ReadyStep("c", 1, ready_at=1.0, seq=2)]
    assert FifoPolicy().pick(ready).task_id == "b"  # 就绪时刻最早;并列时先入队的


def test_policy_decides_dispatch_order_and_is_replaceable(tmp_path, workload, fake_server, monkeypatch):
    wl = {**workload, "tasks": workload["tasks"][:4]}
    for task in wl["tasks"]:  # 全部同时到达,c=1 逼出排队
        task["unit_gap_s"] = 0.0
        task["steps"] = task["steps"][:1]
    monkeypatch.setitem(POLICIES, "lifo", LifoPolicy)

    _, fifo_tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0, policy="fifo", name="fifo")
    _, lifo_tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0, policy="lifo", name="lifo")

    def order(tasks):
        return [t["task_id"] for t in sorted(tasks, key=lambda t: t["t_first_dispatch"])]

    ids = [t["task_id"] for t in wl["tasks"]]
    assert order(fifo_tasks) == ids
    assert order(lifo_tasks) == [ids[0], ids[3], ids[2], ids[1]]


def only(tasks, *, type_, n=1, steps=None):
    """Workload 里前 n 个指定类型的 Task(可截短步数)。"""
    picked = [t for t in tasks if t["type"] == type_][:n]
    return [{**t, "steps": t["steps"][:steps] if steps else t["steps"]} for t in picked]


def subset(wl, tasks, **gaps):
    return {**wl, "tasks": [{**t, "unit_gap_s": 0.0, **gaps} for t in tasks]}


def wait_until(cond, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.02)
    return cond()


def test_run_refuses_when_server_has_fewer_slots_than_c(tmp_path, workload, fake_server):
    fake_server.n_slots = 2
    with pytest.raises(SystemExit, match="slots"):
        run(tmp_path, subset(workload, only(workload["tasks"], type_="short")), fake_server, c=3)
    assert fake_server.requests == []


def test_first_three_tasks_by_arrival_are_warmup_in_both_logs(tmp_path, workload, fake_server):
    tasks = [{**t, "steps": t["steps"][:1]} for t in workload["tasks"][:5]]
    wl = {**workload, "tasks": [{**t, "unit_gap_s": 1.0} for t in tasks]}

    requests, task_rows = run(tmp_path, wl, fake_server)

    warm = {t["task_id"]: t["warmup"] for t in task_rows}
    assert [warm[t["task_id"]] for t in tasks] == [True, True, True, False, False]
    assert {r["task_id"]: r["warmup"] for r in requests} == warm


def test_request_timeout_disconnects_and_logs_timeout_for_request_and_task(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 1.5}  # 长 prefill 期间超时
    agent = only(workload["tasks"], type_="agent")
    wl = subset(workload, agent)

    requests, tasks = run(tmp_path, wl, fake_server, extra=["--timeout", "1"])

    assert len(tasks) == 1 and tasks[0]["status"] == "timeout" and tasks[0]["quality_pass"] is False
    assert len(requests) == 1  # 剩余 Step 不再执行
    assert len(fake_server.requests) == 1
    r = requests[0]
    assert (r["status"], r["error_type"], r["t_first_token"]) == ("timeout", "request_timeout", None)
    assert r["t_end"] - r["t_start"] == pytest.approx(1.0, abs=0.2)
    assert wait_until(lambda: fake_server.aborted == 1)  # 连接确实被我们断开了(server 在 prefill 结束后写回才发现)


def test_task_queued_past_its_timeout_is_removed_and_logged_without_running_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 1.0} if n == 1 else None
    blocker = only(workload["tasks"], type_="short")
    victim = only(workload["tasks"], type_="agent")
    victim[0]["deadline_s"] = 0.2  # 任务级超时 = min(300, 10 × 0.2/5) = 0.4 s
    wl = {**workload, "tasks": [{**blocker[0], "unit_gap_s": 0.0}, {**victim[0], "unit_gap_s": 0.05}]}

    requests, tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0)

    v = next(t for t in tasks if t["task_id"] == victim[0]["task_id"])
    assert v["status"] == "timeout" and v["t_first_dispatch"] is None
    assert v["t_done"] - v["t_arrive"] == pytest.approx(0.4, abs=0.15)
    assert v["queue_time_total"] == pytest.approx(0.4, abs=0.15)
    assert [r["task_id"] for r in requests] == [blocker[0]["task_id"]]  # 它从未发出 Request
    assert len(fake_server.requests) == 1


def test_task_timeout_interrupts_in_flight_request(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 1.5}
    task = only(workload["tasks"], type_="short")
    task[0]["deadline_s"] = 0.2

    requests, tasks = run(tmp_path, subset(workload, task), fake_server)

    assert tasks[0]["status"] == "timeout"
    assert (requests[0]["status"], requests[0]["error_type"]) == ("timeout", "task_timeout")


def test_task_timeout_during_tool_delay_skips_remaining_steps(tmp_path, workload, fake_server):
    task = only(workload["tasks"], type_="agent")
    task[0]["deadline_s"] = 0.5  # 任务级超时 1 s,第 1 步后的 Tool delay 是 5 s
    task[0]["steps"][0]["delay_after_s"] = 5.0

    requests, tasks = run(tmp_path, subset(workload, task), fake_server)

    assert tasks[0]["status"] == "timeout"
    assert len(requests) == 1 and requests[0]["status"] == "ok"
    assert tasks[0]["t_done"] == pytest.approx(1.0, abs=0.2)


@pytest.mark.parametrize("fault, status, error_type", [
    ({"kind": "http", "status": 503}, "error_http", "http_503"),
    ({"kind": "ctx_exceeded"}, "error_ctx_exceeded", "ctx_exceeded"),
    ({"kind": "pool_exhausted"}, "error_http", "pool_exhausted"),  # 与普通 ctx 超限分开记录
    ({"kind": "disconnect", "after_chunks": 3}, "error_other", "incomplete_stream"),
    ({"kind": "stream_pool_exhausted", "after_chunks": 3}, "error_http", "pool_exhausted"),  # 已发过 chunk:HTTP 200 + SSE error
    ({"kind": "garbage", "after_chunks": 3}, "error_other", "malformed_stream"),  # 不能让一个坏 Task 拖垮整个 Run
])
def test_failures_are_classified_and_never_drop_a_task(tmp_path, workload, fake_server, fault, status, error_type):
    tasks = only(workload["tasks"], type_="agent", n=2)
    bad, good = tasks
    fake_server.fault = lambda body, n: fault if bad["needle"] in body["messages"][0]["content"] else None
    bad_steps_before = len(bad["steps"])
    assert bad_steps_before >= 3

    requests, rows = run(tmp_path, subset(workload, tasks), fake_server)

    by_id = {t["task_id"]: t for t in rows}
    assert set(by_id) == {bad["task_id"], good["task_id"]}  # 分母 = 全部已提交 Task
    assert by_id[bad["task_id"]]["status"] == status and by_id[bad["task_id"]]["quality_pass"] is False
    assert by_id[good["task_id"]]["status"] == "ok"
    bad_reqs = [r for r in requests if r["task_id"] == bad["task_id"]]
    assert len(bad_reqs) == 1 and bad_reqs[0]["error_type"] == error_type and bad_reqs[0]["status"] == status
    assert len([r for r in requests if r["task_id"] == good["task_id"]]) == len(good["steps"])


def agent_task(workload, delay_s, n_steps=3, **overrides):
    task = only(workload["tasks"], type_="agent")[0]
    assert len(task["steps"]) >= n_steps
    task = {**task, "steps": [dict(s) for s in task["steps"][:n_steps]], **overrides}
    for step in task["steps"][:-1]:
        step["delay_after_s"] = delay_s
    task["steps"][-1]["delay_after_s"] = None
    return task


def test_each_step_logs_prompt_tokens_of_the_context_it_actually_sent(tmp_path, workload, fake_server, tokenizer):
    task = agent_task(workload, 0.02, n_steps=4)

    requests, _ = run(tmp_path, {**workload, "tasks": [{**task, "unit_gap_s": 0.0}]}, fake_server)

    logged = [r["prompt_tokens"] for r in sorted(requests, key=lambda r: r["step"])]
    assert logged == [tokenizer.prompt_tokens(b["messages"]) for b in fake_server.requests]
    assert logged == sorted(set(logged))  # 上下文逐步增长
    contents = [m["content"] for m in fake_server.requests[-1]["messages"]]
    assert contents[0] == task["steps"][0]["user_content"] and contents[2] == task["steps"][1]["user_content"]
    assert contents[1] == task["needle"] + "." * 127  # 前一步的真实输出


def test_tool_delay_releases_the_slot_and_continuation_requeues_behind_ready_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 0.5} if n == 2 else None  # n=2 是 B 的请求
    a = agent_task(workload, 0.2, n_steps=2, unit_gap_s=0.0)
    b, c = (dict(t, unit_gap_s=g) for t, g in zip(only(workload["tasks"], type_="short", n=2), (0.05, 0.2)))
    wl = {**workload, "tasks": [a, b, c]}  # C 在 A 的续步就绪之前入队(0.25 s < 约 0.4 s)

    requests, tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0)

    def req(task, step=1):
        return next(r for r in requests if r["task_id"] == task["task_id"] and r["step"] == step)

    a1, a2, b1, c1 = req(a), req(a, 2), req(b), req(c)
    assert b1["t_start"] == pytest.approx(a1["t_end"], abs=0.05)  # B 在 A 的 Tool delay 里拿到名额
    assert a2["t_submit"] == pytest.approx(a1["t_end"] + 0.2, abs=0.05)  # delay 结束 = 续步就绪
    assert c1["t_submit"] < a2["t_submit"] < b1["t_end"]  # 续步就绪时 C 已在排队,B 还占着名额
    assert c1["t_start"] == pytest.approx(b1["t_end"], abs=0.05)  # 队尾:C 先于续步
    assert a2["t_start"] == pytest.approx(c1["t_end"], abs=0.05)
    assert next(t for t in tasks if t["task_id"] == a["task_id"])["queue_time_total"] > 0.3


def test_continuation_is_dispatched_right_after_delay_when_nothing_else_is_queued(tmp_path, workload, fake_server):
    a = agent_task(workload, 0.3, n_steps=3, unit_gap_s=0.0)

    requests, tasks = run(tmp_path, {**workload, "tasks": [a]}, fake_server, c=1)

    r = sorted(requests, key=lambda r: r["step"])
    for prev, nxt in zip(r, r[1:]):
        assert nxt["t_submit"] - prev["t_end"] == pytest.approx(0.3, abs=0.05)  # 只隔 Tool delay
        assert nxt["t_start"] - nxt["t_submit"] < 0.05  # 没有排队之外的耽误,也没有排队
    assert tasks[0]["idle_time_total"] == pytest.approx(0.6) and tasks[0]["queue_time_total"] < 0.1


def test_ctx_exceeded_at_a_later_step_fails_the_whole_task_and_stops_remaining_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "ctx_exceeded"} if len(body["messages"]) >= 3 else None
    a = agent_task(workload, 0.02, n_steps=4, unit_gap_s=0.0)

    requests, tasks = run(tmp_path, {**workload, "tasks": [a]}, fake_server)

    assert (tasks[0]["status"], tasks[0]["quality_pass"]) == ("error_ctx_exceeded", False)
    assert [(r["step"], r["status"]) for r in requests] == [(1, "ok"), (2, "error_ctx_exceeded")]
    assert len(fake_server.requests) == 2  # 第 3、4 步不再发出
    assert tasks[0]["idle_time_total"] == pytest.approx(0.02)  # 失败之后不再等 Tool delay


def test_quality_pass_requires_the_needle_in_the_final_output(tmp_path, workload, fake_server):
    wrong = agent_task(workload, 0.02, unit_gap_s=0.0, needle="KEY-zzzzzzzz")  # 模型复述的是别的 key

    _, tasks = run(tmp_path, {**workload, "tasks": [wrong]}, fake_server)

    assert (tasks[0]["status"], tasks[0]["quality_pass"]) == ("ok", False)


def test_quality_pass_judges_only_the_final_step_output(tmp_path, workload, fake_server):
    a = agent_task(workload, 0.02, n_steps=3, unit_gap_s=0.0)
    last_turn = lambda body: len(body["messages"]) == 5  # 第 3 步的请求带 5 条消息

    fake_server.fault = lambda body, n: {"kind": "no_key"} if last_turn(body) else None  # 前两步含 key,末步不含
    _, forgot_at_end = run(tmp_path, {**workload, "tasks": [a]}, fake_server, name="forgot")
    fake_server.fault = lambda body, n: None if last_turn(body) else {"kind": "no_key"}  # 前两步不含,末步含
    _, recalled_at_end = run(tmp_path, {**workload, "tasks": [a]}, fake_server, name="recalled")

    assert (forgot_at_end[0]["status"], forgot_at_end[0]["quality_pass"]) == ("ok", False)
    assert (recalled_at_end[0]["status"], recalled_at_end[0]["quality_pass"]) == ("ok", True)


def test_run_writes_system_log_and_manifest_next_to_the_other_logs(tmp_path, workload, fake_server):
    fake_server.ttft_s = 0.8  # 让至少一次 0.5 s 采样落在请求在飞期间
    i = first_index(workload, "short")
    wl = with_warmup_prefix(workload, i)

    run(tmp_path, wl, fake_server, c=1, name="sys")
    out = tmp_path / "sys"

    system = jsonl(out / "system.jsonl")
    assert len(system) >= 2
    # 假 server 的 /metrics 与 /slots 都可读;没给 --server-pid,所以只有内存这一路报错(不静默)
    assert all(r["error"] is None or r["error"].startswith("memory:") for r in system)
    assert all("metrics" not in (r["error"] or "") and "slots" not in (r["error"] or "") for r in system)
    assert all(len(r["slots"]) == fake_server.n_slots for r in system)
    assert any(r["requests_processing"] == 1 and r["slots"][0]["is_processing"] for r in system)
    # 在飞时 live_tokens 至少是该 Task 第 1 步的 prompt 估计;在飞 Request 只有 c=1 个
    in_flight = [r for r in system if r["requests_processing"] == 1]
    assert max(r["live_tokens"] for r in in_flight) >= min(t["prompt_tokens_measured"] for t in wl["tasks"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["c"], manifest["lambda"], manifest["seed"]) == (1, 1000.0, wl["seed"])
    assert manifest["workload_sha256"] == hashlib.sha256((tmp_path / "sys.json").read_bytes()).hexdigest()
    assert manifest["repo_git_sha"]
    assert manifest["memory_platform"] == system[0]["platform"]
    assert manifest["t_end"] >= manifest["t_start"]
    assert manifest["t_start_iso"] and manifest["t_end_iso"]
