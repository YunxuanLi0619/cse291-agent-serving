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
    for task in wl["tasks"]:  # speed-up: squeeze Tool delay to 0.02 s
        for step in task["steps"]:
            if step["delay_after_s"] is not None:
                step["delay_after_s"] = 0.02
    return wl


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(tmp_path, wl, server, *, c=4, lam=1000.0, policy="fifo", extra=(), name="run"):
    """Run one Run and return (requests, tasks). A very large lam = nearly simultaneous arrivals."""
    path, out = tmp_path / f"{name}.json", tmp_path / name
    path.write_text(json.dumps(wl))
    main(["--workload", str(path), "--base-url", server.url, "--out", str(out), "--c", str(c),
          "--lambda", str(lam), "--policy", policy, *extra])
    return jsonl(out / "requests.jsonl"), jsonl(out / "tasks.jsonl")


def with_warmup_prefix(wl, index):
    """The first 3 Tasks act as warmup; the 4th is the Task under test."""
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
    assert r["t_first_token"] - r["t_start"] >= 0.15  # the server slept 0.2 s before the first token
    assert r["prompt_tokens"] == task["prompt_tokens_measured"]  # what the server reports = what the generator calibrated
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
    # the prompt of the later Request appends the previous step's real output and the new tool result
    assert [len(b["messages"]) for b in fake_server.requests] == [2 * k - 1 for k in range(1, n + 1)]
    assert fake_server.requests[1]["messages"][1] == {"role": "assistant", "content": task["needle"] + "." * 127}
    assert tasks[0]["quality_pass"] is True


def test_every_task_gets_exactly_one_task_row_and_in_flight_never_exceeds_c(tmp_path, workload, fake_server):
    requests, tasks = run(tmp_path, workload, fake_server, c=2)

    assert sorted(t["task_id"] for t in tasks) == sorted(t["task_id"] for t in workload["tasks"])
    assert fake_server.max_in_flight == 2  # c is saturated but never exceeded
    spans = sorted((r["t_start"], r["t_end"]) for r in requests)
    for k, (start, _) in enumerate(spans):  # self-evident from the log: when any Request starts, fewer than c earlier ones are unfinished
        assert sum(1 for _, end in spans[:k] if end > start) < 2


def test_arrival_times_are_unit_gaps_divided_by_lambda(tmp_path, workload, fake_server):
    wl = {**workload, "tasks": workload["tasks"][:3]}
    for task, gap in zip(wl["tasks"], [0.5, 0.25, 0.25]):
        task["unit_gap_s"] = gap
        task["steps"] = task["steps"][:1]

    _, slow = run(tmp_path, wl, fake_server, lam=2.5, name="slow")  # unit interval / λ = 0.2, 0.1, 0.1 s
    _, fast = run(tmp_path, wl, fake_server, lam=5.0, name="fast")  # the same Workload, λ doubled

    arrivals = lambda rows: [t["t_arrive"] for t in sorted(rows, key=lambda t: t["task_id"])]  # the log is written in completion order
    assert sorted(t["task_id"] for t in slow) == sorted(t["task_id"] for t in fast) == [t["task_id"] for t in wl["tasks"]]
    assert arrivals(slow) == pytest.approx([0.2, 0.3, 0.4])
    assert arrivals(fast) == pytest.approx([0.1, 0.15, 0.2])  # same shape, time axis scaled


class LifoPolicy:
    def pick(self, ready):
        return max(ready, key=lambda s: (s.ready_at, s.seq))


def test_fifo_policy_picks_earliest_ready_step():
    ready = [ReadyStep("a", 1, ready_at=2.0, seq=0), ReadyStep("b", 1, ready_at=1.0, seq=1),
             ReadyStep("c", 1, ready_at=1.0, seq=2)]
    assert FifoPolicy().pick(ready).task_id == "b"  # earliest ready time; on ties, the one enqueued first


def test_policy_decides_dispatch_order_and_is_replaceable(tmp_path, workload, fake_server, monkeypatch):
    wl = {**workload, "tasks": workload["tasks"][:4]}
    for task in wl["tasks"]:  # all arrive at the same time; c=1 forces queueing
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
    """The first n Tasks of the given type in the Workload (number of steps can be truncated)."""
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
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 1.5}  # timeout during a long prefill
    agent = only(workload["tasks"], type_="agent")
    wl = subset(workload, agent)

    requests, tasks = run(tmp_path, wl, fake_server, extra=["--timeout", "1"])

    assert len(tasks) == 1 and tasks[0]["status"] == "timeout" and tasks[0]["quality_pass"] is False
    assert len(requests) == 1  # the remaining Steps are not executed
    assert len(fake_server.requests) == 1
    r = requests[0]
    assert (r["status"], r["error_type"], r["t_first_token"]) == ("timeout", "request_timeout", None)
    assert r["t_end"] - r["t_start"] == pytest.approx(1.0, abs=0.2)
    assert wait_until(lambda: fake_server.aborted == 1)  # the connection really was closed by us (the server notices only when writing back after prefill ends)


def test_task_queued_past_its_timeout_is_removed_and_logged_without_running_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 1.0} if n == 1 else None
    blocker = only(workload["tasks"], type_="short")
    victim = only(workload["tasks"], type_="agent")
    victim[0]["deadline_s"] = 0.2  # task-level timeout = min(300, 10 × 0.2/5) = 0.4 s
    wl = {**workload, "tasks": [{**blocker[0], "unit_gap_s": 0.0}, {**victim[0], "unit_gap_s": 0.05}]}

    requests, tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0)

    v = next(t for t in tasks if t["task_id"] == victim[0]["task_id"])
    assert v["status"] == "timeout" and v["t_first_dispatch"] is None
    assert v["t_done"] - v["t_arrive"] == pytest.approx(0.4, abs=0.15)
    assert v["queue_time_total"] == pytest.approx(0.4, abs=0.15)
    assert [r["task_id"] for r in requests] == [blocker[0]["task_id"]]  # it never sent a Request
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
    task[0]["deadline_s"] = 0.5  # task-level timeout 1 s, and the Tool delay after step 1 is 5 s
    task[0]["steps"][0]["delay_after_s"] = 5.0

    requests, tasks = run(tmp_path, subset(workload, task), fake_server)

    assert tasks[0]["status"] == "timeout"
    assert len(requests) == 1 and requests[0]["status"] == "ok"
    assert tasks[0]["t_done"] == pytest.approx(1.0, abs=0.2)


@pytest.mark.parametrize("fault, status, error_type", [
    ({"kind": "http", "status": 503}, "error_http", "http_503"),
    ({"kind": "ctx_exceeded"}, "error_ctx_exceeded", "ctx_exceeded"),
    ({"kind": "pool_exhausted"}, "error_http", "pool_exhausted"),  # recorded separately from ordinary ctx overflow
    ({"kind": "disconnect", "after_chunks": 3}, "error_other", "incomplete_stream"),
    ({"kind": "stream_pool_exhausted", "after_chunks": 3}, "error_http", "pool_exhausted"),  # chunks already sent: HTTP 200 + SSE error
    ({"kind": "garbage", "after_chunks": 3}, "error_other", "malformed_stream"),  # one bad Task must not bring down the whole Run
])
def test_failures_are_classified_and_never_drop_a_task(tmp_path, workload, fake_server, fault, status, error_type):
    tasks = only(workload["tasks"], type_="agent", n=2)
    bad, good = tasks
    fake_server.fault = lambda body, n: fault if bad["needle"] in body["messages"][0]["content"] else None
    bad_steps_before = len(bad["steps"])
    assert bad_steps_before >= 3

    requests, rows = run(tmp_path, subset(workload, tasks), fake_server)

    by_id = {t["task_id"]: t for t in rows}
    assert set(by_id) == {bad["task_id"], good["task_id"]}  # denominator = all submitted Tasks
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
    assert logged == sorted(set(logged))  # context grows step by step
    contents = [m["content"] for m in fake_server.requests[-1]["messages"]]
    assert contents[0] == task["steps"][0]["user_content"] and contents[2] == task["steps"][1]["user_content"]
    assert contents[1] == task["needle"] + "." * 127  # the previous step's real output


def test_tool_delay_releases_the_slot_and_continuation_requeues_behind_ready_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "slow", "ttft_s": 0.5} if n == 2 else None  # n=2 is B's request
    a = agent_task(workload, 0.2, n_steps=2, unit_gap_s=0.0)
    b, c = (dict(t, unit_gap_s=g) for t, g in zip(only(workload["tasks"], type_="short", n=2), (0.05, 0.2)))
    wl = {**workload, "tasks": [a, b, c]}  # C is enqueued before A's continuation step is ready (0.25 s < about 0.4 s)

    requests, tasks = run(tmp_path, wl, fake_server, c=1, lam=1.0)

    def req(task, step=1):
        return next(r for r in requests if r["task_id"] == task["task_id"] and r["step"] == step)

    a1, a2, b1, c1 = req(a), req(a, 2), req(b), req(c)
    assert b1["t_start"] == pytest.approx(a1["t_end"], abs=0.05)  # B gets the slot during A's Tool delay
    assert a2["t_submit"] == pytest.approx(a1["t_end"] + 0.2, abs=0.05)  # delay ends = continuation step is ready
    assert c1["t_submit"] < a2["t_submit"] < b1["t_end"]  # when the continuation is ready C is already queued and B still holds the slot
    assert c1["t_start"] == pytest.approx(b1["t_end"], abs=0.05)  # tail of the queue: C goes before the continuation step
    assert a2["t_start"] == pytest.approx(c1["t_end"], abs=0.05)
    assert next(t for t in tasks if t["task_id"] == a["task_id"])["queue_time_total"] > 0.3


def test_continuation_is_dispatched_right_after_delay_when_nothing_else_is_queued(tmp_path, workload, fake_server):
    a = agent_task(workload, 0.3, n_steps=3, unit_gap_s=0.0)

    requests, tasks = run(tmp_path, {**workload, "tasks": [a]}, fake_server, c=1)

    r = sorted(requests, key=lambda r: r["step"])
    for prev, nxt in zip(r, r[1:]):
        assert nxt["t_submit"] - prev["t_end"] == pytest.approx(0.3, abs=0.05)  # only the Tool delay in between
        assert nxt["t_start"] - nxt["t_submit"] < 0.05  # no delay other than queueing, and no queueing either
    assert tasks[0]["idle_time_total"] == pytest.approx(0.6) and tasks[0]["queue_time_total"] < 0.1


def test_ctx_exceeded_at_a_later_step_fails_the_whole_task_and_stops_remaining_steps(tmp_path, workload, fake_server):
    fake_server.fault = lambda body, n: {"kind": "ctx_exceeded"} if len(body["messages"]) >= 3 else None
    a = agent_task(workload, 0.02, n_steps=4, unit_gap_s=0.0)

    requests, tasks = run(tmp_path, {**workload, "tasks": [a]}, fake_server)

    assert (tasks[0]["status"], tasks[0]["quality_pass"]) == ("error_ctx_exceeded", False)
    assert [(r["step"], r["status"]) for r in requests] == [(1, "ok"), (2, "error_ctx_exceeded")]
    assert len(fake_server.requests) == 2  # steps 3 and 4 are not sent
    assert tasks[0]["idle_time_total"] == pytest.approx(0.02)  # no more waiting for Tool delay after the failure


def test_quality_pass_requires_the_needle_in_the_final_output(tmp_path, workload, fake_server):
    wrong = agent_task(workload, 0.02, unit_gap_s=0.0, needle="KEY-zzzzzzzz")  # the model repeats a different key

    _, tasks = run(tmp_path, {**workload, "tasks": [wrong]}, fake_server)

    assert (tasks[0]["status"], tasks[0]["quality_pass"]) == ("ok", False)


def test_quality_pass_judges_only_the_final_step_output(tmp_path, workload, fake_server):
    a = agent_task(workload, 0.02, n_steps=3, unit_gap_s=0.0)
    last_turn = lambda body: len(body["messages"]) == 5  # the request of step 3 carries 5 messages

    fake_server.fault = lambda body, n: {"kind": "no_key"} if last_turn(body) else None  # first two steps contain the key, the last does not
    _, forgot_at_end = run(tmp_path, {**workload, "tasks": [a]}, fake_server, name="forgot")
    fake_server.fault = lambda body, n: None if last_turn(body) else {"kind": "no_key"}  # first two steps do not contain it, the last does
    _, recalled_at_end = run(tmp_path, {**workload, "tasks": [a]}, fake_server, name="recalled")

    assert (forgot_at_end[0]["status"], forgot_at_end[0]["quality_pass"]) == ("ok", False)
    assert (recalled_at_end[0]["status"], recalled_at_end[0]["quality_pass"]) == ("ok", True)


def test_run_writes_system_log_and_manifest_next_to_the_other_logs(tmp_path, workload, fake_server):
    fake_server.ttft_s = 0.8  # make at least one 0.5 s sample fall while a request is in flight
    i = first_index(workload, "short")
    wl = with_warmup_prefix(workload, i)

    run(tmp_path, wl, fake_server, c=1, name="sys")
    out = tmp_path / "sys"

    system = jsonl(out / "system.jsonl")
    assert len(system) >= 2
    # the fake server's /metrics and /slots are both readable; --server-pid was not given, so only the memory source errors (not silent)
    assert all(r["error"] is None or r["error"].startswith("memory:") for r in system)
    assert all("metrics" not in (r["error"] or "") and "slots" not in (r["error"] or "") for r in system)
    assert all(len(r["slots"]) == fake_server.n_slots for r in system)
    assert any(r["requests_processing"] == 1 and r["slots"][0]["is_processing"] for r in system)
    # while in flight live_tokens is at least the prompt estimate of that Task's step 1; there are only c=1 in-flight Requests
    in_flight = [r for r in system if r["requests_processing"] == 1]
    assert max(r["live_tokens"] for r in in_flight) >= min(t["prompt_tokens_measured"] for t in wl["tasks"])

    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["c"], manifest["lambda"], manifest["seed"]) == (1, 1000.0, wl["seed"])
    assert manifest["workload_sha256"] == hashlib.sha256((tmp_path / "sys.json").read_bytes()).hexdigest()
    assert manifest["repo_git_sha"]
    assert manifest["memory_platform"] == system[0]["platform"]
    assert manifest["t_end"] >= manifest["t_start"]
    assert manifest["t_start_iso"] and manifest["t_end_iso"]


def test_cache_prompt_flag_is_sent_in_every_request_and_recorded_in_manifest(tmp_path, workload, fake_server):
    wl = with_warmup_prefix(workload, first_index(workload, "short"))
    run(tmp_path, wl, fake_server, c=1, name="on")
    run(tmp_path, wl, fake_server, c=1, name="off", extra=["--cache-prompt", "off"])
    on_bodies, off_bodies = fake_server.requests[:4], fake_server.requests[4:]
    assert all(b["cache_prompt"] is True for b in on_bodies)  # true is sent explicitly even by default; the request bodies of both E4 sides differ in only this key
    assert all(b["cache_prompt"] is False for b in off_bodies)
    assert json.loads((tmp_path / "on/manifest.json").read_text())["cache_prompt"] is True
    assert json.loads((tmp_path / "off/manifest.json").read_text())["cache_prompt"] is False


def test_manifest_records_cold_start_seconds_and_workload_tool_delay(tmp_path, workload, fake_server):
    wl = {**with_warmup_prefix(workload, first_index(workload, "short")), "tool_delay_s": 3.0}
    run(tmp_path, wl, fake_server, c=1, name="m", extra=["--cold-start-s", "7"])
    manifest = json.loads((tmp_path / "m/manifest.json").read_text())
    assert manifest["cold_start_s"] == 7.0
    assert manifest["tool_delay_s"] == 3.0
    run(tmp_path, with_warmup_prefix(workload, first_index(workload, "short")), fake_server, c=1, name="n")
    plain = json.loads((tmp_path / "n/manifest.json").read_text())
    assert plain["cold_start_s"] is None and plain["tool_delay_s"] is None
