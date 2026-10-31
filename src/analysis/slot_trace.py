"""把 llama-server 日志里的 slot 选择对回 requests.jsonl(OAI 接口的响应没有 id_slot)。

server 日志每个被接受的请求按序出现一对 `selected slot by ...` + `launch_slot_`;请求按 t_start 排序后
与之一一对应。数量对不上(有请求没到 server 就被取消)时拒绝对应,不猜。

对应是按顺序做的,所以会用 server 释放 slot 时的 n_tokens(= prompt + 输出)逐个核对 ok 请求;
对不上(c>1 时同时启动的请求可能在 server 里换了顺序)就报错,不给出可能错位的结果。

用法: python -m src.analysis.slot_trace --run results/raw/x/run --server-log results/raw/x/server.log
"""
import argparse
import json
import re
from pathlib import Path

SELECT = re.compile(r"selected slot by (LCP|LRU)(?: similarity, f_sim_best = ([\d.]+))?")
LAUNCH = re.compile(r"launch_slot_: id\s+(\d+) \| task (\d+) \|")
RELEASE = re.compile(r"release: id\s+\d+ \| task (\d+) \| stop processing: n_tokens = (\d+)")
TOKEN_TOLERANCE = 2


def parse_server_log(lines) -> list[dict]:
    launches, pending, released = [], None, {}
    for line in lines:
        if m := RELEASE.search(line):
            released[int(m.group(1))] = int(m.group(2))
        elif m := SELECT.search(line):
            pending = {"how": m.group(1), "f_sim": float(m.group(2)) if m.group(2) else None}
        elif (m := LAUNCH.search(line)) and pending is not None:
            launches.append({"slot": int(m.group(1)), "server_task": int(m.group(2)), **pending})
            pending = None
    for launch in launches:
        launch["n_tokens_at_release"] = released.get(launch["server_task"])
    return launches


def join_requests(requests: list[dict], launches: list[dict]) -> list[dict]:
    if len(requests) != len(launches):
        raise ValueError(f"{len(requests)} requests vs {len(launches)} slot launches in the server log; "
                         "cannot match them by order (pass --skip N if the log also holds other requests)")
    rows, last_slot = [], {}
    for req, launch in zip(sorted(requests, key=lambda r: r["t_start"]), launches):
        expected, seen = None, launch["n_tokens_at_release"]
        if req.get("status") == "ok" and req.get("prompt_tokens") is not None:
            expected = req["prompt_tokens"] + req["output_tokens"]
        if expected is not None and seen is not None and abs(expected - seen) > TOKEN_TOLERANCE:
            raise ValueError(f"{req['task_id']} step {req['step']}: client saw {expected} tokens but the matched "
                             f"server task released with n_tokens = {seen}; the order-based pairing is wrong")
        prev = last_slot.get(req["task_id"])
        rows.append({"task_id": req["task_id"], "step": req["step"], "cached_tokens": req.get("cached_tokens"),
                     "slot": launch["slot"], "how": launch["how"], "f_sim": launch["f_sim"],
                     "same_slot_as_prev_step": None if prev is None else prev == launch["slot"]})
        last_slot[req["task_id"]] = launch["slot"]
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="含 requests.jsonl 的 Run 目录")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--skip", type=int, help="server 日志里本 Run 之前已有的 launch 数(日志被别的请求共用时必填)")
    a = ap.parse_args(argv)
    requests = [json.loads(line) for line in (Path(a.run) / "requests.jsonl").read_text().splitlines() if line]
    launches = parse_server_log(Path(a.server_log).read_text().splitlines())
    if a.skip is not None:
        launches = launches[a.skip:a.skip + len(requests)]
    rows = join_requests(requests, launches)
    for r in sorted(rows, key=lambda r: (r["task_id"], r["step"])):
        print(json.dumps(r))


if __name__ == "__main__":
    main()
