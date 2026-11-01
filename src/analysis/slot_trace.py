"""Map the slot selections in the llama-server log back to requests.jsonl (the OAI endpoint's responses have no id_slot).

For every accepted request the server log has a pair `selected slot by ...` + `launch_slot_` in order; requests sorted by t_start
are matched one-to-one with them. If the counts differ (some request was cancelled before reaching the server), refuse to match rather than guess.

Matching is done in order, so each ok request is checked against n_tokens (= prompt + output) when the server releases the slot;
on a mismatch (with c>1, requests started at the same time may be reordered inside the server) it raises an error instead of returning a possibly misaligned result.

Usage: python -m src.analysis.slot_trace --run results/raw/x/run --server-log results/raw/x/server.log
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
    ap.add_argument("--run", required=True, help="Run directory containing requests.jsonl")
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--skip", type=int, help="number of launches already in the server log before this Run (required when the log is shared with other requests)")
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
