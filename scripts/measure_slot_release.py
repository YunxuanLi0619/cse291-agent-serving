"""Measure how long llama-server takes to release a slot after the client disconnects (additional acceptance criterion from the issue #3 comments).

The server must be started with -np 1 so there is only one slot and no attribution of who released it is needed. Each trial:
  1. wait for the slot to be idle; send a streaming long-prompt request
  2. actively disconnect at the specified moment ("prefill:S" = S seconds after sending, before the first token; "decode:S" = S seconds after the first token)
  3. poll /slots every 50 ms until is_processing returns to false; release_s = release time - disconnect time
Each trial is one JSONL row and is also printed.

Usage: python scripts/measure_slot_release.py --prompt-file P.txt --out results/raw/slot_release/trials.jsonl \
        prefill:1 prefill:5 prefill:10 decode:1
"""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

POLL_S = 0.05


async def slot_busy(client: httpx.AsyncClient) -> bool:
    slots = (await client.get("/slots")).raise_for_status().json()
    return any(s["is_processing"] for s in slots)


async def wait_idle(client: httpx.AsyncClient, limit_s: float) -> float:
    t0 = time.monotonic()
    while await slot_busy(client):
        if time.monotonic() - t0 > limit_s:
            raise SystemExit(f"slot still busy after {limit_s}s")
        await asyncio.sleep(POLL_S)
    return time.monotonic() - t0


async def trial(client: httpx.AsyncClient, prompt: str, spec: str, max_tokens: int, idle_limit_s: float) -> dict:
    phase, _, secs = spec.partition(":")
    at = float(secs)
    await wait_idle(client, idle_limit_s)
    # Add a random prefix at the start: otherwise the same prompt would fully hit the KV left in the slot (LCP) and prefill would finish instantly
    nonce = uuid.uuid4().hex
    body = {"messages": [{"role": "user", "content": f"[{nonce}] {prompt}"}], "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True}
    t_send = time.monotonic()
    t_first = None
    done = False  # the request finished naturally before we disconnected (prefill was faster than expected)
    try:
        async with asyncio.timeout(at if phase == "prefill" else None) as budget:
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if t_first is None and line.startswith("data: ") and '"content":"' in line:
                        t_first = time.monotonic()
                        if phase == "decode":
                            budget.reschedule(asyncio.get_running_loop().time() + at)
                    if line == "data: [DONE]":
                        done = True
    except TimeoutError:
        pass
    t_disc = time.monotonic()  # the connection is already closed when the async with exits
    t_release = t_disc
    if not done:
        while await slot_busy(client):
            await asyncio.sleep(POLL_S)
        t_release = time.monotonic()
    row = {"spec": spec, "disconnect_after_send_s": round(t_disc - t_send, 3),
           "first_token_s": None if t_first is None else round(t_first - t_send, 3),
           "ended_naturally": done, "release_s": None if done else round(t_release - t_disc, 3)}
    if phase == "prefill" and t_first is not None:
        row["note"] = "first token appeared before the disconnect: this was not a disconnect during prefill"
    return row


async def main_async(a) -> None:
    prompt = Path(a.prompt_file).read_text()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=a.base_url, timeout=None) as client:
        n = len((await client.get("/slots")).raise_for_status().json())
        if n != 1:
            raise SystemExit(f"/slots reports {n} slots; start the server with -np 1 so the release needs no attribution")
        with out.open("w") as f:
            for spec in a.specs:
                for _ in range(a.repeat):
                    row = await trial(client, prompt, spec, a.max_tokens, a.idle_limit_s)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="prefill:<seconds after sending> or decode:<seconds after the first token>")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--idle-limit-s", type=float, default=120.0)
    asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
