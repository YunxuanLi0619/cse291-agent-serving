"""测量客户端断连后 llama-server 释放 slot 的耗时(issue #3 评论的补充验收条件)。

要求 server 以 -np 1 启动,这样只有一个 slot,谁释放了不用归因。每个 trial:
  1. 等 slot 空闲;发一个流式长 prompt 请求
  2. 在指定时刻主动断连(「prefill:S」= 发出后 S 秒,此时还没有首个 token;「decode:S」= 首个 token 后 S 秒)
  3. 每 50 ms 轮询 /slots,直到 is_processing 回到 false,release_s = 释放时刻 - 断连时刻
结果每个 trial 一行 JSONL,同时打印。

用法: python scripts/measure_slot_release.py --prompt-file P.txt --out results/raw/slot_release/trials.jsonl \
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
    # 开头加随机前缀:否则同一 prompt 会整段命中 slot 里留下的 KV(LCP),prefill 瞬间结束
    nonce = uuid.uuid4().hex
    body = {"messages": [{"role": "user", "content": f"[{nonce}] {prompt}"}], "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True}
    t_send = time.monotonic()
    t_first = None
    done = False  # 请求在我们断连之前就自然结束了(prefill 比预期快)
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
    t_disc = time.monotonic()  # async with 退出时连接已被关闭
    t_release = t_disc
    if not done:
        while await slot_busy(client):
            await asyncio.sleep(POLL_S)
        t_release = time.monotonic()
    row = {"spec": spec, "disconnect_after_send_s": round(t_disc - t_send, 3),
           "first_token_s": None if t_first is None else round(t_first - t_send, 3),
           "ended_naturally": done, "release_s": None if done else round(t_release - t_disc, 3)}
    if phase == "prefill" and t_first is not None:
        row["note"] = "首个 token 先于断连出现:这次不是 prefill 期间断连"
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
    ap.add_argument("specs", nargs="+", help="prefill:<发出后秒数> 或 decode:<首个 token 后秒数>")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--idle-limit-s", type=float, default=120.0)
    asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
