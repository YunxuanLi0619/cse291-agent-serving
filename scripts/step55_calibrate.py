#!/usr/bin/env python3
"""Step 5.5-A speed calibration: single-stream speed, aggregate throughput (c=1/2/4), and T_solo for the four task types.

Usage (the server must already be started with config A: -np 4 -c 16384 --no-kv-unified):
    .venv/bin/python scripts/step55_calibrate.py [--out results/raw/step55/calibration.json]

This is a rough calibration prototype, not the Step 7 workload generator; prompts are built from random words with a fixed seed,
and every request has different content, to avoid prefix reuse in the slot making T_solo optimistic.
"""
import argparse, json, random, subprocess, time
from concurrent.futures import ThreadPoolExecutor
import httpx

BASE = "http://127.0.0.1:8080"
WORDS = ("alpha river stone market window garden copper signal harbor violet engine paper "
         "lantern meadow thunder basket orbit canyon ribbon falcon marble velvet anchor cobalt "
         "saddle timber quartz meteor candle summit pebble tunnel harvest compass glacier").split()


def tok_count(c, text):
    return len(c.post(f"{BASE}/tokenize", json={"content": text}).json()["tokens"])


def make_text(c, rng, n_tokens):
    """Concatenate random words to exactly about n_tokens tokens (counted with /tokenize)."""
    words, n = [], 0
    while n < n_tokens:
        chunk = [rng.choice(WORDS) + str(rng.randrange(100)) for _ in range(max(1, (n_tokens - n) // 3))]
        words += chunk
        n = tok_count(c, " ".join(words))
    # if it overshoots, trim from the tail
    while n > n_tokens and words:
        words = words[:-1]
        n = tok_count(c, " ".join(words))
    return " ".join(words)


def stream_chat(c, messages, max_tokens):
    """One streaming request; returns t_total, t_first_token, timings, and id_slot hints."""
    body = {"messages": messages, "max_tokens": max_tokens, "ignore_eos": True,
            "temperature": 0, "stream": True}
    t0 = time.time(); tft = None; timings = None; text = []; extra_keys = set()
    with c.stream("POST", f"{BASE}/v1/chat/completions", json=body, timeout=300) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[6:])
            extra_keys |= set(d.keys())
            ch = d.get("choices") or []
            if ch and ch[0].get("delta", {}).get("content"):
                if tft is None:
                    tft = time.time() - t0
                text.append(ch[0]["delta"]["content"])
            if "timings" in d:
                timings = d["timings"]
    return {"t_total": time.time() - t0, "t_first_token": tft, "timings": timings,
            "out_text": "".join(text), "chunk_keys": sorted(extra_keys)}


def run_task(c, rng, ttype, tool_delay=1.0, agent_steps=4):
    """Run one task without contention and return the timing breakdown."""
    t0 = time.time(); steps = []; idle = 0.0
    if ttype == "agent":
        msgs = [{"role": "user", "content": make_text(c, rng, 800)}]
        for k in range(agent_steps):
            r = stream_chat(c, msgs, 128); steps.append(r)
            if k < agent_steps - 1:
                time.sleep(tool_delay); idle += tool_delay
                msgs += [{"role": "assistant", "content": r["out_text"]},
                         {"role": "user", "content": make_text(c, rng, 150)}]
    else:
        n_in, n_out = {"short": (256, 64), "longctx": (3000, 64), "longgen": (256, 512)}[ttype]
        steps.append(stream_chat(c, [{"role": "user", "content": make_text(c, rng, n_in)}], n_out))
    return {"type": ttype, "t_solo": time.time() - t0, "idle": idle,
            "prompt_tokens": [s["timings"]["prompt_n"] + s["timings"]["cache_n"] for s in steps],
            "cached": [s["timings"]["cache_n"] for s in steps],
            "pps": [s["timings"]["prompt_per_second"] for s in steps],
            "dps": [s["timings"]["predicted_per_second"] for s in steps],
            "ttft": [s["t_first_token"] for s in steps], "chunk_keys": steps[0]["chunk_keys"]}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/raw/step55/calibration.json")
    a = ap.parse_args()
    import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rng = random.Random(55)
    res = {"started": time.strftime("%F %T"), "batt_start": sh("pmset -g batt | sed -n 2p"),
           "therm_start": sh("pmset -g therm | head -3")}
    with httpx.Client() as c:
        slots = c.get(f"{BASE}/slots").json(); res["n_slots"] = len(slots); res["n_ctx"] = [s["n_ctx"] for s in slots]
        res["slot_keys"] = sorted(slots[0].keys())
        # warm-up (discarded)
        stream_chat(c, [{"role": "user", "content": make_text(c, rng, 200)}], 32)

        # A.1 single stream: ~3000-token prompt + 256 outputs
        a1 = []
        for _ in range(3):
            r = stream_chat(c, [{"role": "user", "content": make_text(c, rng, 3000)}], 256)
            a1.append({k: r["timings"][k] for k in ("prompt_n", "cache_n", "prompt_per_second", "predicted_per_second")}
                      | {"ttft": r["t_first_token"], "t_total": r["t_total"]})
        res["A1_single_stream_3000in_256out"] = a1; print("A1", json.dumps(a1, indent=1), flush=True)

        # A.2 aggregate throughput: 8 identically shaped requests each at c=1/2/4 (256 in, 128 out)
        a2 = {}
        for conc in (1, 2, 4):
            reqs = [[{"role": "user", "content": make_text(c, rng, 256)}] for _ in range(8)]
            t0 = time.time()
            with ThreadPoolExecutor(conc) as ex:
                outs = list(ex.map(lambda m: stream_chat(httpx.Client(), m, 128), reqs))
            wall = time.time() - t0
            tot = sum(o["timings"]["predicted_n"] for o in outs)
            a2[conc] = {"wall_s": wall, "out_tokens": tot, "agg_tok_per_s": tot / wall,
                        "per_req_dps_mean": sum(o["timings"]["predicted_per_second"] for o in outs) / len(outs),
                        "ttft_mean": sum(o["t_first_token"] for o in outs) / len(outs)}
            print("A2 c=%d" % conc, a2[conc], flush=True)
        res["A2_aggregate_256in_128out_x8"] = a2

        # A.3 T_solo: 3 runs per type, c=1
        a3 = {}
        for t in ("short", "longctx", "longgen", "agent"):
            runs = [run_task(c, rng, t) for _ in range(3)]
            a3[t] = {"runs": runs, "t_solo_mean": sum(r["t_solo"] for r in runs) / 3,
                     "t_solo_all": [r["t_solo"] for r in runs]}
            print("A3", t, [round(x, 2) for x in a3[t]["t_solo_all"]], "prompt_tokens", runs[0]["prompt_tokens"], flush=True)
        res["A3_T_solo"] = a3
    res |= {"batt_end": sh("pmset -g batt | sed -n 2p"), "therm_end": sh("pmset -g therm | head -3"),
            "ended": time.strftime("%F %T")}
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
