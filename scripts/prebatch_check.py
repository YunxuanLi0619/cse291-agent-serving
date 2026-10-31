#!/usr/bin/env python3
"""批跑前检查(#7):线程数比较、T_solo 复测、c=8 聚合吞吐、冷盘启动。

用法(须插电、关低电量模式;自己启停 llama-server,启动前确保 8080 没被占用):
    .venv/bin/python scripts/prebatch_check.py threads [--threads 8 4 10 8]
    .venv/bin/python scripts/prebatch_check.py freeze          # 按 threads 结果写 configs/server.json
    .venv/bin/python scripts/prebatch_check.py tsolo           # 用冻结的线程数复测 T_solo
    .venv/bin/python scripts/prebatch_check.py c8              # np=8 上比较 c=4 与 c=8 的聚合吞吐
    .venv/bin/python scripts/prebatch_check.py cold            # 先 `sudo purge`,再立刻运行

每个阶段的原始数据写 results/raw/prebatch/<阶段>.json,含电源状态与 pmset -g therm。
决策逻辑在 src/calibration/prebatch.py(有单测);prompt 构造和请求沿用 step55_calibrate.py。
"""
import argparse, json, os, random, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import step55_calibrate as cal  # noqa: E402
from src.calibration import prebatch  # noqa: E402

OUT = ROOT / "results/raw/prebatch"
CTX = 16384


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def env_state():
    return {"time": time.strftime("%F %T"), "batt": sh("pmset -g batt | sed -n 1,2p"),
            "lowpowermode": sh("pmset -g | grep lowpowermode"), "therm": sh("pmset -g therm | head -4"),
            "top_cpu": sh("ps -Ao pcpu,comm -r | sed -n 2,4p | cut -c1-120")}  # 找出抢 CPU 的后台进程


def save(phase, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{phase}.json").write_text(json.dumps(data, indent=1))
    print("saved", OUT / f"{phase}.json", flush=True)


class Server:
    """start_server.sh 末尾是 exec,所以 Popen 的 pid 就是 llama-server。"""

    def __init__(self, np_, threads, extra=()):
        OUT.mkdir(parents=True, exist_ok=True)
        cfg = OUT / f"server_t{threads}.json"
        cfg.write_text(json.dumps({"threads": threads}))
        self.log = open(OUT / f"server_np{np_}_t{threads}_{int(time.time())}.log", "w")
        self.np, self.threads, self.extra, self.cfg = np_, threads, list(extra), cfg

    def __enter__(self):
        self.t0 = time.time()
        self.proc = subprocess.Popen([str(ROOT / "scripts/start_server.sh"), str(self.np), str(CTX), *self.extra],
                                     env={**os.environ, "SERVER_CONFIG": str(self.cfg)},
                                     stdout=self.log, stderr=subprocess.STDOUT)
        out = subprocess.run([str(ROOT / "scripts/wait_health.sh"), "300"], capture_output=True, text=True)
        if out.returncode:
            self.__exit__()
            raise SystemExit("server never became healthy")
        self.health_wait_s = int(out.stdout.strip())  # wait_health.sh 的整秒读数
        self.ready_s = time.time() - self.t0
        return self

    def __exit__(self, *exc):
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        time.sleep(2)


def warmup(c, rng):
    cal.stream_chat(c, [{"role": "user", "content": cal.make_text(c, rng, 200)}], 32)


def aggregate(rng, c, conc, n_reqs=8):
    reqs = [[{"role": "user", "content": cal.make_text(c, rng, 256)}] for _ in range(n_reqs)]
    t0 = time.time()
    with ThreadPoolExecutor(conc) as ex:
        outs = list(ex.map(lambda m: cal.stream_chat(httpx.Client(), m, 128), reqs))
    wall = time.time() - t0
    return {"conc": conc, "wall_s": wall, "agg_tok_per_s": sum(o["timings"]["predicted_n"] for o in outs) / wall}


def phase_threads(a):
    rng = random.Random(77)
    rounds = []
    for i, t in enumerate(a.threads):
        if i:
            time.sleep(a.cooldown)  # 给 MacBook Air(无风扇)散热,减少前一档对后一档的影响
        with Server(4, t, ["--no-kv-unified"]), httpx.Client() as c:
            warmup(c, rng)
            r = {"threads": t, "env_start": env_state()}
            r["short_s"] = [cal.run_task(c, rng, "short")["t_solo"] for _ in range(3)]
            r["longctx_s"] = [cal.run_task(c, rng, "longctx")["t_solo"] for _ in range(3)]
            r["agg_c4"] = aggregate(rng, c, 4)
            r["env_end"] = env_state()
        rounds.append(r)
        print(f"-t {t}: short {r['short_s']} longctx {r['longctx_s']} c4 {r['agg_c4']['agg_tok_per_s']:.1f} tok/s", flush=True)
        save("threads", {"rounds": rounds})


def pooled_threads():
    rounds = json.loads((OUT / "threads.json").read_text())["rounds"]
    pooled = {}
    for r in rounds:
        p = pooled.setdefault(r["threads"], {"short_s": [], "longctx_s": [], "agg_tok_s": []})
        p["short_s"] += r["short_s"]; p["longctx_s"] += r["longctx_s"]; p["agg_tok_s"].append(r["agg_c4"]["agg_tok_per_s"])
    return pooled


def phase_freeze(a):
    pooled = pooled_threads()
    d = prebatch.decide_threads(pooled)
    save("threads_decision", {"decision": d, "pooled": {str(k): v for k, v in pooled.items()}})
    t_solo = json.loads((ROOT / "configs/t_solo.json").read_text())
    prebatch.freeze(ROOT / "configs", d["threads"], t_solo)  # T_solo 表在 tsolo 阶段之后再覆盖
    print("frozen:", d, flush=True)


def phase_tsolo(a):
    threads = json.loads((ROOT / "configs/server.json").read_text())["threads"]
    rng = random.Random(78)
    old = json.loads((ROOT / "configs/t_solo.json").read_text())
    with Server(4, threads, ["--no-kv-unified"]), httpx.Client() as c:
        warmup(c, rng)
        env0 = env_state()
        runs = {t: [cal.run_task(c, rng, t) for _ in range(a.reps)] for t in ("short", "longctx", "longgen", "agent")}
        env1 = env_state()
    mean_t = {t: mean(r["t_solo"] for r in rs) for t, rs in runs.items()}
    idle = runs["agent"][0]["idle"]
    new = {"short": round(mean_t["short"], 2), "longctx": round(mean_t["longctx"], 2),
           "longgen": round(mean_t["longgen"], 2),
           "agent_per_step_s": round((mean_t["agent"] - idle) / prebatch.AGENT_CAL_STEPS, 2)}
    cmp_ = prebatch.compare_t_solo(old, new)
    mix = json.loads((ROOT / "configs/mix_dev.json").read_text())["mix"]
    b_old, b_new = prebatch.budget(old, mix, 1.5), prebatch.budget(new, mix, 1.5)
    save("tsolo", {"threads": threads, "old": old, "new": new, "comparison": cmp_,
                   "all_t_solo_s": {t: [r["t_solo"] for r in rs] for t, rs in runs.items()},
                   "budget_old_p_eff_1.5": b_old, "budget_new_p_eff_1.5": b_new, "env_start": env0, "env_end": env1})
    print(json.dumps({"new": new, "comparison": cmp_, "budget_new": b_new}, indent=1), flush=True)
    if a.write:
        prebatch.freeze(ROOT / "configs", threads, new)
        print("configs/t_solo.json overwritten with the new table", flush=True)


def phase_c8(a):
    threads = json.loads((ROOT / "configs/server.json").read_text())["threads"]
    rng = random.Random(79)
    res = []
    with Server(8, threads, ["--no-kv-unified"]), httpx.Client() as c:
        warmup(c, rng)
        env0 = env_state()
        for _ in range(a.reps):  # 交替 c=4 / c=8,抵消降频漂移
            for conc in (4, 8):
                res.append(aggregate(rng, c, conc, n_reqs=16))
                print(res[-1], flush=True)
        env1 = env_state()
    g = {k: mean(r["agg_tok_per_s"] for r in res if r["conc"] == k) for k in (4, 8)}
    save("c8", {"threads": threads, "np": 8, "runs": res, "mean_agg_tok_per_s": g, "c8_over_c4": g[8] / g[4],
                "env_start": env0, "env_end": env1})
    print(f"mean agg tok/s: c=4 {g[4]:.1f}, c=8 {g[8]:.1f} (c8/c4 = {g[8] / g[4]:.3f})", flush=True)


def phase_cold(a):
    threads = json.loads((ROOT / "configs/server.json").read_text())["threads"]
    out = {"threads": threads, "env_start": env_state()}
    for label in ("cold", "warm"):  # 先冷(须已 purge),再热(同一份文件刚被读过)
        with Server(4, threads, ["--no-kv-unified"]) as s:
            out[label] = {"wait_health_s": s.health_wait_s, "ready_s": s.ready_s}
        print(label, out[label], flush=True)
    save("cold_start", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("threads"); p.add_argument("--threads", type=int, nargs="+", default=[8, 4, 10, 8])
    p.add_argument("--cooldown", type=int, default=180)
    sub.add_parser("freeze")
    p = sub.add_parser("tsolo"); p.add_argument("--reps", type=int, default=3); p.add_argument("--write", action="store_true")
    p = sub.add_parser("c8"); p.add_argument("--reps", type=int, default=2)
    sub.add_parser("cold")
    a = ap.parse_args()
    {"threads": phase_threads, "freeze": phase_freeze, "tsolo": phase_tsolo, "c8": phase_c8, "cold": phase_cold}[a.phase](a)


if __name__ == "__main__":
    main()
