import json

from src.batch.plan import RunSpec


def write_run(d, n=5, *, sample_gap=0.5, error_rows=0, warmup=3, statuses=None, manifest=None, cached=0, spec: RunSpec | None = None):
    """合成一个 Run 目录(与 runner 的产出同形),用来测校验、批跑驱动和汇总。"""
    d.mkdir(parents=True, exist_ok=True)
    m = {"t_end": 10.0, "errors": {}, "cold_start_s": 2.0, "power_source": "AC Power", "low_power_mode": False,
         "cache_prompt": True, "tool_delay_s": None, "c": 4, "lambda": 0.088, "seed": 1,
         "server_cmdline": "llama-server -np 4 -c 16384 --no-kv-unified", **(manifest or {})}
    if spec:
        m.update(c=spec.c, seed=spec.seed, cache_prompt=spec.cache_prompt, tool_delay_s=spec.tool_delay_s,
                 server_cmdline=f"llama-server -np {spec.np} -c {spec.ctx} {' '.join(spec.server_flags)}", **{"lambda": spec.lam})
    (d / "manifest.json").write_text(json.dumps(m))
    statuses = statuses or ["ok"] * n
    tasks = [{"task_id": f"t{i}", "type": "short", "status": s, "warmup": i < warmup, "t_arrive": i * 10.0,
              "t_done": i * 10.0 + 5.0, "deadline_s": 10.0, "queue_time_total": 0.0, "idle_time_total": 0.0,
              "n_steps": 1, "quality_pass": s == "ok", "t_first_dispatch": i * 10.0}
             for i, s in enumerate(statuses)]
    (d / "tasks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in tasks))
    reqs = [{"task_id": t["task_id"], "step": 1, "status": t["status"], "cached_tokens": cached,
             "prompt_tokens": 100, "error_type": None, "warmup": t["warmup"]} for t in tasks]
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reqs))
    rows = [{"t": i * sample_gap, "error": "memory: boom" if i < error_rows else None,
             "footprint_mb": 3800.0, "rss_mb": 900.0, "live_tokens": 0} for i in range(40)]
    (d / "system.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
