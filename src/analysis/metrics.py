"""Metrics: all pure functions over parsed log rows (list[dict]); they never touch files.

Definitions (terms are in GLOSSARY.md; the "total duration" in the Goodput definition is the steady-state window here):
- Steady state = Tasks with `warmup=false`; the denominator is always all steady-state Tasks (failures and timeouts count).
- Steady-state window = earliest arrival to latest completion, in seconds; throughput and goodput are both divided by it.
- Completion time = t_done - t_arrive; p50/p99 count only Tasks with status=ok; percentiles use linear interpolation.
- Throughput = number of ok Tasks / window (no deadline or quality-pass requirement).
- Goodput = (number of Tasks that are ok, quality_pass and completed within deadline_s) / window.
- Queue time = queue_time_total from the Task log, over all steady-state Tasks (timed-out Tasks queue longest under overload and cannot be excluded).
"""
import math

import numpy as np

NAN = math.nan


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else NAN


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else NAN


def steady(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r["warmup"]]


def run_metrics(tasks: list[dict]) -> dict:
    ts = steady(tasks)
    n = len(ts)
    ok = [t for t in ts if t["status"] == "ok"]
    completion = [t["t_done"] - t["t_arrive"] for t in ok]
    good = [t for t in ok if t["quality_pass"] and t["t_done"] - t["t_arrive"] <= t["deadline_s"]]
    window = max(t["t_done"] for t in ts) - min(t["t_arrive"] for t in ts) if ts else NAN
    rate = lambda k: k / window if window and window > 0 else NAN  # noqa: E731
    share = lambda k: k / n if n else NAN  # noqa: E731
    queue = [t["queue_time_total"] for t in ts]
    return {
        "n_tasks": n, "n_ok": len(ok), "window_s": window,
        "p50_s": percentile(completion, 50), "p99_s": percentile(completion, 99),
        "throughput_per_s": rate(len(ok)), "goodput_per_s": rate(len(good)),
        "fail_rate": share(n - len(ok)), "timeout_rate": share(sum(t["status"] == "timeout" for t in ts)),
        "queue_time_mean_s": mean(queue), "queue_time_p99_s": percentile(queue, 99),
    }


def step_slice(requests: list[dict], tasks: list[dict]) -> list[dict]:
    """One row per step k of agent Tasks (replaces a separate E2). Only Requests of steady-state agent Tasks are used.

    fail_share = share of non-ok Requests at that step; latency metrics (ttft, latency, queue_wait) and token metrics only count ok Requests;
    reuse_ratio = sum(cached_tokens) / sum(prompt_tokens) (the share of the prompt covered by Prefix reuse).
    Beware survivor bias: step k only contains Tasks that reached step k; Tasks that failed earlier do not appear in later steps, so fail_share is a conditional failure share.
    """
    agent_ids = {t["task_id"] for t in tasks if t["type"] == "agent"}
    by_k: dict[int, list[dict]] = {}
    for r in steady(requests):
        if r["task_id"] in agent_ids:
            by_k.setdefault(r["step"], []).append(r)
    rows = []
    for k, rs in sorted(by_k.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        prompt = sum(r["prompt_tokens"] for r in ok)
        rows.append({
            "k": k, "n_requests": len(rs), "fail_share": (len(rs) - len(ok)) / len(rs),
            "ttft_s": mean([r["t_first_token"] - r["t_start"] for r in ok if r["t_first_token"] is not None]),
            "latency_s": mean([r["t_end"] - r["t_start"] for r in ok]),
            "queue_wait_s": mean([r["t_start"] - r["t_submit"] for r in ok]),
            "prompt_tokens": mean([r["prompt_tokens"] for r in ok]),
            "reuse_ratio": sum(r["cached_tokens"] for r in ok) / prompt if prompt else NAN,
        })
    return rows


def peak_memory(system: list[dict]) -> tuple[float, float]:
    """Maximum of footprint (primary metric, ADR 0004) and RSS (secondary) during sampling; rows where sampling failed (null) are ignored."""
    peak = lambda key: max((r[key] for r in system if r.get(key) is not None), default=NAN)  # noqa: E731
    return peak("footprint_mb"), peak("rss_mb")


def kv_utilization(system: list[dict], budget_tokens: int) -> list[tuple[float, float]]:
    """(t, live_tokens / B); sample rows with missing live_tokens are skipped."""
    return [(r["t"], r["live_tokens"] / budget_tokens) for r in system if r.get("live_tokens") is not None]
