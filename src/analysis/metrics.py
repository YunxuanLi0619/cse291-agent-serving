"""指标:全是纯函数,输入是已解析的日志行(list[dict]),不碰文件。

口径(术语见 GLOSSARY.md;Goodput 定义里的「总时长」在这里取稳态窗口):
- 稳态 = `warmup=false` 的 Task;分母永远是全部稳态 Task(失败和超时都算)。
- 稳态窗口 = 最早到达到最晚完成,单位秒;吞吐和 goodput 都除以它。
- 完成时间 = t_done - t_arrive,p50/p99 只统计 status=ok 的 Task;分位数用线性插值。
- 吞吐 = ok Task 数 / 窗口(不要求 deadline 和 quality pass)。
- goodput = (ok 且 quality_pass 且完成时间 ≤ deadline_s 的 Task 数) / 窗口。
- 排队时间 = Task 日志的 queue_time_total,统计全部稳态 Task(过载时的超时 Task 排队最久,不能排除)。
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
    """agent Task 的第 k 步一行(代替单独的 E2)。只看稳态 agent Task 的 Request。

    fail_share = 该步 Request 里非 ok 的占比;时延类(ttft、latency、queue_wait)和 token 类只统计 ok Request;
    reuse_ratio = Σcached_tokens / Σprompt_tokens(Prefix reuse 占 prompt 的比例)。
    注意幸存者偏差:第 k 步只含走到第 k 步的 Task,中途失败的 Task 不出现在后面的步里,fail_share 是条件失败比例。
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
    """采样期间 footprint(主口径,ADR 0004)和 RSS(次口径)的最大值;采样失败的行(null)不参与。"""
    peak = lambda key: max((r[key] for r in system if r.get(key) is not None), default=NAN)  # noqa: E731
    return peak("footprint_mb"), peak("rss_mb")


def kv_utilization(system: list[dict], budget_tokens: int) -> list[tuple[float, float]]:
    """(t, live_tokens / B);live_tokens 缺失的采样行跳过。"""
    return [(r["t"], r["live_tokens"] / budget_tokens) for r in system if r.get("live_tokens") is not None]
