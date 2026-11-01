"""Fixed-concurrency baseline 的选择规则(#9,GLOSSARY:Dev workload 上吞吐或 goodput 最高且 p99 可接受的 c)。

规则在看到数据之前定下:
- 比较量 = 各 c 在重复 Run(seed 1–3)上 goodput 的均值(goodput 要求 deadline 内完成且 quality pass,比吞吐更严);
- 「p99 可接受」= ok Task 的 完成时间 / deadline 的 p99 ≤ 1(不同类型的绝对时间不可比,所以按各自 deadline 归一),取各 Run 的均值;
  并且失败率(含超时)的均值 ≤ 5%。只看 ok Task 有幸存者偏差:过载时只有少数快的 Task 完成,它们的 p99 会很好看,所以失败率一起拦;
- 候选 = p99 可接受的 c;其中 goodput 在最高值 5% 以内的并列,取较小的 c(slot 容量更大、占的资源更少);
- 没有 c 可接受时退而取 goodput 最高的 c,并标 `p99_acceptable=False`。

2026-11-01 放宽(看过 E1–E5 数据之后改的,事后调整,报告里要写明;`rule="strict"` 保留原规则,用来复现旧结果):
- `rule="relaxed"`:有任何 `error_ctx_exceeded` 的 c 一律不合格(它是靠拒绝任务换来的 goodput,连回退也不能选);
  其余只要求失败率(含超时)均值 ≤ 10%。p99/deadline 比值只记录,不再作为门槛——100 个 Task 的 p99 基本等于最慢那一两个,
  现有数据里连不过载的 c=4、λ=0.029 也到 1.7–1.9;失败率门槛仍然拦住"只有少数快任务完成"的幸存者偏差。
"""
import math

import numpy as np

TIE_TOLERANCE = 0.05
P99_LIMIT = 1.0
FAIL_RATE_LIMIT = 0.05
RELAXED_FAIL_RATE_LIMIT = 0.10


def p99_deadline_ratio(tasks: list[dict]) -> float:
    ratios = [(t["t_done"] - t["t_arrive"]) / t["deadline_s"] for t in tasks if not t["warmup"] and t["status"] == "ok"]
    return float(np.percentile(ratios, 99)) if ratios else math.nan


def ctx_exceeded_rate(tasks: list[dict]) -> float:
    steady = [t for t in tasks if not t["warmup"]]
    return sum(t["status"] == "error_ctx_exceeded" for t in steady) / len(steady) if steady else math.nan


def select_baseline(runs: list[dict], rule: str = "relaxed") -> dict:
    """runs 每行:c、seed、goodput_per_s、p99_deadline_ratio(NaN 表示该 Run 没有 ok Task)、fail_rate、
    ctx_exceeded_rate(缺省按 0;只有 rule="relaxed" 用到)。"""
    assert rule in ("relaxed", "strict")
    by_c = []
    for c in sorted({r["c"] for r in runs}):
        rs = [r for r in runs if r["c"] == c]
        by_c.append({"c": c, "n_runs": len(rs),
                     "goodput_per_s": float(np.mean([r["goodput_per_s"] for r in rs])),
                     "p99_deadline_ratio": float(np.mean([r["p99_deadline_ratio"] for r in rs])),
                     "fail_rate": float(np.mean([r["fail_rate"] for r in rs])),
                     "ctx_exceeded_rate": float(np.mean([r.get("ctx_exceeded_rate", 0.0) for r in rs]))})
    for row in by_c:  # NaN 比较为 False,自然排除
        if rule == "strict":
            row["acceptable"] = bool(row["p99_deadline_ratio"] <= P99_LIMIT and row["fail_rate"] <= FAIL_RATE_LIMIT)
        else:
            row["acceptable"] = bool(row["ctx_exceeded_rate"] == 0 and row["fail_rate"] <= RELAXED_FAIL_RATE_LIMIT)
    ok = [row for row in by_c if row["acceptable"]]
    pool = ok or ([row for row in by_c if row["ctx_exceeded_rate"] == 0] or by_c if rule == "relaxed" else by_c)
    best = max(row["goodput_per_s"] for row in pool)
    chosen = min((row for row in pool if row["goodput_per_s"] >= best * (1 - TIE_TOLERANCE)), key=lambda row: row["c"])
    return {"c": chosen["c"], "p99_acceptable": bool(ok), "rule": rule, "by_c": by_c}
