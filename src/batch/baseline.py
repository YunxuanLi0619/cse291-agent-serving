"""Fixed-concurrency baseline 的选择规则(#9,GLOSSARY:Dev workload 上吞吐或 goodput 最高且 p99 可接受的 c)。

规则在看到数据之前定下:
- 比较量 = 各 c 在重复 Run(seed 1–3)上 goodput 的均值(goodput 要求 deadline 内完成且 quality pass,比吞吐更严);
- 「p99 可接受」= ok Task 的 完成时间 / deadline 的 p99 ≤ 1(不同类型的绝对时间不可比,所以按各自 deadline 归一),取各 Run 的均值;
  并且失败率(含超时)的均值 ≤ 5%。只看 ok Task 有幸存者偏差:过载时只有少数快的 Task 完成,它们的 p99 会很好看,所以失败率一起拦;
- 候选 = p99 可接受的 c;其中 goodput 在最高值 5% 以内的并列,取较小的 c(slot 容量更大、占的资源更少);
- 没有 c 可接受时退而取 goodput 最高的 c,并标 `p99_acceptable=False`。
"""
import math

import numpy as np

TIE_TOLERANCE = 0.05
P99_LIMIT = 1.0
FAIL_RATE_LIMIT = 0.05


def p99_deadline_ratio(tasks: list[dict]) -> float:
    ratios = [(t["t_done"] - t["t_arrive"]) / t["deadline_s"] for t in tasks if not t["warmup"] and t["status"] == "ok"]
    return float(np.percentile(ratios, 99)) if ratios else math.nan


def select_baseline(runs: list[dict]) -> dict:
    """runs 每行:c、seed、goodput_per_s、p99_deadline_ratio(NaN 表示该 Run 没有 ok Task)、fail_rate。"""
    by_c = []
    for c in sorted({r["c"] for r in runs}):
        rs = [r for r in runs if r["c"] == c]
        by_c.append({"c": c, "n_runs": len(rs),
                     "goodput_per_s": float(np.mean([r["goodput_per_s"] for r in rs])),
                     "p99_deadline_ratio": float(np.mean([r["p99_deadline_ratio"] for r in rs])),
                     "fail_rate": float(np.mean([r["fail_rate"] for r in rs]))})
    for row in by_c:  # NaN 比较为 False,自然排除
        row["acceptable"] = bool(row["p99_deadline_ratio"] <= P99_LIMIT and row["fail_rate"] <= FAIL_RATE_LIMIT)
    ok = [row for row in by_c if row["acceptable"]]
    pool = ok or by_c
    best = max(row["goodput_per_s"] for row in pool)
    chosen = min((row for row in pool if row["goodput_per_s"] >= best * (1 - TIE_TOLERANCE)), key=lambda row: row["c"])
    return {"c": chosen["c"], "p99_acceptable": bool(ok), "by_c": by_c}
