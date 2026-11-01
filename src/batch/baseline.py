"""Selection rule for the Fixed-concurrency baseline (#9, GLOSSARY: the c with the highest throughput or goodput and an acceptable p99 on the Dev workload).

The rule was fixed before seeing the data:
- Compared quantity = mean goodput of each c over the repeated Runs (seeds 1-3) (goodput requires completion within the deadline and quality pass, so it is stricter than throughput);
- "acceptable p99" = p99 of (completion time / deadline) over ok Tasks <= 1 (absolute times of different types are not comparable, so each is normalised by its own deadline), averaged over Runs;
    and the mean failure rate (including timeouts) <= 5%. Looking only at ok Tasks has survivor bias: under overload only a few fast Tasks complete and their p99 looks good, so the failure rate is a second gate;
- Candidates = the c values with an acceptable p99; among them, those within 5% of the highest goodput tie and the smaller c wins (larger slot capacity, fewer resources);
- If no c is acceptable, fall back to the c with the highest goodput and set `p99_acceptable=False`.

2026-11-01 relaxation (changed after seeing the E1-E5 data, i.e. a post-hoc adjustment that the report must state; `rule="strict"` keeps the original rule to reproduce the old result):
- `rule="relaxed"`: any c with an `error_ctx_exceeded` is ineligible (its goodput comes from rejecting tasks, so it cannot be picked even as the fallback);
    otherwise only the mean failure rate (including timeouts) <= 10% is required. The p99/deadline ratio is only recorded, no longer a gate: with 100 Tasks the p99 is essentially the slowest one or two,
    and in the existing data even the non-overloaded c=4 at lambda=0.029 reaches 1.7-1.9; the failure-rate gate still blocks the survivor bias of "only a few fast tasks complete".
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
    """Each row of runs: c, seed, goodput_per_s, p99_deadline_ratio (NaN means the Run has no ok Task), fail_rate,
    ctx_exceeded_rate (0 by default; only used by rule="relaxed")."""
    assert rule in ("relaxed", "strict")
    by_c = []
    for c in sorted({r["c"] for r in runs}):
        rs = [r for r in runs if r["c"] == c]
        by_c.append({"c": c, "n_runs": len(rs),
                     "goodput_per_s": float(np.mean([r["goodput_per_s"] for r in rs])),
                     "p99_deadline_ratio": float(np.mean([r["p99_deadline_ratio"] for r in rs])),
                     "fail_rate": float(np.mean([r["fail_rate"] for r in rs])),
                     "ctx_exceeded_rate": float(np.mean([r.get("ctx_exceeded_rate", 0.0) for r in rs]))})
    for row in by_c:  # NaN comparisons are False, so NaN rows are excluded naturally
        if rule == "strict":
            row["acceptable"] = bool(row["p99_deadline_ratio"] <= P99_LIMIT and row["fail_rate"] <= FAIL_RATE_LIMIT)
        else:
            row["acceptable"] = bool(row["ctx_exceeded_rate"] == 0 and row["fail_rate"] <= RELAXED_FAIL_RATE_LIMIT)
    ok = [row for row in by_c if row["acceptable"]]
    pool = ok or ([row for row in by_c if row["ctx_exceeded_rate"] == 0] or by_c if rule == "relaxed" else by_c)
    best = max(row["goodput_per_s"] for row in pool)
    chosen = min((row for row in pool if row["goodput_per_s"] >= best * (1 - TIE_TOLERANCE)), key=lambda row: row["c"])
    return {"c": chosen["c"], "p99_acceptable": bool(ok), "rule": rule, "by_c": by_c}
