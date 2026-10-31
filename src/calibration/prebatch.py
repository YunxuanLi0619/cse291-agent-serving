"""批跑前检查(#7)的决策逻辑:线程数冻结、T_solo 复测比对、冻结配置读写。

实机测量在 scripts/prebatch_check.py;这里只放可以离线测试的纯逻辑。
"""
import json
from math import prod
from pathlib import Path
from statistics import mean, stdev

DEFAULT_THREADS = 8
THREADS_MIN_GAIN = 0.10  # 最快档比 -t 8 快不到 10% 就保持 -t 8
MAX_CV = 0.15  # 单流耗时的变异系数超过它就算不稳


def _speed_index(cfg: dict, base: dict) -> float:
    """相对 base 的几何平均加速比:耗时指标取倒数比,吞吐取正比。"""
    ratios = [mean(base["short_s"]) / mean(cfg["short_s"]),
              mean(base["longctx_s"]) / mean(cfg["longctx_s"]),
              mean(cfg["agg_tok_s"]) / mean(base["agg_tok_s"])]
    return prod(ratios) ** (1 / len(ratios))


def _cv(cfg: dict) -> float:
    return max(stdev(cfg[k]) / mean(cfg[k]) for k in ("short_s", "longctx_s"))


def decide_threads(results: dict[int, dict]) -> dict:
    """results[t] = {"short_s": [...], "longctx_s": [...], "agg_tok_s": [...]}(同一线程数的多轮数据已合并)。"""
    index = {t: _speed_index(cfg, results[DEFAULT_THREADS]) for t, cfg in results.items()}
    stable = {t: i for t, i in index.items() if _cv(results[t]) <= MAX_CV}
    best = max(stable, key=stable.get, default=DEFAULT_THREADS)
    if stable.get(best, 0) < 1 + THREADS_MIN_GAIN:
        top = max(index, key=index.get)
        reason = "unstable" if index[top] >= 1 + THREADS_MIN_GAIN else "within_10pct"
        return {"threads": DEFAULT_THREADS, "reason": reason, "speed_index": index}
    return {"threads": best, "reason": "fastest", "speed_index": index}


T_SOLO_TOLERANCE = 0.15  # 任一类偏差超过它就重算 λ 网格、deadline、预算


def compare_t_solo(old: dict, new: dict) -> dict:
    deviation = {k: new[k] / old[k] - 1 for k in old}
    exceeded = [k for k, d in deviation.items() if abs(d) > T_SOLO_TOLERANCE]
    return {"deviation": deviation, "exceeded": exceeded, "recalibrate": bool(exceeded)}


AGENT_CAL_STEPS = 4  # 校准用的 agent Task:4 步、Tool delay 1 s
AGENT_CAL_DELAY_S = 1.0
LAMBDA_GRID_FACTORS = (0.3, 0.6, 0.9, 1.2)


def budget(t_solo: dict, mix: dict, p_eff: float) -> dict:
    """平均服务时间 S̄、容量 μ = P_eff / S̄、λ 网格(μ 的倍数)。"""
    service = {**t_solo, "agent": t_solo["agent_per_step_s"] * AGENT_CAL_STEPS
               + AGENT_CAL_DELAY_S * (AGENT_CAL_STEPS - 1)}
    s_bar = sum(w * service[type_] for type_, w in mix.items())
    mu = p_eff / s_bar
    return {"mean_service_s": s_bar, "mu": mu, "lambda_grid": [f * mu for f in LAMBDA_GRID_FACTORS]}


def freeze(config_dir, threads: int, t_solo: dict) -> None:
    """写出冻结的线程数(start_server.sh 读)和 T_solo 表(Workload 生成器读)。"""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "server.json").write_text(json.dumps({"threads": threads}, indent=2) + "\n")
    (config_dir / "t_solo.json").write_text(json.dumps(t_solo, indent=2) + "\n")
