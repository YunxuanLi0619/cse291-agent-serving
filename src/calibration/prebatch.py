"""Decision logic of the pre-batch checks (#7): freezing the thread count, comparing the T_solo re-measurement, reading/writing the frozen config.

The real-hardware measurement is in scripts/prebatch_check.py; only pure logic that can be tested offline lives here.
"""
import json
from math import prod
from pathlib import Path
from statistics import mean, stdev

DEFAULT_THREADS = 8
THREADS_MIN_GAIN = 0.10  # if the fastest setting is less than 10% faster than -t 8, keep -t 8
MAX_CV = 0.15  # a coefficient of variation of single-stream time above this counts as unstable


def _speed_index(cfg: dict, base: dict) -> float:
    """Geometric-mean speedup relative to base: time metrics use the inverse ratio, throughput the direct ratio."""
    ratios = [mean(base["short_s"]) / mean(cfg["short_s"]),
              mean(base["longctx_s"]) / mean(cfg["longctx_s"]),
              mean(cfg["agg_tok_s"]) / mean(base["agg_tok_s"])]
    return prod(ratios) ** (1 / len(ratios))


def _cv(cfg: dict) -> float:
    return max(stdev(cfg[k]) / mean(cfg[k]) for k in ("short_s", "longctx_s"))


def decide_threads(results: dict[int, dict]) -> dict:
    """results[t] = {"short_s": [...], "longctx_s": [...], "agg_tok_s": [...]} (several rounds of the same thread count already merged)."""
    index = {t: _speed_index(cfg, results[DEFAULT_THREADS]) for t, cfg in results.items()}
    stable = {t: i for t, i in index.items() if _cv(results[t]) <= MAX_CV}
    best = max(stable, key=stable.get, default=DEFAULT_THREADS)
    if stable.get(best, 0) < 1 + THREADS_MIN_GAIN:
        top = max(index, key=index.get)
        reason = "unstable" if index[top] >= 1 + THREADS_MIN_GAIN else "within_10pct"
        return {"threads": DEFAULT_THREADS, "reason": reason, "speed_index": index}
    return {"threads": best, "reason": "fastest", "speed_index": index}


T_SOLO_TOLERANCE = 0.15  # if any type deviates by more than this, recompute the lambda grid, deadlines and budget


def compare_t_solo(old: dict, new: dict) -> dict:
    deviation = {k: new[k] / old[k] - 1 for k in old}
    exceeded = [k for k, d in deviation.items() if abs(d) > T_SOLO_TOLERANCE]
    return {"deviation": deviation, "exceeded": exceeded, "recalibrate": bool(exceeded)}


AGENT_CAL_STEPS = 4  # the agent Task used for calibration: 4 steps, Tool delay 1 s
AGENT_CAL_DELAY_S = 1.0
LAMBDA_GRID_FACTORS = (0.3, 0.6, 0.9, 1.2)


def budget(t_solo: dict, mix: dict, p_eff: float) -> dict:
    """Mean service time S-bar, capacity mu = P_eff / S-bar, and the lambda grid (multiples of mu)."""
    service = {**t_solo, "agent": t_solo["agent_per_step_s"] * AGENT_CAL_STEPS
               + AGENT_CAL_DELAY_S * (AGENT_CAL_STEPS - 1)}
    s_bar = sum(w * service[type_] for type_, w in mix.items())
    mu = p_eff / s_bar
    return {"mean_service_s": s_bar, "mu": mu, "lambda_grid": [f * mu for f in LAMBDA_GRID_FACTORS]}


def freeze(config_dir, threads: int, t_solo: dict) -> None:
    """Write the frozen thread count (read by start_server.sh) and the T_solo table (read by the Workload generator)."""
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "server.json").write_text(json.dumps({"threads": threads}, indent=2) + "\n")
    (config_dir / "t_solo.json").write_text(json.dumps(t_solo, indent=2) + "\n")
