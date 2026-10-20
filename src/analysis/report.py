"""分析命令:从原始日志目录重新生成全部指标 CSV 和图。每张图 = 同名的 .csv + .png,报告引用 CSV。

用法: python -m src.analysis --raw results/raw/e1 --out results/analysis
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .metrics import kv_utilization, peak_memory, run_metrics, step_slice
from .plots import Panel, line_panels, memory_bars, time_curves
from .runs import load_runs

# 一个 Run 的配置键;platform 单独处理(不同平台的数据不混画),seed 是重复编号
CONFIG_KEYS = ["platform", "series", "B", "c", "lambda", "tool_delay_s", "cache_prompt"]
SWEEP_AXES = ["c", "lambda", "tool_delay_s"]
METRICS = ["p50_s", "p99_s", "throughput_per_s", "goodput_per_s", "fail_rate", "timeout_rate",
           "queue_time_mean_s", "queue_time_p99_s"]
PERF_PANELS: list[Panel] = [
    ("throughput vs goodput (tasks/s)", [("throughput_per_s", "throughput"), ("goodput_per_s", "goodput")]),
    ("completion time of ok tasks (s)", [("p50_s", "p50"), ("p99_s", "p99")]),
    ("queue time, all tasks (s)", [("queue_time_mean_s", "mean"), ("queue_time_p99_s", "p99")]),
    ("failure / timeout rate", [("fail_rate", "fail"), ("timeout_rate", "timeout")]),
]


def run_table(runs) -> pd.DataFrame:
    return pd.DataFrame([{"run": r.name, **r.config, **run_metrics(r.tasks)} for r in runs])


def mean_std(g, columns: list[str]) -> pd.DataFrame:
    """分组后每列给均值和样本标准差(n=1 时为空),列名 `<m>_mean` / `<m>_std`。"""
    agg = g[columns].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg


def aggregate_seeds(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """同配置的重复 Run(不同 seed)合并。"""
    g = df.groupby(keys, dropna=False)
    agg = mean_std(g, metrics)
    agg.insert(0, "n_seeds", g["seed"].nunique(dropna=False))
    return agg.reset_index()


def with_line_label(df: pd.DataFrame, line_cols: list[str]) -> pd.DataFrame:
    """加一列 `line`:线的图例名 = 各 line_cols 取值拼接(没有可分的列则为 all)。"""
    data = df.copy()
    data.insert(0, "line", df[line_cols].astype(str).agg(" ".join, axis=1) if line_cols else "all")
    return data


def platform_slug(platform) -> str:
    return "unknown" if pd.isna(platform) else str(platform)


def write_figure(df: pd.DataFrame, out: Path, stem: str, **plot_kwargs) -> None:
    df.to_csv(out / f"{stem}.csv", index=False)
    line_panels(df, path=out / f"{stem}.png", **plot_kwargs)


def perf_figures(runs: pd.DataFrame, out: Path) -> None:
    agg = aggregate_seeds(runs, CONFIG_KEYS, METRICS)
    for platform, df in agg.groupby("platform", dropna=False):
        slug = platform_slug(platform)
        for axis in SWEEP_AXES:
            if df[axis].nunique() < 2:
                continue
            others = [k for k in CONFIG_KEYS if k not in ("platform", axis) and df[k].nunique(dropna=False) > 1]
            write_figure(with_line_label(df, others), out, f"perf_by_{axis}_{slug}", x=axis, line_cols=others, panels=PERF_PANELS,
                         title=f"{axis} sweep [{slug}]")


PAIRED_PANELS: list[Panel] = [
    ("p50 completion time, on - off (s)", [("p50_s_delta", "p50")]),
    ("goodput, on - off (tasks/s)", [("goodput_per_s_delta", "goodput")]),
    ("throughput, on - off (tasks/s)", [("throughput_per_s_delta", "throughput")]),
    ("fail rate, on - off", [("fail_rate_delta", "fail")]),
]


def paired_cache_prompt(runs: pd.DataFrame, out: Path) -> None:
    """cache_prompt 开 - 关,同 seed(同一份 Workload)内配对相减,再跨 seed 取均值和标准差;只有一侧的 seed 不参与。"""
    keys = [k for k in CONFIG_KEYS if k != "cache_prompt"] + ["seed"]
    deltas = ["p50_s", "goodput_per_s", "throughput_per_s", "fail_rate"]
    df = runs[runs["cache_prompt"].notna()]
    for platform, d in df.groupby("platform", dropna=False):
        on = d[d["cache_prompt"].astype(bool)].set_index(keys)[deltas]
        off = d[~d["cache_prompt"].astype(bool)].set_index(keys)[deltas]
        diff = (on - off.reindex(on.index))[on.index.isin(off.index)]  # 缺 off 一侧的 seed 不配对
        if diff.empty:
            continue
        diff.columns = [f"{m}_delta" for m in deltas]
        diff = diff.reset_index()
        group_keys = [k for k in keys if k != "seed"]
        g = diff.groupby(group_keys, dropna=False)
        agg = mean_std(g, [f"{m}_delta" for m in deltas])
        agg.insert(0, "n_pairs", g["seed"].nunique())
        agg = agg.reset_index()
        others = [k for k in group_keys if k not in ("platform", "c") and agg[k].nunique(dropna=False) > 1]
        slug = platform_slug(platform)
        write_figure(with_line_label(agg, others), out, f"paired_cache_prompt_{slug}", x="c", line_cols=others, panels=PAIRED_PANELS,
                     title=f"cache_prompt on - off, paired by seed [{slug}]")


STEP_METRICS = ["n_requests", "fail_share", "ttft_s", "latency_s", "queue_wait_s", "prompt_tokens", "reuse_ratio"]
STEP_PANELS: list[Panel] = [
    ("prefix reuse ratio (cached / prompt tokens)", [("reuse_ratio", "reuse")]),
    ("time to first token, ok requests (s)", [("ttft_s", "ttft")]),
    ("request latency and queue wait, ok requests (s)", [("latency_s", "latency"), ("queue_wait_s", "queue wait")]),
    ("failed share of requests", [("fail_share", "fail")]),
]


def step_slice_figures(runs, out: Path) -> None:
    rows = [{"run": r.name, **r.config, **row} for r in runs for row in step_slice(r.requests, r.tasks)]
    if not rows:
        return
    agg = aggregate_seeds(pd.DataFrame(rows), CONFIG_KEYS + ["k"], STEP_METRICS)
    for platform, df in agg.groupby("platform", dropna=False):
        slug = platform_slug(platform)
        others = [k for k in CONFIG_KEYS if k != "platform" and df[k].nunique(dropna=False) > 1]
        write_figure(with_line_label(df, others), out, f"step_slice_{slug}", x="k", line_cols=others, panels=STEP_PANELS,
                     title=f"agent task, by step index k [{slug}]", x_label="step k")


MEMORY_KEYS = ["platform", "series", "B", "np", "c"]
RSS_NOTE = {"macos": "RSS (secondary; unreliable on macOS: compressed memory is not counted, ADR 0004)"}


def memory_figures(runs, out: Path) -> None:
    rows = []
    for r in runs:
        footprint, rss = peak_memory(r.system)
        rows.append({"run": r.name, **{k: r.config[k] for k in MEMORY_KEYS}, "footprint_mb_peak": footprint,
                     "rss_mb_peak": rss})
    df = pd.DataFrame(rows)
    for platform, d in df.groupby("platform", dropna=False):
        slug = platform_slug(platform)
        g = d.groupby(MEMORY_KEYS, dropna=False)
        agg = mean_std(g, ["footprint_mb_peak", "rss_mb_peak"])
        agg.insert(0, "n_runs", g.size())
        agg = agg.reset_index().dropna(subset=["footprint_mb_peak_mean", "rss_mb_peak_mean"], how="all")
        if agg.empty:  # 这些 Run 没有任何成功的内存采样:不出图
            continue
        memory_bars(agg, label_cols=MEMORY_KEYS[1:], title=f"peak memory by server config [{slug}]",
                    path=out / f"memory_peak_{slug}.png", rss_note=RSS_NOTE.get(slug, "RSS (secondary)"))
        agg.to_csv(out / f"memory_peak_{slug}.csv", index=False)


def kv_util_figures(runs, out: Path) -> None:
    """每个配置取 seed 最小的那个 Run 画一条 KV utilization 曲线(B 取自该 Run 自己的 -c)。"""
    config_keys = [k for k in CONFIG_KEYS if k != "platform"]
    first: dict[tuple, object] = {}
    for r in sorted(runs, key=lambda r: (r.config["seed"] is None, r.config["seed"], r.name)):
        first.setdefault((r.config["platform"], *(r.config[k] for k in config_keys)), r)
    for platform in sorted({k[0] for k in first}, key=str):
        chosen = [r for k, r in first.items() if k[0] == platform and r.config["B"]]
        varying = [k for k in config_keys if len({str(r.config[k]) for r in chosen}) > 1]
        rows = [{"line": " ".join(f"{k}={r.config[k]}" for k in varying) or "all", "run": r.name, "t": t, "kv_util": u}
                for r in chosen for t, u in kv_utilization(r.system, r.config["B"])]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        slug = platform_slug(platform)
        df.to_csv(out / f"kv_util_{slug}.csv", index=False)
        time_curves(df, title=f"KV utilization over time [{slug}]", path=out / f"kv_util_{slug}.png")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="原始日志目录(递归查找 Run 目录)")
    ap.add_argument("--out", required=True, help="输出目录:每张图一个同名 .csv 和 .png,另有 runs.csv")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    loaded = load_runs(Path(a.raw))
    runs = run_table(loaded)
    runs.to_csv(out / "runs.csv", index=False)
    if len(runs) and (dup := runs[runs.duplicated(CONFIG_KEYS + ["seed"], keep=False)]).shape[0]:
        print(f"warning: same config and seed in several runs, they are averaged as repeats: {sorted(dup['run'])}",
              file=sys.stderr)
    if len(runs):
        perf_figures(runs, out)
        paired_cache_prompt(runs, out)
        step_slice_figures(loaded, out)
        memory_figures(loaded, out)
        kv_util_figures(loaded, out)


if __name__ == "__main__":
    main()
