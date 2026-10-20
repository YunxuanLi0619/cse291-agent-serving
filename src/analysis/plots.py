"""作图:一张图 = 一个 DataFrame,调用方负责把同一个 DataFrame 写成同名 CSV(报告引用 CSV)。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

Panel = tuple[str, list[tuple[str, str]]]  # (子图标题, [(指标列前缀, 图例名)]);均值列 `<m>_mean`,误差列 `<m>_std`


def _label(key, line_cols: list[str]) -> str:
    return " ".join(f"{c}={v}" for c, v in zip(line_cols, key)) or "all"


def line_panels(df: pd.DataFrame, *, x: str, line_cols: list[str], panels: list[Panel], title: str,
                path: Path, x_label: str | None = None) -> None:
    """每个子图 = 若干指标;每条线 = line_cols 的一个取值组合;误差条 = 重复 Run(不同 seed)间的样本标准差。"""
    ncols = 2
    fig, axes = plt.subplots((len(panels) + 1) // ncols, ncols, figsize=(11, 4 * ((len(panels) + 1) // ncols)),
                             squeeze=False)
    groups = list(df.groupby(line_cols, dropna=False, sort=True)) if line_cols else [((), df)]
    for ax, (panel_title, metrics) in zip(axes.flat, panels):
        for key, g in groups:
            key = key if isinstance(key, tuple) else (key,)
            g = g.sort_values(x)
            for metric, name in metrics:
                err = g[f"{metric}_std"] if f"{metric}_std" in g else None  # n=1 时 std 为 NaN:不画误差条,而不是画成零方差
                ax.errorbar(g[x], g[f"{metric}_mean"], yerr=err, marker="o", capsize=3,
                            label=f"{name} [{_label(key, line_cols)}]")
        ax.set_title(panel_title)
        ax.set_xlabel(x_label or x)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def memory_bars(df: pd.DataFrame, *, label_cols: list[str], title: str, path: Path, rss_note: str) -> None:
    """每个服务配置一组柱:footprint(主)和 RSS(次,图例里注明不可靠的原因);误差条 = 跨 Run 标准差。"""
    labels = [" ".join(f"{c}={row[c]}" for c in label_cols) for _, row in df.iterrows()]
    pos = range(len(df))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(df) + 3), 4.5))
    for off, metric, name in [(-0.2, "footprint_mb_peak", "footprint (primary)"), (0.2, "rss_mb_peak", rss_note)]:
        ax.bar([p + off for p in pos], df[f"{metric}_mean"], width=0.4, label=name,
               yerr=df[f"{metric}_std"].fillna(0), capsize=3)
    ax.set_xticks(list(pos), labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("peak MB")
    ax.set_title(title)
    ax.set_ylim(top=max(df["footprint_mb_peak_mean"].max(), df["rss_mb_peak_mean"].max()) * 1.3)  # 给图例留位置
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def time_curves(df: pd.DataFrame, *, title: str, path: Path) -> None:
    """KV utilization 随 Run 内时间的曲线;每个 `line` 一条。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for line, g in df.groupby("line", sort=True):
        ax.plot(g["t"], g["kv_util"], label=line, linewidth=1)
    ax.set_xlabel("time since run start (s)")
    ax.set_ylabel("KV utilization (live tokens / B)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
