"""Regenerate the Phase 1 report figures from the raw batch logs and analysis CSVs.

    .venv/bin/python report/phase1/make_breakdown.py   # first: writes tables/derived_numbers.json
    .venv/bin/python report/phase1/make_figs.py

Fig. 1(a) is recomputed from results/raw/batch/*/tasks.jsonl (steady-state tasks, 3 seeds);
Fig. 1(b) and (c) read the CSVs written by `python -m src.analysis` (results/analysis/batch);
Fig. 2 reads tables/derived_numbers.json.
"""
import collections
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "results", "raw", "batch")
CSV = os.path.join(ROOT, "results", "analysis", "batch")
OUT = os.path.join(os.path.dirname(__file__), "figs")
DERIVED = os.path.join(os.path.dirname(__file__), "tables", "derived_numbers.json")
os.makedirs(OUT, exist_ok=True)

# categorical slots 1-3 of the reference palette (validated, adjacent CVD dE >= 9)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def outcome_shares(c):
    n = collections.Counter()
    for d in glob.glob(os.path.join(RAW, f"S_np{c}_c{c}_lam0.088_cpon_dvar_s*")):
        for line in open(os.path.join(d, "tasks.jsonl")):
            t = json.loads(line)
            if not t["warmup"]:
                n[t["status"]] += 1
    tot = sum(n.values())
    return {k: 100 * n[k] / tot for k in ("ok", "timeout", "error_ctx_exceeded")}, tot


def completed_sd(c):
    """Sample sd across seeds of the share of steady-state tasks completed, in percentage points."""
    shares = []
    for d in glob.glob(os.path.join(RAW, f"S_np{c}_c{c}_lam0.088_cpon_dvar_s*")):
        rows = [json.loads(line) for line in open(os.path.join(d, "tasks.jsonl"))]
        rows = [t for t in rows if not t["warmup"]]
        shares.append(100 * sum(t["status"] == "ok" for t in rows) / len(rows))
    return float(pd.Series(shares).std(ddof=1))


fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.0), gridspec_kw={"width_ratios": [1.15, 1, 1]})

# (a) task outcome vs c (series S, lambda = 0.088)
cs = [1, 2, 4, 8]
shares = {c: outcome_shares(c)[0] for c in cs}
bottom = [0.0] * len(cs)
for key, color, label in (("ok", BLUE, "completed"), ("timeout", ORANGE, "timeout"),
                          ("error_ctx_exceeded", AQUA, "ctx exceeded")):
    vals = [shares[c][key] for c in cs]
    ax[0].bar(range(len(cs)), vals, bottom=bottom, color=color, width=0.62, label=label,
              edgecolor="white", linewidth=1.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 8:
            # the completed segment carries a sd whisker at its top, so its label sits low
            y = b + v * (0.3 if key == "ok" else 0.5)
            ax[0].text(i, y, f"{v:.0f}", ha="center", va="center", color=INK, fontsize=6.5)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax[0].errorbar(range(len(cs)), [shares[c]["ok"] for c in cs], yerr=[completed_sd(c) for c in cs], fmt="none",
               ecolor=INK, elinewidth=0.8, capsize=2)
ax[0].set_xticks(range(len(cs)))
ax[0].set_xticklabels([f"{c}\n({16384 // c // 1024}k)" for c in cs])
ax[0].set_xlabel("concurrency c (KV tokens per slot)")
ax[0].set_ylabel("% of tasks")
ax[0].set_ylim(0, 140)
ax[0].set_yticks([0, 25, 50, 75, 100])
ax[0].grid(axis="x", visible=False)
ax[0].legend(loc="upper center", ncol=3, frameon=False,
             handlelength=1, columnspacing=0.9)
ax[0].set_title("(a) Outcomes, $\\lambda$=0.088", y=1.04, fontsize=8)

# (b) failure rate vs arrival rate, c = 4, series S
lam = pd.read_csv(os.path.join(CSV, "perf_by_lambda_macos.csv"))
d = lam[(lam.series == "S") & (lam.c == 4) & (lam.cache_prompt == True) & (lam.tool_delay_s.isna())]  # noqa: E712
d = d.sort_values("lambda")
ax[1].errorbar(d["lambda"], 100 * d.fail_rate_mean, yerr=100 * d.fail_rate_std, color=BLUE,
               marker="o", ms=4, lw=1.5, capsize=2, elinewidth=0.8)
for x, y in zip(d["lambda"], 100 * d.fail_rate_mean):
    ax[1].annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(-10, 4), ha="center", fontsize=6.5)
ax[1].set_xlabel("arrival rate $\\lambda$ (task/s)")
ax[1].set_ylabel("failed tasks (%)")
ax[1].set_ylim(0, 105)
ax[1].set_xlim(0.018, 0.128)
ax[1].set_xticks(d["lambda"])
ax[1].set_xticklabels([f"{x:.3f}" for x in d["lambda"]])
ax[1].set_title("(b) Saturation, c=4 (mean$\\pm$sd)", y=1.04, fontsize=8)

# (c) peak footprint vs configuration
mem = pd.read_csv(os.path.join(CSV, "memory_peak_macos.csv")).sort_values(["series", "c"])
labels = [f"{r.series} c={r.c}" for r in mem.itertuples()]
ax[2].bar(range(len(mem)), mem.footprint_mb_peak_mean, color=BLUE, width=0.62,
          yerr=mem.footprint_mb_peak_std, error_kw={"elinewidth": 0.8, "capsize": 2})
for i, v in enumerate(mem.footprint_mb_peak_mean):
    ax[2].text(i, v + 120, f"{v:.0f}", ha="center", fontsize=5.5)
ax[2].set_xticks(range(len(mem)))
ax[2].set_xticklabels(labels, rotation=35, ha="right")
ax[2].set_ylabel("peak footprint (MiB)")
ax[2].set_ylim(0, 5000)
ax[2].grid(axis="x", visible=False)
ax[2].set_title("(c) Peak memory (45 runs)", y=1.04, fontsize=8)

fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(OUT, "fig1.pdf"))
print("wrote", os.path.join(OUT, "fig1.pdf"))

# Figure 2: where task time goes, and what context growth costs at the Request level
with open(DERIVED) as f:
    derived = json.load(f)

fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15), gridspec_kw={"width_ratios": [1.3, 1, 1]})

# (a) mean per-task time accounting by type, c = 4, low vs high arrival rate
types = ("short", "longctx", "longgen", "agent")
positions, tick_labels = [], []
for g, ty in enumerate(types):
    for b, (key, lam_label) in enumerate((("L029", "0.029"), ("S4", "0.088"))):
        x = g * 2.6 + b
        row = derived["time_accounting"][key][ty]
        bottom = 0.0
        for part, color, label in (("queue", ORANGE, "queue"), ("service", BLUE, "service"), ("tool", AQUA, "tool delay")):
            ax[0].bar(x, row[part], bottom=bottom, width=0.8, color=color, edgecolor="white", linewidth=1,
                      label=label if g == 0 and b == 0 else None)
            bottom += row[part]
        ax[0].errorbar(x, bottom, yerr=row["total_sd"], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
        positions.append(x)
        tick_labels.append(lam_label)
    ax[0].text(g * 2.6 + 0.5, -80, ty, ha="center", va="top", fontsize=7, color=INK)
ax[0].set_xticks(positions)
ax[0].set_xticklabels(tick_labels, fontsize=5, rotation=90)
ax[0].tick_params(axis="x", length=0, pad=1)
ax[0].set_ylabel("mean seconds per task")
ax[0].set_ylim(0, 340)
ax[0].set_yticks([0, 50, 100, 150, 200, 250])
ax[0].grid(axis="x", visible=False)
ax[0].legend(loc="upper left", ncol=3, frameon=False, handlelength=1, columnspacing=0.9)
ax[0].set_title("(a) Task time, c=4 (bars: $\\lambda$; sd of total)", y=1.04, fontsize=8)

# (b) median TTFT per agent step; TTFT excludes runner queueing
series = (("c4_lam0.029_cache_on", BLUE, "o", "cache on, $\\lambda$=0.029"),
          ("c4_lam0.088_cache_on", ORANGE, "s", "cache on, $\\lambda$=0.088"),
          ("c4_lam0.088_cache_off", AQUA, "^", "cache off, $\\lambda$=0.088"))
for key, color, marker, label in series:
    rows = derived["steps"][key]
    ks = sorted(int(k) for k in rows)
    med = [rows[str(k)]["ttft"] for k in ks]
    err = [[rows[str(k)]["ttft"] - rows[str(k)]["ttft_q25"] for k in ks], [rows[str(k)]["ttft_q75"] - rows[str(k)]["ttft"] for k in ks]]
    ax[1].errorbar(ks, med, yerr=err, color=color, marker=marker, ms=3.5, lw=1.5, capsize=1.5, elinewidth=0.6,
                   label=label, markeredgecolor="white", markeredgewidth=0.5)
prompt = derived["steps"]["c4_lam0.029_cache_on"]
step_ticks = sorted(int(k) for k in prompt)
ax[1].set_xticks(step_ticks)
ax[1].set_xticklabels([f"{k}\n{prompt[str(k)]['prompt_tokens'] / 1000:.1f}k" for k in step_ticks])
ax[1].set_xlabel("agent step k (median prompt tokens)")
ax[1].set_ylabel("TTFT (s; median, IQR)")
ax[1].set_ylim(0, 52)
ax[1].set_yticks([0, 10, 20, 30])
ax[1].legend(loc="upper center", frameon=False, handlelength=1.4, labelspacing=0.2, borderaxespad=0)
ax[1].set_title("(b) Context growth vs TTFT, c=4", y=1.04, fontsize=8)

# (c) share of the prompt served from retained context, per agent step
for key, color, marker, label in series[:2]:
    rows = derived["steps"][key]
    ks = sorted(int(k) for k in rows)
    ax[2].plot(ks, [100 * rows[str(k)]["reuse"] for k in ks], color=color, marker=marker, ms=3.5, lw=1.5,
               label=label, markeredgecolor="white", markeredgewidth=0.5)
ax[2].set_xticks(step_ticks)
ax[2].set_xlabel("agent step k")
ax[2].set_ylabel("prompt tokens reused (%)")
ax[2].set_ylim(0, 150)
ax[2].set_yticks([0, 25, 50, 75, 100])
ax[2].legend(loc="upper center", frameon=False, handlelength=1.4, labelspacing=0.2, borderaxespad=0)
ax[2].set_title("(c) Prefix reuse, c=4", y=1.04, fontsize=8)

fig.tight_layout(w_pad=1.0)
fig.savefig(os.path.join(OUT, "fig2.pdf"))
print("wrote", os.path.join(OUT, "fig2.pdf"))
