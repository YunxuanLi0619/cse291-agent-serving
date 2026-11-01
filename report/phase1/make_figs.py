"""Regenerate the Phase 1 report figure from the raw batch logs and analysis CSVs.

    .venv/bin/python report/phase1/make_figs.py

Panel (a) is recomputed from results/raw/batch/*/tasks.jsonl (steady-state tasks, 3 seeds);
(b) and (c) read the CSVs written by `python -m src.analysis` (results/analysis/batch).
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


fig, ax = plt.subplots(1, 3, figsize=(7.1, 1.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})

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
            ax[0].text(i, b + v / 2, f"{v:.0f}", ha="center", va="center", color=INK, fontsize=6.5)
    bottom = [b + v for b, v in zip(bottom, vals)]
ax[0].set_xticks(range(len(cs)))
ax[0].set_xticklabels([f"{c}\n({16384 // c // 1024}k)" for c in cs])
ax[0].set_xlabel("concurrency c (KV tokens per slot)")
ax[0].set_ylabel("% of steady-state tasks")
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
    ax[1].annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
ax[1].set_xlabel("arrival rate $\\lambda$ (task/s)")
ax[1].set_ylabel("failed tasks (%)")
ax[1].set_ylim(0, 105)
ax[1].set_xticks(d["lambda"])
ax[1].set_xticklabels([f"{x:.3f}" for x in d["lambda"]])
ax[1].set_title("(b) Saturation, c=4 (mean$\\pm$sd)", y=1.04, fontsize=8)

# (c) peak footprint vs configuration
mem = pd.read_csv(os.path.join(CSV, "memory_peak_macos.csv")).sort_values(["series", "c"])
labels = [f"{r.series} c={r.c}" for r in mem.itertuples()]
ax[2].bar(range(len(mem)), mem.footprint_mb_peak_mean / 1000, color=BLUE, width=0.62,
          yerr=mem.footprint_mb_peak_std / 1000, error_kw={"elinewidth": 0.8, "capsize": 2})
for i, v in enumerate(mem.footprint_mb_peak_mean / 1000):
    ax[2].text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=6.5)
ax[2].set_xticks(range(len(mem)))
ax[2].set_xticklabels(labels, rotation=35, ha="right")
ax[2].set_ylabel("peak footprint (GB)")
ax[2].set_ylim(0, 5)
ax[2].grid(axis="x", visible=False)
ax[2].set_title("(c) Peak memory (45 runs)", y=1.04, fontsize=8)

fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(OUT, "fig1.pdf"))
print("wrote", os.path.join(OUT, "fig1.pdf"))
