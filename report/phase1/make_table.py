"""Write tables/perf_table.tex (Appendix B) from the analysis CSVs.

    .venv/bin/python report/phase1/make_table.py

Cells are mean +/- sample sd over the 3 seeds. `footprint` prints binary units, so its "MB" is MiB.
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(ROOT, "results", "analysis", "batch")
OUT = os.path.join(os.path.dirname(__file__), "tables", "perf_table.tex")

perf = pd.read_csv(os.path.join(CSV, "perf_by_c_macos.csv"))
mem = pd.read_csv(os.path.join(CSV, "memory_peak_macos.csv")).set_index(["series", "c"])

rows = []  # (experiment, label, row)


def pick(series, c, lam, cache, delay=None):
    d = perf[(perf.series == series) & (perf.c == c) & (abs(perf["lambda"] - lam) < 1e-9)
             & (perf.cache_prompt == cache)]
    d = d[d.tool_delay_s.isna()] if delay is None else d[d.tool_delay_s == delay]
    assert len(d) == 1, (series, c, lam, cache, delay, len(d))
    return d.iloc[0]


def pm(mean, sd, digits=0, scale=1.0):
    return f"${scale * mean:.{digits}f}\\pm{scale * sd:.{digits}f}$"


for s, c in (("S", 1), ("S", 2), ("S", 4), ("S", 8), ("U", 4), ("U", 8)):
    rows.append(("E1", f"{s}, $c{{=}}{c}$, $\\lambda{{=}}0.088$", s, c, pick(s, c, 0.088, True)))
for c in (1, 4):
    rows.append(("E4", f"S, $c{{=}}{c}$, cache off", "S", c, pick("S", c, 0.088, False)))
for lam in (0.029, 0.059, 0.118):
    rows.append(("E5", f"S, $c{{=}}4$, $\\lambda{{=}}{lam}$", "S", 4, pick("S", 4, lam, True)))
for dl in (0.0, 0.5, 1.0, 3.0):
    rows.append(("E3", f"S, $c{{=}}4$, delay {dl:g}\\,s", "S", 4, pick("S", 4, 0.088, True, dl)))

lines = [
    "\\begin{tabular}{@{}llrrrrrrr@{}}", "\\toprule",
    "Exp. & Configuration & p50 (s) & p99 (s) & Queue (s) & Thr. (task/s) & Goodput (task/s) & Fail (\\%) & Peak (MiB) \\\\",
    "\\midrule",
]
prev = None
for exp, label, s, c, r in rows:
    if prev and exp != prev:
        lines.append("\\addlinespace[2pt]")
    prev = exp
    m = mem.loc[(s, c)]
    lines.append(
        f"{exp} & {label} & {pm(r.p50_s_mean, r.p50_s_std)} & {pm(r.p99_s_mean, r.p99_s_std)} & "
        f"{pm(r.queue_time_mean_s_mean, r.queue_time_mean_s_std)} & "
        f"{pm(r.throughput_per_s_mean, r.throughput_per_s_std, 3)} & "
        f"{pm(r.goodput_per_s_mean, r.goodput_per_s_std, 3)} & "
        f"{pm(r.fail_rate_mean, r.fail_rate_std, 0, 100)} & "
        f"{pm(m.footprint_mb_peak_mean, m.footprint_mb_peak_std)} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT, len(rows), "rows")
