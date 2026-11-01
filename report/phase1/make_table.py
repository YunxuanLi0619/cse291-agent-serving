"""Write tables/perf_table.tex (Appendix C) from the analysis CSVs.

    .venv/bin/python report/phase1/make_table.py
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(ROOT, "results", "analysis", "batch")
OUT = os.path.join(os.path.dirname(__file__), "tables", "perf_table.tex")

perf = pd.read_csv(os.path.join(CSV, "perf_by_c_macos.csv"))
mem = pd.read_csv(os.path.join(CSV, "memory_peak_macos.csv"))
mem = mem.set_index(["series", "c"]).footprint_mb_peak_mean / 1000

rows = []  # (experiment, label, row)


def pick(series, c, lam, cache, delay=None):
    d = perf[(perf.series == series) & (perf.c == c) & (abs(perf["lambda"] - lam) < 1e-9)
             & (perf.cache_prompt == cache)]
    d = d[d.tool_delay_s.isna()] if delay is None else d[d.tool_delay_s == delay]
    assert len(d) == 1, (series, c, lam, cache, delay, len(d))
    return d.iloc[0]


for s, c in (("S", 1), ("S", 2), ("S", 4), ("S", 8), ("U", 4), ("U", 8)):
    rows.append(("E1", f"{s}, $c{{=}}{c}$, $\\lambda{{=}}0.088$", s, c, pick(s, c, 0.088, True)))
for c in (1, 4):
    rows.append(("E4", f"S, $c{{=}}{c}$, cache off", "S", c, pick("S", c, 0.088, False)))
for lam in (0.029, 0.059, 0.118):
    rows.append(("E5", f"S, $c{{=}}4$, $\\lambda{{=}}{lam}$", "S", 4, pick("S", 4, lam, True)))
for dl in (0.0, 0.5, 1.0, 3.0):
    rows.append(("E3", f"S, $c{{=}}4$, delay {dl:g}\\,s", "S", 4, pick("S", 4, 0.088, True, dl)))

lines = [
    "\\begin{tabular}{@{}llrrrrrr@{}}", "\\toprule",
    "Exp. & Configuration & p50 (s) & p99 (s) & Thr. & Goodput & Fail (\\%) & Peak (GB) \\\\",
    "\\midrule",
]
prev = None
for exp, label, s, c, r in rows:
    if prev and exp != prev:
        lines.append("\\addlinespace[2pt]")
    prev = exp
    lines.append(
        f"{exp} & {label} & {r.p50_s_mean:.0f} & {r.p99_s_mean:.0f} & {r.throughput_per_s_mean:.3f} & "
        f"{r.goodput_per_s_mean:.3f} & {100 * r.fail_rate_mean:.0f} & {mem[(s, c)]:.2f} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT, len(rows), "rows")
