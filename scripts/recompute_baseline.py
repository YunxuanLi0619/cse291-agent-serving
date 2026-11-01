"""Recompute results/batch/baseline_c.json from the finished E1-S runs under the relaxed rule.

    .venv/bin/python scripts/recompute_baseline.py

Keeps the original strict-rule result next to it (the rule was relaxed after seeing the data).
Also reports c=4 at the lowest lambda (E5) as supplementary evidence; it is NOT part of the choice,
because c=1/2 were never run at that lambda.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.analysis.metrics import run_metrics  # noqa: E402
from src.analysis.runs import load_run  # noqa: E402
from src.batch.baseline import ctx_exceeded_rate, p99_deadline_ratio, select_baseline  # noqa: E402

RAW = Path("results/batch/../raw/batch").resolve()
OUT = Path("results/batch/baseline_c.json")


def rows_for(pattern: str) -> list[dict]:
    rows = []
    for d in sorted(RAW.iterdir()):
        m = re.fullmatch(pattern, d.name)
        if not m:
            continue
        run = load_run(d)
        met = run_metrics(run.tasks)
        rows.append({"c": int(m["c"]), "seed": int(m["seed"]), "run": d.name, "goodput_per_s": met["goodput_per_s"],
                     "p99_deadline_ratio": p99_deadline_ratio(run.tasks), "fail_rate": met["fail_rate"],
                     "ctx_exceeded_rate": ctx_exceeded_rate(run.tasks)})
    return rows


e1 = rows_for(r"S_np(?P<c>\d+)_c\d+_lam0\.088_cpon_dvar_s(?P<seed>\d)")
low = rows_for(r"S_np(?P<c>4)_c4_lam0\.029_cpon_dvar_s(?P<seed>\d)")
result = select_baseline(e1) | {
    "experiment": "E1-S", "lambda": 0.088, "n_runs": len(e1), "complete": len(e1) == 12,
    "strict_rule_result": select_baseline(e1, rule="strict"),
    "supplementary_c4_lambda_0.029": select_baseline(low) if low else None,
}
OUT.write_text(json.dumps(result, indent=1))
print("relaxed:", result["c"], "acceptable:", result["p99_acceptable"])
for r in result["by_c"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("strict :", result["strict_rule_result"]["c"], result["strict_rule_result"]["p99_acceptable"])
print("c=4 @0.029 (relaxed):", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result["supplementary_c4_lambda_0.029"]["by_c"][0].items()})
