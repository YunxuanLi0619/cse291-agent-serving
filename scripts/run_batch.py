#!/usr/bin/env python3
"""Batch E1–E5 (#9): can be run in overnight batches and is resumable. Usually started via scripts/run_batch.sh.

    .venv/bin/python scripts/run_batch.py --dry-run          # only print the plan and estimated duration; AC power not required
    .venv/bin/python scripts/run_batch.py [--only E1-S E4]   # official batch; after an interruption, rerun as-is to continue from the next Run

Exit codes: 0 all done; 1 a Run failed or an experiment was blocked by its pilot; 2 bad arguments; 3 gate failed (not on AC / low-power mode), refusing to start; 4 pre-check failed (Workload missing/modified, port 8080 in use).
For details (directory layout, retries, pilot gate) see the top of src/batch/driver.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.batch.driver import THERM_SAMPLE_S, Batch  # noqa: E402
from src.batch.plan import build_plan  # noqa: E402
from src.batch.system import MacHooks, port_in_use  # noqa: E402
from src.batch.workloads import verify_workloads  # noqa: E402


def print_plan(cfg: dict, only: list[str] | None) -> None:
    total = 0.0
    for st in build_plan(cfg):
        if only and st.experiment not in only:
            continue
        secs = st.pilot.estimated_s + sum(r.estimated_s for r in st.runs)
        total += secs
        print(f"{st.experiment:5s} pilot {st.pilot.name} + {len(st.runs)} runs, reusing {len(st.reused)}: ~{secs / 3600:.1f} h")
    print(f"total ~{total / 3600:.1f} h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="run only these experiments (E1-S E4 E5 E3 E1-U)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilots-only", action="store_true", help="only do the 30-Task pilot of each experiment (a passed pilot is not redone afterwards)")
    a = ap.parse_args()
    cfg = json.loads((ROOT / "configs/batch.json").read_text())
    if a.only and not set(a.only) <= {s.experiment for s in build_plan(cfg)}:
        sys.exit(2)
    if a.dry_run:
        return print_plan(cfg, a.only)
    problems = verify_workloads(ROOT, cfg) + (["port 8080 is already in use"] if port_in_use() else [])
    if problems:
        print("preflight failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        sys.exit(4)
    # AC power also has idle sleep (it slept 644 s and 900 s in #7): do not allow system idle sleep while this process is alive
    subprocess.Popen(["caffeinate", "-is", "-w", str(os.getpid())])
    print_plan(cfg, a.only)
    code = Batch(cfg, ROOT, MacHooks(ROOT).hooks(), only=a.only, therm_sample_s=THERM_SAMPLE_S,
                 pilots_only=a.pilots_only).run()
    if a.pilots_only:
        sys.exit(code)
    r = subprocess.run([sys.executable, "-m", "src.analysis", "--raw", "results/raw/batch", "--out", "results/analysis/batch"],
                       cwd=ROOT, capture_output=True, text=True)
    print("analysis:", "ok" if r.returncode == 0 else f"failed ({r.stderr.strip()[-300:]})", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
