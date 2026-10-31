#!/usr/bin/env python3
"""批跑 E1–E5(#9):可分批过夜、可恢复。通常经 scripts/run_batch.sh 启动。

    .venv/bin/python scripts/run_batch.py --dry-run          # 只打印计划和估计时长,不要求插电
    .venv/bin/python scripts/run_batch.py [--only E1-S E4]   # 正式批跑;中断后原样重跑即从下一个 Run 继续

退出码:0 全部完成;1 有 Run 失败或实验被试跑拦下;2 参数错;3 门禁不过(未插电/低电量模式)拒绝启动;4 预检不过(Workload 缺失/被改、8080 被占)。
细节(目录布局、重试、试跑门禁)见 src/batch/driver.py 顶部。
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
    ap.add_argument("--only", nargs="+", help="只跑这些实验(E1-S E4 E5 E3 E1-U)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pilots-only", action="store_true", help="只做各实验的 30 Task 试跑(通过的试跑之后不会重做)")
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
    # 插电也会空闲睡眠(#7 里睡了 644 s 和 900 s):本进程存活期间不允许系统空闲睡眠
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
