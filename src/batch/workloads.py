"""批跑用到的 Workload 文件:E3 的固定 Tool delay 变体由 default_s<seed>.json 派生;所有文件的 sha256 冻结在 configs/batch.json。"""
import hashlib
import json
from pathlib import Path

from src.runner.gen_workload import with_fixed_tool_delay, write_workload

from .plan import build_plan


def write_e3_variants(root: Path, cfg: dict) -> None:
    """每个 seed、每个 delay 一份;同输入字节一致,可重复调用。"""
    for seed in range(1, cfg["reps"] + 1):
        base = json.loads((root / f"workloads/dev/default_s{seed}.json").read_text())
        for delay in cfg["e3_delays_s"]:
            write_workload(with_fixed_tool_delay(base, delay), root / f"workloads/dev/e3_d{delay:g}_s{seed}.json")


def needed_workloads(cfg: dict) -> list[str]:
    plan = build_plan(cfg)
    return sorted({r.workload for st in plan for r in st.runs} | {st.pilot.workload for st in plan})


def verify_workloads(root: Path, cfg: dict) -> list[str]:
    """批跑要用的每个 Workload 必须存在,且 sha256 等于冻结值(之后不得改动)。"""
    problems = []
    frozen = cfg.get("workload_sha256", {})
    for rel in needed_workloads(cfg):
        path = root / rel
        if not path.exists():
            problems.append(f"{rel}: missing")
        elif rel not in frozen:
            problems.append(f"{rel}: no frozen sha256 in configs/batch.json")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != frozen[rel]:
            problems.append(f"{rel}: sha256 differs from the frozen value")
    return problems
