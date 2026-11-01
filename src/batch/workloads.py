"""Workload files used by the batch: the fixed-Tool-delay variants of E3 are derived from default_s<seed>.json; the sha256 of every file is frozen in configs/batch.json."""
import hashlib
import json
from pathlib import Path

from src.runner.gen_workload import with_fixed_tool_delay, write_workload

from .plan import build_plan


def write_e3_variants(root: Path, cfg: dict) -> None:
    """One per seed and per delay; same input gives identical bytes, safe to call repeatedly."""
    for seed in range(1, cfg["reps"] + 1):
        base = json.loads((root / f"workloads/dev/default_s{seed}.json").read_text())
        for delay in cfg["e3_delays_s"]:
            write_workload(with_fixed_tool_delay(base, delay), root / f"workloads/dev/e3_d{delay:g}_s{seed}.json")


def needed_workloads(cfg: dict) -> list[str]:
    plan = build_plan(cfg)
    return sorted({r.workload for st in plan for r in st.runs} | {st.pilot.workload for st in plan})


def verify_workloads(root: Path, cfg: dict) -> list[str]:
    """Every Workload the batch needs must exist and its sha256 must equal the frozen value (it must not change afterwards)."""
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
