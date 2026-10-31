"""批跑的门禁与 Run 完整性校验。返回问题列表(空 = 通过),读不到的一律当成有问题,不假设没事。

门禁:电源必须是 AC、低电量模式关(#9);降频只记录,不拒绝(无风扇机器长时间满载几乎必然降频)。
校验:Run 目录产出齐全;有问题的 Run 不能算「已完成」,批跑会把它挪开重跑。
"""
import json
import re
from pathlib import Path

from src.analysis.runs import read_jsonl
from src.runner.manifest import low_power_mode, power_source

MAX_SAMPLE_GAP_S = 5.0  # 采样间隔 0.5 s;大缺口 = 系统睡眠或进程被冻住
MAX_SAMPLE_ERROR_SHARE = 0.02
MAX_ERROR_OTHER_SHARE = 0.05  # 超时是过载的正常结果;error_other 多半是 server 挂了
WARMUP_TASKS = 3
_LIMIT_RE = re.compile(r"^\s*CPU_(?:Speed|Scheduler)_Limit\s*=\s*(\d+)", re.MULTILINE)
_WARNING_RE = re.compile(r"^(?!Note:).*warning level", re.MULTILINE | re.IGNORECASE)


def gate_problems(batt: str, settings: str) -> list[str]:
    problems = []
    try:
        source = power_source(batt)
        if source != "AC Power":
            problems.append(f"power source is '{source}', not AC Power")
    except ValueError as e:
        problems.append(f"cannot read power source: {e}")
    try:
        if low_power_mode(settings):
            problems.append("low power mode is on")
    except ValueError as e:
        problems.append(f"cannot read low power mode: {e}")
    return problems


def is_throttled(therm: str) -> bool:
    """`pmset -g therm` 没有降频时只有三行 `Note: No ... recorded`;降频时出现 CPU_*_Limit < 100 或 warning level。"""
    return any(int(v) < 100 for v in _LIMIT_RE.findall(therm)) or bool(_WARNING_RE.search(therm))


def validate_run(run_dir: Path, expected_tasks: int) -> list[str]:
    problems = []
    for name in ("manifest.json", "tasks.jsonl", "requests.jsonl", "system.jsonl"):
        if not (run_dir / name).exists():
            problems.append(f"{name} is missing")
    if problems:
        return problems
    manifest = json.loads((run_dir / "manifest.json").read_text())
    tasks, system = read_jsonl(run_dir / "tasks.jsonl"), read_jsonl(run_dir / "system.jsonl")
    if "t_end" not in manifest:
        problems.append("manifest has no t_end (run did not finish)")
    if manifest.get("errors"):
        problems.append(f"manifest errors: {', '.join(sorted(manifest['errors']))}")
    if manifest.get("cold_start_s") is None:
        problems.append("manifest has no cold_start_s")
    if manifest.get("power_source") != "AC Power":
        problems.append(f"run started on '{manifest.get('power_source')}'")
    if manifest.get("low_power_mode") is not False:
        problems.append("run started in low power mode")
    if len(tasks) != expected_tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        problems.append(f"tasks.jsonl has {len(tasks)} rows, expected {expected_tasks}")
    n_warmup = sum(bool(t["warmup"]) for t in tasks)
    if tasks and n_warmup != WARMUP_TASKS:
        problems.append(f"{n_warmup} warmup tasks, expected {WARMUP_TASKS}")
    n_other = sum(t["status"] == "error_other" for t in tasks)
    if tasks and n_other / len(tasks) > MAX_ERROR_OTHER_SHARE:
        problems.append(f"{n_other}/{len(tasks)} tasks ended in error_other (server died?)")
    if not system:
        problems.append("system.jsonl is empty")
    else:
        gap = max((b["t"] - a["t"] for a, b in zip(system, system[1:])), default=0.0)
        if gap > MAX_SAMPLE_GAP_S:
            problems.append(f"system.jsonl has a {gap:.1f} s gap (system slept?)")
        n_err = sum(bool(r.get("error")) for r in system)
        if n_err / len(system) > MAX_SAMPLE_ERROR_SHARE:
            problems.append(f"{n_err}/{len(system)} system samples have errors")
    return problems
