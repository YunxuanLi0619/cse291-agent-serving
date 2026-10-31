"""批跑驱动(#9):按 plan 逐个 Run 执行,可分批过夜、可恢复。

目录(root 下):
  results/raw/batch/<run>/        已完成且通过校验的 Run(分析读这里);存在即视为完成,恢复时跳过
  results/batch/running/<run>/    正在跑的 Run;启动时发现残留 = 上次被打断,挪到 invalid 后重跑
  results/batch/pilots/<E>/<run>/ 每个实验的 30 Task 试跑
  results/batch/invalid/<run>__<时间>/  校验没过的尝试(含 problems.json),保留作证据
  results/batch/state.json        失败的 Run、被试跑拦下的实验
  results/batch/batch_start_<时间>.json  每批开始前的电源、低电量模式、pmset -g therm

所有副作用(pmset、server、runner、睡眠)经 Hooks 注入,测试里全是假的。
"""
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .baseline import select_baseline
from .checks import gate_problems, is_throttled, validate_run
from .plan import RunSpec, Stage, build_plan

EXIT_REFUSED = 3  # 门禁不过,拒绝启动
MAX_ATTEMPTS = 3
PILOT_ATTEMPTS = 2
POWER_POLL_S = 60
COOLDOWN_POLL_S = 30
COOLDOWN_MAX_S = 600
THERM_SAMPLE_S = 30


class ServerHandle(Protocol):
    pid: int
    cold_start_s: float

    def alive(self) -> bool: ...
    def stop(self) -> None: ...


@dataclass
class Hooks:
    batt: Callable[[], str]
    settings: Callable[[], str]
    therm: Callable[[], str]
    sleep: Callable[[float], None]
    start_server: Callable[[RunSpec, Path], ServerHandle]  # (spec, server.log 路径),启动失败抛异常
    run_runner: Callable[[RunSpec, Path, ServerHandle], int]  # 返回 runner 退出码
    smoke_analysis: Callable[[Path], list[str]]  # 试跑后用分析命令读一遍,返回问题


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


class Batch:
    def __init__(self, cfg: dict, root: Path, hooks: Hooks, *, only: list[str] | None = None,
                 max_attempts: int = MAX_ATTEMPTS, pilot_attempts: int = PILOT_ATTEMPTS,
                 power_poll_s: float = POWER_POLL_S, cooldown_poll_s: float = COOLDOWN_POLL_S,
                 cooldown_max_s: float = COOLDOWN_MAX_S, therm_sample_s: float = 0.0, pilots_only: bool = False,
                 log: Callable[[str], None] = lambda msg: print(msg, flush=True)):
        self.cfg, self.root, self.h, self.only = cfg, Path(root), hooks, only
        self.max_attempts, self.pilot_attempts = max_attempts, pilot_attempts
        self.power_poll_s, self.cooldown_poll_s, self.cooldown_max_s = power_poll_s, cooldown_poll_s, cooldown_max_s
        self.pilots_only = pilots_only  # 只做各实验的 30 Task 试跑(无人值守前当面验证真实流水线)
        self.therm_sample_s = therm_sample_s  # 0 = 不开后台采样线程(测试)
        self._log = log
        self.raw = self.root / "results/raw/batch"
        self.work = self.root / "results/batch"
        self.state = {"failed_runs": {}, "blocked_stages": {}}

    def log(self, msg: str) -> None:
        self._log(f"[{datetime.now().strftime('%F %T')}] {msg}")

    # ---- 入口 ----
    def run(self) -> int:
        for d in (self.raw, self.work / "running", self.work / "pilots", self.work / "invalid"):
            d.mkdir(parents=True, exist_ok=True)
        self._start_snapshot()
        self._set_aside_leftovers()
        stages = [s for s in build_plan(self.cfg) if self.only is None or s.experiment in self.only]
        for stage in stages:
            self._run_stage(stage)
        self._write_state()
        if not self.pilots_only:
            self._finish(stages)
            self._write_summary(stages)
        return 1 if self.state["failed_runs"] or self.state["blocked_stages"] else 0

    def _start_snapshot(self) -> None:
        batt, settings, therm = self.h.batt(), self.h.settings(), self.h.therm()
        problems = gate_problems(batt, settings)
        snap = {"time": datetime.now(timezone.utc).isoformat(), "batt": batt, "settings": settings, "therm": therm,
                "gate_problems": problems}
        (self.work).mkdir(parents=True, exist_ok=True)
        (self.work / f"batch_start_{_stamp()}.json").write_text(json.dumps(snap, indent=1))
        if problems:
            self.log(f"refusing to start the batch: {'; '.join(problems)}")
            raise SystemExit(EXIT_REFUSED)
        self.log(f"batch start: AC power, low power mode off, throttled={is_throttled(therm)}")

    def _set_aside_leftovers(self) -> None:
        for d in sorted((self.work / "running").iterdir()):
            self.log(f"leftover from an interrupted run: {d.name}")
            self._move_invalid(d, ["interrupted: left in running/ when the batch started"])

    # ---- Stage ----
    def _run_stage(self, stage: Stage) -> None:
        self.log(f"== {stage.experiment}: {len(stage.runs)} runs (reusing {len(stage.reused)} from earlier stages)")
        if not self._pilot_ok(stage) or self.pilots_only:
            return
        for spec in stage.runs:
            if (self.raw / spec.name / "batch_meta.json").exists():
                continue
            if not self._attempt_until_valid(spec, self.raw, self.max_attempts, stage):
                self.state["failed_runs"][spec.name] = self.state.get("last_problems", [])
                self._write_state()
                self.log(f"FAILED after {self.max_attempts} attempts: {spec.name}")

    def _pilot_ok(self, stage: Stage) -> bool:
        pilots = self.work / "pilots" / stage.experiment
        if (pilots / stage.pilot.name / "batch_meta.json").exists():
            return True
        pilots.mkdir(parents=True, exist_ok=True)
        if self._attempt_until_valid(stage.pilot, pilots, self.pilot_attempts, stage):
            return True
        self.state["blocked_stages"][stage.experiment] = "; ".join(self.state.get("last_problems", []))
        self._write_state()
        self.log(f"pilot failed, {stage.experiment} is blocked: {self.state['blocked_stages'][stage.experiment]}")
        return False

    # ---- 单个 Run ----
    def _attempt_until_valid(self, spec: RunSpec, dest_root: Path, attempts: int, stage: Stage) -> bool:
        for attempt in range(1, attempts + 1):
            self._wait_for_ac()
            therm_start, throttled_wait = self._cool_down()
            running = self.work / "running" / spec.name
            running.mkdir(parents=True)
            problems, meta = self._execute(spec, running, therm_start)
            if spec.pilot and not problems:
                problems = self._pilot_problems(spec, running)
                meta["cache_prompt_off_effective"] = self._cache_off_effective(spec, running)
            if problems:
                self.log(f"attempt {attempt}/{attempts} invalid: {spec.name}: {'; '.join(problems)}")
                self.state["last_problems"] = problems
                self._move_invalid(running, problems)
                continue
            meta.update(valid=True, attempt=attempt, experiment=stage.experiment, spec=spec.__dict__ | {"name": spec.name},
                        throttled_wait_s=throttled_wait, problems=[])
            (running / "batch_meta.json").write_text(json.dumps(meta, indent=1, default=str))
            dest = dest_root / spec.name
            os.replace(running, dest)  # 同一文件系统上的原子改名:出现在 dest = 完成
            self.log(f"done: {spec.name} (attempt {attempt}, throttled={meta['throttled']})")
            return True
        return False

    def _execute(self, spec: RunSpec, run_dir: Path, therm_start: str) -> tuple[list[str], dict]:
        meta = {"therm_start": therm_start, "throttled": is_throttled(therm_start)}
        handle = None
        sampler = None
        problems: list[str] = []
        try:
            handle = self.h.start_server(spec, run_dir / "server.log")
            meta["cold_start_s"] = handle.cold_start_s
            if self.therm_sample_s > 0:
                sampler = _ThermSampler(self.h.therm, run_dir / "therm.jsonl", self.therm_sample_s)
                sampler.start()
            rc = self.h.run_runner(spec, run_dir, handle)
            if rc != 0:
                problems.append(f"runner exited with code {rc}")
            if not handle.alive():
                problems.append("server died during the run")
        except Exception as e:  # 启动失败、runner 异常:这次尝试作废,批跑继续
            problems.append(f"{type(e).__name__}: {e}")
        finally:
            if sampler:
                sampler.stop()
                meta["throttled"] = meta["throttled"] or sampler.throttled
            if handle:
                handle.stop()
        meta["therm_end"] = self.h.therm()
        meta["throttled"] = meta["throttled"] or is_throttled(meta["therm_end"])
        problems += [f"at end: {p}" for p in gate_problems(self.h.batt(), self.h.settings())]
        problems += validate_run(run_dir, spec.n_tasks)
        return problems, meta

    # ---- 试跑特有检查 ----
    def _pilot_problems(self, spec: RunSpec, run_dir: Path) -> list[str]:
        from src.analysis.runs import read_jsonl
        problems = self.h.smoke_analysis(run_dir)
        requests = read_jsonl(run_dir / "requests.jsonl")
        if any(r["status"] == "ok" and r.get("cached_tokens") is None for r in requests):
            problems.append("ok requests without cached_tokens (E4 needs it for every request)")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("cache_prompt") != spec.cache_prompt:
            problems.append(f"manifest cache_prompt={manifest.get('cache_prompt')}, expected {spec.cache_prompt}")
        if manifest.get("tool_delay_s") != spec.tool_delay_s:
            problems.append(f"manifest tool_delay_s={manifest.get('tool_delay_s')}, expected {spec.tool_delay_s}")
        return problems

    @staticmethod
    def _cache_off_effective(spec: RunSpec, run_dir: Path) -> bool | None:
        """cache_prompt=false 时 cached_tokens 应全为 0;不是 0 就是 E4 的前提有问题(只记录,不拦)。"""
        if spec.cache_prompt:
            return None
        from src.analysis.runs import read_jsonl
        return all((r.get("cached_tokens") or 0) == 0 for r in read_jsonl(run_dir / "requests.jsonl"))

    # ---- 等待 ----
    def _wait_for_ac(self) -> None:
        waited = False
        while problems := gate_problems(self.h.batt(), self.h.settings()):
            if not waited:
                self.log(f"waiting for AC power / normal power mode: {'; '.join(problems)}")
                waited = True
            self.h.sleep(self.power_poll_s)

    def _cool_down(self) -> tuple[str, float]:
        """降频中就等一会儿(有上限,不无限等);返回开跑前的 therm 文本和等待秒数。"""
        waited, therm = 0.0, self.h.therm()
        while is_throttled(therm) and waited < self.cooldown_max_s:
            self.h.sleep(self.cooldown_poll_s)
            waited += self.cooldown_poll_s
            therm = self.h.therm()
        return therm, waited

    # ---- 整理 ----
    def _move_invalid(self, d: Path, problems: list[str]) -> None:
        dest = self.work / "invalid" / f"{d.name}__{_stamp()}"
        n = 0
        while dest.exists():
            n += 1
            dest = self.work / "invalid" / f"{d.name}__{_stamp()}_{n}"
        shutil.move(str(d), dest)
        (dest / "problems.json").write_text(json.dumps(problems, indent=1))

    def _write_state(self) -> None:
        state = {k: v for k, v in self.state.items() if k != "last_problems"}
        (self.work / "state.json").write_text(json.dumps(state, indent=1))

    def _finish(self, stages: list[Stage]) -> None:
        if not any(s.experiment == "E1-S" for s in stages):
            return
        rows = self._e1_rows([s for s in stages if s.experiment == "E1-S"][0])
        if not rows:
            return
        result = select_baseline(rows) | {"experiment": "E1-S", "n_runs": len(rows), "rule": "see src/batch/baseline.py"}
        (self.work / "baseline_c.json").write_text(json.dumps(result, indent=1))
        self.log(f"Fixed-concurrency baseline: c={result['c']} (p99 acceptable: {result['p99_acceptable']})")

    def _write_summary(self, stages: list[Stage]) -> None:
        planned = [r for s in stages for r in s.runs]
        metas = {r.name: json.loads((self.raw / r.name / "batch_meta.json").read_text())
                 for r in planned if (self.raw / r.name / "batch_meta.json").exists()}
        summary = {"planned": len(planned), "done": len(metas), **{k: self.state[k] for k in ("failed_runs", "blocked_stages")},
                   "throttled_runs": sorted(n for n, m in metas.items() if m["throttled"]),
                   "retried_runs": {n: m["attempt"] for n, m in metas.items() if m["attempt"] > 1}}
        (self.work / "summary.json").write_text(json.dumps(summary, indent=1))

    def _e1_rows(self, stage: Stage) -> list[dict]:
        from src.analysis.metrics import run_metrics
        from src.analysis.runs import load_run
        from .baseline import p99_deadline_ratio
        rows = []
        for spec in stage.runs:
            d = self.raw / spec.name
            if not d.exists():
                continue
            run = load_run(d)
            rows.append({"c": spec.c, "seed": spec.seed, "run": spec.name,
                         "goodput_per_s": run_metrics(run.tasks)["goodput_per_s"],
                         "p99_deadline_ratio": p99_deadline_ratio(run.tasks)})
        return rows


class _ThermSampler(threading.Thread):
    """Run 期间每隔几十秒记一次 pmset -g therm 到 therm.jsonl;任何一次降频都算该 Run 降频。"""

    def __init__(self, therm: Callable[[], str], path: Path, interval_s: float):
        super().__init__(daemon=True)
        self._therm, self._path, self._interval, self._stop_evt = therm, path, interval_s, threading.Event()
        self.throttled = False

    def run(self) -> None:
        with open(self._path, "a") as f:
            while True:
                try:
                    text = self._therm()
                    hot = is_throttled(text)
                    self.throttled = self.throttled or hot
                    f.write(json.dumps({"time": time.time(), "throttled": hot, "therm": text}) + "\n")
                except Exception as e:
                    f.write(json.dumps({"time": time.time(), "error": str(e)}) + "\n")
                f.flush()
                if self._stop_evt.wait(self._interval):
                    return

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=30)
