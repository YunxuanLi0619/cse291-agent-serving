"""Batch driver (#9): execute the Runs of the plan one by one, resumable and splittable over several overnight batches.

Directories (under root):
    results/raw/batch/<run>/        completed Runs that passed validation (analysis reads here); existing = done, skipped on resume
    results/batch/running/<run>/    Runs in progress; a leftover found at start = interrupted last time, moved to invalid and rerun
    results/batch/pilots/<E>/<run>/ the 30-Task pilot of each experiment
    results/batch/invalid/<run>__<time>/  attempts that failed validation (with problems.json), kept as evidence
    results/batch/state.json        failed Runs and experiments blocked by their pilots
    results/batch/batch_start_<time>.json  power, low-power mode and pmset -g therm before each batch

All side effects (pmset, server, runner, sleeping) are injected through Hooks and are all fakes in the tests.
"""
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from .baseline import select_baseline
from .checks import gate_problems, is_throttled, validate_run
from .plan import RunSpec, Stage, build_plan

EXIT_REFUSED = 3  # gate failed, refuse to start
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
    start_server: Callable[[RunSpec, Path], ServerHandle]  # (spec, server.log path); raises if the start fails
    run_runner: Callable[[RunSpec, Path, ServerHandle], int]  # returns the runner exit code
    smoke_analysis: Callable[[Path], list[str]]  # after a pilot, read it back with the analysis command and return the problems


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
        self.pilots_only = pilots_only  # only the 30-Task pilots of each experiment (verify the real pipeline in person before going unattended)
        self.therm_sample_s = therm_sample_s  # 0 = no background sampling thread (tests)
        self._log = log
        self.raw = self.root / "results/raw/batch"
        self.work = self.root / "results/batch"
        self.state = {"failed_runs": {}, "blocked_stages": {}}
        self._stages: list[Stage] = []

    def log(self, msg: str) -> None:
        self._log(f"[{datetime.now().strftime('%F %T')}] {msg}")

    def _read(self, fn: Callable[[], str], what: str) -> str:
        """pmset occasionally times out or fails: what cannot be read counts as an empty string (the gate treats it as "unreadable") so the batch does not crash."""
        try:
            return fn()
        except Exception as e:
            self.log(f"cannot read {what}: {type(e).__name__}: {e}")
            return ""

    def _batt(self) -> str:
        return self._read(self.h.batt, "pmset -g batt")

    def _settings(self) -> str:
        return self._read(self.h.settings, "pmset -g")

    def _therm(self) -> str:
        return self._read(self.h.therm, "pmset -g therm")

    # ---- entry point ----
    def run(self) -> int:
        for d in (self.raw, self.work / "running", self.work / "pilots", self.work / "invalid"):
            d.mkdir(parents=True, exist_ok=True)
        self._start_snapshot()
        self._set_aside_leftovers()
        stages = self._stages = [s for s in build_plan(self.cfg) if self.only is None or s.experiment in self.only]
        for stage in stages:
            self._run_stage(stage)
        self._write_state()
        if not self.pilots_only:
            self._finish(stages)
            self._write_summary(stages)
        return 1 if self.state["failed_runs"] or self.state["blocked_stages"] else 0

    def _start_snapshot(self) -> None:
        batt, settings, therm = self._batt(), self._settings(), self._therm()
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

    # ---- a single Run ----
    def _attempt_until_valid(self, spec: RunSpec, dest_root: Path, attempts: int, stage: Stage) -> bool:
        for attempt in range(1, attempts + 1):
            self._wait_for_ac()
            therm_start, throttled_wait = self._cool_down()
            running = self.work / "running" / spec.name
            if running.exists():
                self._move_invalid(running, ["stale directory in running/ before the attempt"])
            running.mkdir(parents=True)
            problems, meta = self._execute(spec, running, therm_start)
            if spec.pilot and not problems:
                try:
                    problems = self._pilot_problems(spec, running)
                    meta["cache_prompt_off_effective"] = self._cache_off_effective(spec, running)
                except Exception as e:
                    problems = [f"pilot checks crashed: {type(e).__name__}: {e}"]
            if problems:
                self.log(f"attempt {attempt}/{attempts} invalid: {spec.name}: {'; '.join(problems)}")
                self.state["last_problems"] = problems
                self._move_invalid(running, problems)
                continue
            meta.update(valid=True, attempt=attempt, experiment=stage.experiment, spec=spec.__dict__ | {"name": spec.name},
                        throttled_wait_s=throttled_wait, problems=[])
            (running / "batch_meta.json").write_text(json.dumps(meta, indent=1, default=str))
            dest = dest_root / spec.name
            if dest.exists():  # leftover without a completion marker (crashed around the rename last time): move it aside so it does not block the valid result just produced
                self._move_invalid(dest, ["stale destination without batch_meta.json"])
            os.replace(running, dest)  # atomic rename on the same filesystem: appearing at dest = completed
            self.log(f"done: {spec.name} (attempt {attempt}, throttled={meta['throttled']}); {self._eta()}")
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
        except Exception as e:  # start failure or runner exception: this attempt is void, the batch continues
            problems.append(f"{type(e).__name__}: {e}")
        finally:
            if sampler:
                sampler.stop()
                meta["throttled"] = meta["throttled"] or sampler.throttled
            if handle:
                handle.stop()
        meta["therm_end"] = self._therm()
        meta["throttled"] = meta["throttled"] or is_throttled(meta["therm_end"])
        problems += [f"at end: {p}" for p in gate_problems(self._batt(), self._settings())]
        try:  # after the runner is killed the manifest / JSONL may be half written: this attempt is void and an exception must not take down the whole batch
            problems += validate_run(run_dir, spec.n_tasks)
        except Exception as e:
            problems.append(f"validation crashed: {type(e).__name__}: {e}")
        return problems, meta

    # ---- pilot-specific checks ----
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
        """With cache_prompt=false, cached_tokens should all be 0; otherwise the premise of E4 is broken (recorded only, not blocking)."""
        if spec.cache_prompt:
            return None
        from src.analysis.runs import read_jsonl
        return all((r.get("cached_tokens") or 0) == 0 for r in read_jsonl(run_dir / "requests.jsonl"))

    # ---- waiting ----
    def _wait_for_ac(self) -> None:
        waited = False
        while problems := gate_problems(self._batt(), self._settings()):
            if not waited:
                self.log(f"waiting for AC power / normal power mode: {'; '.join(problems)}")
                waited = True
            self.h.sleep(self.power_poll_s)

    def _cool_down(self) -> tuple[str, float]:
        """If throttled, wait a while (bounded, never forever); return the therm text from before the start and the seconds waited."""
        waited, therm = 0.0, self._therm()
        while is_throttled(therm) and waited < self.cooldown_max_s:
            self.h.sleep(self.cooldown_poll_s)
            waited += self.cooldown_poll_s
            therm = self._therm()
        return therm, waited

    # ---- tidying up ----
    def _move_invalid(self, d: Path, problems: list[str]) -> None:
        dest = self.work / "invalid" / f"{d.name}__{_stamp()}"
        n = 0
        while dest.exists():
            n += 1
            dest = self.work / "invalid" / f"{d.name}__{_stamp()}_{n}"
        shutil.move(str(d), dest)
        (dest / "problems.json").write_text(json.dumps(problems, indent=1))

    def _eta(self) -> str:
        left = [r for st in self._stages for r in st.runs if not (self.raw / r.name / "batch_meta.json").exists()]
        return f"{len(left)} runs left, ~{sum(r.estimated_s for r in left) / 3600:.1f} h"

    def _write_state(self) -> None:
        state = {k: v for k, v in self.state.items() if k != "last_problems"}
        (self.work / "state.json").write_text(json.dumps(state, indent=1))

    def _finish(self, stages: list[Stage]) -> None:
        if not any(s.experiment == "E1-S" for s in stages):
            return
        rows = self._e1_rows([s for s in stages if s.experiment == "E1-S"][0])
        if not rows:
            return
        expected = {c: self.cfg["reps"] for c in self.cfg["e1_concurrency"]}
        have = {c: sum(r["c"] == c for r in rows) for c in expected}
        result = select_baseline(rows) | {"experiment": "E1-S", "n_runs": len(rows), "complete": have == expected,
                                          "strict_rule_result": select_baseline(rows, rule="strict")}
        if not result["complete"]:
            self.log(f"WARNING: E1-S is incomplete ({have}, expected {expected}); the baseline c is provisional")
        (self.work / "baseline_c.json").write_text(json.dumps(result, indent=1))
        self.log(f"Fixed-concurrency baseline: c={result['c']} (p99 acceptable: {result['p99_acceptable']})")

    def _write_summary(self, stages: list[Stage]) -> None:
        planned = [r for s in stages for r in s.runs]
        metas = {r.name: json.loads((self.raw / r.name / "batch_meta.json").read_text())
                 for r in planned if (self.raw / r.name / "batch_meta.json").exists()}
        summary = {"planned": len(planned), "done": len(metas), **{k: self.state[k] for k in ("failed_runs", "blocked_stages")},
                   "throttled_runs": sorted(n for n, m in metas.items() if m["throttled"]),
                   "retried_runs": {n: m["attempt"] for n, m in metas.items() if m["attempt"] > 1},
                   # E4's cache_prompt=on and E5's lambda=0.9*mu reuse Runs of E1-S; if those Runs are missing, the paired analysis has holes
                   "missing_reused": {st.experiment: missing for st in stages
                                      if (missing := [n for n in st.reused if not (self.raw / n / "batch_meta.json").exists()])}}
        (self.work / "summary.json").write_text(json.dumps(summary, indent=1))

    def _e1_rows(self, stage: Stage) -> list[dict]:
        from src.analysis.metrics import run_metrics
        from src.analysis.runs import load_run
        from .baseline import ctx_exceeded_rate, p99_deadline_ratio
        rows = []
        for spec in stage.runs:
            d = self.raw / spec.name
            if not d.exists():
                continue
            run = load_run(d)
            rows.append({"c": spec.c, "seed": spec.seed, "run": spec.name,
                         "goodput_per_s": run_metrics(run.tasks)["goodput_per_s"],
                         "p99_deadline_ratio": p99_deadline_ratio(run.tasks),
                         "fail_rate": run_metrics(run.tasks)["fail_rate"],
                         "ctx_exceeded_rate": ctx_exceeded_rate(run.tasks)})
        return rows


class _ThermSampler(threading.Thread):
    """During a Run, record `pmset -g therm` to therm.jsonl every few tens of seconds; any single throttling event counts as the Run being throttled."""

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
