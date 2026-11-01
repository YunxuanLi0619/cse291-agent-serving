import json
import shutil
from pathlib import Path

import pytest

from batch_helpers import write_run
from src.batch.driver import Batch, Hooks
from src.batch.plan import build_plan

BATT_AC = "Now drawing from 'AC Power'\n"
BATT_BAT = "Now drawing from 'Battery Power'\n"
SETTINGS = " lowpowermode         0\n"
NO_THERM = "Note: No thermal warning level has been recorded\n"
HOT = "CPU_Scheduler_Limit \t= 100\nCPU_Speed_Limit \t= 70\n"
CFG = {**json.loads((Path(__file__).resolve().parent.parent / "configs/batch.json").read_text()),
       "reps": 2, "n_tasks": 5, "pilot_n_tasks": 4, "e1_concurrency": [1, 4]}


class Handle:
    def __init__(self, fake):
        self.pid, self.cold_start_s, self.fake = 4242, 2.5, fake

    def alive(self):
        return not self.fake.server_dies

    def stop(self):
        self.fake.stopped += 1


class Fake:
    """All side effects of the batch driver: power/thermal commands, server, runner. The runner directly writes synthetic Run directories."""

    def __init__(self):
        self.batt, self.therm, self.sleeps = [BATT_AC], [NO_THERM], []
        self.started, self.ran, self.stopped = [], [], 0
        self.bad_runs: dict[str, int] = {}  # name → the first few times produce incomplete data
        self.raise_in_runner: set[str] = set()
        self.server_dies = False
        self.cached_tokens = None  # None = automatic, following cache_prompt

    @staticmethod
    def _next(seq):
        return seq.pop(0) if len(seq) > 1 else seq[0]  # the last value repeats forever

    def hooks(self, **override) -> Hooks:
        h = dict(batt=lambda: self._next(self.batt), settings=lambda: SETTINGS, therm=lambda: self._next(self.therm),
                 sleep=self.sleeps.append, start_server=self.start_server, run_runner=self.run_runner,
                 smoke_analysis=lambda d: [])
        return Hooks(**{**h, **override})

    def start_server(self, spec, log_path):
        self.started.append(spec.name)
        return Handle(self)

    def run_runner(self, spec, run_dir, handle):
        self.ran.append(spec.name)
        if spec.name in self.raise_in_runner:
            raise RuntimeError("runner exploded")
        bad = self.bad_runs.get(spec.name, 0) > 0
        if bad:
            self.bad_runs[spec.name] -= 1
        write_run(run_dir, n=spec.n_tasks - 1 if bad else spec.n_tasks, spec=spec,
                  cached=self.cached_tokens if self.cached_tokens is not None else (40 if spec.cache_prompt else 0),
                  manifest={"cold_start_s": handle.cold_start_s})
        return 0


def make(tmp_path, fake, **kw):
    return Batch(CFG, tmp_path, kw.pop("hooks", None) or fake.hooks(), log=lambda *_: None, **kw)


def plan_names(only=None):
    return [r.name for st in build_plan(CFG) if only is None or st.experiment in only for r in st.runs]


def done(tmp_path):
    return sorted(p.name for p in (tmp_path / "results/raw/batch").iterdir())


def state(tmp_path):
    return json.loads((tmp_path / "results/batch/state.json").read_text())


def test_refuses_to_start_a_batch_on_battery_and_runs_nothing(tmp_path):
    fake = Fake()
    fake.batt = [BATT_BAT]
    with pytest.raises(SystemExit) as e:
        make(tmp_path, fake).run()
    assert e.value.code == 3
    assert fake.started == [] and fake.ran == []


def test_runs_every_stage_with_its_pilot_first_and_records_the_batch_start_environment(tmp_path):
    fake = Fake()
    assert make(tmp_path, fake).run() == 0
    assert done(tmp_path) == sorted(plan_names())
    assert fake.stopped == len(fake.started)  # started before each Run and closed after it, no server left behind
    for st in build_plan(CFG):  # this Stage's pilot comes before any of its full Runs
        pilot_at = fake.ran.index(st.pilot.name)
        assert all(fake.ran.index(r.name) > pilot_at for r in st.runs)
        assert (tmp_path / f"results/batch/pilots/{st.experiment}/{st.pilot.name}/batch_meta.json").exists()
    meta = json.loads((tmp_path / "results/raw/batch" / plan_names()[0] / "batch_meta.json").read_text())
    assert meta["valid"] is True and meta["attempt"] == 1 and meta["cold_start_s"] == 2.5
    start = json.loads(next((tmp_path / "results/batch").glob("batch_start_*.json")).read_text())
    assert start["gate_problems"] == [] and "AC Power" in start["batt"] and start["therm"] == NO_THERM
    assert "lowpowermode" in start["settings"]


def test_resume_skips_completed_runs_and_reruns_only_the_missing_one(tmp_path):
    make(tmp_path, Fake()).run()
    again = Fake()
    assert make(tmp_path, again).run() == 0
    assert again.started == [] and again.ran == []  # pilots are not redone either
    victim = plan_names()[5]
    shutil.rmtree(tmp_path / "results/raw/batch" / victim)
    third = Fake()
    make(tmp_path, third).run()
    assert third.ran == [victim]


def test_an_interrupted_run_left_in_running_is_set_aside_and_redone(tmp_path):
    name = plan_names()[0]
    leftover = tmp_path / "results/batch/running" / name
    leftover.mkdir(parents=True)
    (leftover / "tasks.jsonl").write_text("{}\n")
    make(tmp_path, Fake(), only=["E1-S"]).run()
    assert name in done(tmp_path)
    assert not leftover.exists()
    assert any(p.name.startswith(name) for p in (tmp_path / "results/batch/invalid").iterdir())


def test_a_run_with_bad_output_is_moved_aside_and_retried(tmp_path):
    fake = Fake()
    name = plan_names(only=["E1-S"])[2]
    fake.bad_runs[name] = 1
    assert make(tmp_path, fake, only=["E1-S"]).run() == 0
    assert fake.ran.count(name) == 2
    assert json.loads((tmp_path / "results/raw/batch" / name / "batch_meta.json").read_text())["attempt"] == 2
    invalid = [p for p in (tmp_path / "results/batch/invalid").iterdir() if p.name.startswith(name)]
    assert len(invalid) == 1
    assert "tasks.jsonl has 4 rows, expected 5" in (invalid[0] / "problems.json").read_text()


def test_a_run_that_keeps_failing_is_recorded_as_failed_and_the_batch_moves_on(tmp_path):
    fake = Fake()
    name = plan_names(only=["E1-S"])[2]
    fake.bad_runs[name] = 99
    assert make(tmp_path, fake, only=["E1-S"], max_attempts=2).run() == 1  # a failed Run → non-zero exit
    assert fake.ran.count(name) == 2
    assert set(done(tmp_path)) == set(plan_names(only=["E1-S"])) - {name}
    assert name in state(tmp_path)["failed_runs"]


def test_an_exception_in_the_runner_counts_as_a_failed_attempt_and_the_server_is_still_stopped(tmp_path):
    fake = Fake()
    name = plan_names(only=["E1-S"])[0]
    fake.raise_in_runner.add(name)
    assert make(tmp_path, fake, only=["E1-S"], max_attempts=2).run() == 1
    assert fake.stopped == len(fake.started)
    assert set(done(tmp_path)) == set(plan_names(only=["E1-S"])) - {name}


def test_a_server_that_died_during_the_run_invalidates_it(tmp_path):
    fake = Fake()
    fake.server_dies = True
    assert make(tmp_path, fake, only=["E1-S"], pilot_attempts=1).run() == 1
    assert done(tmp_path) == []
    assert "server died" in state(tmp_path)["blocked_stages"]["E1-S"]


def test_a_stage_whose_pilot_fails_is_blocked_and_its_full_runs_never_start(tmp_path):
    fake = Fake()
    e4 = build_plan(CFG)[1]
    fake.bad_runs[e4.pilot.name] = 99
    assert make(tmp_path, fake, pilot_attempts=2).run() == 1
    assert not ({r.name for r in e4.runs} & set(fake.ran))
    assert fake.ran.count(e4.pilot.name) == 2
    assert list(state(tmp_path)["blocked_stages"]) == ["E4"]
    assert set(plan_names(only=["E3"])) <= set(done(tmp_path))  # the other Stages finish as usual


def test_pilot_must_log_cached_tokens_for_the_cache_experiment(tmp_path):
    fake = Fake()
    orig = fake.run_runner

    def without_cached_tokens(spec, run_dir, handle):
        rc = orig(spec, run_dir, handle)
        rows = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines()]
        (run_dir / "requests.jsonl").write_text("".join(json.dumps({**r, "cached_tokens": None}) + "\n" for r in rows))
        return rc

    fake.run_runner = without_cached_tokens
    assert make(tmp_path, fake, only=["E4"], pilot_attempts=1).run() == 1
    assert "cached_tokens" in state(tmp_path)["blocked_stages"]["E4"]


def test_pilot_records_whether_cache_prompt_off_really_stopped_prefix_reuse(tmp_path):
    fake = Fake()
    fake.cached_tokens = 30  # there is still reuse with cache_prompt off: E4's premise is flawed, but that is a finding, not a batch failure
    make(tmp_path, fake, only=["E4"]).run()
    pilot = build_plan(CFG)[1].pilot
    meta = json.loads((tmp_path / f"results/batch/pilots/E4/{pilot.name}/batch_meta.json").read_text())
    assert meta["cache_prompt_off_effective"] is False


def test_waits_for_ac_power_between_runs_instead_of_running_on_battery(tmp_path):
    fake = Fake()
    fake.batt = [BATT_AC, BATT_BAT, BATT_BAT, BATT_BAT, BATT_AC]  # plugged in when the batch starts, unplugged before the first Run, plugged back in a few minutes later
    make(tmp_path, fake, only=["E1-S"], power_poll_s=60).run()
    assert fake.sleeps.count(60) >= 2
    assert set(done(tmp_path)) == set(plan_names(only=["E1-S"]))


def test_a_run_that_ends_on_battery_is_invalid(tmp_path):
    fake = Fake()
    reads = {"n": 0}

    def batt():  # readings: batch start 1, gate 2, Run end 3…; so the 3rd reading hitting battery = at the end of the first Run
        reads["n"] += 1
        return BATT_BAT if reads["n"] == 3 else BATT_AC

    make(tmp_path, fake, only=["E1-S"], hooks=fake.hooks(batt=batt), pilot_attempts=2).run()
    problems = [p.read_text() for p in (tmp_path / "results/batch/invalid").glob("*/problems.json")]
    assert any("at end" in p and "Battery Power" in p for p in problems)


def test_throttling_before_a_run_is_waited_out_up_to_a_cap_and_flagged_in_the_run_meta(tmp_path):
    fake = Fake()
    fake.therm = [HOT]  # throttled all the time
    make(tmp_path, fake, only=["E1-S"], cooldown_max_s=120, cooldown_poll_s=30).run()
    assert fake.sleeps.count(30) >= 4  # waits up to the limit before starting, and does not wait forever
    meta = json.loads((tmp_path / "results/raw/batch" / plan_names(only=["E1-S"])[0] / "batch_meta.json").read_text())
    assert meta["throttled"] is True and meta["therm_start"] == HOT


def test_finishing_writes_the_baseline_c_chosen_from_the_e1_series_s_runs(tmp_path):
    assert make(tmp_path, Fake(), only=["E1-S"]).run() == 0
    baseline = json.loads((tmp_path / "results/batch/baseline_c.json").read_text())
    assert baseline["c"] in (1, 4) and [row["c"] for row in baseline["by_c"]] == [1, 4]
    assert baseline["experiment"] == "E1-S"


def test_summary_lists_done_failed_blocked_and_throttled_runs(tmp_path):
    fake = Fake()
    fake.therm = [HOT]
    bad = plan_names(only=["E1-S"])[1]
    fake.bad_runs[bad] = 99
    make(tmp_path, fake, only=["E1-S"], max_attempts=1, cooldown_max_s=0).run()
    summary = json.loads((tmp_path / "results/batch/summary.json").read_text())
    planned = plan_names(only=["E1-S"])
    assert summary["planned"] == len(planned) and summary["done"] == len(planned) - 1
    assert list(summary["failed_runs"]) == [bad] and summary["blocked_stages"] == {}
    assert sorted(summary["throttled_runs"]) == sorted(set(planned) - {bad})


def test_pilots_only_runs_one_pilot_per_stage_and_no_full_run(tmp_path):
    fake = Fake()
    assert make(tmp_path, fake, pilots_only=True).run() == 0
    assert fake.ran == [st.pilot.name for st in build_plan(CFG)]
    assert not any((tmp_path / "results/raw/batch").iterdir())
    make(tmp_path, fake).run()  # later official batches do not redo pilots that already passed
    assert [st.pilot.name for st in build_plan(CFG)] == fake.ran[:5] and fake.ran.count(build_plan(CFG)[0].pilot.name) == 1


def test_a_truncated_manifest_after_a_killed_runner_is_an_invalid_attempt_not_a_crash(tmp_path):
    fake = Fake()
    name = plan_names(only=["E1-S"])[0]
    orig = fake.run_runner

    def truncates(spec, run_dir, handle):
        rc = orig(spec, run_dir, handle)
        if spec.name == name:
            (run_dir / "manifest.json").write_text('{"c": 1, "lam')
        return rc

    fake.run_runner = truncates
    assert make(tmp_path, fake, only=["E1-S"], max_attempts=2).run() == 1
    assert name in state(tmp_path)["failed_runs"]
    assert set(done(tmp_path)) == set(plan_names(only=["E1-S"])) - {name}


def test_failing_pmset_commands_are_treated_as_unreadable_not_as_a_crash(tmp_path):
    fake = Fake()
    calls = {"n": 0}

    def flaky_therm():
        calls["n"] += 1
        if calls["n"] % 2:
            raise RuntimeError("pmset timed out")
        return NO_THERM

    make(tmp_path, fake, only=["E1-S"], hooks=fake.hooks(therm=flaky_therm)).run()
    assert set(done(tmp_path)) == set(plan_names(only=["E1-S"]))


def test_a_leftover_destination_without_completion_marker_does_not_block_the_finished_run(tmp_path):
    name = plan_names(only=["E1-S"])[0]
    stale = tmp_path / "results/raw/batch" / name
    stale.mkdir(parents=True)
    (stale / "tasks.jsonl").write_text("{}\n")
    assert make(tmp_path, Fake(), only=["E1-S"]).run() == 0
    assert (stale / "batch_meta.json").exists()
    assert any(p.name.startswith(name) for p in (tmp_path / "results/batch/invalid").iterdir())


def test_summary_flags_reused_runs_that_are_missing_and_baseline_flags_incomplete_data(tmp_path):
    fake = Fake()
    victim = [r.name for r in build_plan(CFG)[0].runs if r.c == 4][0]  # E1-S c=4: reused by both E4 and E5
    fake.bad_runs[victim] = 99
    make(tmp_path, fake, max_attempts=1).run()
    summary = json.loads((tmp_path / "results/batch/summary.json").read_text())
    assert summary["missing_reused"]["E4"] == [victim] and summary["missing_reused"]["E5"] == [victim]
    assert json.loads((tmp_path / "results/batch/baseline_c.json").read_text())["complete"] is False
