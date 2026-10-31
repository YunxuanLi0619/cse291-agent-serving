import json

import pytest

from batch_helpers import write_run
from src.batch.checks import gate_problems, is_throttled, validate_run

BATT_AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t99%; charged; 0:00 remaining present: true\n"
BATT_BAT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging; 3:00 remaining present: true\n"
SETTINGS = "System-wide power settings:\n lowpowermode         {}\n sleep 1\n"
NO_THERM = ("Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\n"
            "Note: No CPU power status has been recorded\n")


def test_gate_accepts_ac_power_with_low_power_mode_off():
    assert gate_problems(BATT_AC, SETTINGS.format(0)) == []


def test_gate_refuses_battery_and_low_power_mode():
    assert gate_problems(BATT_BAT, SETTINGS.format(0)) == ["power source is 'Battery Power', not AC Power"]
    assert gate_problems(BATT_AC, SETTINGS.format(1)) == ["low power mode is on"]
    assert len(gate_problems(BATT_BAT, SETTINGS.format(1))) == 2


def test_gate_refuses_when_pmset_output_is_unreadable_instead_of_assuming_ok():
    assert gate_problems("garbage", SETTINGS.format(0))
    assert gate_problems(BATT_AC, "garbage")


def test_throttling_is_detected_from_cpu_limits_and_warning_levels():
    assert not is_throttled(NO_THERM)
    assert not is_throttled("CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 100\n")
    assert is_throttled("CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 10\nCPU_Speed_Limit \t= 75\n")
    assert is_throttled("CPU_Scheduler_Limit \t= 80\nCPU_Speed_Limit \t= 100\n")
    assert is_throttled("Thermal warning level: 70\nNote: No performance warning level has been recorded\n")


def test_a_complete_run_has_no_problems(tmp_path):
    write_run(tmp_path / "r")
    assert validate_run(tmp_path / "r", expected_tasks=5) == []


@pytest.mark.parametrize("kwargs, expected", [
    ({"n": 4}, "tasks.jsonl has 4 rows, expected 5"),
    ({"warmup": 2}, "2 warmup tasks, expected 3"),
    ({"sample_gap": 6.0}, "system.jsonl has a 6.0 s gap (system slept?)"),
    ({"error_rows": 10}, "10/40 system samples have errors"),
    ({"manifest": {"errors": {"pmset_therm": "x"}}}, "manifest errors: pmset_therm"),
    ({"manifest": {"cold_start_s": None}}, "manifest has no cold_start_s"),
    ({"manifest": {"power_source": "Battery Power"}}, "run started on 'Battery Power'"),
    ({"manifest": {"low_power_mode": True}}, "run started in low power mode"),
    ({"statuses": ["ok"] * 3 + ["error_other"] * 2}, "2/5 tasks ended in error_other (server died?)"),
])
def test_validation_names_each_problem(tmp_path, kwargs, expected):
    write_run(tmp_path / "r", **kwargs)
    assert expected in validate_run(tmp_path / "r", expected_tasks=5)


def test_an_unfinished_run_without_t_end_is_invalid(tmp_path):
    write_run(tmp_path / "r")
    m = json.loads((tmp_path / "r/manifest.json").read_text())
    del m["t_end"]
    (tmp_path / "r/manifest.json").write_text(json.dumps(m))
    assert "manifest has no t_end (run did not finish)" in validate_run(tmp_path / "r", expected_tasks=5)


def test_missing_files_are_reported_not_raised(tmp_path):
    (tmp_path / "r").mkdir()
    assert any("missing" in p for p in validate_run(tmp_path / "r", expected_tasks=5))
