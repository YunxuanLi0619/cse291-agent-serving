import json
from pathlib import Path

from src.batch.plan import build_plan

CFG = json.loads((Path(__file__).resolve().parent.parent / "configs/batch.json").read_text())


def stages():
    return {s.experiment: s for s in build_plan(CFG)}


def test_stages_follow_priority_order_s_then_e4_e5_then_e3_then_u():
    assert [s.experiment for s in build_plan(CFG)] == ["E1-S", "E4", "E5", "E3", "E1-U"]


def test_every_config_is_repeated_three_times_with_seed_equal_to_repetition_number():
    e1 = stages()["E1-S"].runs
    assert len(e1) == 4 * 3
    for c in (1, 2, 4, 8):
        assert sorted(r.seed for r in e1 if r.c == c) == [1, 2, 3]
    assert all(r.workload == f"workloads/dev/default_s{r.seed}.json" for r in e1)


def test_series_s_sets_np_to_c_and_series_u_shares_one_pool_of_eight_slots():
    s = stages()
    assert all(r.np == r.c and r.server_flags == ("--no-kv-unified",) for r in s["E1-S"].runs)
    assert {(r.np, r.c, r.server_flags) for r in s["E1-U"].runs} == {(8, 4, ("-kvu",)), (8, 8, ("-kvu",))}
    assert all(r.c <= r.np and r.ctx == 16384 for st in s.values() for r in st.runs)  # ADR 0003


def test_e4_reuses_e1_runs_for_cache_prompt_on_and_only_adds_the_off_side():
    s = stages()
    assert {(r.c, r.cache_prompt) for r in s["E4"].runs} == {(1, False), (4, False)}
    assert len(s["E4"].runs) == 6
    assert sorted(s["E4"].reused) == sorted(r.name for r in s["E1-S"].runs if r.c in (1, 4))
    assert all(r.lam == CFG["lambda_by_factor"]["0.9"] for r in s["E4"].runs)


def test_e5_reuses_e1_c4_for_the_0_9_mu_point_and_adds_the_other_three_rates():
    s = stages()
    assert {r.lam for r in s["E5"].runs} == {0.029, 0.059, 0.118}
    assert len(s["E5"].runs) == 9 and all(r.c == r.np == 4 for r in s["E5"].runs)
    assert sorted(s["E5"].reused) == sorted(r.name for r in s["E1-S"].runs if r.c == 4)


def test_e3_varies_fixed_tool_delay_on_derived_workloads_at_c4():
    e3 = stages()["E3"].runs
    assert {r.tool_delay_s for r in e3} == {0, 0.5, 1, 3}
    assert len(e3) == 12 and all(r.c == 4 for r in e3)
    assert all(r.workload == f"workloads/dev/e3_d{r.tool_delay_s:g}_s{r.seed}.json" for r in e3)


def test_no_run_appears_in_two_stages_and_names_are_unique():
    names = [r.name for st in build_plan(CFG) for r in st.runs]
    assert len(names) == len(set(names)) == 12 + 6 + 9 + 12 + 6


def test_runs_are_ordered_repetition_by_repetition_with_alternating_config_order():
    e1 = stages()["E1-S"].runs
    assert [r.seed for r in e1] == [1] * 4 + [2] * 4 + [3] * 4  # drift (throttling) is spread across configurations instead of landing on the last one
    assert [r.c for r in e1[:4]] == [1, 2, 4, 8]
    assert [r.c for r in e1[4:8]] == [8, 4, 2, 1]


def test_each_stage_has_a_30_task_pilot_on_seed_1_that_is_not_one_of_the_full_runs():
    for st in build_plan(CFG):
        assert st.pilot.n_tasks == 30 and st.pilot.seed == 1
        assert all(r.n_tasks == 100 for r in st.runs)
        assert st.pilot.name not in {r.name for r in st.runs}
    assert stages()["E4"].pilot.cache_prompt is False  # the pilot must cover the side newly added by this experiment


def test_run_name_encodes_the_whole_configuration():
    r = stages()["E4"].runs[0]
    assert r.name == f"S_np{r.c}_c{r.c}_lam0.088_cpoff_dvar_s{r.seed}"


def test_estimated_total_time_matches_the_budget_in_the_lab_notebook():
    plan = build_plan(CFG)
    hours = sum(r.estimated_s for st in plan for r in st.runs) / 3600
    assert 17 < hours < 22  # the handbook budget "within about 20 h"; rough per-Run estimate = arrival period n/λ + about 150 s for draining and restart
    assert plan[0].runs[0].estimated_s == 100 / 0.088 + 150
