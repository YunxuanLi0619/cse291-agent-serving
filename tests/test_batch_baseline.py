import math

from src.batch.baseline import p99_deadline_ratio, select_baseline


def task(done, deadline=10.0, status="ok", warmup=False, arrive=0.0):
    return {"status": status, "warmup": warmup, "t_arrive": arrive, "t_done": done, "deadline_s": deadline}


def test_p99_deadline_ratio_uses_steady_ok_tasks_only():
    tasks = [task(5.0), task(10.0), task(100.0, warmup=True), task(90.0, status="timeout")]
    assert math.isclose(p99_deadline_ratio(tasks), 0.995)  # linear-interpolated p99 of the ratios [0.5, 1.0]
    assert math.isnan(p99_deadline_ratio([task(1.0, status="timeout")]))


def runs(c, goodputs, p99s, fail=0.0):
    return [{"c": c, "seed": i + 1, "goodput_per_s": g, "p99_deadline_ratio": p, "fail_rate": fail}
            for i, (g, p) in enumerate(zip(goodputs, p99s))]


def test_picks_highest_mean_goodput_among_c_whose_p99_meets_the_deadline():
    table = runs(1, [0.03] * 3, [0.8] * 3) + runs(2, [0.05] * 3, [0.9] * 3) + runs(4, [0.07] * 3, [1.4] * 3)
    r = select_baseline(table, rule="strict")
    assert r["c"] == 2 and r["p99_acceptable"] is True
    assert [row["c"] for row in r["by_c"]] == [1, 2, 4]  # the full table is also handed out, for the record


def test_goodput_within_5pct_of_the_best_prefers_the_smaller_c():
    table = runs(2, [0.0500] * 3, [0.7] * 3) + runs(4, [0.0515] * 3, [0.9] * 3) + runs(8, [0.0516] * 3, [0.9] * 3)
    assert select_baseline(table, rule="strict")["c"] == 2


def test_falls_back_to_best_goodput_and_says_p99_is_not_acceptable_when_nobody_meets_it():
    table = runs(1, [0.02] * 3, [2.0] * 3) + runs(4, [0.04] * 3, [3.0] * 3)
    r = select_baseline(table, rule="strict")
    assert (r["c"], r["p99_acceptable"]) == (4, False)


def test_runs_with_nan_metrics_do_not_break_the_choice():
    table = runs(1, [0.03] * 3, [0.8] * 3) + runs(2, [0.05, 0.05, 0.05], [float("nan")] * 3)
    r = select_baseline(table, rule="strict")
    assert r["c"] == 1


def test_a_c_where_most_tasks_fail_is_not_acceptable_even_if_the_few_ok_ones_are_fast():
    # under overload only a few fast Tasks complete, so the p99 of ok Tasks looks good; the failure rate must block as well (survivor bias)
    table = runs(2, [0.04] * 3, [0.9] * 3, fail=0.02) + runs(8, [0.05] * 3, [0.3] * 3, fail=0.6)
    r = select_baseline(table, rule="strict")
    assert (r["c"], r["p99_acceptable"]) == (2, True)
    assert {row["c"]: row["acceptable"] for row in r["by_c"]} == {2: True, 8: False}


def runs_ctx(c, goodputs, fail, ctx, p99=3.0):
    return [{"c": c, "seed": i + 1, "goodput_per_s": g, "p99_deadline_ratio": p99, "fail_rate": fail, "ctx_exceeded_rate": ctx}
            for i, g in enumerate(goodputs)]


def test_relaxed_rule_ignores_p99_but_still_blocks_high_failure_rates():
    table = runs_ctx(2, [0.03] * 3, fail=0.08, ctx=0.0) + runs_ctx(4, [0.05] * 3, fail=0.30, ctx=0.0)
    r = select_baseline(table)
    assert (r["c"], r["p99_acceptable"], r["rule"]) == (2, True, "relaxed")  # p99 = 3.0 is no longer a threshold


def test_relaxed_rule_never_picks_a_c_that_rejects_tasks_even_as_the_fallback():
    # c=8 has the highest goodput, but it rejects a whole class of Tasks with error_ctx_exceeded; even when all c are overloaded we must not fall back to it
    table = runs_ctx(1, [0.005] * 3, 0.64, 0.0) + runs_ctx(4, [0.015] * 3, 0.59, 0.0) + runs_ctx(8, [0.052] * 3, 0.30, 0.25)
    r = select_baseline(table)
    assert (r["c"], r["p99_acceptable"]) == (4, False)
    assert {row["c"]: row["acceptable"] for row in r["by_c"]}[8] is False


def test_strict_rule_still_reproduces_the_original_choice():
    table = runs_ctx(4, [0.015] * 3, 0.59, 0.0) + runs_ctx(8, [0.052] * 3, 0.30, 0.25)
    assert select_baseline(table, rule="strict")["c"] == 8
