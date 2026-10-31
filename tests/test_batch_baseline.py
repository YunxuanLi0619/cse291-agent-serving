import math

from src.batch.baseline import p99_deadline_ratio, select_baseline


def task(done, deadline=10.0, status="ok", warmup=False, arrive=0.0):
    return {"status": status, "warmup": warmup, "t_arrive": arrive, "t_done": done, "deadline_s": deadline}


def test_p99_deadline_ratio_uses_steady_ok_tasks_only():
    tasks = [task(5.0), task(10.0), task(100.0, warmup=True), task(90.0, status="timeout")]
    assert math.isclose(p99_deadline_ratio(tasks), 0.995)  # 比值 [0.5, 1.0] 的线性插值 p99
    assert math.isnan(p99_deadline_ratio([task(1.0, status="timeout")]))


def runs(c, goodputs, p99s):
    return [{"c": c, "seed": i + 1, "goodput_per_s": g, "p99_deadline_ratio": p}
            for i, (g, p) in enumerate(zip(goodputs, p99s))]


def test_picks_highest_mean_goodput_among_c_whose_p99_meets_the_deadline():
    table = runs(1, [0.03] * 3, [0.8] * 3) + runs(2, [0.05] * 3, [0.9] * 3) + runs(4, [0.07] * 3, [1.4] * 3)
    r = select_baseline(table)
    assert r["c"] == 2 and r["p99_acceptable"] is True
    assert [row["c"] for row in r["by_c"]] == [1, 2, 4]  # 完整的表也交出去,供记录


def test_goodput_within_5pct_of_the_best_prefers_the_smaller_c():
    table = runs(2, [0.0500] * 3, [0.7] * 3) + runs(4, [0.0515] * 3, [0.9] * 3) + runs(8, [0.0516] * 3, [0.9] * 3)
    assert select_baseline(table)["c"] == 2


def test_falls_back_to_best_goodput_and_says_p99_is_not_acceptable_when_nobody_meets_it():
    table = runs(1, [0.02] * 3, [2.0] * 3) + runs(4, [0.04] * 3, [3.0] * 3)
    r = select_baseline(table)
    assert (r["c"], r["p99_acceptable"]) == (4, False)


def test_runs_with_nan_metrics_do_not_break_the_choice():
    table = runs(1, [0.03] * 3, [0.8] * 3) + runs(2, [0.05, 0.05, 0.05], [float("nan")] * 3)
    r = select_baseline(table)
    assert r["c"] == 1
