"""分析命令的测试:手写 JSONL 夹具进、CSV/图出;期望值都是手算的。"""
import csv
import json

import pytest

from src.analysis.report import main

CMDLINE = "/bin/llama-server -m m.gguf --host 127.0.0.1 --port 8080 -np {np} -c {b} -t 8 --cache-ram 0 --metrics {kv}"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def task(task_id, *, t_arrive, t_done, status="ok", quality_pass=True, deadline_s=10.0, queue=0.0,
         type="short", warmup=False, n_steps=1):
    return {"task_id": task_id, "type": type, "t_arrive": t_arrive, "t_first_dispatch": t_arrive + queue,
            "t_done": t_done, "queue_time_total": queue, "idle_time_total": 0.0, "n_steps": n_steps,
            "status": status, "quality_pass": quality_pass and status == "ok", "deadline_s": deadline_s,
            "warmup": warmup}


def make_run(root, name, tasks, *, c=2, lam=0.5, seed=1, np_=None, b=16384, kv="--no-kv-unified",
             platform="macos", requests=(), system=(), complete=True, **manifest_extra):
    d = root / name
    d.mkdir(parents=True)
    manifest = {"seed": seed, "c": c, "lambda": lam, "memory_platform": platform,
                "server_cmdline": CMDLINE.format(np=np_ or c, b=b, kv=kv), "t_start": 0.0, **manifest_extra}
    if complete:
        manifest["t_end"] = 100.0
    (d / "manifest.json").write_text(json.dumps(manifest))
    write_jsonl(d / "tasks.jsonl", tasks)
    write_jsonl(d / "requests.jsonl", list(requests))
    write_jsonl(d / "system.jsonl", list(system))
    return d


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# 手算夹具:前 3 个 warmup(故意很慢/失败,若被计入会改变结果),后 7 个稳态。
#   稳态窗口 = 最早到达 10 ~ 最晚完成 40 = 30 s;N = 7
#   ok 的完成时间 [4,6,8,12,10] → 排序 [4,6,8,10,12]:p50 = 8,p99 = 10 + 0.96 × 2 = 11.92
#   吞吐 = 5 ok / 30;失败率 = 2/7(timeout + ctx_exceeded);timeout 率 = 1/7
#   goodput:deadline 10 内且 quality pass 的只有 3 个(完成 12 超期;T8 没 pass)→ 3/30 = 0.1
#   排队时间(含失败 Task)[.5,1,2,3,1.5,20,0]:均值 28/7 = 4.0;p99 = 3 + 0.94 × 17 = 18.98
FIXTURE = [
    task("w1", t_arrive=0, t_done=100, queue=50.0, warmup=True),
    task("w2", t_arrive=1, t_done=90, status="timeout", warmup=True),
    task("w3", t_arrive=2, t_done=80, warmup=True),
    task("t4", t_arrive=10, t_done=14, queue=0.5),
    task("t5", t_arrive=11, t_done=17, queue=1.0),
    task("t6", t_arrive=12, t_done=20, queue=2.0),
    task("t7", t_arrive=13, t_done=25, queue=3.0),
    task("t8", t_arrive=14, t_done=24, queue=1.5, quality_pass=False),
    task("t9", t_arrive=15, t_done=40, status="timeout", queue=20.0),
    task("t10", t_arrive=16, t_done=18, status="error_ctx_exceeded"),
]


@pytest.fixture
def raw(tmp_path):
    return tmp_path / "raw"


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def test_run_metrics_match_hand_computed_values_and_exclude_warmup(raw, out):
    make_run(raw, "r1", FIXTURE)
    main(["--raw", str(raw), "--out", str(out)])
    (row,) = read_csv(out / "runs.csv")
    assert row["n_tasks"] == "7"
    assert float(row["p50_s"]) == pytest.approx(8.0)
    assert float(row["p99_s"]) == pytest.approx(11.92)
    assert float(row["throughput_per_s"]) == pytest.approx(5 / 30)
    assert float(row["fail_rate"]) == pytest.approx(2 / 7)
    assert float(row["timeout_rate"]) == pytest.approx(1 / 7)
    assert float(row["goodput_per_s"]) == pytest.approx(0.1)
    assert float(row["queue_time_mean_s"]) == pytest.approx(4.0)
    assert float(row["queue_time_p99_s"]) == pytest.approx(18.98)


def simple_run(raw, name, *, c=1, seed=1, x=4.0, **kw):
    """3 个 warmup + 2 个稳态 ok Task(完成时间都是 x):p50 = x,窗口 = x + 1,吞吐 = 2 / (x + 1)。"""
    tasks = [task(f"w{i}", t_arrive=i, t_done=i + 99, warmup=True) for i in range(3)]
    tasks += [task("a", t_arrive=10, t_done=10 + x), task("b", t_arrive=11, t_done=11 + x)]
    return make_run(raw, name, tasks, c=c, seed=seed, **kw)


def test_unfinished_runs_and_non_run_directories_are_skipped(raw, out):
    make_run(raw, "done", FIXTURE)
    make_run(raw, "crashed", FIXTURE, complete=False)
    (raw / "step5").mkdir()
    (raw / "step5" / "summary.csv").write_text("x\n")
    main(["--raw", str(raw), "--out", str(out)])
    assert [r["run"] for r in read_csv(out / "runs.csv")] == ["done"]


def test_run_without_steady_tasks_does_not_crash(raw, out):
    make_run(raw, "r", [task("w", t_arrive=0, t_done=1, warmup=True)])
    main(["--raw", str(raw), "--out", str(out)])
    (row,) = read_csv(out / "runs.csv")
    assert row["n_tasks"] == "0" and row["p50_s"] == ""


def test_repeats_are_aggregated_over_seeds_per_config_with_mean_and_std(raw, out):
    for seed, x in [(1, 4.0), (2, 6.0), (3, 8.0)]:
        simple_run(raw, f"c1_s{seed}", c=1, seed=seed, x=x)
        simple_run(raw, f"c4_s{seed}", c=4, seed=seed, x=2.0)
    main(["--raw", str(raw), "--out", str(out)])
    rows = {r["c"]: r for r in read_csv(out / "perf_by_c_macos.csv")}
    assert set(rows) == {"1", "4"}
    assert rows["1"]["n_seeds"] == "3"
    assert float(rows["1"]["p50_s_mean"]) == pytest.approx(6.0)
    assert float(rows["1"]["p50_s_std"]) == pytest.approx(2.0)  # 样本标准差 (4,6,8)
    assert float(rows["1"]["throughput_per_s_mean"]) == pytest.approx((2 / 5 + 2 / 7 + 2 / 9) / 3)
    assert float(rows["4"]["p50_s_std"]) == pytest.approx(0.0)
    assert (out / "perf_by_c_macos.png").stat().st_size > 0


def test_a_swept_factor_gets_a_figure_only_when_it_has_two_or_more_values(raw, out):
    simple_run(raw, "a", c=1, lam=0.5)
    simple_run(raw, "b", c=2, lam=0.5)
    main(["--raw", str(raw), "--out", str(out)])
    assert (out / "perf_by_c_macos.csv").exists()
    assert not (out / "perf_by_lambda_macos.csv").exists()
    assert not (out / "perf_by_tool_delay_s_macos.csv").exists()


def test_other_factors_split_into_separate_lines_not_mixed(raw, out):
    for c in (1, 2):
        for lam in (0.5, 1.0):
            simple_run(raw, f"c{c}_l{lam}", c=c, lam=lam, x=4.0 * lam)
    main(["--raw", str(raw), "--out", str(out)])
    rows = read_csv(out / "perf_by_c_macos.csv")
    assert len(rows) == 4  # (c, λ) 各一行,不会被平均到一起
    assert len({r["line"] for r in rows}) == 2
    assert (out / "perf_by_lambda_macos.csv").exists()


def test_data_from_different_platforms_is_never_in_the_same_csv_or_figure(raw, out):
    for platform, x in [("macos", 4.0), ("linux", 9.0)]:
        simple_run(raw, f"{platform}1", c=1, x=x, platform=platform)
        simple_run(raw, f"{platform}2", c=2, x=x, platform=platform)
    main(["--raw", str(raw), "--out", str(out)])
    for platform, x in [("macos", 4.0), ("linux", 9.0)]:
        rows = read_csv(out / f"perf_by_c_{platform}.csv")
        assert {r["platform"] for r in rows} == {platform}
        assert all(float(r["p50_s_mean"]) == x for r in rows)
        assert (out / f"perf_by_c_{platform}.png").exists()


def test_cache_prompt_and_tool_delay_split_lines_and_cache_prompt_is_compared_per_seed(raw, out):
    off, on = {1: 6.0, 2: 10.0}, {1: 4.0, 2: 5.0, 3: 99.0}  # seed 3 只有 on 一侧,不能配对
    for seed, x in off.items():
        simple_run(raw, f"off{seed}", c=1, seed=seed, x=x, cache_prompt=False, tool_delay_s=1.0)
        simple_run(raw, f"off4_{seed}", c=4, seed=seed, x=x, cache_prompt=False, tool_delay_s=1.0)
    for seed, x in on.items():
        simple_run(raw, f"on{seed}", c=1, seed=seed, x=x, cache_prompt=True, tool_delay_s=1.0)
    simple_run(raw, "delay3", c=1, seed=1, x=4.0, cache_prompt=True, tool_delay_s=3.0)
    main(["--raw", str(raw), "--out", str(out)])

    assert {r["cache_prompt"] for r in read_csv(out / "perf_by_c_macos.csv")} == {"True", "False"}
    assert {r["tool_delay_s"] for r in read_csv(out / "perf_by_tool_delay_s_macos.csv")} == {"1.0", "3.0"}

    # 同 seed 内 on − off:seed1 4−6 = −2,seed2 5−10 = −5 → 均值 −3.5,样本标准差 √4.5
    (row,) = [r for r in read_csv(out / "paired_cache_prompt_macos.csv")
              if r["c"] == "1" and r["tool_delay_s"] == "1.0"]
    assert row["n_pairs"] == "2"
    assert float(row["p50_s_delta_mean"]) == pytest.approx(-3.5)
    assert float(row["p50_s_delta_std"]) == pytest.approx(4.5 ** 0.5)
    assert (out / "paired_cache_prompt_macos.png").stat().st_size > 0


def test_runs_with_and_without_a_tool_delay_in_one_batch_still_get_a_figure(raw, out):
    # 批跑里只有 E3 的 Run 在 manifest 里带 tool_delay_s,其余没有:缺失值也要能当图例名
    for c in (1, 4):
        simple_run(raw, f"plain{c}", c=c)
        simple_run(raw, f"delay{c}", c=c, tool_delay_s=3.0)
    main(["--raw", str(raw), "--out", str(out)])
    assert (out / "perf_by_c_macos.png").stat().st_size > 0
    assert {r["line"] for r in read_csv(out / "perf_by_c_macos.csv")} == {"nan", "3.0"}


def req(task_id, step, *, submit, start, first, end, prompt, cached, status="ok", warmup=False):
    return {"task_id": task_id, "step": step, "t_submit": submit, "t_start": start, "t_first_token": first,
            "t_end": end, "prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": 10, "status": status,
            "error_type": None if status == "ok" else status, "warmup": warmup, "id_slot": None}


def test_agent_requests_are_sliced_by_step_index_k(raw, out):
    tasks = [task("a0", type="agent", t_arrive=0, t_done=5, warmup=True, n_steps=2),
             task("a1", type="agent", t_arrive=10, t_done=15, n_steps=2),
             task("a2", type="agent", t_arrive=20, t_done=30, status="timeout", n_steps=2),
             task("s1", type="short", t_arrive=12, t_done=40)]
    requests = [
        req("a0", 1, submit=0, start=0, first=90, end=99, prompt=9999, cached=0, warmup=True),  # warmup 不计
        req("s1", 1, submit=12, start=12, first=50, end=60, prompt=9999, cached=0),  # 非 agent 不计
        req("a1", 1, submit=10, start=10.5, first=11.0, end=12, prompt=1000, cached=0),
        req("a1", 2, submit=14, start=14.0, first=14.2, end=15, prompt=1100, cached=1000),
        req("a2", 1, submit=20, start=21, first=21.5, end=23, prompt=1000, cached=0),
        req("a2", 2, submit=26, start=27, first=None, end=30, prompt=None, cached=None, status="timeout"),
    ]
    make_run(raw, "r", tasks, requests=requests)
    main(["--raw", str(raw), "--out", str(out)])
    rows = {r["k"]: r for r in read_csv(out / "step_slice_macos.csv")}
    assert set(rows) == {"1", "2"}
    # k=1:两个 ok Request;TTFT (0.5, 0.5),延迟 (1.5, 2.0),排队 (0.5, 1.0),无前缀复用
    assert rows["1"]["n_requests_mean"] == "2.0"
    assert float(rows["1"]["ttft_s_mean"]) == pytest.approx(0.5)
    assert float(rows["1"]["latency_s_mean"]) == pytest.approx(1.75)
    assert float(rows["1"]["queue_wait_s_mean"]) == pytest.approx(0.75)
    assert float(rows["1"]["prompt_tokens_mean"]) == pytest.approx(1000)
    assert float(rows["1"]["reuse_ratio_mean"]) == pytest.approx(0.0)
    assert float(rows["1"]["fail_share_mean"]) == pytest.approx(0.0)
    # k=2:一个 ok、一个 timeout:失败占 1/2;时延类指标只看 ok;复用 = 1000/1100
    assert float(rows["2"]["fail_share_mean"]) == pytest.approx(0.5)
    assert float(rows["2"]["ttft_s_mean"]) == pytest.approx(0.2)
    assert float(rows["2"]["reuse_ratio_mean"]) == pytest.approx(1000 / 1100)
    assert (out / "step_slice_macos.png").stat().st_size > 0


def sys_row(t, footprint, rss, live):
    return {"t": t, "platform": "macos", "footprint_mb": footprint, "rss_mb": rss, "requests_processing": 1,
            "requests_deferred": 0, "live_tokens": live, "slots": [], "error": None if footprint else "memory: x"}


def test_peak_footprint_and_rss_per_server_config_skip_failed_samples(raw, out):
    def rows(peak_fp, peak_rss):
        return [sys_row(0.5, 3700, 200, 0), sys_row(1.0, peak_fp, 100, 0), sys_row(1.5, None, None, 0),
                sys_row(2.0, 3600, peak_rss, 0)]
    for seed, fp, rss in [(1, 3800, 1000), (2, 3900, 3000)]:
        simple_run(raw, f"c4_s{seed}", c=4, seed=seed, system=rows(fp, rss))
    simple_run(raw, "c1", c=1, system=rows(3750, 500))
    simple_run(raw, "linux", c=4, platform="linux", system=[sys_row(1.0, 2000, 1900, 0)])
    main(["--raw", str(raw), "--out", str(out)])
    got = {(r["c"], r["np"]): r for r in read_csv(out / "memory_peak_macos.csv")}
    assert float(got[("4", "4")]["footprint_mb_peak_mean"]) == pytest.approx(3850)
    assert float(got[("4", "4")]["footprint_mb_peak_std"]) == pytest.approx(100 / 2 ** 0.5)
    assert float(got[("4", "4")]["rss_mb_peak_mean"]) == pytest.approx(2000)
    assert got[("4", "4")]["n_runs"] == "2" and got[("1", "1")]["B"] == "16384"
    assert [r["footprint_mb_peak_mean"] for r in read_csv(out / "memory_peak_linux.csv")] == ["2000.0"]
    assert (out / "memory_peak_macos.png").stat().st_size > 0


def test_kv_utilization_curve_is_live_tokens_over_B_for_the_first_seed_of_each_config(raw, out):
    live = [(0.5, 0), (1.0, 4096), (1.5, None), (2.0, 8192)]
    for seed in (1, 2):
        simple_run(raw, f"c4_s{seed}", c=4, seed=seed, b=16384, system=[sys_row(t, 3800, 1, v) for t, v in live])
    simple_run(raw, "c1", c=1, b=8192, system=[sys_row(1.0, 3800, 1, 4096)])
    main(["--raw", str(raw), "--out", str(out)])
    rows = read_csv(out / "kv_util_macos.csv")
    c4 = [r for r in rows if r["run"].startswith("c4")]
    assert {r["run"] for r in c4} == {"c4_s1"}
    assert [(float(r["t"]), float(r["kv_util"])) for r in c4] == [(0.5, 0.0), (1.0, 0.25), (2.0, 0.5)]  # live_tokens 缺失的行跳过
    assert [float(r["kv_util"]) for r in rows if r["run"] == "c1"] == [0.5]  # B 取自各 Run 自己的 -c
    assert (out / "kv_util_macos.png").stat().st_size > 0


def test_runs_without_any_memory_samples_produce_no_memory_outputs_and_do_not_crash(raw, out):
    simple_run(raw, "r", c=1, system=[sys_row(0.5, None, None, None)])
    main(["--raw", str(raw), "--out", str(out)])
    assert not list(out.glob("memory_peak_*")) and not list(out.glob("kv_util_*"))


def test_same_config_and_seed_in_two_runs_is_warned_about(raw, out, capsys):
    simple_run(raw, "a", c=1, seed=1)
    simple_run(raw, "b", c=1, seed=1)
    main(["--raw", str(raw), "--out", str(out)])
    assert "same config and seed" in capsys.readouterr().err
