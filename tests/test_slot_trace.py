from src.analysis.slot_trace import join_requests, parse_server_log

LOG = """\
0.34.721.402 I slot get_availabl: id  0 | task -1 | selected slot by LRU, n_tokens = 0, t_last = -1
0.34.721.419 I slot launch_slot_: id  0 | task 81 | processing task, is_child = 0
0.55.900.960 I slot      release: id  0 | task 81 | stop processing: n_tokens = 928, truncated = 0
0.55.926.559 I slot get_availabl: id  1 | task -1 | selected slot by LCP similarity, f_sim_best = 0.121 (> 0.100 thold), f_keep = 0.040
0.55.926.597 I slot launch_slot_: id  1 | task 680 | processing task, is_child = 0
0.59.773.256 I slot get_availabl: id  2 | task -1 | selected slot by LRU, n_tokens = 928, t_last = 636404235672
0.59.773.300 I slot launch_slot_: id  2 | task 762 | processing task, is_child = 0
"""


def test_parse_server_log_pairs_each_selection_with_its_launch():
    launches = parse_server_log(LOG.splitlines())
    assert [(x["slot"], x["server_task"], x["how"], x["f_sim"]) for x in launches] == [
        (0, 81, "LRU", None), (1, 680, "LCP", 0.121), (2, 762, "LRU", None)]


def test_join_requests_matches_by_start_order_and_flags_agent_slot_change():
    launches = parse_server_log(LOG.splitlines())
    requests = [  # 故意乱序:按 t_start 对应
        {"task_id": "t2", "step": 1, "t_start": 9.0, "cached_tokens": 0},
        {"task_id": "t1", "step": 1, "t_start": 1.0, "cached_tokens": 0, "status": "ok",
         "prompt_tokens": 800, "output_tokens": 128},
        {"task_id": "t1", "step": 2, "t_start": 5.0, "cached_tokens": 30},
    ]
    rows = join_requests(requests, launches)
    by_key = {(r["task_id"], r["step"]): r for r in rows}
    assert by_key[("t1", 1)]["slot"] == 0 and by_key[("t1", 2)]["slot"] == 1
    assert by_key[("t1", 2)]["same_slot_as_prev_step"] is False
    assert by_key[("t1", 1)]["same_slot_as_prev_step"] is None


def test_join_requests_refuses_when_counts_differ():
    launches = parse_server_log(LOG.splitlines())
    try:
        join_requests([{"task_id": "t", "step": 1, "t_start": 0.0, "cached_tokens": 0}], launches)
    except ValueError as e:
        assert "1 requests vs 3 slot launches" in str(e) and "--skip" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_join_requests_rejects_pairing_whose_token_counts_disagree_with_server_release():
    launches = parse_server_log(LOG.splitlines())  # 第 1 个 launch 释放时 n_tokens = 928
    requests = [
        {"task_id": "a", "step": 1, "t_start": 1.0, "cached_tokens": 0, "status": "ok",
         "prompt_tokens": 500, "output_tokens": 128},  # 500+128 ≠ 928:顺序对错了
        {"task_id": "b", "step": 1, "t_start": 2.0, "cached_tokens": 0},
        {"task_id": "c", "step": 1, "t_start": 3.0, "cached_tokens": 0},
    ]
    try:
        join_requests(requests, launches)
    except ValueError as e:
        assert "928" in str(e) and "a" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_main_skip_selects_the_launches_that_belong_to_the_run(tmp_path, capsys):
    from src.analysis.slot_trace import main
    log = tmp_path / "server.log"
    log.write_text(LOG)
    run = tmp_path / "run"
    run.mkdir()
    (run / "requests.jsonl").write_text(
        '{"task_id": "t1", "step": 1, "t_start": 1.0, "cached_tokens": 0}\n'
        '{"task_id": "t2", "step": 1, "t_start": 2.0, "cached_tokens": 0}\n')
    main(["--run", str(run), "--server-log", str(log), "--skip", "1"])
    rows = [__import__("json").loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(r["task_id"], r["slot"], r["how"]) for r in rows] == [("t1", 1, "LCP"), ("t2", 2, "LRU")]
