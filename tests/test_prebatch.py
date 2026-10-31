import json
import os
import subprocess
from pathlib import Path

from src.calibration.prebatch import budget, compare_t_solo, decide_threads, freeze


def _cfg(short, longctx, agg):
    return {"short_s": short, "longctx_s": longctx, "agg_tok_s": agg}


def test_keeps_t8_when_no_candidate_is_10pct_faster():
    results = {
        4: _cfg([4.5, 4.4, 4.6], [36, 36.5, 35.5], [30.0, 30.4]),
        8: _cfg([4.3, 4.3, 4.4], [35, 35.2, 35.1], [32.8, 32.5]),
        10: _cfg([4.2, 4.3, 4.2], [34.5, 34.9, 34.7], [33.0, 33.4]),
    }
    d = decide_threads(results)
    assert d["threads"] == 8
    assert d["reason"] == "within_10pct"


def test_picks_clearly_faster_stable_candidate():
    results = {
        4: _cfg([4.3, 4.4, 4.3], [35, 35.5, 35.2], [32.0, 32.4]),
        8: _cfg([4.3, 4.3, 4.4], [35, 35.2, 35.1], [32.8, 32.5]),
        10: _cfg([3.7, 3.8, 3.7], [30.5, 30.8, 30.6], [37.5, 37.9]),  # 约 +15%
    }
    d = decide_threads(results)
    assert d["threads"] == 10
    assert d["reason"] == "fastest"


def test_skips_faster_candidate_whose_runs_are_unstable():
    results = {
        8: _cfg([4.3, 4.3, 4.4], [35, 35.2, 35.1], [32.8, 32.5]),
        10: _cfg([2.6, 4.2, 5.2], [22, 40, 31], [37.5, 37.9]),  # 均值快,但 3 次差一倍
    }
    d = decide_threads(results)
    assert d["threads"] == 8
    assert d["reason"] == "unstable"


OLD = {"short": 4.3, "longctx": 35.0, "longgen": 22.5, "agent_per_step_s": 8.05}


def test_small_t_solo_drift_keeps_the_calibration():
    new = {"short": 4.0, "longctx": 31.0, "longgen": 22.0, "agent_per_step_s": 8.5}  # 最大偏差 -11.4%
    r = compare_t_solo(OLD, new)
    assert r["recalibrate"] is False
    assert r["exceeded"] == []
    assert round(r["deviation"]["longctx"], 3) == -0.114


def test_t_solo_deviation_over_15pct_in_any_type_triggers_recalibration():
    new = {"short": 4.3, "longctx": 25.0, "longgen": 22.5, "agent_per_step_s": 9.5}  # longctx -28.6%, agent +18%
    r = compare_t_solo(OLD, new)
    assert r["recalibrate"] is True
    assert r["exceeded"] == ["longctx", "agent_per_step_s"]


def test_budget_reproduces_step_55_numbers():
    # 来自 lab-notebook 5.5-B:S̄=20.26 s、P_eff=1.5、μ≈0.074、λ 网格 {0.022,0.044,0.067,0.089}
    mix = {"short": 0.4, "longctx": 0.2, "longgen": 0.2, "agent": 0.2}
    b = budget(OLD, mix, p_eff=1.5)
    assert round(b["mean_service_s"], 2) == 20.26
    assert round(b["mu"], 3) == 0.074
    assert [round(x, 3) for x in b["lambda_grid"]] == [0.022, 0.044, 0.067, 0.089]


START = Path(__file__).resolve().parent.parent / "scripts" / "start_server.sh"


def test_frozen_t_solo_is_read_by_the_workload_generator(tokenizer, tmp_path):
    from src.runner.gen_workload import generate_workload
    new = {"short": 4.0, "longctx": 31.0, "longgen": 22.0, "agent_per_step_s": 8.5}
    freeze(tmp_path, threads=10, t_solo=new)
    t_solo = json.loads((tmp_path / "t_solo.json").read_text())
    mix = {"profile": "dev", "seeds": [1], "mix": {"short": 1.0}}
    wl = generate_workload(mix, t_solo, seed=1, n_tasks=1, tokenizer=tokenizer)
    assert wl["tasks"][0]["deadline_s"] == 5 * 4.0


def test_start_server_uses_frozen_thread_count(tmp_path):
    freeze(tmp_path, threads=10, t_solo=OLD)
    stub = tmp_path / "llama-server"
    stub.write_text('#!/usr/bin/env bash\necho "$@"\n')
    stub.chmod(0o755)
    r = subprocess.run([str(START), "4", "16384"],
                       env={**os.environ, "LLAMA_SERVER_BIN": str(stub), "MODEL": "m.gguf",
                            "SERVER_CONFIG": str(tmp_path / "server.json")},
                       capture_output=True, text=True)
    args = r.stdout.split()
    assert args[args.index("-t") + 1] == "10"
