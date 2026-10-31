import json
from pathlib import Path

import pytest

from src.runner.gen_workload import main

from fake_tokenizer import FakeTokenizer

RECALL_QUESTION = "What is the secret key mentioned at the beginning of this conversation? Reply with the key only."
ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def gen(tmp_path, name, profile="dev", seed=1, n_tasks=20):
    out = tmp_path / name
    main(
        ["--mix", str(CONFIGS / f"mix_{profile}.json"), "--t-solo", str(CONFIGS / "t_solo.json"),
         "--seed", str(seed), "--n-tasks", str(n_tasks), "--out", str(out)],
        tokenizer=FakeTokenizer(),
    )
    return out


def test_same_seed_gives_byte_identical_file(tmp_path):
    a = gen(tmp_path, "a.json")
    b = gen(tmp_path, "b.json")
    assert a.read_bytes() == b.read_bytes()


def load(path):
    return json.loads(path.read_text())


def test_dev_and_eval_use_different_seeds_and_mixes(tmp_path):
    dev = load(gen(tmp_path, "dev.json", "dev", seed=1))
    ev = load(gen(tmp_path, "eval.json", "eval", seed=101))
    assert dev["profile"] == "dev" and ev["profile"] == "eval"
    assert dev["mix"] != ev["mix"]
    assert dev["seed"] != ev["seed"]


def test_seed_outside_profile_range_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        gen(tmp_path, "x.json", "dev", seed=101)
    with pytest.raises(SystemExit):
        gen(tmp_path, "y.json", "eval", seed=1)


def tasks_of(wl, type_):
    return [t for t in wl["tasks"] if t["type"] == type_]


def test_every_task_has_needle_steps_and_deadline(tmp_path):
    wl = load(gen(tmp_path, "w.json", n_tasks=200))
    assert {t["type"] for t in wl["tasks"]} == {"short", "longctx", "longgen", "agent"}
    for t in wl["tasks"]:
        assert t["needle"].startswith("KEY-")
        assert t["steps"][0]["user_content"].count(t["needle"]) == 1
        assert RECALL_QUESTION in t["steps"][-1]["user_content"]
        assert t["steps"][-1]["delay_after_s"] is None
    assert {len(t["steps"]) for t in tasks_of(wl, "agent")} == {3, 4, 5, 6}
    assert all(len(t["steps"]) == 1 for t in wl["tasks"] if t["type"] != "agent")
    assert {s["max_tokens"] for t in tasks_of(wl, "short") for s in t["steps"]} == {64}
    assert {s["max_tokens"] for t in tasks_of(wl, "longgen") for s in t["steps"]} == {512}
    # 只有 agent 的非末步有 Tool delay,取值 0.2–3 s
    delays = [s["delay_after_s"] for t in tasks_of(wl, "agent") for s in t["steps"][:-1]]
    assert all(0.2 <= d <= 3.0 for d in delays)


def test_deadline_is_five_times_t_solo_from_input_table(tmp_path):
    table = tmp_path / "t_solo.json"  # 自带固定的表:configs/t_solo.json 每次重新校准都会变
    table.write_text(json.dumps({"short": 4.3, "longctx": 35.0, "longgen": 22.5, "agent_per_step_s": 8.05}))
    out = tmp_path / "w.json"
    main(["--mix", str(CONFIGS / "mix_dev.json"), "--t-solo", str(table), "--seed", "1",
          "--n-tasks", "200", "--out", str(out)], tokenizer=FakeTokenizer())
    wl = load(out)
    assert {t["deadline_s"] for t in tasks_of(wl, "short")} == {21.5}
    assert {t["deadline_s"] for t in tasks_of(wl, "longctx")} == {175.0}
    assert {t["deadline_s"] for t in tasks_of(wl, "longgen")} == {112.5}
    for t in tasks_of(wl, "agent"):  # 5 × (每步 8.05 s × 步数 + Σ Tool delay)
        total_delay = sum(s["delay_after_s"] for s in t["steps"][:-1])
        assert t["deadline_s"] == pytest.approx(5 * (8.05 * len(t["steps"]) + total_delay), abs=1e-3)


def test_t_solo_table_is_an_input_not_hardcoded(tmp_path):
    custom = tmp_path / "t_solo.json"
    custom.write_text(json.dumps({"short": 10.0, "longctx": 1.0, "longgen": 1.0, "agent_per_step_s": 1.0}))
    out = tmp_path / "w.json"
    main(["--mix", str(CONFIGS / "mix_dev.json"), "--t-solo", str(custom), "--seed", "1",
          "--n-tasks", "50", "--out", str(out)], tokenizer=FakeTokenizer())
    assert {t["deadline_s"] for t in tasks_of(load(out), "short")} == {50.0}


def test_arrival_gaps_are_unit_rate_exponential(tmp_path):
    wl = load(gen(tmp_path, "w.json", n_tasks=3000))
    gaps = [t["unit_gap_s"] for t in wl["tasks"]]
    assert all(g > 0 for g in gaps)
    assert sum(gaps) / len(gaps) == pytest.approx(1.0, rel=0.1)  # λ=1 → 均值 1 s


TARGETS = {"short": 256, "longctx": 3000, "longgen": 256, "agent": 800}


def test_prompt_tokens_of_every_task_are_within_5_percent_of_target(tmp_path):
    tok = FakeTokenizer()
    wl = load(gen(tmp_path, "w.json", n_tasks=100))
    for t in wl["tasks"]:
        target = TARGETS[t["type"]]
        measured = tok.prompt_tokens([{"role": "user", "content": t["steps"][0]["user_content"]}])
        assert abs(measured - target) <= 0.05 * target, (t["task_id"], t["type"], measured)


class CoarseTokenizer:
    """token 数只按 100 一档变化 → 256 的 ±5% 区间(243–268)内没有任何可达值。"""

    def prompt_tokens(self, messages):
        return -(-FakeTokenizer().prompt_tokens(messages) // 100) * 100


def test_generation_fails_when_a_prompt_cannot_be_calibrated(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        main(["--mix", str(CONFIGS / "mix_dev.json"), "--t-solo", str(CONFIGS / "t_solo.json"),
              "--seed", "1", "--n-tasks", "5", "--out", str(tmp_path / "w.json")],
             tokenizer=CoarseTokenizer())


def test_agent_tool_results_are_distinct_and_do_not_repeat_the_first_prompt(tmp_path):
    wl = load(gen(tmp_path, "w.json", n_tasks=100))
    for t in tasks_of(wl, "agent"):
        first = t["steps"][0]["user_content"]
        results = [s["user_content"].split("\n\n")[0] for s in t["steps"][1:]]
        assert len(set(results)) == len(results), t["task_id"]
        for r in results:
            assert r[:60] not in first, t["task_id"]
