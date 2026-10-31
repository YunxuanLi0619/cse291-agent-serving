import hashlib
import json

from src.batch.plan import build_plan
from src.batch.workloads import verify_workloads, write_e3_variants
from src.runner.gen_workload import generate_workload, write_workload
from test_batch_driver import CFG


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path, tokenizer):
    mix = {"profile": "dev", "seeds": [1, 2, 3, 4, 5], "mix": {"short": 0.5, "agent": 0.5}}
    t_solo = {"short": 3.0, "longctx": 20.0, "longgen": 10.0, "agent_per_step_s": 6.0}
    for seed in (1, 2):
        write_workload(generate_workload(mix, t_solo, seed, 8, tokenizer), tmp_path / f"workloads/dev/default_s{seed}.json")


def test_e3_variants_are_derived_for_every_seed_and_delay_and_are_byte_stable(tmp_path, tokenizer):
    setup(tmp_path, tokenizer)
    write_e3_variants(tmp_path, CFG)
    files = sorted(p.name for p in (tmp_path / "workloads/dev").glob("e3_*"))
    assert files == sorted(f"e3_d{d:g}_s{s}.json" for d in CFG["e3_delays_s"] for s in (1, 2))
    before = {f: (tmp_path / "workloads/dev" / f).read_bytes() for f in files}
    write_e3_variants(tmp_path, CFG)
    assert before == {f: (tmp_path / "workloads/dev" / f).read_bytes() for f in files}
    assert json.loads((tmp_path / "workloads/dev/e3_d0_s1.json").read_text())["tool_delay_s"] == 0


def test_verify_reports_missing_and_changed_workloads_against_frozen_hashes(tmp_path, tokenizer):
    setup(tmp_path, tokenizer)
    cfg = {**CFG, "reps": 2}
    write_e3_variants(tmp_path, cfg)
    needed = sorted({r.workload for st in build_plan(cfg) for r in st.runs} | {st.pilot.workload for st in build_plan(cfg)})
    cfg["workload_sha256"] = {w: sha(tmp_path / w) for w in needed}
    assert verify_workloads(tmp_path, cfg) == []
    (tmp_path / "workloads/dev/default_s2.json").write_text("{}")
    (tmp_path / "workloads/dev/e3_d3_s1.json").unlink()
    problems = verify_workloads(tmp_path, cfg)
    assert "workloads/dev/default_s2.json: sha256 differs from the frozen value" in problems
    assert "workloads/dev/e3_d3_s1.json: missing" in problems
    cfg["workload_sha256"].pop("workloads/dev/default_s1.json")
    assert "workloads/dev/default_s1.json: no frozen sha256 in configs/batch.json" in verify_workloads(tmp_path, cfg)
