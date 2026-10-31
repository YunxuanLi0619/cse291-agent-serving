import json
import os
from types import SimpleNamespace

from src.batch.plan import RunSpec
from src.batch.system import MacHooks
from src.runner.gen_workload import generate_workload, write_workload

from fake_server import FakeServer


def test_real_runner_hook_passes_every_run_variable_through_to_the_manifest_and_requests(tmp_path, tokenizer):
    mix = {"profile": "dev", "seeds": [1], "mix": {"short": 1.0}}
    t_solo = {"short": 3.0, "longctx": 20.0, "longgen": 10.0, "agent_per_step_s": 6.0}
    spec = RunSpec("S", 4, 16384, 2, 1000.0, False, None, 1, 6)
    write_workload(generate_workload(mix, t_solo, 1, 6, tokenizer), tmp_path / spec.workload)
    with FakeServer(n_slots=4) as server:
        hooks = MacHooks(tmp_path, base_url=server.url).hooks()
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        handle = SimpleNamespace(pid=os.getpid(), cold_start_s=1.5)
        assert hooks.run_runner(spec, run_dir, handle) == 0
        assert hooks.smoke_analysis(run_dir) == []
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (manifest["c"], manifest["lambda"], manifest["cache_prompt"], manifest["cold_start_s"]) == (2, 1000.0, False, 1.5)
    assert len((run_dir / "tasks.jsonl").read_text().splitlines()) == 6
    assert all(b["cache_prompt"] is False for b in server.requests)


def test_smoke_analysis_reports_a_run_directory_the_analysis_cannot_read(tmp_path):
    (tmp_path / "empty").mkdir()
    assert MacHooks(tmp_path).hooks().smoke_analysis(tmp_path / "empty")
