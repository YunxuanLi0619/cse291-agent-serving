import hashlib
import subprocess

from src.runner.manifest import collect_manifest

THERM = "Note: No thermal warning level has been recorded\nNote: No CPU power status has been recorded\n"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t67%; discharging; 3:57 remaining present: true\n"
PMSET = " lowpowermode         1\n sleep 1\n"


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def make_repo(path, name="f.txt"):
    path.mkdir()
    git(path, "init", "-q")
    (path / name).write_text("x")
    git(path, "add", name)
    git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return git(path, "rev-parse", "HEAD")


def fake_cmds(cmdline):
    table = {("pmset", "-g", "therm"): THERM, ("pmset", "-g", "batt"): BATT, ("pmset", "-g"): PMSET,
             ("ps", "-ww", "-o", "command=", "-p", "4321"): cmdline + "\n"}

    def run(argv):
        if tuple(argv) in table:
            return table[tuple(argv)]
        raise RuntimeError(f"unexpected command {argv}")
    return run


def collect(tmp_path, make_repo_dir=True, **over):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"weights")
    wl = tmp_path / "wl.json"
    wl.write_text('{"seed": 7, "tasks": []}')
    llama = tmp_path / "llama.cpp"
    llama_sha = make_repo(llama)
    cmdline = f"/bin/llama-server -m {model} --host 127.0.0.1 --port 8080 -np 4 -c 16384 --metrics"
    args = dict(repo_dir=tmp_path / "repo", llama_cpp_dir=llama, server_pid=4321, workload_path=wl, seed=7,
                c=2, lam=0.5, timeout_s=120.0, run_cmd=fake_cmds(cmdline))
    repo_sha = make_repo(args["repo_dir"]) if make_repo_dir else None
    args.update(over)
    return collect_manifest(**args), dict(repo_sha=repo_sha, llama_sha=llama_sha, model=model, wl=wl, cmdline=cmdline)


def test_manifest_records_everything_needed_to_reproduce_and_audit(tmp_path):
    m, exp = collect(tmp_path)
    assert m["repo_git_sha"] == exp["repo_sha"] and m["repo_dirty"] is False
    assert m["llama_cpp_commit"] == exp["llama_sha"]
    assert m["model_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert m["server_cmdline"] == exp["cmdline"]
    assert m["workload_sha256"] == hashlib.sha256(exp["wl"].read_bytes()).hexdigest()
    assert (m["seed"], m["c"], m["lambda"], m["timeout_s"]) == (7, 2, 0.5, 120.0)
    assert m["pmset_therm"] == THERM
    assert m["power_source"] == "Battery Power" and m["low_power_mode"] is True
    assert m["platform"]["system"] and m["platform"]["machine"]
    assert m["errors"] == {}


def test_repo_dirty_flag_follows_uncommitted_changes(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    (repo / "f.txt").write_text("changed")
    m, _ = collect(tmp_path, make_repo_dir=False)
    assert m["repo_dirty"] is True


def test_untracked_run_output_under_results_does_not_make_repo_dirty(tmp_path):
    repo = tmp_path / "repo"
    make_repo(repo)
    (repo / "results" / "raw").mkdir(parents=True)
    (repo / "results" / "raw" / "run.jsonl").write_text("{}")
    m, _ = collect(tmp_path, make_repo_dir=False)
    assert m["repo_dirty"] is False


def test_unreadable_field_is_null_and_named_in_errors_not_dropped(tmp_path):
    def broken(argv):
        raise RuntimeError("pmset not found")
    m, _ = collect(tmp_path, run_cmd=broken)
    assert m["pmset_therm"] is None and m["power_source"] is None and m["low_power_mode"] is None
    assert m["server_cmdline"] is None and m["model_sha256"] is None
    assert {"pmset_therm", "power_source", "low_power_mode", "server_cmdline", "model_sha256"} <= set(m["errors"])
    assert m["repo_git_sha"]  # unrelated fields are unaffected
