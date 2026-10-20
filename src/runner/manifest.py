"""每次 Run 的 manifest.json:复现和审计所需的信息。

某个字段读不到时置 null 并把原因写进 `errors`,不丢字段、不让整个 Run 起不来。
`run_cmd` 只用于依赖本机的命令(ps、pmset),测试里注入假实现;git 命令直接真跑。
"""
import hashlib
import platform
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable

RunCmd = Callable[[list[str]], str]  # 执行命令返回 stdout,失败抛异常


def default_run_cmd(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=True).stdout


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo_dir: Path) -> str:
    return default_run_cmd(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).strip()


def git_dirty(repo_dir: Path) -> bool:
    # results/ 里是各次 Run 的输出,不算代码改动
    return bool(default_run_cmd(["git", "-C", str(repo_dir), "status", "--porcelain", "--", ".", ":!results"]).strip())


def model_path(cmdline: str) -> Path:
    args = shlex.split(cmdline)
    for flag in ("-m", "--model"):
        if flag in args[:-1]:
            return Path(args[args.index(flag) + 1])
    raise ValueError("no -m/--model in server command line")


def power_source(batt: str) -> str:
    m = re.search(r"Now drawing from '([^']+)'", batt)
    if not m:
        raise ValueError("unrecognized `pmset -g batt` output")
    return m.group(1)


def low_power_mode(settings: str) -> bool:
    m = re.search(r"^\s*lowpowermode\s+(\d)", settings, re.MULTILINE)
    if not m:
        raise ValueError("no lowpowermode in `pmset -g` output")
    return m.group(1) == "1"


def collect_manifest(*, repo_dir: Path, llama_cpp_dir: Path | None, server_pid: int | None,
                     workload_path: Path, seed: int | None, c: int, lam: float, timeout_s: float,
                     run_cmd: RunCmd = default_run_cmd) -> dict:
    errors: dict[str, str] = {}

    def field(name: str, fn: Callable[[], object]):
        try:
            return fn()
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            return None

    def need(value, what: str):
        if value is None:
            raise ValueError(what)
        return value

    cmdline = field("server_cmdline", lambda: run_cmd(
        ["ps", "-ww", "-o", "command=", "-p", str(need(server_pid, "no --server-pid given"))]).strip())
    return {
        "repo_git_sha": field("repo_git_sha", lambda: git_head(repo_dir)),
        "repo_dirty": field("repo_dirty", lambda: git_dirty(repo_dir)),
        "llama_cpp_commit": field("llama_cpp_commit", lambda: git_head(need(llama_cpp_dir, "no --llama-cpp-dir given"))),
        "model_sha256": field("model_sha256", lambda: sha256_file(model_path(need(cmdline, "server command line unknown")))),
        "server_pid": server_pid, "server_cmdline": cmdline,
        "workload_sha256": field("workload_sha256", lambda: sha256_file(workload_path)),
        "seed": seed, "c": c, "lambda": lam, "timeout_s": timeout_s,
        "pmset_therm": field("pmset_therm", lambda: run_cmd(["pmset", "-g", "therm"])),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "python": platform.python_version()},
        "power_source": field("power_source", lambda: power_source(run_cmd(["pmset", "-g", "batt"]))),
        "low_power_mode": field("low_power_mode", lambda: low_power_mode(run_cmd(["pmset", "-g"]))),
        "errors": errors,
    }
