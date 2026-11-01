"""Read all completed Runs from the raw log directory.

Run directory = a directory containing manifest.json and tasks.jsonl (may be nested); a manifest without `t_end` is a Run that crashed midway: skipped and reported on stderr.
All Run-level configuration comes from the manifest: `c`, `lambda`, `seed`, `server_cmdline` (parsed into np / B / series),
and the optional `tool_delay_s`, `cache_prompt` (written by the E3/E4 batch scripts; None by default).
"""
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PLATFORM_ALIASES = {"darwin": "macos"}  # align manifest.platform.system with the memory probe's platform label


@dataclass
class Run:
    name: str
    config: dict
    tasks: list[dict]
    requests: list[dict]
    system: list[dict]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _flag(args: list[str], *names: str) -> int | None:
    for name in names:
        if name in args[:-1]:
            return int(args[args.index(name) + 1])
    return None


def parse_server_config(cmdline: str | None) -> dict:
    """np, B (= -c, ADR 0002), series (S = --no-kv-unified, U = -kvu); None if it cannot be read."""
    args = cmdline.split() if cmdline else []
    series = "S" if "--no-kv-unified" in args else "U" if {"-kvu", "--kv-unified"} & set(args) else None
    return {"np": _flag(args, "-np", "--parallel"), "B": _flag(args, "-c", "--ctx-size"), "series": series}


def platform_of(manifest: dict) -> str | None:
    if manifest.get("memory_platform"):
        return manifest["memory_platform"]
    system = (manifest.get("platform") or {}).get("system")
    return PLATFORM_ALIASES.get(system.lower(), system.lower()) if system else None


def load_run(d: Path) -> Run:
    m = json.loads((d / "manifest.json").read_text())
    config = {"platform": platform_of(m), **parse_server_config(m.get("server_cmdline")), "c": m["c"],
              "lambda": m["lambda"], "tool_delay_s": m.get("tool_delay_s"),
              "cache_prompt": m.get("cache_prompt"), "seed": m.get("seed")}
    return Run(d.name, config, read_jsonl(d / "tasks.jsonl"), read_jsonl(d / "requests.jsonl"),
               read_jsonl(d / "system.jsonl"))


def load_runs(raw: Path) -> list[Run]:
    runs = []
    for manifest in sorted(raw.rglob("manifest.json")):
        d = manifest.parent
        if not (d / "tasks.jsonl").exists():
            continue
        if "t_end" not in json.loads(manifest.read_text()):
            print(f"skipping unfinished run (no t_end in manifest): {d}", file=sys.stderr)
            continue
        runs.append(load_run(d))
    return runs
