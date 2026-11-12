"""Derive the per-type and per-step numbers cited in the Phase 1 report from the raw batch logs.

    .venv/bin/python report/phase1/make_breakdown.py

Reads results/raw/batch/*/ (and the single-seed c=1 run in results/raw/baseline_8/run). Writes
tables/type_table.tex (Table 1), tables/queue_table.tex (Table 2), tables/type_full_table.tex
(Appendix B) and tables/derived_numbers.json (every other log-derived number in the prose).

Definitions:
- All statistics use steady-state tasks (warmup=false), pooled over the 3 seeds unless noted.
- Queue wait = queue_time_total. A task that times out while queued is censored at its timeout.
- Service = t_done - t_arrive - queue wait - tool delay, reported for completed tasks.
- Truncated = an ok agent Request that produced fewer tokens than its max_tokens of 128.
"""
import collections
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "results", "raw", "batch")
TABLES = os.path.join(os.path.dirname(__file__), "tables")
TYPES = ("short", "longctx", "longgen", "agent")
AGENT_MAX_TOKENS = 128
SOLO_GUARD_S = 40  # a cancelled prefill can keep the server busy for up to ~37 s (lab notebook, #3 addendum)

CONFIGS = {
    "S1": ("S, $c{=}1$, $\\lambda{=}0.088$", "S_np1_c1_lam0.088_cpon_dvar_s*"),
    "S2": ("S, $c{=}2$, $\\lambda{=}0.088$", "S_np2_c2_lam0.088_cpon_dvar_s*"),
    "S4": ("S, $c{=}4$, $\\lambda{=}0.088$", "S_np4_c4_lam0.088_cpon_dvar_s*"),
    "S8": ("S, $c{=}8$, $\\lambda{=}0.088$", "S_np8_c8_lam0.088_cpon_dvar_s*"),
    "U4": ("U, $c{=}4$, $\\lambda{=}0.088$", "U_np8_c4_lam0.088_cpon_dvar_s*"),
    "U8": ("U, $c{=}8$, $\\lambda{=}0.088$", "U_np8_c8_lam0.088_cpon_dvar_s*"),
    "L029": ("S, $c{=}4$, $\\lambda{=}0.029$", "S_np4_c4_lam0.029_cpon_dvar_s*"),
    "L059": ("S, $c{=}4$, $\\lambda{=}0.059$", "S_np4_c4_lam0.059_cpon_dvar_s*"),
    "L118": ("S, $c{=}4$, $\\lambda{=}0.118$", "S_np4_c4_lam0.118_cpon_dvar_s*"),
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_runs(pattern):
    """One (tasks, requests) pair per seed; task ids repeat across seeds, so runs stay separate."""
    return [(read_jsonl(os.path.join(d, "tasks.jsonl")), read_jsonl(os.path.join(d, "requests.jsonl")))
            for d in sorted(glob.glob(os.path.join(RAW, pattern)))]


def steady(tasks):
    return [t for t in tasks if not t["warmup"]]


def median(values):
    return float(np.median(values)) if len(values) else None


def is_good(t):
    return t["status"] == "ok" and t["quality_pass"] and t["t_done"] - t["t_arrive"] <= t["deadline_s"]


def type_stats(pattern):
    """Per-type outcome counts and queue/service medians, plus the per-seed completed count."""
    runs = load_runs(pattern)
    tasks = [t for ts, _ in runs for t in steady(ts)]
    out = {"per_seed_completed": [sum(t["status"] == "ok" for t in steady(ts)) for ts, _ in runs],
           "n": len(tasks), "completed": sum(t["status"] == "ok" for t in tasks)}
    for ty in TYPES:
        a = [t for t in tasks if t["type"] == ty]
        ok = [t for t in a if t["status"] == "ok"]
        out[ty] = {
            "n": len(a), "completed": len(ok), "good": sum(is_good(t) for t in a),
            "quality_pass": sum(t["quality_pass"] for t in ok),
            "timeout": sum(t["status"] == "timeout" for t in a),
            "ctx": sum(t["status"] == "error_ctx_exceeded" for t in a),
            "never_dispatched": sum(t["t_first_dispatch"] is None for t in a),
            "queue_all": median([t["queue_time_total"] for t in a]),
            "queue_completed": median([t["queue_time_total"] for t in ok]),
            "service_completed": median([t["t_done"] - t["t_arrive"] - t["queue_time_total"] - t["idle_time_total"]
                                         for t in ok]),
        }
    return out


def time_accounting(pattern):
    """Mean seconds per task in queue / service / tool delay, for every steady-state task including timeouts."""
    runs = load_runs(pattern)
    out = {}
    pooled = [t for ts, _ in runs for t in steady(ts)]
    total = sum(t["t_done"] - t["t_arrive"] for t in pooled)
    out["queue_share_of_task_time"] = sum(t["queue_time_total"] for t in pooled) / total
    for ty in TYPES:
        rows = [[(t["queue_time_total"], t["t_done"] - t["t_arrive"] - t["queue_time_total"] - t["idle_time_total"],
                  t["idle_time_total"]) for t in steady(ts) if t["type"] == ty] for ts, _ in runs]
        per_seed = np.array([np.mean(r, axis=0) for r in rows])
        out[ty] = {"queue": per_seed[:, 0].mean(), "service": per_seed[:, 1].mean(), "tool": per_seed[:, 2].mean(),
                   "total_sd": float(per_seed.sum(axis=1).std(ddof=1))}
    return out


def step_rows(pattern):
    """Request-level view of agent steps: TTFT (excludes queueing), queue wait and prompt reuse per step k."""
    by_k = collections.defaultdict(list)
    for ts, rs in load_runs(pattern):
        agents = {t["task_id"] for t in steady(ts) if t["type"] == "agent"}
        for r in rs:
            if r["task_id"] in agents and r["status"] == "ok":
                by_k[r["step"]].append(r)
    rows = {}
    for k, v in sorted(by_k.items()):
        ttft = [r["t_first_token"] - r["t_start"] for r in v]
        rows[k] = {
            "n": len(v), "prompt_tokens": median([r["prompt_tokens"] for r in v]),
            "ttft": median(ttft), "ttft_q25": float(np.percentile(ttft, 25)), "ttft_q75": float(np.percentile(ttft, 75)),
            "queue_wait": median([r["t_start"] - r["t_submit"] for r in v]),
            "service": median([r["t_end"] - r["t_start"] for r in v]),
            "reuse": sum(r["cached_tokens"] for r in v) / sum(r["prompt_tokens"] for r in v),
        }
    return rows


def continuation_reuse(pattern):
    """Share of the prompt reused for agent steps k >= 2: pooled and per seed."""
    per_seed, cached_all, prompt_all = [], 0, 0
    for ts, rs in load_runs(pattern):
        agents = {t["task_id"] for t in steady(ts) if t["type"] == "agent"}
        v = [r for r in rs if r["task_id"] in agents and r["status"] == "ok" and r["step"] >= 2]
        cached, prompt = sum(r["cached_tokens"] for r in v), sum(r["prompt_tokens"] for r in v)
        per_seed.append(cached / prompt)
        cached_all, prompt_all = cached_all + cached, prompt_all + prompt
    return {"pooled": cached_all / prompt_all, "per_seed": per_seed}


def c8_truncation():
    """Where agents hit the 2048-token slot cap at c=8, and what truncation does to goodput."""
    runs = load_runs(CONFIGS["S8"][1])
    failed, trunc_ok, trunc_failed, goodput_all, goodput_excl = [], [], [], [], []
    for ts, rs in runs:
        by_id = {t["task_id"]: t for t in ts}
        s = steady(ts)
        window = max(t["t_done"] for t in s) - min(t["t_arrive"] for t in s)
        truncated_tasks = set()
        for r in rs:
            t = by_id[r["task_id"]]
            if r["warmup"] or t["type"] != "agent":
                continue
            if r["status"] == "error_ctx_exceeded":
                failed.append({"step": r["step"], "n_steps": t["n_steps"]})
            if r["status"] == "ok" and r["output_tokens"] < AGENT_MAX_TOKENS:
                row = {"step": r["step"], "n_steps": t["n_steps"], "prompt": r["prompt_tokens"],
                       "output": r["output_tokens"], "quality_pass": t["quality_pass"]}
                (trunc_ok if t["status"] == "ok" else trunc_failed).append(row)
                truncated_tasks.add(r["task_id"])
        good = [t for t in s if is_good(t)]
        goodput_all.append(len(good) / window)
        goodput_excl.append(sum(t["task_id"] not in truncated_tasks for t in good) / window)
    outputs = [r["output"] for r in trunc_ok]
    stats = collections.Counter(t["n_steps"] for ts, _ in runs for t in steady(ts) if t["type"] == "agent")
    return {
        "ctx_failures": len(failed), "ctx_failure_steps": sorted({f["step"] for f in failed}),
        "ctx_failure_n_steps": sorted({f["n_steps"] for f in failed}), "agents_by_n_steps": dict(sorted(stats.items())),
        "truncated_completed_tasks": len(trunc_ok), "truncated_completed_output_range": [min(outputs), max(outputs)],
        "truncated_completed_quality_pass": sum(r["quality_pass"] for r in trunc_ok),
        "truncated_completed_steps": sorted({r["step"] for r in trunc_ok}),
        "truncated_before_failure_tasks": len(trunc_failed),
        "truncated_before_failure_prompt_range": [min(r["prompt"] for r in trunc_failed), max(r["prompt"] for r in trunc_failed)],
        "truncated_before_failure_output_range": [min(r["output"] for r in trunc_failed), max(r["output"] for r in trunc_failed)],
        "goodput_mean": float(np.mean(goodput_all)), "goodput_excluding_truncated_mean": float(np.mean(goodput_excl)),
    }


def other_truncation():
    """Agent/short/longctx/longgen output below max_tokens outside S c=8 (expected: none)."""
    limit = {"short": 64, "longctx": 64, "longgen": 512, "agent": AGENT_MAX_TOKENS}
    count = 0
    for d in glob.glob(os.path.join(RAW, "*")):
        if os.path.basename(d).startswith("S_np8"):
            continue
        by_id = {t["task_id"]: t for t in read_jsonl(os.path.join(d, "tasks.jsonl"))}
        count += sum(r["status"] == "ok" and r["output_tokens"] < limit[by_id[r["task_id"]]["type"]]
                     for r in read_jsonl(os.path.join(d, "requests.jsonl")))
    return count


def shared_pool_events():
    """Pool-pressure lines in the server logs of the six series U runs."""
    purge = retry = exceeded = 0
    largest = 0
    for d in glob.glob(os.path.join(RAW, "U_*")):
        with open(os.path.join(d, "server.log"), errors="ignore") as f:
            for line in f:
                if "purging slot" in line:
                    purge += 1
                    largest = max(largest, int(line.split(" with ")[1].split()[0]))
                retry += "failed to find free space" in line
                exceeded += "Context size has been exceeded" in line
    return {"purges": purge, "smaller_batch_retries": retry, "context_exceeded": exceeded, "largest_purge_tokens": largest}


def solo_requests():
    """Requests that ran alone, as an after-the-fact stand-in for T_solo over the batch.

    Solo = overlaps no other Request of its run and starts more than SOLO_GUARD_S after any non-ok Request
    ended; prompts with a reused prefix are excluded. This is not a dedicated T_solo measurement: the
    bins mix configurations, because the batch order changes configuration over time.
    """
    rows = []
    for d in glob.glob(os.path.join(RAW, "*")):
        with open(os.path.join(d, "manifest.json")) as f:
            t_start = json.load(f)["t_start"]
        types = {t["task_id"]: t["type"] for t in read_jsonl(os.path.join(d, "tasks.jsonl"))}
        rs = [r for r in read_jsonl(os.path.join(d, "requests.jsonl")) if r["t_start"] is not None]
        for r in rs:
            if r["status"] != "ok" or r["cached_tokens"] >= 100 or types[r["task_id"]] == "agent":
                continue
            clean = all(o is r or not (o["t_start"] < r["t_end"] and o["t_end"] > r["t_start"]) for o in rs) and not any(
                o["status"] != "ok" and o["t_end"] <= r["t_start"] and r["t_start"] - o["t_end"] < SOLO_GUARD_S for o in rs)
            if clean:
                rows.append((types[r["task_id"]], t_start + r["t_start"], r["t_end"] - r["t_start"]))
    t0 = min(m["t_start"] for m in manifests())
    out = {}
    for ty in ("short", "longctx", "longgen"):
        bins = collections.defaultdict(list)
        for row_type, t, dur in rows:
            if row_type == ty:
                bins[int((t - t0) // (3 * 3600)) * 3].append(dur)
        out[ty] = {f"h{b}-{b + 3}": {"n": len(v), "median_s": median(v)} for b, v in sorted(bins.items())}
    return out


def manifests():
    result = []
    for d in glob.glob(os.path.join(RAW, "*")):
        with open(os.path.join(d, "manifest.json")) as f:
            result.append(json.load(f))
    return result


def batch_facts():
    ms = manifests()
    peaks, throttled = [], False
    for d in glob.glob(os.path.join(RAW, "*")):
        peaks.append(max(r["footprint_mb"] or 0 for r in read_jsonl(os.path.join(d, "system.jsonl"))))
        throttled = throttled or any(r["throttled"] for r in read_jsonl(os.path.join(d, "therm.jsonl")))
    return {
        "n_runs": len(ms), "wall_clock_hours": (max(m["t_end"] for m in ms) - min(m["t_start"] for m in ms)) / 3600,
        "repo_git_sha": sorted({m["repo_git_sha"] for m in ms}), "any_dirty": any(m["repo_dirty"] for m in ms),
        "peak_footprint_mib_range": [min(peaks), max(peaks)], "any_thermal_warning": throttled,
    }


def single_seed_c1():
    """c=1 at lambda=0.029 on seed 1 (quality-baseline run, -np 4) vs the batch c=4 run on the same trace."""
    out = {}
    for name, d in (("c1_np4_baseline_run", os.path.join(ROOT, "results", "raw", "baseline_8", "run")),
                    ("c4_batch_run", os.path.join(RAW, "S_np4_c4_lam0.029_cpon_dvar_s1"))):
        s = steady(read_jsonl(os.path.join(d, "tasks.jsonl")))
        window = max(t["t_done"] for t in s) - min(t["t_arrive"] for t in s)
        with open(os.path.join(d, "manifest.json")) as f:
            sha = json.load(f)["workload_sha256"]
        failed = sum(t["status"] != "ok" for t in s)
        out[name] = {"workload_sha256": sha, "n": len(s), "failed": failed, "fail_rate": failed / len(s),
                     "goodput": sum(is_good(t) for t in s) / window}
    return out


def baseline_gap():
    """What the missing c=2 run at lambda=0.029 can change, bounded from measured runs only."""
    tasks = [t for ts, _ in load_runs(CONFIGS["L029"][1]) for t in steady(ts)]
    good = sum(is_good(t) for t in tasks)
    late = [t for t in tasks if t["status"] == "ok" and t["t_done"] - t["t_arrive"] > t["deadline_s"]]
    goodput_by_seed = {}
    for c in (1, 2, 4):
        goodput_by_seed[f"c{c}"] = []
        for ts, _ in load_runs(f"S_np{c}_c{c}_lam0.088_cpon_dvar_s*"):
            s = steady(ts)
            window = max(t["t_done"] for t in s) - min(t["t_arrive"] for t in s)
            goodput_by_seed[f"c{c}"].append(sum(is_good(t) for t in s) / window)
    return {
        "c4_lam0.029_n": len(tasks), "c4_lam0.029_good": good, "max_additional_good_tasks": len(tasks) - good,
        "max_gain_in_good_tasks": len(tasks) / good - 1,
        "c4_lam0.029_failed": sum(t["status"] != "ok" for t in tasks),
        "c4_lam0.029_late_by_type": dict(collections.Counter(t["type"] for t in late)),
        "goodput_by_seed_lam0.088": goodput_by_seed,
    }


def num(x, digits=1):
    return "--" if x is None else f"{x:.{digits}f}"


def write_type_table(stats):
    cols = ("S1", "S2", "S4", "S8", "L029")
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             " & \\multicolumn{4}{c}{$\\lambda{=}0.088$} & $\\lambda{=}0.029$ \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(l){6-6}",
             "Type (total) & $c{=}1$ & $c{=}2$ & $c{=}4$ & $c{=}8$ & $c{=}4$ \\\\", "\\midrule"]
    for ty in TYPES:
        cells = []
        for c in cols:
            s = stats[c][ty]
            cells.append(f"{s['completed']}" + (f" ({s['ctx']} ctx)" if s["ctx"] else ""))
        lines.append(f"{ty} ({stats['S1'][ty]['n']}) & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", f"all ({stats['S1']['n']}) & " + " & ".join(str(stats[c]["completed"]) for c in cols) + " \\\\",
              "per-seed sd & " + " & ".join(f"{np.std(stats[c]['per_seed_completed'], ddof=1):.1f}" for c in cols) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TABLES, "type_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_queue_table(stats):
    cols = ("L029", "S4", "S8")
    lines = ["\\begin{tabular}{@{}lrrrrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{$c{=}4$, $\\lambda{=}0.029$} & \\multicolumn{3}{c}{$c{=}4$, $\\lambda{=}0.088$}"
             " & \\multicolumn{3}{c}{$c{=}8$, $\\lambda{=}0.088$} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-10}",
             "Type" + " & Q$_{\\mathrm{all}}$ & Q$_{\\mathrm{ok}}$ & S" * 3 + " \\\\", "\\midrule"]
    for ty in TYPES:
        cells = []
        for c in cols:
            s = stats[c][ty]
            cells += [num(s["queue_all"]), num(s["queue_completed"]), num(s["service_completed"])]
        lines.append(f"{ty} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TABLES, "queue_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_full_table(stats):
    lines = ["\\begin{tabular}{@{}llrrrrrrrrrr@{}}", "\\toprule",
             "Configuration & Type & Tasks & Compl. & Good & Quality & Timeout & Ctx & Never disp. & "
             "Q$_{\\mathrm{all}}$ (s) & Q$_{\\mathrm{ok}}$ (s) & S (s) \\\\", "\\midrule"]
    for i, key in enumerate(("S1", "S4", "S8", "U4", "U8", "L029")):  # the rest are in derived_numbers.json
        label = CONFIGS[key][0]
        if i:
            lines.append("\\addlinespace[2pt]")
        for j, ty in enumerate(TYPES):
            s = stats[key][ty]
            lines.append(f"{label if j == 0 else ''} & {ty} & {s['n']} & {s['completed']} & {s['good']} & "
                         f"{s['quality_pass']} & {s['timeout']} & {s['ctx']} & {s['never_dispatched']} & "
                         f"{num(s['queue_all'])} & {num(s['queue_completed'])} & {num(s['service_completed'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(TABLES, "type_full_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(TABLES, exist_ok=True)
    stats = {key: type_stats(pattern) for key, (_, pattern) in CONFIGS.items()}
    write_type_table(stats)
    write_queue_table(stats)
    write_full_table(stats)
    derived = {
        "batch": batch_facts(),
        "type_stats": stats,
        "time_accounting": {k: time_accounting(CONFIGS[k][1]) for k in ("L029", "L059", "S4", "L118")},
        "steps": {
            "c4_lam0.029_cache_on": step_rows(CONFIGS["L029"][1]),
            "c4_lam0.088_cache_on": step_rows(CONFIGS["S4"][1]),
            "c4_lam0.088_cache_off": step_rows("S_np4_c4_lam0.088_cpoff_dvar_s*"),
            "c1_lam0.088_cache_off": step_rows("S_np1_c1_lam0.088_cpoff_dvar_s*"),
        },
        "continuation_reuse": {
            "c1_lam0.088": continuation_reuse(CONFIGS["S1"][1]),
            "c4_lam0.088": continuation_reuse(CONFIGS["S4"][1]),
            "c4_lam0.029": continuation_reuse(CONFIGS["L029"][1]),
            **{f"e3_delay_{d}": continuation_reuse(f"S_np4_c4_lam0.088_cpon_d{d}_s*") for d in ("0", "0.5", "1", "3")},
        },
        "c8_truncation": c8_truncation(),
        "truncated_requests_outside_S_c8": other_truncation(),
        "shared_pool_events": shared_pool_events(),
        "solo_requests": solo_requests(),
        "single_seed_c1_lam0.029": single_seed_c1(),
        "baseline_gap": baseline_gap(),
    }
    path = os.path.join(TABLES, "derived_numbers.json")
    with open(path, "w") as f:
        json.dump(derived, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
