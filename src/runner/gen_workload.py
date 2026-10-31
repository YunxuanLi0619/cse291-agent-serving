"""Workload 生成器:固定种子 → 纯数据 JSON 文件,同种子字节一致。

用法: python -m src.runner.gen_workload --mix configs/mix_dev.json --t-solo configs/t_solo.json \
        --seed 1 --n-tasks 100 --out workloads/dev/default_s1.json --base-url http://127.0.0.1:8080
"""
import argparse
import json
import random
from pathlib import Path


WORDS = ("alpha river stone market window garden copper signal harbor violet engine paper "
         "lantern meadow thunder basket orbit canyon ribbon falcon marble velvet anchor cobalt "
         "saddle timber quartz meteor candle summit pebble tunnel harvest compass glacier").split()
# 措辞是 #8 冒烟后改的:Llama-3.2-3B 对 "secret key" 常拒答("I can't provide ... illegal or harmful"),
# 改成 "reference code" 后同一批 prompt 全部答对
RECALL_QUESTION = ("What is the reference code mentioned at the beginning of this conversation? "
                   "Reply with the code only.")
TOLERANCE = 0.05
TOOL_RESULT_TOKENS = 150
# 每个 Task type 的第 1 步 prompt_tokens 目标(含 chat template)与每步输出上限
TYPE_SPECS = {
    "short": {"prompt_tokens": 256, "max_tokens": 64},
    "longctx": {"prompt_tokens": 3000, "max_tokens": 64},
    "longgen": {"prompt_tokens": 256, "max_tokens": 512},
    "agent": {"prompt_tokens": 800, "max_tokens": 128},
}


def _task_rng(seed: int, index: int, purpose: str) -> random.Random:
    # 每个 Task 的每个用途一条独立随机流:Task 内容不依赖 tokenizer 调用了多少次
    return random.Random(f"{seed}:{index}:{purpose}")


class _WordStream:
    """同一条随机流上按需增长的词表;words(n) 总是同一序列的前缀。"""

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._words: list[str] = []

    def words(self, n: int) -> list[str]:
        while len(self._words) < n:
            self._words.append(self._rng.choice(WORDS) + str(self._rng.randrange(100)))
        return self._words[:n]


def _fit(count_fn, target: int) -> int:
    """返回使 count_fn(n) ≤ target 的最大 n(count_fn 随 n 单调不减),并自检落在 ±5% 内。"""
    hi = 1
    while count_fn(hi) <= target:
        hi *= 2
        if hi > 1_000_000:
            raise ValueError(f"cannot reach {target} tokens")
    lo = 0  # count_fn(lo) ≤ target 或 lo=0;count_fn(hi) > target
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if count_fn(mid) <= target else (lo, mid)
    measured = count_fn(lo)
    if abs(measured - target) > TOLERANCE * target:
        raise ValueError(f"prompt_tokens {measured} is outside ±{TOLERANCE:.0%} of target {target}")
    return lo


def _needle(rng: random.Random) -> str:
    return "KEY-" + "".join(rng.choices("abcdefghjkmnpqrstuvwxyz23456789", k=8))


def _first_prompt(tokenizer, ws: _WordStream, needle: str, needle_frac: float, suffix: str, target: int):
    def content(n: int) -> str:
        words = ws.words(n)
        at = round(n * needle_frac)
        return (" ".join(words[:at]) + f" The reference code is {needle}. " + " ".join(words[at:])
                + "\n\n" + suffix)

    def count(n: int) -> int:
        return tokenizer.prompt_tokens([{"role": "user", "content": content(n)}])

    n = _fit(count, target)
    return content(n), count(n)


def _tool_result(tokenizer, rng: random.Random) -> str:
    ws = _WordStream(rng)
    overhead = tokenizer.prompt_tokens([{"role": "user", "content": ""}])

    def count(n: int) -> int:
        return tokenizer.prompt_tokens([{"role": "user", "content": " ".join(ws.words(n))}]) - overhead

    return " ".join(ws.words(_fit(count, TOOL_RESULT_TOKENS)))


def _make_task(t_solo: dict, seed: int, index: int, type_: str, gap: float, tokenizer) -> dict:
    spec = TYPE_SPECS[type_]
    shape = _task_rng(seed, index, "shape")
    ws = _WordStream(_task_rng(seed, index, "filler"))
    needle = _needle(_task_rng(seed, index, "needle"))
    n_steps = shape.randint(3, 6) if type_ == "agent" else 1
    delays = [round(shape.uniform(0.2, 3.0), 2) for _ in range(n_steps - 1)]
    suffix = RECALL_QUESTION if n_steps == 1 else "Work through the task step by step; tool results will follow."
    first, measured = _first_prompt(tokenizer, ws, needle, shape.uniform(0.2, 0.8), suffix, spec["prompt_tokens"])
    contents = [first]
    for k in range(1, n_steps):
        text = _tool_result(tokenizer, _task_rng(seed, index, f"tool{k}"))  # 每步独立随机流
        contents.append(text + "\n\n" + RECALL_QUESTION if k == n_steps - 1 else text)
    if type_ == "agent":
        deadline = 5 * (t_solo["agent_per_step_s"] * n_steps + sum(delays))
    else:
        deadline = 5 * t_solo[type_]
    return {
        "task_id": f"t{index:04d}", "type": type_, "unit_gap_s": gap, "needle": needle,
        "deadline_s": round(deadline, 3),
        "prompt_tokens_target": spec["prompt_tokens"], "prompt_tokens_measured": measured,
        "steps": [{"user_content": c, "max_tokens": spec["max_tokens"],
                   "delay_after_s": delays[k] if k < n_steps - 1 else None}
                  for k, c in enumerate(contents)],
    }


def generate_workload(mix_cfg: dict, t_solo: dict, seed: int, n_tasks: int, tokenizer) -> dict:
    if seed not in mix_cfg["seeds"]:
        raise SystemExit(f"seed {seed} is not allowed for profile {mix_cfg['profile']!r} "
                         f"(allowed: {mix_cfg['seeds']})")
    types = list(mix_cfg["mix"])
    weights = [mix_cfg["mix"][t] for t in types]
    head = random.Random(f"{seed}:head")
    tasks = []
    for i in range(n_tasks):
        type_ = head.choices(types, weights)[0]
        gap = round(head.expovariate(1.0), 6)
        tasks.append(_make_task(t_solo, seed, i, type_, gap, tokenizer))
    return {"profile": mix_cfg["profile"], "seed": seed, "mix": mix_cfg["mix"],
            "t_solo": t_solo, "tasks": tasks}


def with_fixed_tool_delay(workload: dict, delay_s: float) -> dict:
    """E3 变体:所有 Tool delay 取同一个值,agent 的 deadline 按 Workload 自带的 T_solo 表和新 delay 重算。

    其余内容(prompt、Needle、到达间隔)不变,所以同 seed 的不同 delay 之间可以配对。
    """
    t_solo = workload["t_solo"]
    tasks = []
    for task in workload["tasks"]:
        steps = [{**s, "delay_after_s": delay_s if s["delay_after_s"] is not None else None} for s in task["steps"]]
        task = {**task, "steps": steps}
        if task["type"] == "agent":
            task["deadline_s"] = round(5 * (t_solo["agent_per_step_s"] * len(steps) + delay_s * (len(steps) - 1)), 3)
        tasks.append(task)
    return {**workload, "tasks": tasks, "tool_delay_s": delay_s}


def write_workload(workload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workload, sort_keys=True, indent=1, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None, tokenizer=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--t-solo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-tasks", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="用于 token 校准的 llama-server")
    a = ap.parse_args(argv)
    if tokenizer is None:
        from .tokenizer import LlamaServerTokenizer
        tokenizer = LlamaServerTokenizer(a.base_url)
    mix_cfg = json.loads(Path(a.mix).read_text())
    t_solo = json.loads(Path(a.t_solo).read_text())
    wl = generate_workload(mix_cfg, t_solo, a.seed, a.n_tasks, tokenizer)
    out = Path(a.out)
    write_workload(wl, out)
    return out


if __name__ == "__main__":
    main()
