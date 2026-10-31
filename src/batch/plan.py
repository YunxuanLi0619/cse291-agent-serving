"""批跑矩阵(#9):把 E1–E5 展开成有序的 Run 列表。纯逻辑,不碰文件、进程和网络。

- 优先级(手册 Step 9):E1-系列 S → E4 → E5 → E3 → E1-系列 U;E2 用 E1 数据按 Step 切片,不单独跑。
- 重复 i 用 seed i(Workload `default_s<i>.json`),所以同 seed 的不同配置是同一份 Task 序列,可以配对。
- E4 的 cache_prompt=on 和 E5 的 λ=0.9μ 就是 E1-S 的 Run,不重跑,只记在 `reused`。
- Stage 内按「重复轮次」排序、轮内配置顺序隔轮反向:降频之类的漂移分摊到各配置,中途被迫停下时每个配置也都有数据。
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RunSpec:
    series: str  # "S":np = c、每 slot 容量 B/c;"U":共享池 -kvu
    np: int
    ctx: int  # B,总 KV token 数
    c: int
    lam: float
    cache_prompt: bool
    tool_delay_s: float | None  # None = Workload 自带的随机 Tool delay
    seed: int
    n_tasks: int
    pilot: bool = False

    @property
    def server_flags(self) -> tuple[str, ...]:
        return ("--no-kv-unified",) if self.series == "S" else ("-kvu",)

    @property
    def workload(self) -> str:
        if self.tool_delay_s is None:
            return f"workloads/dev/default_s{self.seed}.json"
        return f"workloads/dev/e3_d{self.tool_delay_s:g}_s{self.seed}.json"

    @property
    def estimated_s(self) -> float:
        """粗估墙钟:到达期 n/λ + 排空与重启约 150 s(只用来在日志里报 ETA,不参与任何判断)。"""
        return self.n_tasks / self.lam + 150

    @property
    def name(self) -> str:
        delay = "var" if self.tool_delay_s is None else f"{self.tool_delay_s:g}"
        base = (f"{self.series}_np{self.np}_c{self.c}_lam{self.lam:g}_cp{'on' if self.cache_prompt else 'off'}"
                f"_d{delay}_s{self.seed}")
        return f"pilot_{base}" if self.pilot else base


@dataclass(frozen=True)
class Stage:
    experiment: str
    pilot: RunSpec
    runs: list[RunSpec]  # 只含本 Stage 新增的 Run
    reused: list[str]  # 复用的、别的 Stage 已有的 Run 名


def _order(specs: list[RunSpec]) -> list[RunSpec]:
    """配置按出现顺序编号;第 seed 轮里奇数轮正序、偶数轮倒序。"""
    configs = list(dict.fromkeys(replace(s, seed=0) for s in specs))
    by_seed: dict[int, dict] = {}
    for s in specs:
        by_seed.setdefault(s.seed, {})[replace(s, seed=0)] = s
    out = []
    for i, seed in enumerate(sorted(by_seed)):
        keys = configs if i % 2 == 0 else configs[::-1]
        out += [by_seed[seed][k] for k in keys if k in by_seed[seed]]
    return out


def build_plan(cfg: dict) -> list[Stage]:
    lam = cfg["lambda_by_factor"]
    lam_e1 = lam[cfg["e1_factor"]]
    seeds = range(1, cfg["reps"] + 1)

    def spec(series="S", c=4, lam_=lam_e1, cp=True, delay=None, seed=1, np_=None, n=cfg["n_tasks"], pilot=False):
        return RunSpec(series, np_ or c if series == "S" else np_, cfg["ctx"], c, lam_, cp, delay, seed, n, pilot)

    def pilot_of(s: RunSpec) -> RunSpec:
        return replace(s, seed=1, n_tasks=cfg["pilot_n_tasks"], pilot=True)

    experiments = [
        ("E1-S", [spec(c=c, seed=i) for i in seeds for c in cfg["e1_concurrency"]],
         spec(c=4)),
        ("E4", [spec(c=c, cp=cp, seed=i) for i in seeds for c in cfg["e4_concurrency"] for cp in (True, False)],
         spec(c=cfg["e4_concurrency"][0], cp=False)),
        ("E5", [spec(c=cfg["e5_concurrency"], lam_=l, seed=i) for i in seeds for l in lam.values()],
         spec(c=cfg["e5_concurrency"], lam_=max(lam.values()))),
        ("E3", [spec(c=cfg["e3_concurrency"], delay=d, seed=i) for i in seeds for d in cfg["e3_delays_s"]],
         spec(c=cfg["e3_concurrency"], delay=max(cfg["e3_delays_s"]))),
        ("E1-U", [spec("U", c=c, np_=cfg["e1u_np"], seed=i) for i in seeds for c in cfg["e1u_concurrency"]],
         spec("U", c=cfg["e1u_concurrency"][0], np_=cfg["e1u_np"])),
    ]
    seen: set[str] = set()
    stages = []
    for name, wanted, pilot in experiments:
        wanted = list(dict.fromkeys(wanted))
        runs = [s for s in wanted if s.name not in seen]
        reused = [s.name for s in wanted if s.name in seen]
        seen |= {s.name for s in runs}
        stages.append(Stage(name, pilot_of(pilot), _order(runs), reused))
    return stages
