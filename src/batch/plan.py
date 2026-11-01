"""Batch matrix (#9): expand E1-E5 into an ordered list of Runs. Pure logic, no files, processes or network.

- Priority (manual Step 9): E1-series S -> E4 -> E5 -> E3 -> E1-series U; E2 is sliced from the E1 data by Step and is not run separately.
- Repeat i uses seed i (Workload `default_s<i>.json`), so different configurations with the same seed see the same Task sequence and can be paired.
- E4's cache_prompt=on and E5's lambda=0.9*mu are Runs of E1-S: not rerun, only recorded in `reused`.
- Within a Stage, order by "repeat round", and reverse the configuration order every other round: drift such as throttling is spread across configurations, and every configuration still has data if the batch is forced to stop midway.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RunSpec:
    series: str  # "S": np = c, per-slot capacity B/c; "U": shared pool -kvu
    np: int
    ctx: int  # B, total number of KV tokens
    c: int
    lam: float
    cache_prompt: bool
    tool_delay_s: float | None  # None = the random Tool delay that ships with the Workload
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
        """Rough wall clock: arrival period n/lambda + about 150 s for draining and restart (only used to report an ETA in the log, never in any decision)."""
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
    runs: list[RunSpec]  # only the Runs added by this Stage
    reused: list[str]  # names of reused Runs that already exist in other Stages


def _order(specs: list[RunSpec]) -> list[RunSpec]:
    """Configurations are numbered in order of appearance; in round `seed`, odd rounds go forward and even rounds backward."""
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
        return RunSpec(series, c if series == "S" else np_, cfg["ctx"], c, lam_, cp, delay, seed, n, pilot)

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
