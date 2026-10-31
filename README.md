# Memory-Aware Agent Edge Serving — Phase 1 baseline

术语见 `GLOSSARY.md`,决策见 `docs/adr/`。环境:`uv venv .venv --python 3.12 && uv pip install -r requirements.lock.txt`。

## 测试

```bash
.venv/bin/python -m pytest          # 全部测试(不需要真的 llama-server)
```

## 流水线(当前已有:Workload 生成 + runner:FIFO、并发上限、开环 Poisson、超时/失败落盘、系统采样器与 manifest + 分析脚本)

```bash
scripts/start_server.sh 4 16384 --no-kv-unified    # 终端 A:启动 llama-server(参数集中在脚本里)
scripts/wait_health.sh                              # 轮询 /health,成功时打印冷启动等待秒数

# 生成 Workload(token 长度用运行中的 llama-server 校准并自检 ±5%);同种子两次输出字节一致
python -m src.runner.gen_workload --mix configs/mix_dev.json --t-solo configs/t_solo.json \
  --seed 1 --n-tasks 100 --out workloads/dev/default_s1.json

# 执行 Workload:并发上限 c=4、到达率 λ=0.05 个/秒、FIFO,写 requests.jsonl、tasks.jsonl、system.jsonl(每 0.5 s 采样)与 manifest.json
# (--server-pid 给 llama-server 进程号才能采 footprint/RSS 并记录启动命令行;不给则对应字段为 null 且写入 error)
# (启动前校验 /slots 条数 ≥ c;--timeout 覆盖请求级超时,默认 120 s;--n-tasks 只提交前 N 个)
python -m src.runner.run --workload workloads/dev/default_s1.json --c 4 --lambda 0.05 \
  --policy fifo --out results/raw/try1 --server-pid $(pgrep -f llama-server)

# OAI 接口的响应没有 id_slot:把 server 日志里每个请求的 slot 选择(LRU / LCP 及相似度)按顺序对回 requests.jsonl,
# 用来解释 agent 续步的 cached_tokens(--skip = 日志里本 Run 之前已有的 launch 数,日志被别的请求共用时必填)
python -m src.analysis.slot_trace --run results/raw/try1 --server-log server.log --skip 0

# 分析:一条命令从原始日志目录(递归查找已完成的 Run 目录)重新生成全部 CSV 和图
python -m src.analysis --raw results/raw/e1 --out results/analysis
```

## 批跑 E1–E5(#9)

```bash
.venv/bin/python scripts/run_batch.py --dry-run               # 计划与估计时长(约 19 h),不要求插电
.venv/bin/python scripts/run_batch.py --pilots-only           # 只做各实验的 30 Task 试跑
caffeinate -is scripts/run_batch.sh                           # 正式批跑;中断后原样重跑,从下一个 Run 继续
```

- 矩阵与排序在 `src/batch/plan.py`(参数在 `configs/batch.json`):E1-S(c=1/2/4/8,λ=0.9μ)→ E4(cache_prompt 关,c=1/4)→ E5(λ=0.3/0.6/1.2μ,c=4)→ E3(Tool delay 0/0.5/1/3 s,c=4)→ E1-U(`-kvu`,np=8,c=4/8);每个配置 3 次,重复 i 用 seed i;E4 的「开」和 E5 的 0.9μ 复用 E1-S 的 Run,E2 从 E1 数据按 Step 切片。
- 每个实验先用 30 Task 试跑,过了才放开全量;试跑没过的实验记入 `results/batch/state.json` 的 `blocked_stages`,其余实验继续。
- 每个 Run 前重启服务、记录冷启动秒数(manifest 的 `cold_start_s`);前 3 个 Task 标 `warmup`。每批开始前把电源、低电量模式和 `pmset -g therm` 写进 `results/batch/batch_start_*.json`,未插电则拒绝启动(退出码 3);批跑中途被拔电会暂停等待。
- 通过校验的 Run 在 `results/raw/batch/<run>/`(存在 `batch_meta.json` 即视为完成);没过校验的尝试挪到 `results/batch/invalid/` 并重试(最多 3 次)。降频只记录(`batch_meta.json` 的 `throttled`),不拒绝。
- 结束时写 `results/batch/summary.json`、`baseline_c.json`(Dev workload 上的 Fixed-concurrency baseline,规则见 `src/batch/baseline.py`)并跑分析到 `results/analysis/batch`。

- 分析输出:`runs.csv`(每个 Run 一行);每张图 = 同名 `.csv` + `.png`,报告引用 CSV:`perf_by_{c,lambda,tool_delay_s}_<platform>`(吞吐/goodput、p50/p99、排队、失败/超时率,按其余配置分线,误差条 = 重复间样本标准差)、`paired_cache_prompt_<platform>`(同 seed 内 on − off)、`step_slice_<platform>`(agent 第 k 步)、`memory_peak_<platform>`、`kv_util_<platform>`。不同平台各出一套,不混画。
- 指标口径写在 `src/analysis/metrics.py` 顶部:稳态排除 `warmup=true`;p50/p99 只算 ok Task;失败/超时进分母;吞吐和 goodput 除以稳态窗口(最早到达 → 最晚完成)。
- 分析从 manifest 读配置:`c`、`lambda`、`seed`、`server_cmdline`(解析 `-np`、`-c`、系列 S/U)。分组还用到可选字段 `tool_delay_s`、`cache_prompt`,由批跑脚本写入 manifest;缺省为空,不影响其他分析。manifest 没有 `t_end` 的 Run 视为未完成,跳过。

- `configs/mix_dev.json` 与 `mix_eval.json`:种子范围(1–5 / 101–105)与混合比例各不相同;种子不在所属范围内会拒绝生成。
- `configs/t_solo.json`:T_solo 表(输入配置,不写死在代码里);`deadline_s = 5 × T_solo`,agent 按自己的步数与 Tool delay 计算。
- Workload 是 JSON(`sort_keys`,稳定字节),每个 Task 的 `steps` 已是完整的用户消息、`max_tokens` 和 Step 后的 Tool delay。
