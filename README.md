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

# 分析:一条命令从原始日志目录(递归查找已完成的 Run 目录)重新生成全部 CSV 和图
python -m src.analysis --raw results/raw/e1 --out results/analysis
```

- 分析输出:`runs.csv`(每个 Run 一行);每张图 = 同名 `.csv` + `.png`,报告引用 CSV:`perf_by_{c,lambda,tool_delay_s}_<platform>`(吞吐/goodput、p50/p99、排队、失败/超时率,按其余配置分线,误差条 = 重复间样本标准差)、`paired_cache_prompt_<platform>`(同 seed 内 on − off)、`step_slice_<platform>`(agent 第 k 步)、`memory_peak_<platform>`、`kv_util_<platform>`。不同平台各出一套,不混画。
- 指标口径写在 `src/analysis/metrics.py` 顶部:稳态排除 `warmup=true`;p50/p99 只算 ok Task;失败/超时进分母;吞吐和 goodput 除以稳态窗口(最早到达 → 最晚完成)。
- 分析从 manifest 读配置:`c`、`lambda`、`seed`、`server_cmdline`(解析 `-np`、`-c`、系列 S/U)。分组还用到可选字段 `tool_delay_s`、`cache_prompt`,由批跑脚本写入 manifest;缺省为空,不影响其他分析。manifest 没有 `t_end` 的 Run 视为未完成,跳过。

- `configs/mix_dev.json` 与 `mix_eval.json`:种子范围(1–5 / 101–105)与混合比例各不相同;种子不在所属范围内会拒绝生成。
- `configs/t_solo.json`:T_solo 表(输入配置,不写死在代码里);`deadline_s = 5 × T_solo`,agent 按自己的步数与 Tool delay 计算。
- Workload 是 JSON(`sort_keys`,稳定字节),每个 Task 的 `steps` 已是完整的用户消息、`max_tokens` 和 Step 后的 Tool delay。
