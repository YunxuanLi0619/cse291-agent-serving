# Memory-Aware Agent Edge Serving — Phase 1 baseline

术语见 `GLOSSARY.md`,决策见 `docs/adr/`。环境:`uv venv .venv --python 3.12 && uv pip install -r requirements.lock.txt`。

## 测试

```bash
.venv/bin/python -m pytest          # 全部测试(不需要真的 llama-server)
```

## 流水线(当前已有:Workload 生成 + runner:FIFO、并发上限、开环 Poisson、超时/失败落盘)

```bash
scripts/start_server.sh 4 16384 --no-kv-unified    # 终端 A:启动 llama-server(参数集中在脚本里)
scripts/wait_health.sh                              # 轮询 /health,成功时打印冷启动等待秒数

# 生成 Workload(token 长度用运行中的 llama-server 校准并自检 ±5%);同种子两次输出字节一致
python -m src.runner.gen_workload --mix configs/mix_dev.json --t-solo configs/t_solo.json \
  --seed 1 --n-tasks 100 --out workloads/dev/default_s1.json

# 执行 Workload:并发上限 c=4、到达率 λ=0.05 个/秒、FIFO,写 requests.jsonl 与 tasks.jsonl
# (启动前校验 /slots 条数 ≥ c;--timeout 覆盖请求级超时,默认 120 s;--n-tasks 只提交前 N 个)
python -m src.runner.run --workload workloads/dev/default_s1.json --c 4 --lambda 0.05 \
  --policy fifo --out results/raw/try1
```

- `configs/mix_dev.json` 与 `mix_eval.json`:种子范围(1–5 / 101–105)与混合比例各不相同;种子不在所属范围内会拒绝生成。
- `configs/t_solo.json`:T_solo 表(输入配置,不写死在代码里);`deadline_s = 5 × T_solo`,agent 按自己的步数与 Tool delay 计算。
- Workload 是 JSON(`sort_keys`,稳定字节),每个 Task 的 `steps` 已是完整的用户消息、`max_tokens` 和 Step 后的 Tool delay。
