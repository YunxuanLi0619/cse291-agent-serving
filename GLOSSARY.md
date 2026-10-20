# Memory-Aware Agent Edge Serving

在内存受限的机器上,用 llama.cpp 服务多步 agent 任务;度量并发、上下文增长和步间空闲如何影响延迟、内存和失败,并为 Phase 2 的内存感知调度提供基线。

## Language

### Work

**Task**:
一个完整的工作单元,由一个或多个 Step 组成,属于四种 Task type 之一。
_Avoid_: job, session

**Task type**:
Task 的类别:`short`、`longctx`、`longgen`、`agent`,决定初始上下文长度、输出长度和步数。
_Avoid_: category, class

**Step**:
Task 的一次模型调用,加上随后的 Tool delay(最后一步没有 delay)。FIFO 排队的单位。
_Avoid_: turn, round, stage

**Request**:
Step 里发往 llama-server 的那次 HTTP 调用;日志按 Request 记录,领域讨论里用 Step。
_Avoid_: call, query

**Tool delay**:
agent Task 的相邻两个 Step 之间受控的等待时间,模拟工具执行;期间不占并发名额。
_Avoid_: think time, sleep

**Needle**:
埋在 Task 第一步上下文里的随机 key,最后一步要求模型复述;程序据此判定 quality pass。
_Avoid_: canary, probe

**Quality pass**:
Task 最终输出包含正确 Needle。
_Avoid_: correctness, accuracy

### Load

**Workload**:
由固定种子生成的一份纯数据文件:Task 列表、λ=1 的单位到达间隔、tool 文本、Needle、deadline。同种子字节一致。
_Avoid_: dataset, trace

**Dev workload / Eval workload**:
Dev 用来设计和调参(种子 1–5),Eval 用来最终比较(种子 101–105),两者混合比例不同。
_Avoid_: train/test set

**Arrival rate (λ)**:
开环 Poisson 过程下每秒到达的 Task 数;runner 用 `单位间隔 / λ` 缩放。
_Avoid_: load, QPS

**Capacity (μ)**:
默认混合下系统能持续完成的 Task 速率估计(≈ 有效并行度 / 平均服务时间)。λ 以 μ 的倍数表示。
_Avoid_: max throughput

**Warm-up task**:
每次运行最先到达的 3 个 Task;原始日志保留,稳态统计排除。
_Avoid_: burn-in

### Memory

**Slot**:
llama-server 内一个并行序列的上下文位置,持有自己的 KV。
_Avoid_: lane, worker

**Memory budget (B)**:
服务启动时的总 KV token 数(`-c`);字节 = B × 114,688。
_Avoid_: memory limit, RAM budget

**KV utilization**:
运行中被在飞 Request 占用的 token 数除以 B。
_Avoid_: memory usage

**Retained context**:
Step 完成后仍留在某个 Slot 里的 KV 前缀,可能被续步复用,也可能被别的 Request 占走。
_Avoid_: cache, session state

**Prefix reuse**:
Request 命中 Slot 内已有前缀,只需处理其余 token(`cached_tokens` > 0)。
_Avoid_: cache hit

**Series S**:
固定 B、令 np = c,每个 Slot 的容量 = B / c 的实验系列。
_Avoid_: split config

**Series U**:
所有 Slot 共享一个容量为 B 的 KV 池(`-kvu`)的实验系列。
_Avoid_: shared config

### Scheduling

**Concurrency (c)**:
runner 允许同时在飞的 Request 数上限;不变量 c ≤ np。
_Avoid_: parallelism, workers

**FIFO queue**:
runner 持有的、按"Step 就绪时刻"先进先出的队列。
_Avoid_: backlog

**Scheduling policy**:
决定下一个派发哪个就绪 Step 的规则;FIFO 是第一个实现,Phase 2 的内存感知策略是另一个。
_Avoid_: scheduler(scheduler 指持有队列和并发上限的整体,Policy 只是其中的决策)

**Fixed-concurrency baseline**:
在 Dev workload 上选出的、吞吐或 goodput 最高且 p99 可接受的 c;Phase 2 必须打败的调优基线。
_Avoid_: best config

### Measurement

**T_solo**:
某 Task type 在 c=1、无竞争下的完成时间。
_Avoid_: baseline latency

**Deadline**:
Task 的完成期限 = 5 × T_solo(agent 按自己的步数和 delay 计算),在 Dev workload 上冻结。
_Avoid_: SLO, SLA

**Goodput**:
在 deadline 内完成且 quality pass 的 Task 数除以总时长。
_Avoid_: throughput(throughput 不要求 deadline 和 quality pass)

**Run**:
一次固定配置(服务参数、c、λ、workload、seed)的完整实验,产出三类日志和一个 manifest。
_Avoid_: trial, experiment(Experiment 指 E1–E5 这类系列)
