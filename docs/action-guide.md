# CSE 291A 项目行动指南:Memory-Aware Agent Edge Serving

> 生成日期:2026-10-19(周一)。依据:`Memory Aware Agent Edge Serving.pdf`、`CSE 291A Fall 2026 Course Projects.pdf`,以及 llama.cpp `llama-server` 当前文档。
> 课程文档标注为 **TENTATIVE**,细节以课程 Piazza/公告为准。

---

## 0. 一页纸总览

**核心问题(原文):** 在内存受限的机器上,admission(准入)和 scheduling(调度)决策如何在保持内存占用有界的前提下,提高"完成的 agent 工作量"?

**关键日期**

| 节点 | 日期 | 距今 |
|---|---|---|
| Phase 1:实现 + 测量 + Phase 1 writeup | **周三 2026-10-28** | 9 天 |
| Phase 2:实现 + 评估 + 最终报告 | **周四 2026-12-10** | 52 天 |

**资源:** 每人 $50 AWS credits(组队可合并使用);LLM API 用 TritonAI(本项目跑本地模型,基本用不到);可用 GPU 实例清单在课程 "Useful Resources" 标签页(我看不到,需要你自己确认)。可用笔记本做开发。组队 1–5 人,**全组同分**,不要求个人贡献说明。

**评分维度(课程文档原文归纳)**

1. Baseline 复现与测量质量
2. Phase 2 设计的合理性(要由测量数据推出来)和实现质量
3. 实验控制与对比质量
4. 分析深度
5. 可复现性与报告清晰度

**一句话策略:** 题目原文说"实现很多策略不如把一个策略理解透"。所以走 **一个聚焦机制 + 调优过的简单基线 + 严格对照 + 诚实报告(包括没收益的情形)** 这条路。

---

## 1. 先读懂题目:交付物逐条拆解

### Phase 1(10/28 截止)

| 要求 | 具体要交什么 |
|---|---|
| FIFO 队列 + 固定并发 | 一个自己控制的 scheduler(不是直接用 server 自带排队) |
| 混合负载:短/长上下文、短/长输出、多步任务(受控 tool delay) | 一份**负载定义文件**(YAML/JSON + 固定随机种子) |
| 同时 instrument **单次 model request** 和**完整 agent task** | 两级日志:request-level、task-level |
| 指标:task completion time、queueing time、tail latency、throughput、peak memory、failures/timeouts | 原始 JSONL 日志 + 分析脚本 |
| 搞清:并发、上下文增长、步骤间空闲期如何影响内存压力和完成量 | 3 组以上的扫描实验(见 §4) |
| 判断 retained context / reusable prefix 是否实质影响所选 runtime | prefix cache on/off 对照 |
| **可复现 baseline + 性能曲线 + 主导瓶颈的解释** | writeup 里一个明确的"瓶颈结论" |
| Phase 2 方向 | writeup 末尾给出**有数据支撑**的 Phase 2 方案 |

### Phase 2(12/10 截止)

- 选**一个**聚焦机制,对比 **FIFO** 和 **在开发负载上选出的最佳固定并发基线**。
- 若做 cache-aware 设计,还必须对比一个**简单的固定 cache 策略**。
- 在**变化的到达率和负载混合**下评估,**必须包含策略收益很小的情形**。
- 检查增益不是来自:跳过任务、缩短输出、降低任务质量。
- 计入策略自身开销;检验长任务是否被不公平地拖延。
- 提交:scheduler、workload runner、原始测量、分析脚本、报告。

### 课程层面的硬性规则(容易丢分的地方)

- **开发负载 vs 评估负载分开**:用开发负载设计/调参,用另一组评估负载打分。
- **固定版本**:model、runtime(llama.cpp commit)、dataset、代码版本都要 pin。
- 计入 **setup / 编译时间**(如果影响使用场景);区分 **cold-start** 与 **steady-state**。
- **失败和超时要记录,不能悄悄剔除**。
- 随机实验要**重复**并**报告方差**。
- 声称"更好"必须有受控实验;"证明简单基线已足够"也是合格结论。

---

## 2. 研究方向建议

### 2.1 先建立对问题的正确心智模型

在 llama.cpp 这类 runtime 里,"内存"主要由两部分构成:

- **模型权重**:常量,与并发无关。
- **KV cache**:随"同时驻留的 token 数"线性增长。每个 token 的 KV 字节数约为

  `2 × n_layer × n_kv_head × head_dim × bytes_per_elem`(f16 时 bytes=2)

  这条公式让你可以在**请求到达前就估算内存**,是 admission 策略的基础。

Agent 任务和普通请求的区别在于:**步骤之间有 tool delay**。这段空闲期里上下文有两种命运:

- 继续占着 slot / KV → 内存被"闲置的上下文"吃掉,新任务进不来;
- 被别的请求覆盖 → 下一步要重新 prefill,浪费算力,延迟变长。

这个 **"占着不用 vs. 释放后重算"的权衡**就是本题的核心张力,Phase 1 要用数据把它量出来。

### 2.2 一个必须先回答的风险:内存真的是瓶颈吗?

在小型 CPU 机器上,decode 往往先撞上**算力/内存带宽**,吞吐在并发 2–4 时就饱和,再加并发只是拉长延迟。这时"内存压力"不一定出现。

处理办法(题目也允许):**主动制造内存约束**,例如

- 选内存小的实例,或用 cgroup / Docker `--memory` / `systemd-run -p MemoryMax=` 限制可用内存(OOM kill 记为 failure);
- 或设定一个 **KV token 预算 B**(由 `--ctx-size` 和 slot 数决定),把它当作"内存预算"并在报告中说明口径。

Phase 1 的"主导瓶颈"结论可以是"在预算 X 下是 KV 容量"或"在预算 Y 下是算力",两种都合格,**关键是用曲线证明**。

### 2.3 Phase 2 候选机制(按推荐度排序)

| 优先级 | 机制 | 思路 | 难度 | 风险 |
|---|---|---|---|---|
| **★ 推荐** | **KV 预算准入 + 续步优先** | 用估算的 KV token 数做 admission(`已承诺 + 新任务估算 ≤ B` 才放行);已在执行的任务的"下一步"优先于新任务(finish-what-you-start),并加 aging 防长任务饥饿 | 中 | 估算不准;需要证明不是靠拒绝任务取胜 |
| 备选 A | 自适应并发(AIMD) | 根据观测的压力信号(`requests_deferred`、RSS、延迟)增减并发 | 低–中 | 信号噪声大;容易被"调优过的固定并发"打平 |
| 备选 B | Cache-aware 保留策略 | tool delay 期间决定保留还是释放某任务的前缀 | 高 | 依赖 runtime 暴露的控制项;需额外对比固定 cache 策略 |
| 不推荐 | 同时做多个 | 题目明确说多个不如一个深 | — | 分析会变浅 |

**我的建议:** 主线做"KV 预算准入 + 续步优先"。这是一个**可以拆成两个消融**(只做预算 / 只做续步优先 / 两者都做)的机制,天然提供"深度分析"的材料,而且完全在你自己的 scheduler 里实现,**不依赖 runtime 暴露什么控制项**,风险最低。Cache-aware 只在 Phase 1 数据显示"前缀复用收益显著"时才作为扩展。

> 做机制选择的最终依据必须是 Phase 1 的数据(课程要求"justify the policy from measurements")。上表是先验,10/28 的 writeup 里要用你的曲线确认或推翻它。

---

## 3. 技术方案(llama.cpp 要点)

以下来自当前 `llama-server` 文档(通过 Context7 查询),**请在你 pin 的 commit 上再核对一遍**,flag 在版本间会变。

### 3.1 启动与并发

```bash
llama-server -m model.gguf --host 127.0.0.1 --port 8080 \
  -np 4 -c 16384 --metrics
```

- `-np, --parallel N`:slot 数,即 server 端并发上限。默认 `-1` = auto(此时 `n_parallel=4` 且强制 `kv_unified=true`)。
- `-c, --ctx-size`:总上下文。**非 unified KV 时按 slot 均分**(16384/4 = 4096/slot,历史行为,需验证);unified 时所有序列共享一块 KV buffer。**务必用 `GET /slots` 里的 `n_ctx` 字段确认每个 slot 实际拿到多少**,并把 `-kvu / --no-kv-unified` 显式写进实验配置,别依赖默认值。
- `-cb, --cont-batching`:默认开启(continuous batching)。
- 单个 batch 由所有活跃 slot 共享,`llama_decode` 是主要算力瓶颈。

### 3.2 缓存控制(只用 runtime 真正暴露的)

| 控制项 | 作用 |
|---|---|
| `--cache-prompt` / `--no-cache-prompt`(默认开) | 请求间复用公共前缀 |
| `--cache-reuse N` | 通过 KV shifting 复用的最小 chunk 大小(默认 0) |
| `-cram, --cache-ram N` | prompt cache 的最大 MiB(默认 8192) |
| `--cache-idle-slots` | 新任务到来时把空闲 slot 存入 prompt cache(默认开,依赖 cache-ram) |
| `--slot-save-path` + `POST /slots/{id}?action=save\|restore` | 把某 slot 的 KV 存盘/恢复,返回 `n_restored`、`restore_ms` |

题目原文:*"Any cache-aware extension must use cache controls and measurements that the runtime actually exposes."* 所以 cache-aware 方案只能建立在上表这些控制项之上。

### 3.3 可观测性

- `GET /metrics`(需 `--metrics`):`llamacpp:requests_processing`、`requests_deferred`、`n_tokens_max`、`n_busy_slots_per_decode`、`prompt_tokens_seconds`、`predicted_tokens_seconds` 等。
- `GET /slots`:每个 slot 的 `is_processing`、`n_ctx`、`id_task` 等。
- **文档里没有直接的"KV 占用字节数"指标** → 峰值内存要靠**外部采样 server 进程 RSS**(`psutil`,100 ms–1 s 一次)。
  ⚠️ **注意:** llama.cpp 很可能在启动时就一次性分配并清零整块 KV,此时 RSS 基本是平的(由 `-np × 每 slot 上下文` 决定,而不是由负载决定)。这个假设要在 `docs/phase1-steps.md` 的 Step 5 里实测验证;如果成立,就要同时记录 runner 统计的"在飞 token 数 / KV 容量"来表示内存压力,不能只看 RSS 增量。
- 每个响应里的 `timings` 字段(`prompt_n`、`cache_n` 等,我没在文档里核实,请自行确认)可用来计算"前缀命中了多少 token"。

### 3.4 模型与机器选择

- 模型和精度**全程固定**。选一个在所选机器上"舒服放得下"的小模型(如 1B–3B 级 instruct,Q4_K_M 量化),再通过并发/上下文来制造压力。
- **你的笔记本是 macOS(Darwin)**:Apple Silicon 是统一内存,很难做硬性内存上限,且 Metal 与 AWS CPU 行为差别大。建议 **开发在笔记本、最终测量在 AWS 小型 CPU 实例**,并用 cgroup 限内存。具体可用实例以课程 Useful Resources 为准。
- **花钱纪律:** 先在本地把脚本调通,再开云实例跑批;跑完立即 stop。把"AWS 上跑完整扫描一次需要多少小时 × 单价"算出来,确保能重跑 ≥3 次。

---

## 4. Phase 1 行动计划(10/19 → 10/28)

### Day 1–2(10/19–10/20):环境与"能跑"

- [ ] **读两份课程提供的样例报告**(高分/低分各一份,链接在课程 PDF 第 2 页)。这是最快搞清评分口径的方式。
- [ ] 确认组队情况(1–5 人)并在课程渠道确认选题。
- [ ] 拉取并**记录 llama.cpp commit hash**,编译并**记录编译耗时**;下载模型并记录 SHA256。
- [ ] 本地起 `llama-server --metrics`,手动 curl 通 `/v1/chat/completions`、`/metrics`、`/slots`。
- [ ] 在笔记本上确认 RSS 采样脚本可用。

### Day 3–5(10/21–10/23):workload 与 runner

**负载设计(开发集):**

| 任务类型 | 上下文 | 输出 | 步数 | 作用 |
|---|---|---|---|---|
| Short | ~256 tok | ~64 tok | 1 | 对照 |
| LongCtx | 3–4k tok | 短 | 1 | 测 prefill/KV 压力 |
| LongGen | 短 | ~512 tok | 1 | 测 decode 占用时间 |
| Agent | 起始 ~1k tok,每步追加上步输出 + tool 结果 | 中 | 3–6 | 测上下文增长 + 空闲期 |

关键设计选择:

- **开环到达(Poisson,速率 λ)**,而不是闭环。开环才能测出排队时间和尾延迟,并扫出"拐点"。
- **tool delay 用受控 `sleep`**(如 0.2–3 s 的固定分布 + 种子),不调用真实工具。
- **输出长度要受控**:固定 `max_tokens`、`temperature=0`,必要时用 `ignore_eos` 让长度可复现(并在文档中说明)。
- **给任务加可程序化校验的"质量检查"**:如 needle-in-context(在上下文里埋一个 key,最终步要求复述)。这样 Phase 2 才能证明"增益不是靠降低质量得来的"。**现在就设计好,别等 Phase 2 才补。**
- 种子固定,负载文件进版本控制。

**Runner / 日志:**

- request-level:`task_id, step, t_submit, t_start, t_first_token, t_end, prompt_tokens, cached_tokens, output_tokens, status`
- task-level:`task_id, type, t_arrive, t_first_dispatch, t_done, queue_time_total, n_steps, status(ok/timeout/oom/error), quality_pass`
- system-level:`t, rss_mb, requests_processing, requests_deferred, n_busy_slots_per_decode`
- 所有失败/超时必须落盘,分析脚本的分母里要包含它们。
- scheduler 是自己写的 FIFO + 固定并发上限(例如一个 semaphore),不要依赖 server 的排队。

### Day 6–8(10/24–10/26):实验

| 实验 | 变量 | 回答的问题 |
|---|---|---|
| E1 并发扫描 | 并发 c ∈ {1,2,4,8,…} | 吞吐拐点在哪?p50/p99 与峰值内存如何随 c 变化?何时开始失败? |
| E2 上下文增长 | 起始上下文/步数 | 上下文增长如何推高内存与延迟? |
| E3 空闲期 | tool delay 大小 | 步骤间空闲是否让 slot 被"闲置的上下文"占用,导致排队? |
| E4 前缀复用 | `--cache-prompt` 开/关,共享/不共享 system prompt | 前缀复用是否**实质**影响该 runtime?省了多少 prefill? |
| E5 到达率 | λ 扫描 | 找到系统饱和点,为 Phase 2 选工作区间 |

- 每个配置**重复 ≥3 次**,报告均值与方差/区间;
- **cold start 与 steady-state 分开报**(丢弃或单列预热段);
- 在 E1 里选出"开发集上的**最佳固定并发**",这是 Phase 2 必须打败的基线。

### Day 9(10/27–10/28):writeup 与打包

**Phase 1 writeup 结构(建议 3–5 页):**

1. 研究问题与环境(机器、模型、commit、flag,全部 pin 住)
2. Workload 定义与指标口径
3. 测量方法与 instrumentation
4. 结果:性能曲线(c vs 吞吐/p99/峰值内存/失败率)
5. **主导瓶颈结论**(一句话 + 证据曲线)
6. Prefix/retained context 的影响
7. **Phase 2 方向与理由**(必须引用上面的数据)
8. 局限性

**提交包:** 源码、负载定义、原始日志、分析脚本、**一条命令复现主要图的说明**、report。

---

## 5. Phase 2 行动计划(10/29 → 12/10)

| 时间 | 任务 |
|---|---|
| 10/29–11/05 | 根据 Phase 1 反馈和数据**冻结设计**;写下策略的伪代码、估算方法、超参及调参协议 |
| 11/06–11/19 | 实现 scheduler;单元测试(估算、admission、aging);在开发集上调参 |
| 11/20–12/01 | **冻结参数后**在**独立评估集**上跑全部对比(避开感恩节 11/26 当周的集中冲刺) |
| 12/02–12/08 | 消融、公平性、开销分析;出图;写报告 |
| 12/09 | 复现演练(干净环境按 README 重跑一遍) |
| 12/10 | 提交 |

### 5.1 推荐策略的具体设计(可按 Phase 1 数据修改)

```
状态: committed_tokens = Σ 在飞任务的 KV token 估算
B = KV token 预算(由内存限制/ctx 推出)

on 新任务到达:
    est = estimate(task)           # 按任务类型 + 历史滑动均值
    if committed + est ≤ B: admit  else: 入等待队列
on 某任务下一步就绪:
    入"续步队列"                   # 优先于新任务队列
dispatch:
    先取续步队列(带 aging 上限),再取新任务队列
on 任务结束: committed -= 实际占用
```

- 估算器要报告**误差**(估算 vs 实际 token),这是"justify from measurements"的一部分。
- 可选扩展:用 RSS 反馈在线校正 B。

### 5.2 对比矩阵(课程强制要求)

| 配置 | 说明 |
|---|---|
| FIFO | Phase 1 baseline |
| Best-fixed-concurrency | 在**开发集**上调出的最优 c(调优过的简单基线) |
| 提议策略 | 冻结参数后上评估集 |
| 消融 | 只预算 / 只续步优先 / 两者 |
| (若做 cache-aware)简单固定 cache 策略 | 如"始终释放"/"始终保留"/固定 TTL |

**变化维度:** 到达率 λ(低/中/高/过载)× 负载混合(偏短、偏长、偏多步、均匀),**必须包含无收益区**(低 λ、同质短任务、内存宽松),并如实报告。

### 5.3 防"作弊"检查清单(课程明确要求)

- [ ] 各策略**完成任务数**相同或明确报告差异;超时/被拒任务计入分母
- [ ] 每步**输出 token 数**与基线一致(分布对比)
- [ ] **quality_pass 率**不下降
- [ ] 主指标建议用 **goodput = 在 deadline 内完成且通过校验的任务数 / 秒**,避免只比平均延迟
- [ ] **策略开销**:对比"空策略直通"的 scheduler CPU 时间和每次决策耗时
- [ ] **长任务公平性**:按任务类型分别报 p50/p99 和 slowdown(完成时间 / 无竞争时间),看最长任务是否被饿死
- [ ] 评估集与开发集**种子、混合比例均不同**

---

## 6. 报告结构(最终,建议)

1. 问题与动机
2. 系统与环境(全部版本 pin)
3. 负载与指标
4. Phase 1:baseline 与瓶颈
5. Phase 2:策略设计与理由(引用 Phase 1)
6. 实验设置(控制变量、重复次数、冷/热)
7. 结果:主对比 → 消融 → 公平性 → 开销 → **无收益区**
8. 失败/超时的统计
9. 局限性与威胁有效性
10. 复现说明

评分看重"**把观察到的行为和内存管理决策连起来**"。每张图都要回答"这说明了内存的什么行为,所以策略为什么这样设计"。

---

## 7. 建议的仓库结构

```
/
├── CLAUDE.md
├── docs/ (agents/, action-guide.md)
├── workloads/      # 负载定义 + 种子(dev / eval 分开)
├── src/
│   ├── scheduler/  # FIFO、固定并发、提议策略
│   ├── runner/     # 开环到达、agent 任务执行、tool delay 模拟
│   └── monitor/    # RSS 与 /metrics 采样
├── experiments/    # 每个实验一个配置文件
├── results/raw/    # 原始 JSONL(不要手改)
├── analysis/       # 从 raw 生成所有图表的脚本
├── report/
└── README.md       # 一条命令复现
```

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 内存不是瓶颈,策略无收益 | 主动施加内存预算(§2.2);"在何种条件下无收益"本身就是合格结论 |
| 基线被"调优过的固定并发"打平 | 这是课程允许的结论;用消融和 λ 扫描说明边界 |
| 实验噪声大 | 固定种子、重复 ≥3 次、报告区间;云上不同实例类型不混用 |
| AWS 预算耗尽 | 本地调通再上云;跑完立刻 stop;提前估算每轮成本 |
| 输出长度不可控导致对比失真 | 固定 `max_tokens`、`temperature=0`,必要时 `ignore_eos` |
| flag/行为随 llama.cpp 版本变化 | pin commit;实验配置显式写出所有 flag(尤其 `-np`、`-c`、`-kvu`、`--cache-*`) |
| 到 Phase 2 才发现缺质量校验 | 在 Phase 1 就把 needle 式校验写进 workload |
| 多机制并行导致分析变浅 | 坚持一个主机制 + 消融 |

---

## 9. 需要你尽快确认/决定的事

1. **组队**:几个人?(影响 credits 总量和分工;全组同分。)
2. **机器**:笔记本型号(Apple Silicon?内存多大?)与 Useful Resources 里的可用 AWS 实例。
3. **最终测量放在哪里**:建议 AWS 小型 CPU 实例 + 内存限制;请向助教确认笔记本结果是否可作为最终证据。
4. **模型选择**:定一个并固定到结束。
5. 课程文档是 **TENTATIVE**,留意公告是否更改选项编号或截止日期(文档里选项名为 "Memory-Aware Agent Serving",你的选题文件为 "Option 1")。

---

## 10. 未能核实的内容(诚实声明)

- 课程的 **Useful Resources 标签页**、样例报告(Google Drive 链接)我没有访问,AWS 实例清单与评分细节请以它们为准。
- `llama-server` 的 `timings.cache_n` 字段、请求里的 `id_slot` 参数、非 unified 模式下 ctx 均分的具体行为,**未在刚才的文档查询中直接确认**,请在你 pin 的 commit 上实测验证。
- Phase 2 的最优机制取决于 Phase 1 的实测,上面的推荐是先验判断。
