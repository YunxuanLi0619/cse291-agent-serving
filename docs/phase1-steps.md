# Phase 1 逐步操作手册(10/19 → 10/28)

> 给新手的使用说明
> - 每一步都有:**目标 → 命令 → 怎么判断成功 → 要记录什么**。按顺序做,做完一步打一个勾。
> - 所有命令在 **macOS 终端(zsh)** 里执行。项目目录名里有空格,所以路径要加引号:`cd ~/"CSE 291"`。
> - 难度和指标完全按课程要求,没有简化。课程要求的"可复现"意味着:**每一步的版本、参数、耗时都要记下来**(见 Step 0 的实验记录本)。
> - 你的机器:Apple M4、16 GB 内存、10 核、剩余磁盘 98 GB。已有 git / uv / jq / curl / gh / python3(3.14);**没有 cmake**。
> - 📌 标 ✅ 的命令我已核对过当前文档;标 ⚠️ 的是我不能 100% 保证、需要你实测确认的地方。
> - 🔧 **v2 审核修订(2026-10-19):** 对照课程 PDF 原文与 llama.cpp 文档补了一批缺口:新增 Step 5.5(校准 + 口径定稿)、Step 4 的流式/`ignore_eos`/取消三项冒烟测试、c ≤ np 不变量、deadline 与超时语义、每次运行的 manifest、`/health` 轮询代替 `sleep`、实验优先级。改动处用 🆕 标注;文末新增「如何落实(Skill 与方法)」。

---

## 总进度表

| 步骤 | 内容 | 预计耗时 | 建议完成日 |
|---|---|---|---|
| 0 | 读样例报告 + 建实验记录本 | 1 h | 10/19 |
| 1 | 装工具、建 Python 环境 | 20 min | 10/19 |
| 2 | 编译 llama.cpp | 15 min | 10/19 |
| 3 | 下载并校验模型 | 15 min | 10/19 |
| 4 | 启动服务并做冒烟测试 | 30 min | 10/20 |
| 5 | **内存基线实验**(决定后面整个设计) | 1–2 h | 10/20 |
| 5.5 | 🆕 **容量校准 + 口径定稿**(内存预算、λ 网格、deadline、实验总时长) | 1–2 h | 10/20 |
| 6 | 建项目骨架与配置 | 30 min | 10/21 |
| 7 | 写 workload(负载) | 半天 | 10/21–22 |
| 8 | 写 runner + FIFO 调度器 + 监控 | 1 天 | 10/22–23 |
| 9 | 跑实验 E1–E5 | 1.5 天 | 10/24–25 |
| 10 | 分析、出图 | 半天 | 10/26 |
| 11 | 写 writeup、打包、提交 | 1 天 | 10/27–28 |

> **关于 Step 7–10 的代码:** 这几步需要写 Python 代码。我在这份手册里给出每一步的**目标、文件接口和运行命令**,代码我会在你完成 Step 5 后接着写并测试(我可以用一个假的服务器先验证逻辑),避免你对着一堆没跑过的代码调试。

---

## Step 0 — 读样例报告 + 建实验记录本(10/19)

**目标:** 知道"高分"长什么样;从第一分钟开始记录,满足"可复现"要求。

1. 打开课程 PDF 第 2 页里的两个 Google Drive 链接,各读一遍:**高分报告**、**低分报告**。对比它们差在哪(图表?对照?对失败的处理?)。
2. 建记录本:

```bash
cd ~/"CSE 291"
mkdir -p docs
cat > docs/lab-notebook.md <<'EOF'
# 实验记录本(每次操作都追加,不要删改)

## 环境
- 日期:
- 机器:Apple M4, 16GB, 10 cores, macOS (uname -a: )
- llama.cpp commit:
- 编译命令与耗时:
- 模型文件与 sha256:
- 启动参数:

## 日志
EOF
```

**记录:** 你从两份样例报告里学到的 3 点(写进记录本)。

---

## Step 1 — 装工具、建 Python 环境(10/19)

**目标:** 有编译器、cmake、一个固定版本的 Python 环境。

```bash
# 1) 确认命令行开发工具(应输出 /Library/Developer/CommandLineTools)
xcode-select -p

# 2) 装 cmake
brew install cmake
cmake --version            # 应能看到版本号

# 3) 在项目里建 Python 环境
#    不用系统的 3.14,用 3.12 以避免部分库没有对应安装包
cd ~/"CSE 291"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install httpx psutil pandas numpy matplotlib pyyaml pytest
python --version           # 应显示 3.12.x
```

> 以后每次开新终端窗口,都要先 `cd ~/"CSE 291" && source .venv/bin/activate`。

**判断成功:** `cmake --version` 和 `python --version` 都有输出。
**记录:** cmake 版本、Python 版本(`uv pip freeze > requirements.lock.txt` 保存依赖版本)。

🆕 **同时把项目变成 git 仓库**(当前目录还不是仓库;后面的 issue 流程和"可复现"都依赖它,不要拖到 Step 6):

```bash
cd ~/"CSE 291" && git init
gh auth status        # 确认 gh 已登录。是否 `gh repo create`(建议 private)由你决定
```

---

## Step 2 — 编译 llama.cpp(10/19)

**目标:** 得到固定 commit 的 `llama-server`。

**重要的设计决定:** 我建议 **关闭 Metal(GPU)、纯 CPU 编译**。理由:

1. 课程最终大概率在 AWS **CPU 实例**上测,CPU 版本与之行为更接近;
2. 用 Metal 时模型和 KV 在 GPU 统一内存里,进程 RSS 不能准确反映内存占用,会让"峰值内存"这个核心指标失真。

```bash
mkdir -p ~/src ~/models
cd ~/src
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# 记录 commit(以后所有实验都基于它;不要再 git pull)
git rev-parse HEAD

# 编译,并计时(课程要求把编译耗时算进来)
cmake -B build -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
time cmake --build build --config Release -t llama-server -j 10
```

✅ 编译 server 的命令与 `-DGGML_METAL=OFF` 选项来自官方文档。

**判断成功:**

```bash
~/src/llama.cpp/build/bin/llama-server --version
~/src/llama.cpp/build/bin/llama-server --help | grep -E "parallel|kv-unified|cache-ram|metrics"
```

能看到版本和这几个参数就成功。

**记录:** commit hash、`time` 输出的耗时。

---

## Step 3 — 下载并校验模型(10/19)

**目标:** 选定一个模型并**从此固定,不再更换**(课程要求 model 和精度固定)。

选择:**Llama-3.2-3B-Instruct,Q4_K_M 量化(约 2.0 GB)**。理由:比 1B 模型的 KV cache 更大(更容易出现内存压力),但在 16GB 机器上仍很舒适。(我已确认下载地址有效。)

```bash
cd ~/models
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

ls -lh Llama-3.2-3B-Instruct-Q4_K_M.gguf     # 约 1.9G
shasum -a 256 Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**记录:** 文件名、大小、sha256、下载来源 URL。

🆕 **校验而不只是记录:** 打开 Hugging Face 该文件页面,把页面上显示的 SHA256 与本地 `shasum` 结果比对,不一致就重新下载(`curl -L` 中断会得到损坏文件,服务可能起得来但结果不可信)。

---

## Step 4 — 启动服务并做冒烟测试(10/20)

**目标:** 确认服务能跑、四个接口都通。

**终端 A(启动服务,保持运行):**

```bash
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -np 4 -c 16384 --no-kv-unified \
  -t 8 --cache-ram 0 --metrics \
  2>&1 | tee ~/"CSE 291"/server-startup.log
```

参数含义(全部要写进记录本):

| 参数 | 含义 |
|---|---|
| `-np 4` | 4 个并发 slot |
| `-c 16384` | 总上下文 16384 tokens |
| `--no-kv-unified` | 每个 slot 独占一份 KV(此时每 slot ≈ 16384/4 = 4096) |
| `-t 8` | 8 个线程,留 2 核给你的 runner 和监控 |
| `--cache-ram 0` | 关闭主机内存里的额外 prompt cache,避免它混进内存测量 |
| `--metrics` | 开启 `/metrics` 接口 |

> ⚠️ 为了实验稳定:**插上电源、关掉浏览器等大程序、不要让 Mac 休眠**。可另开一个终端运行 `caffeinate -dimsu` 阻止休眠。

**终端 B(测试):**

```bash
cd ~/"CSE 291" && source .venv/bin/activate

# 1) 健康检查
curl -s http://127.0.0.1:8080/health

# 2) 一次对话
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32,"temperature":0}' | jq

# 3) 看 slot 数量和每个 slot 的上下文大小
curl -s http://127.0.0.1:8080/slots | jq '.[] | {id, n_ctx, is_processing}'

# 4) 看 metrics
curl -s http://127.0.0.1:8080/metrics | grep -E "requests_|n_tokens_max|n_busy"
```

**判断成功:**

- `/health` 返回 ok;
- 对话有 `choices` 以及 `timings`、`usage` 字段(✅ 文档确认含 `timings.cache_n`、`timings.prompt_n`、`usage.prompt_tokens_details.cached_tokens`);
- `/slots` 返回 4 项,**看每个 `n_ctx` 是不是 4096**(⚠️ 验证"非 unified 时按 slot 均分"这个假设);
- `/metrics` 有 `requests_processing` 和 `requests_deferred`。

**记录:** 完整启动命令、`n_ctx` 实测值。

> ✅ 文档确认:`--cache-idle-slots` 依赖 `--cache-ram`。本手册设了 `--cache-ram 0`,所以它同时失效,**prompt 复用只来自 slot 自己还留着的 KV**。这一点要写进报告的环境说明,E3/E4 的解释都依赖它。

🆕 **三项必须通过的附加冒烟测试**(后面的测量方法依赖它们,现在不验证,Step 8 会踩坑):

```bash
# 5) 流式:t_first_token 只有流式才能测。确认最后一个 chunk 带 timings
curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to 5."}],"max_tokens":32,"stream":true}' | tail -n 6

# 6) ignore_eos 在 OpenAI 兼容接口上是否生效:要求 usage.completion_tokens 恰好 = 200
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":200,"ignore_eos":true,"temperature":0}' | jq .usage

# 7) 取消:先发一个 max_tokens=512 的请求,几秒后 Ctrl+C 断开;在另一窗口反复执行下面这条
curl -s http://127.0.0.1:8080/slots | jq '.[] | {id, is_processing}'
```

- 测试 5:最后一个数据 chunk 里有 `timings`(含 `prompt_n`、`cache_n`);没有就要改用别的方式拿 prefill 耗时。
- 测试 6:不是 200 → 改走原生 `/completion` 接口或记录"输出长度不可强制"这一限制。
- 测试 7:断开后 `is_processing` 在 ≈1–2 s 内回到 `false` → runner 超时时直接断连即可释放 slot;**不是的话,超时请求会继续占 slot 污染后续测量**,必须在报告里写明并改设计。

---

## Step 5 — 内存基线实验(10/20,**最重要的一步**)

**目标:** 搞清"内存到底由什么决定",它会直接决定你 Phase 2 的方向。

### 5.1 算出每个 token 的 KV 内存

```bash
grep -E "n_layer|n_head_kv|n_embd_head_k|n_embd_head_v" ~/"CSE 291"/server-startup.log | head
grep -iE "kv.*(size|buffer)|KV self" ~/"CSE 291"/server-startup.log | head
```

套公式:`每 token KV 字节 = 2 × n_layer × n_head_kv × head_dim × 2`(f16)。
对 Llama-3.2-3B,我估计约 **112 KB/token**,但**以你日志里的数字为准**。再用 `-c 16384 × 每 token 字节` 与日志里打印的 KV 总大小互相验证,二者应接近。

### 5.2 测"空闲/负载下的进程内存"

```bash
PID=$(pgrep -x llama-server | head -1)     # -x 精确匹配进程名,避免匹配到 tee / 当前 shell
echo "空闲 RSS (MB):  $(( $(ps -o rss= -p $PID) / 1024 ))"
footprint $PID | head -20                   # 🆕 macOS 的物理占用口径(⚠️ 命令输出格式请实测确认)
```

然后另开一个窗口持续采样,同时在终端 B 发几个并发请求:

```bash
# 采样窗口:每 0.5 秒记录一次 RSS,写入文件
while true; do echo "$(date +%s.%N),$(ps -o rss= -p $PID)"; sleep 0.5; done > /tmp/rss_probe.csv
```

```bash
# 终端 B:同时发 4 个请求(并发)
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a 300 word story."}],"max_tokens":300,"temperature":0}' \
    -o /dev/null &
done; wait
```

停掉采样窗口(Ctrl+C),看 `/tmp/rss_probe.csv` 里 RSS 在负载前后有没有明显变化。

### 5.3 你要回答的问题(写进记录本)

1. 空闲 RSS ≈ 模型权重 + **整块 KV**,还是只有权重?
2. 负载下 RSS 是否**随并发上升**,还是几乎不变?
3. 🆕 RSS 与 `footprint` 是否一致?模型权重默认是 mmap 的,RSS 会随页面被访问而增长(冷启动后第一批请求前后可能不同)。用 `--no-mmap` 再测一次空闲值,看差多少,然后**选定一个口径并写进 Step 5.5-C**(建议两个口径都报)。

> ⚠️ **这是我的假设,需要你验证:** llama.cpp 启动时会一次性为 `-c` 个 token 分配并清零 KV。如果成立,RSS 在启动后基本是**平的**——内存由"启动配置"(`-np × 每 slot 上下文`)决定,而不是由负载决定。
> 这对项目意义重大:"峰值内存"不能只看 RSS,还要同时记录 **"实际在用的 token 数 / KV 容量"**(由你的 runner 自己统计)。如果 RSS 确实是平的,Phase 1 的内存压力要通过 **(a) 不同的 `-np`/`-c` 配置对应的内存、(b) 请求因超出 slot 容量而失败/排队** 来表现——这正是课程所说的"限定内存预算"。
> 我之前写的 `docs/action-guide.md` §3.3 假定了 RSS 会随负载增长,**请以这一步的实测为准**。

### 5.4 再试一组配置(各记录 RSS 与 `/slots`)

关掉服务(终端 A 里 Ctrl+C),换成下面两组,各重启后测空闲 RSS:

```bash
# 配置 B:更多 slot
... -np 8 -c 32768 --no-kv-unified ...

# 配置 C:共享 KV 池(所有序列共用一块,长任务可以占更多)
... -np 8 -c 16384 -kvu ...
```

**记录:** 三种配置的空闲 RSS、每 slot `n_ctx`。这张表会成为 Phase 1 报告里"内存由什么决定"的第一个证据。

---

## 🆕 Step 5.5 — 容量校准与口径定稿(10/20,Step 5 之后立刻做)

**为什么需要:** 原方案里 λ、n_tasks、重复次数、批跑总时长都没有依据,"峰值内存"的口径也没写死。不做这一步,Step 9 很可能要么跑不完,要么全程过载、要么看不到任何内存现象。

### A. 速度校准(服务用 `-np 4 -c 16384 --no-kv-unified` 启动)

1. **单流速度:** 一个 ~3000 token prompt + 256 token 输出(`ignore_eos`),记 `timings.prompt_per_second` 与 `predicted_per_second`。
2. **聚合吞吐:** 并发 1 / 2 / 4 各发 4 个同样请求,记"总输出 token ÷ 墙钟时间"。看拐点大概在哪。
3. **每种任务类型单独跑 3 次(c=1,无竞争),记无竞争完成时间 `T_solo(type)`。** deadline 和 slowdown 都要用它。

### B. 算实验预算

- 默认混合下的平均服务时间 `S̄ = Σ 占比 × T_solo`;容量 `μ ≈ (A.2 的最佳聚合吞吐折算的并行度) / S̄`。
- λ 网格取 `{0.3, 0.6, 0.9, 1.2} × μ`(包含过载)。单次运行时长 ≈ `n_tasks / λ`(过载时看排空时间)。
- 总机时 = 运行数 ×(运行时长 + 重启约 30 s)。**Step 9 全部实验合计控制在约 20 h 以内**,超了就降 `n_tasks`(下限约 100)或缩 E2 网格;**不要降重复次数**(< 3 次就不合格)。

### C. 口径定稿(逐条写进记录本,之后不再改;下表是建议默认值)

| 项 | 定义 |
|---|---|
| 内存预算 B | 服务启动的**总 KV token 数**(`-c`);换算字节 = B × 每 token KV 字节(Step 5.1) |
| 峰值内存 | 采样期间 server 进程 `max(RSS)` 与 `max(footprint)` 两个口径都报;另报 **KV 利用率 = live_tokens / B** 的时间曲线和峰值 |
| 并发 c 与 np | **不变量:c ≤ np**,否则排队发生在 server 内部,"自己控制的 FIFO"失效。runner 启动时校验 `/slots` 条数 ≥ c,否则拒绝运行 |
| 超时 | 两级:请求级(单次 HTTP,建议 120 s)、任务级(含排队,建议 `10 × T_solo`,上限 300 s)。超时必须**主动断开连接**(Step 4 测试 7 验证)并记 `status=timeout` |
| deadline / goodput | 任务 deadline = `5 × T_solo(type)`(在 dev 集上定好后冻结);goodput = 在 deadline 内完成**且** `quality_pass` 的任务数 ÷ 总时长 |
| 冷启动 | 记 server 启动 → `/health` 返回 ok 的秒数、第一个请求的延迟;每次运行前 3 个任务标 `warmup=true`,稳态统计排除它们(原始日志仍保留) |
| 失败分类 | `ok / timeout / error_ctx_exceeded / error_http / error_other`;分母永远是**全部已提交**的任务 |

### D. 内存轴的实验设计(依据 Step 5 的结果在这里选定)

如果 Step 5 证实 RSS 由启动配置决定,那么"固定 `-np 8 -c 32768`,只改 c"的 E1 会得到一条平线,**看不到内存与并发的关系**。建议两个系列:

- **系列 S(固定预算、切分 slot):** 总预算 B=16384 固定,`np=c`,c ∈ {1,2,4,8} → 每 slot 容量 = 16384/c。c 大时,长任务因超出 slot 容量而失败 → 自然得到"并发↑ → 失败率↑"曲线。
- **系列 U(共享池):** `-kvu -np 8 -c 16384`,c ∈ {1,2,4,8}。观察**池耗尽时 server 的实际行为**(defer / 报错 / 截断)——Phase 2 的 KV 准入控制针对的正是这种情形。

两个系列都保留;时间紧就先做 S,再做 U 的 c ∈ {4,8}。

---

## Step 6 — 建项目骨架与配置(10/21)

```bash
cd ~/"CSE 291"
git init                      # 若 Step 1 已执行过则无副作用
mkdir -p workloads/dev workloads/eval src/scheduler src/runner src/monitor \
         experiments configs tests results/raw analysis report/tables scripts
touch src/__init__.py src/scheduler/__init__.py src/runner/__init__.py src/monitor/__init__.py   # 🆕 否则 python -m src.runner... 会失败

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.gguf
server-startup.log
results/raw/*.tmp
EOF

# 启动脚本:所有 server 参数集中在一个地方,保证每次一致
cat > scripts/start_server.sh <<'EOF'
#!/usr/bin/env bash
# 用法: scripts/start_server.sh <np> <ctx> [extra flags...]
set -euo pipefail
NP=${1:?np}; CTX=${2:?ctx}; shift 2
exec ~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -np "$NP" -c "$CTX" -t 8 --cache-ram 0 --metrics "$@"
EOF
chmod +x scripts/start_server.sh

# 🆕 轮询 /health 代替固定 sleep(加载中返回 503,-f 会让 curl 失败,从而继续等)
cat > scripts/wait_health.sh <<'EOF'
#!/usr/bin/env bash
# 用法: scripts/wait_health.sh [timeout_s=180];成功时在 stdout 打印冷启动等待秒数
T=${1:-180}; S=$(date +%s)
until curl -sf http://127.0.0.1:8080/health >/dev/null; do
  [ $(( $(date +%s) - S )) -ge "$T" ] && { echo "health timeout" >&2; exit 1; }
  sleep 0.5
done
echo $(( $(date +%s) - S ))
EOF
chmod +x scripts/wait_health.sh

git add -A && git commit -m "Project skeleton and server launch script"
```

之后启动服务统一用:`scripts/start_server.sh 4 16384 --no-kv-unified`。

---

## Step 7 — 写 workload(10/21–22)

**目标:** 一份**固定种子、可复现**的负载定义。

**文件约定:**

- `workloads/dev/*.yaml`:开发集(用来设计和调参)
- `workloads/eval/*.yaml`:评估集(Phase 2 才用;**现在只需先建好目录,并规定它与开发集的种子/比例不同**)

**四种任务类型(课程要求:短/长上下文、短/长输出、多步 + 受控 tool delay):**

| 类型 | 初始上下文 | 每步输出 | 步数 | tool delay |
|---|---|---|---|---|
| `short` | ~256 tok | 64 | 1 | — |
| `longctx` | ~3000 tok(留出输出空间,不超过 slot 4096) | 64 | 1 | — |
| `longgen` | ~256 tok | 512 | 1 | — |
| `agent` | ~800 tok,每步追加上一步输出与 tool 结果 | 128 | 3–6 | 0.2–3 s(种子控制) |

**必须内置的设计:**

1. **到达过程:** 开环 Poisson 到达,速率 λ 可调(不是"跑完一个再来一个")。
2. **输出长度受控:** 固定 `max_tokens`、`temperature=0`;需要时 `ignore_eos`。
3. **质量校验(现在就加,Phase 2 要用):** 每个任务最后一步带一个"needle"——在上下文里埋一个随机 key,要求模型复述,程序检查输出里是否含该 key,得到 `quality_pass`。
4. **文件是纯数据**(YAML),由生成脚本 + 种子生成;同一个种子必须产生同一份负载。
5. 🆕 **token 数必须精确可控:** 用 server 的 `POST /tokenize` 构造 prompt,使"~256 / ~800 / ~3000 token"的实测 `usage.prompt_tokens`(含 chat template 开销)落在目标 ±5% 内;生成器里写一个自检。
6. 🆕 **agent 上下文预算要算清:** 第 k 步 prompt ≈ `800 + (k−1) × (128 + tool_result_tokens)`。取 tool_result ≈ 150 token、最多 6 步 → 末步约 2.5k < 4096,基线内不会超限;E2 再刻意加大 tool_result/步数去超过 slot 容量(此时失败是被测量的现象,不是 bug)。
7. 🆕 **needle 校验覆盖多步:** key 埋在第 1 步的上下文里,最后一步复述;`quality_pass` 只由程序判定,不靠人看。
8. 🆕 **dev / eval 分离写成规则:** dev 用 seed 1–5、eval 用 seed 101–105,且混合比例不同(例如 dev 40/20/20/20,eval 30/30/10/30);workload 文件的 sha256 写入每次运行的 manifest。
9. 🆕 **deadline 字段:** 每个任务在文件里带 `deadline_s`(来自 Step 5.5 的 `5 × T_solo`),Phase 2 直接用。

**运行方式(Step 7 的验收命令,代码届时由我写并测试):**

```bash
python -m src.runner.gen_workload --mix configs/mix_default.yaml --seed 1 --n-tasks 200 \
  --out workloads/dev/default_s1.yaml
# 连续跑两次,diff 应为空:
python -m src.runner.gen_workload ... --seed 1 --out /tmp/a.yaml
diff workloads/dev/default_s1.yaml /tmp/a.yaml && echo "可复现 ✔"
```

---

## Step 8 — runner + FIFO 调度器 + 监控(10/22–23)

**目标:** 一个**自己控制**的 FIFO 队列(不依赖 server 自带排队),并发上限固定为 `c`;同时输出三类日志。

**三类日志(JSONL,一行一条,**失败和超时也必须写**):**

| 文件 | 字段 |
|---|---|
| `requests.jsonl` | task_id, step, t_submit, t_start, t_first_token, t_end, prompt_tokens, cached_tokens, output_tokens, status, error_type, warmup, id_slot(若响应提供) |
| `tasks.jsonl` | task_id, type, t_arrive, t_first_dispatch, t_done, queue_time_total, idle_time_total(tool delay 总和), n_steps, status(ok / timeout / error_*), quality_pass, deadline_s, warmup |
| `system.jsonl` | t, rss_mb, requests_processing, requests_deferred, live_tokens(runner 统计的在飞 token 数), 每 slot 的 is_processing |

🆕 **必须满足的设计约束**(每条都写成测试,见文末「如何落实」):

1. runner 自己持有 FIFO 队列;任何时刻在飞请求数 ≤ c ≤ np(启动时校验 `/slots`)。
2. 请求用 `stream: true`,以首个 token chunk 的到达时间记 `t_first_token`;`timings` 取最后一个 chunk。
3. 超时 = 主动关闭连接并记 `status=timeout`;每个任务无论成功失败都**恰好写一行** `tasks.jsonl`。
4. 每次运行产出 `manifest.json`:repo git sha(+ 是否 dirty)、llama.cpp commit、模型 sha256、server 完整命令行、workload 文件 sha256、seed、c、λ、timeout、起止时间、`pmset -g therm` 输出。
5. 系统采样器独立线程,0.5 s 一次;某次采样失败也要写一行(带 error 字段),不能静默跳过。
6. 先用**假 server**(本地一个会 sleep 的 HTTP 服务,可注入失败 / 慢响应 / 断连)验证 1–5,再接真的 llama-server。

**运行方式(验收命令):**

```bash
# 终端 A:启动服务
scripts/start_server.sh 4 16384 --no-kv-unified
# 终端 B:跑一次小规模试验(30 个任务,并发 2)
python -m src.runner.run --workload workloads/dev/default_s1.yaml \
  --policy fifo --concurrency 2 --lambda 0.5 --timeout 120 \
  --out results/raw/smoke_$(date +%Y%m%d_%H%M%S)
```

**判断成功:** `results/raw/smoke_*/` 下有三个 JSONL;`tasks.jsonl` 行数 = 30;人为把 `--timeout 1` 时能看到 `status=timeout` 的记录(证明不会悄悄丢任务)。

---

## Step 9 — 跑实验 E1–E5(10/24–25)

**通用规则(课程要求):**

- **每个配置重复 ≥ 3 次**(不同随机种子或同种子重复,写明);
- 每次实验前**重启服务**(保证 cold start 一致),先发 1–2 个预热请求,**预热段单独标注**,稳态数据另算;
- **失败/超时都保留**;
- 用 `caffeinate` + 插电;一次实验批跑期间不做别的事。

**批跑模板(以 E1 为例):**

```bash
# 🆕 存成 scripts/run_e1.sh 再运行(不要直接粘进交互 shell)。系列 S:固定预算 B=16384、np=c(见 Step 5.5-D)。
# λ 取自 Step 5.5-B 的网格,不要用拍脑袋的 1.0
LAMBDA=${LAMBDA:?set from Step 5.5-B}
for C in 1 2 4 8; do
  for REP in 1 2 3; do
    OUT=results/raw/e1S_c${C}_rep${REP}; mkdir -p "$OUT"
    scripts/start_server.sh $C 16384 --no-kv-unified > "$OUT/server.log" 2>&1 &
    SRV=$!
    COLD=$(scripts/wait_health.sh) || { echo "server failed to start"; kill $SRV; exit 1; }
    echo "$COLD" > "$OUT/cold_start_s.txt"
    python -m src.runner.run --workload workloads/dev/default_s${REP}.yaml \
      --policy fifo --concurrency $C --lambda $LAMBDA --timeout 180 --out "$OUT"
    kill $SRV; wait $SRV 2>/dev/null; sleep 5
  done
done
```

| 实验 | 变量 | 目的 |
|---|---|---|
| **E1 并发扫描** | c = 1, 2, 4, 8(必要时 16) | 吞吐拐点、p50/p99、内存、失败率随并发的变化 → **找主导瓶颈** |
| **E2 上下文增长** | 初始上下文与步数 | 上下文增长如何推高内存/延迟 |
| **E3 空闲期** | tool delay 0 / 0.5 / 1 / 3 s | 步骤间空闲是否让 slot 被闲置上下文占着,导致排队 |
| **E4 前缀复用** | 请求里 `"cache_prompt": false` vs `true`;共享/不共享 system prompt | 前缀复用是否**实质**影响该 runtime(看 `timings.cache_n` 与 prefill 耗时) |
| **E5 到达率** | λ 扫描(低→过载) | 找系统饱和点,确定 Phase 2 的工作区间 |

🆕 **优先级(时间不够时按此砍):** 必做 E1(系列 S)、E4、E5 → 其次 E3(至少取 tool delay = 0 与 3 s 两点)→ E1 系列 U → E2。E2 可以合并:直接以 agent 任务的"第 k 步"作为上下文长度自变量,从 E1 数据里切片,不一定单独跑。**每个实验先用 `n_tasks=30`、1 次重复试跑一遍**,确认日志和分析脚本都对,再放开跑全量。

🆕 **E4 的注意事项:** 前缀复用只在请求落到"还留着该前缀的 slot"时发生(本配置下 `--cache-idle-slots` 已失效)。c=1 时续步必然回到同一 slot;c=4 且有 tool delay 时,续步可能落到别的 slot 而丢失复用——这恰好是 E3 要量的现象。所以 E4 至少在 c=1 和 c=4 各测一次,并记录每个请求的 `cache_n`。

**结果的"最佳固定并发":** 在 E1 里,按开发集选出**使吞吐(或 goodput)最高且 p99 可接受**的 c。这是 Phase 2 必须打败的**调优过的基线**,现在就写进记录本。

---

## Step 10 — 分析与出图(10/26)

**目标:** 所有图表都由一条命令从 `results/raw/` 重新生成。

```bash
python analysis/make_figures.py --raw results/raw --out report/figs
```

**至少要有的图(课程要求"performance curves"):**

1. 并发 c vs **吞吐**(带重复实验的误差条)
2. 并发 c vs **p50 / p99 任务完成时间**
3. 并发 c vs **排队时间**
4. 并发 c vs **失败 / 超时率**
5. **内存**:不同 server 配置的 RSS(柱状图)+ 运行中 `live_tokens / KV 容量` 的时间曲线
6. E3:tool delay vs 吞吐/排队
7. E4:前缀复用开/关的 prefill 耗时、`cached_tokens` 占比
8. E5:λ vs p99(能看到"拐点")

🆕 **分析脚本也要有测试:** 用一份手写的 ~10 行 JSONL 夹具,断言 p50/p99、吞吐、失败率的**手算值**,并断言"失败 / 超时任务计入分母"。每张图的数据同时输出为 CSV(`report/tables/`),报告里的数字都引用 CSV,避免图和正文对不上。

**分析要回答的问题(报告里要有一句话结论):** 主导瓶颈是 **KV 容量**、**算力(decode 速度)**,还是**空闲占用**?每个结论配一张图作为证据。

---

## Step 11 — Phase 1 writeup 与提交(10/27–28)

**提交内容(课程要求):** 源码、数据集/负载、实验说明、结果与日志、简洁报告。

```bash
# 1) 在干净状态下复现一遍(课程评分看重可复现)
git status                               # 应为干净
rm -rf /tmp/repro && git clone . /tmp/repro && cd /tmp/repro
cat README.md                            # 按 README 的步骤做一遍

# 2) 打 tag 固定提交版本
git tag phase1-submission
```

**README 必须包含:** 环境版本(llama.cpp commit、模型 sha256、Python 依赖)、编译与启动命令、一条命令跑 smoke test、一条命令复现图表、各实验的命令。

**报告结构(3–5 页):**

1. 研究问题与环境(版本全部 pin)
2. 负载定义与指标口径
3. 测量方法与 instrumentation
4. 结果:性能曲线
5. **主导瓶颈结论**
6. 前缀 / retained context 的影响(E4)
7. **Phase 2 方向与理由**(必须引用你的数据)
8. 局限性(笔记本、CPU-only、单机、模型规模等)

---

## 附:遇到问题怎么办

| 现象 | 先检查 |
|---|---|
| `cmake: command not found` | Step 1 的 `brew install cmake` 是否成功 |
| 编译报错 | 把完整报错贴给我,不要猜 |
| 服务起不来 / 秒退 | 看终端 A 的报错;常见原因是模型路径写错或内存不够 |
| `/metrics` 返回 404 | 启动时漏了 `--metrics` |
| `/slots` 里 `n_ctx` 不是 4096 | 记录实测值,按实测值设计负载(这本身是发现) |
| 实验结果抖动大 | 是否插电、是否有后台程序、是否散热降频;重复次数不够就加次数 |
| 请求报 context 超限 | 任务上下文超过 slot 容量;这属于要**记录为 failure** 的真实现象 |

## 向助教确认(不影响现在开工)

- Phase 1 的 baseline 在笔记本上跑是否可接受(题目文档写明"or your laptops")?最终(Phase 2)测量是否必须在 AWS 上?
- 课程文档标注 **TENTATIVE**,留意公告里截止日期和选项是否变动。
- 🆕 组队情况(1–5 人)与是否需要在课程渠道登记选题。

---

## 🆕 如何落实(Skill 与方法)

Phase 1 里有两类性质不同的工作,用不同的方法:

| 工作 | Step | 方法 |
|---|---|---|
| 搭环境 / 编译 / 下载 / 测量 / 校准 | 0–5.5 | **不走 spec 流程。** 让 Claude 直接在终端执行并把结果追加进 `docs/lab-notebook.md`。(`/wizard` 只用于"只有人能做"的步骤;这里 agent 自己能做的不要用。) |
| 查清 llama.cpp 的行为事实 | Step 2 pin commit 之后 | `/research`:后台 agent 读该 commit 的源码/文档,产出带引用的 md |
| 写 runner / scheduler / monitor / 分析脚本 | 6–10 | 主流程:`/grill-with-docs` → `/to-spec` → `/to-tickets` → 每张票 `/clear` 后 `/implement`(内含 `/tdd` 与 `/code-review`) |
| 跑批实验 | 9 | 脚本驱动(`scripts/run_*.sh`),不是 skill |
| 复盘 | Phase 1 结束 | `/retro`,把教训带进 Phase 2 的环境 |

**时间线**

1. **10/19:** Step 0–3(含 `git init`)。Step 2 完成后启动 `/research`(问题见下)。
2. **10/20:** Step 4、5、5.5,结果写进记录本。
3. **10/20 晚:** 新开会话,`/grill-with-docs`(不要中途 clear,直到 `/to-tickets` 结束):把 5.5-C 的口径写进 `GLOSSARY.md`;把"CPU-only 编译"、"以总 KV token 数 B 作为内存预算"、"c ≤ np"记为 ADR。随后 `/to-spec` → `/to-tickets`。
4. **10/21–23:** 每张票 `/clear` → `/implement`。
5. **10/24–26:** 跑实验、出图(先 30 任务试跑,再全量)。
6. **10/27–28:** 写报告 → `/retro`。写报告的骨架从 10/25 就开始填,不要把整个报告压在最后一天。

**`/research` 要回答的问题(都是原文档里标"未核实"的)**

1. `--no-kv-unified` 与 `-kvu` 下,`-c` 与 `-np` 如何决定每个 slot 的 `n_ctx`;共享池耗尽时 server 的行为。
2. KV buffer 在 CPU 后端何时分配、是否清零(RSS 为何平 / 不平)。
3. 请求如何选择 slot(前缀相似度?空闲优先?);响应里是否带 `id_slot`。
4. 客户端断连是否取消正在进行的生成并释放 slot。
5. OpenAI 兼容接口上 `ignore_eos`、流式 `timings` 是否生效 / 出现在哪个 chunk。

**`/tdd` 的 seam(`/implement` 时需你确认)**——只测外部行为,不测 llama-server 本身:

1. `gen_workload` 命令:同 seed → 文件字节完全一致;token 自检落在 ±5%。
2. `run` 命令 + 假 server → JSONL:在飞请求 ≤ c;超时断连且记 `timeout`;每任务恰好一行;失败计入。
3. 分析函数:手写 JSONL 夹具 → 手算的 p50/p99、吞吐、失败率。

**建议的 tracer-bullet 票(`/to-tickets` 时的起点)**

| # | 票 | 被谁阻塞 |
|---|---|---|
| 1 | 确定性 workload 生成 + 单任务 runner 对假 server 跑通,写出 requests/tasks 各一行 | 无 |
| 2 | FIFO + 并发上限 + 开环 Poisson + 超时/失败落盘 | 1 |
| 3 | 多步 agent 任务 + tool delay + needle 校验 | 1 |
| 4 | 系统采样器 + manifest | 1 |
| 5 | 接真 llama-server,30 任务冒烟 | 2, 3, 4 |
| 6 | 分析脚本:指标 + 图 + CSV(用夹具,不依赖真实数据) | 1 |
| 7 | 批跑脚本 E1–E5 | 5, 6 |

遇到"结果抖动 / server 莫名卡死 / 超时不释放 slot"这类难复现问题,用 `/diagnosing-bugs`,不要靠猜。
