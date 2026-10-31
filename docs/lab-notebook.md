# 实验记录本(每次操作都追加,不要删改)

## 环境
- 日期:2026-10-19
- 机器:**MacBook Air**(无风扇,持续负载会散热降频,每次运行记 `pmset -g therm`), Apple M4, 16GB, 10 cores, macOS (uname -a: Darwin Yunxuans-MacBook-Air.local 25.6.0 Darwin Kernel Version 25.6.0: Fri Jul 31 19:11:03 PDT 2026; root:xnu-12377.161.14~5/RELEASE_ARM64_T8132 arm64)
- llama.cpp commit: 1b29621bacc6a446bd523fc790316a24a06e886f (tag b11741, 2026-10-19 12:30 -0700; `llama-server --version`: 0.7.0-dev build 11741, AppleClang 21.0.0)
- 编译命令与耗时: `cmake -B build -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release` + `cmake --build build --config Release -t llama-server -j 10`;耗时 real 2:24.91(user 641.86s)。CPU-only。
- 模型文件与 sha256: Llama-3.2-3B-Instruct-Q4_K_M.gguf, 2019377696 B (≈1.88 GiB), sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff;来源 https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf;已与 HF API (tree/main) 的 lfs.oid 和 size 比对,一致。
- 启动参数:

## 日志

### 2026-10-19 Step 0 — 样例报告
(Drive 预览只能截图浏览,我只快速浏览了每份的前几页,未通读。)
1. 高分(Edge Computer-Use Agent):先把瓶颈隔离出来(先单独研究 grounding,再搭完整系统);对比实验控制严格(同一 100 样本子集、相同 prompt/解析流程,只改一个因素);结果以表格 + 柱状图量化(Acc@5/10/20/50px、延迟、量化位宽权衡),并写明硬件(Ryzen 7 9700X / 32GB)。
2. 低分(Jamie Food Recommender):篇幅主要花在架构/部署/隔离设计(LangGraph、Cloud Run、IAM、GCS 前缀),量化对照实验占比小;"改进"的论证偏描述而非受控测量。→ 我们应该把篇幅花在测量方法与对照上,而不是系统实现细节。
3. 共同点/启示:高分报告从"观察到的失败"出发,再做对照消融,并给出权衡曲线(accuracy vs latency);我们的对应物是 并发 c vs 吞吐/p99/失败率 的曲线 + 主导瓶颈结论。
(课程 PDF 里低分报告链接是 ...EOz3cgTi,实际 ID 为 ...FOEz3cgTi 的 `cgTi`;用 pdftotext 提取链接比肉眼读可靠。)

### 2026-10-19 Step 1 — 环境
- xcode-select -p: /Library/Developer/CommandLineTools
- cmake 4.4.4 (brew);Python 3.12.13 (uv venv .venv);依赖锁定于 requirements.lock.txt (25 行)
- git init 已完成(分支 main,尚无提交);gh 已登录 YunxuanLi0619。是否 `gh repo create` 待用户决定(未创建)。
- 磁盘剩余约 110 GiB。

### 2026-10-19 Step 2 — 编译
- 见上方环境。`--help` 中确认存在 -np / -kvu --no-kv-unified / --cache-ram / --metrics / --cache-idle-slots。
- 新发现:该 commit 有 `--kv-unified-per-slot N`(每 slot 上下文上限,默认不设),Step 5 可考虑对照。

### 2026-10-19 Step 3 — 模型
- 见上方环境。下载约 6 分钟(与编译/clone 并行,占用带宽)。

### 2026-10-19 Step 4 — 启动与冒烟测试(配置 A:`-np 4 -c 16384 --no-kv-unified -t 8 --cache-ram 0 --metrics`)
参数含义见 phase1-steps.md Step 4 表格。服务日志:`server-startup.log`(已 gitignore;第一次无 `-lv`,第二次加了 `-lv 4` 仅为拿到超参/KV buffer 日志,不影响内存)。
| 测试 | 结果 |
|---|---|
| 1 /health | `{"status":"ok"}`;冷启动(页缓存已热)5–18 s |
| 2 对话 | 有 `choices`、`usage.prompt_tokens_details.cached_tokens`、`timings.{cache_n,prompt_n,...}` |
| 3 /slots | **4 项,每项 n_ctx=4096**(非 unified 时均分假设成立);server 日志 `n_ctx_slot = 4096, kv_unified = 'false'` |
| 4 /metrics | 有 requests_processing / requests_deferred / n_tokens_max / n_busy_slots_per_decode |
| 5 流式 | ✅ `timings` 在最后一个数据 chunk(`finish_reason:"stop"` 那个,`[DONE]` 之前) |
| 6 ignore_eos | ✅ OpenAI 兼容接口上生效:`completion_tokens=200`,`finish_reason=length` |
| 7 取消 | ✅ 客户端断开后 ≈0.7 s 内 `is_processing` 回到 false;server 日志 `cancel task` → `release` |
额外发现(llama.cpp 行为,来自日志,待 /research 或源码核实):
- 选 slot:`selected slot by LCP similarity (f_sim_best > 0.100 thold)`;无匹配时 `by LRU`。→ 续步落到哪个 slot 取决于前缀相似度,不是"空闲优先"。(E3/E4 的解释依据)
- `--cache-ram 0` 下 `--cache-idle-slots` 被禁用(日志 `requires --cache-ram, disabling`);但 slot 内 KV 仍会复用:第二个请求 `cached_tokens=30`(chat template 前缀)。
- 该 commit 里 `--no-mmap` **已不存在**,改为 `-lm none`(`-lm mmap|none|mlock|mmap+mlock`)。手册里 5.3 的 `--no-mmap` 请以 `-lm none` 为准。
- 默认日志级别不打印 `n_layer`/KV buffer;需 `-lv 4`。
- 响应里是否带 `id_slot`:未检查(5.5 时确认)。

### 2026-10-19 Step 5 — 内存基线
**5.1 每 token KV**:n_layer=28, n_head_kv=8, n_embd_head_k=n_embd_head_v=128 (n_head=24, n_embd_k_gqa=1024)。2×28×8×128×2 B = **114,688 B = 112 KiB/token**。16384 token → 1792 MiB,与日志 `CPU KV buffer size = 1792.00 MiB`(K 896 + V 896,f16)完全一致。compute buffer 82 MiB。
**5.2/5.3 空闲与负载下的内存**(`results/raw/step5/`,采样器 `scripts/rss_footprint_sampler.sh`):
- 配置 A 空闲:`footprint` 3789 MB;`ps` RSS 692 MB(这次)。
- 负载:4 个并发请求(各 300 token,ignore_eos,共 85.8 s 完成),采样 639 点/411 s:**footprint 3789→3818 MB(仅 +29 MB)**;**RSS 592→2536→170 MB 剧烈起伏**(权重页被访问时涨,之后被 macOS 回收/压缩)。
- 结论:Q1 空闲内存 ≈ 权重 + **整块 KV**(KV 在启动时一次性分配);Q2 footprint 与并发/负载几乎无关,**由启动配置决定**;Q3 RSS 在这台机器上不可信(同一配置不同次启动的 idle RSS 为 689 / 4678 / 3577 MB),**macOS 压缩内存不计入 RSS**。→ 5.5-C:**峰值内存主口径 = footprint**(同时仍记录 RSS 作次口径,并在报告里说明其不可靠的原因)。
- ⚠️ 手册 5.3 的假设被证实:"内存由启动配置决定,而不是由负载决定"。`docs/action-guide.md` §3.3 关于"RSS 随负载增长"的假设应作废。
**5.4 配置表**(idle_t0 = /health 就绪时;footprint 为 `footprint <pid>` 首行 Footprint):
| 配置 | 参数 | 每 slot n_ctx | KV buffer | idle footprint | idle RSS (t0 → 20s 后) | cold start |
|---|---|---|---|---|---|---|
| A (mmap) | -np 4 -c 16384 --no-kv-unified | 4096 | 1792 MiB | 3789 MB | 3577 → 1210 MB | 5 s |
| A (-lm none) | 同上 + `-lm none` | 4096 | 1792 MiB | **4083 MB** | 3712 → 985 MB | 8 s |
| B | -np 8 -c 32768 --no-kv-unified | 4096 | 3584 MiB | 5584 MB | 2763 → 1634 MB | 13 s |
| C | -np 8 -c 16384 -kvu | **16384**(8 个 slot 共享同一池) | 1792 MiB | 3791 MB | 5028 → 3229 MB | 9 s |
- mmap 与 `-lm none` 的 footprint 差 ≈ 294 MB:mmap 时约 300 MiB 的权重是文件映射,不计入 footprint(footprint 输出 "mapped file 677 MB clean")。→ footprint(mmap) 是下界,`-lm none` 是含全部权重的上界;写进报告。
- 与 KV 的关系:A→B footprint 差 1795 MB ≈ KV 差 1792 MiB ✓;A→C footprint 基本不变 ✓。
- C 配置的 /slots 报告每 slot n_ctx = 16384(池容量,不是独占容量);8 个 slot 共 16384 token 的池。
- 冷启动数字都是模型已在页缓存里的情况;真正的冷盘启动未测。
### ⚠️ 环境警告(2026-10-19 13:5x)
- **Mac 在用电池,且 `lowpowermode=1`(低电量模式)**。冒烟与内存实验(只看 footprint)不受影响,但**速度类数据不可用**:单流解码仅 ≈7 tok/s(预期 M4 应明显更高),4 并发合计 ≈14 tok/s。**5.5 的校准必须插电并关闭低电量模式后再做**,否则 λ 网格、deadline、T_solo 都会偏。

### 2026-10-19 Step 5.5 — 容量校准与口径定稿
**条件:** 配置 A(`-np 4 -c 16384 --no-kv-unified -t 8 --cache-ram 0`),低电量模式已关,**但在电池供电下完成**(55%→49%,无热降频记录)。⚠️ Step 9 批跑前必须插电,并先用每类 3 个任务复测 T_solo,与下表偏差 >15% 就重新校准。此前低电量模式下解码仅 ~7 tok/s,关闭后 ~35 tok/s,说明速度对电源模式极敏感。
数据:`results/raw/step55/calibration.json`、`calibrate.stdout`、`server.log`;脚本 `scripts/step55_calibrate.py`(粗原型,prompt 为种子随机词,每次请求内容不同;**不是** Step 7 的生成器)。

**A.1 单流(3000 token prompt + 256 输出,3 次):** prefill 112–123 tok/s(≈26 s),decode 12.9–16.4 tok/s,总耗时 40–47 s。
**A.2 聚合吞吐(8 个 256in/128out 请求):**
| c | 墙钟 s | 聚合 tok/s | 单请求 decode tok/s | 平均 TTFT s |
|---|---|---|---|---|
| 1 | 46.9 | 21.8 | 33.0 | 2.0 |
| 2 | 39.0 | 26.2 | 19.8 | 3.2 |
| 4 | 31.2 | 32.8 | 15.0 | 6.5 |
聚合吞吐随并发增长平缓(c=4 仅为 c=1 的 1.5×),无明显拐点,CPU 算力是主要约束;c=8 未测(假设 ≤ c=4)。
**A.3 T_solo(c=1,各 3 次;prompt_tokens 为实测含 chat template):**
| 类型 | prompt_tokens | 3 次耗时 s | 均值 |
|---|---|---|---|
| short (64 out) | 290(目标 256,+13%,含 ~32 template token) | 4.0 / 4.6 / 4.1 | 4.3 |
| longctx (64 out) | 3035(目标 3000) | 33.5 / 34.2 / 37.3 | 35.0 |
| longgen (512 out) | 290 | 21.9 / 23.0 / 22.7 | 22.5 |
| agent (4 步,tool delay 1.0 s,tool result 150 tok) | 835→1123→1411→1698 | 35.5 / 34.4 / 35.8 | 35.2(含 idle 3.0 s) |
- decode 速度随上下文下降:31 tok/s(290 ctx)→ 26(longgen)→ 13(3000 ctx);agent 各步 27.8→21.5 tok/s。
- 同一 slot 的 agent 续步前缀复用有效:`cached_tokens` = 30 → 962 → 1250 → 1538(c=1,idle 1 s 后仍在)。
- **流式 chunk 与 /slots 都没有 id_slot**(chunk 键:choices, created, id, model, object, system_fingerprint, timings;slot 键:id, is_processing, n_ctx, speculative)。→ requests.jsonl 的 `id_slot` 字段只能留空,或从 server 日志(`slot launch_slot_: id N`)事后关联。
- 要修正 Step 7 的 token 自检:short 目标 256 时实测 290,template 约 +32;生成器应以 `usage.prompt_tokens` 为准校准(±5%)。

**B. 预算(默认混合 40/20/20/20 = short/longctx/longgen/agent):**
- S̄ = 0.4×4.3 + 0.2×35.0 + 0.2×22.5 + 0.2×35.2 = **20.26 s**
- 有效并行度 P_eff = G(4)/G(1) = 1.5(c=2: 1.2);μ ≈ 1.5/20.26 = **0.074 task/s(4.45 task/min)**
- λ 网格 {0.3, 0.6, 0.9, 1.2}×μ = **{0.022, 0.044, 0.067, 0.089} task/s**(平均到达间隔 45 / 22.5 / 15 / 11.3 s)。E1/E3/E4 默认用 λ = 0.9μ = **0.067**。
- n_tasks = 100(下限,不再降),重复 3 次;每次重启 30 s。单次时长 = max(n/λ, n·S̄/P_eff(c)):λ=0.9μ 时 c=1/2/4/8 约 34.3 / 28.6 / 25.5 / 25.5 min。
- 总机时(n=100, 3 次重复):E1 系列 S 5.7 h + E1 系列 U(c=4,8)2.5 h + E3(tool delay 0 与 3 s)2.5 h + E4(仅 cache_prompt=false,c=1 与 c=4)3.0 h + E5(λ=0.3/0.6/1.2μ,c=4)6.8 h = **20.6 h**。
  - **这个数字的前提是复用:** E4 的 cache_prompt=true 用 E1 同配置那一次、E5 的 λ=0.9μ 用 E1 c=4 那一次,不重跑;不复用则 E5 为 8.1 h,E4 再 +3.0 h,合计 ≈ 25 h。
  - 估计的不确定性:P_eff(8) 未测;c=8(系列 S)每 slot 仅 2048 token,longctx 会直接 ctx 超限、提前失败,会使该配置实际更短;校准在电池下完成。
  - 如果实际超出,优先砍 E1 系列 U(只留 c=4,8 之一)和 E3 的第二个 delay 点,不降重复次数。
- 低 λ 的运行由到达过程决定时长(0.3μ 一次 75 min,三次 3.75 h)——E5 是最贵的实验。

**C. 口径定稿(之后不再改):**
| 项 | 定义 |
|---|---|
| 内存预算 B | 服务启动的总 KV token 数(`-c`);字节 = B × **114,688 B/token**(B=16384 → 1792 MiB) |
| 峰值内存 | **主口径 = `footprint`**(采样期间 max);次口径 RSS 也记录但在报告中标注不可靠(macOS 压缩内存不计入 RSS,同配置 idle RSS 在 689–4678 MB 间波动,Step 5)。报告 `-lm none` 与默认 mmap 两个 footprint(差 ≈294 MB:mmap 时文件映射的权重不计入 footprint);主实验用默认 mmap,并写明 footprint 为下界。另报 KV 利用率 = live_tokens / B 的时间曲线与峰值(footprint 本身基本是平的,KV 利用率才反映负载) |
| 并发 c 与 np | 不变量 c ≤ np;runner 启动校验 `/slots` 条数 ≥ c,否则拒绝运行 |
| 超时 | 请求级 120 s;任务级(含排队)= min(300, 10×T_solo):short 43 s、longctx 300 s、longgen 225 s、agent 300 s。超时主动断开连接(Step 4 测试 7:≈0.7 s 释放 slot ✓)并记 `status=timeout` |
| deadline / goodput | `deadline_s = 5 × T_solo(type)`:short 21 s、longctx 175 s、longgen 113 s;agent 按每个任务自己的步数/延迟计算:≈ 5 × (8.05 s × n_steps + Σ tool delay)(8.05 s = (35.2−3.0)/4 步)。dev 集上定好后冻结;goodput = deadline 内完成且 quality_pass 的任务数 ÷ 总时长 |
| 冷启动 | 记 server 启动→/health ok 秒数(页缓存热时 5–14 s,冷盘未测)与首个请求延迟;每次运行前 3 个任务标 `warmup=true`,稳态统计排除(原始日志保留) |
| 失败分类 | ok / timeout / error_ctx_exceeded / error_http / error_other;分母 = 全部已提交任务 |

**D. 内存轴实验设计:** Step 5 证实 footprint 由启动配置决定(与负载无关),所以按手册两个系列都保留:
- 系列 S(B=16384 固定,np=c,c∈{1,2,4,8},每 slot 容量 = 16384/c):c=8 → 2048 token/slot,3035-token 的 longctx 必然超限(确定性失败,是被测现象);c=4 → 4096 刚好容纳 agent 末步 1698+128。
- 系列 U(`-kvu -np 8 -c 16384`,c∈{4,8}):观察池耗尽时的行为(defer/报错/截断)。注意 `-kvu` 下 /slots 对每个 slot 都报 n_ctx=16384(池容量)。
- 时间紧时先做 S,再做 U 的 c∈{4,8}(与上面预算一致)。

### 2026-10-19 #3 补充 — prefill 期间断连的 slot 释放耗时(真机)
配置 `-np 1 -c 4096`(`scripts/start_server.sh 1 4096`),单 slot 所以不用归因。prompt = longctx 3000 token(每次加随机前缀,避免整段命中 slot 里的 KV)。脚本 `scripts/measure_slot_release.py`,原始数据 `results/raw/slot_release/`。断连 = 客户端关闭流式连接;释放 = `/slots` 的 `is_processing` 回到 false(50 ms 轮询)。
| 断连时刻(发出后) | 断连 → 释放 | 发出 → 释放 |
|---|---|---|
| prefill 1 s(4 次) | 16.3 / 17.9 / 18.0 / 18.6 s | 约 17–19 s |
| prefill 5 s | 31.8 s | 36.8 s |
| prefill 10 s | 22.1 s | 32.1 s |
| prefill 15 s | 11.8 s | 26.8 s |
| prefill 20 s | 7.9 s | 27.9 s |
| prefill 24 s | 6.6 s | 30.6 s |
| decode(首个 token 后 1 s) | 0.11 s | — |
- **decode 期间断连:** 立即释放(0.11 s),与 Step 4 测试 7 一致。上面"超时主动断连 ≈0.7 s 释放 slot ✓"**只对 decode 阶段成立**。
- **prefill 期间断连:** slot 一直忙到 prefill 结束。1 s 断连时 server 当场记 `cancel task`,但要等第一个 2048-token 批处理结束才 `release`(`n_tokens ≈ 2078`),约 17–19 s;5–24 s 断连时 `cancel task` 要等整个 prompt 处理完才出现(`release` 时 `n_tokens ≈ 3022`),发出→释放 ≈ 27–37 s ≈ 完整 prefill 时间。
- **没弄清的:** 为什么 1 s 断连被立刻发现而 ≥5 s 的要等到 prefill 结束才被发现(只有现象,没验证机制)。样本很少(单个 prompt、每个时刻 1 次,1 s 重复 4 次),且在电池供电下测。
- **对 runner 的影响:** 长 prompt(longctx)超时断连后,server 的 slot 可能还被占 10–30 s,而 runner 在断连瞬间就释放了自己的并发名额,下一个请求可能撞上 server 内部排队(ADR 0003 担心的情况)。runner 目前不检测这一点。

### 2026-10-30 #7 线程数对比 — 冻结 `-t 8`
配置 `-np 4 -c 16384 --no-kv-unified --cache-ram 0`,插电、低电量模式关闭、无热降频记录。交错顺序 `8 4 10 8 4 10`,每档之间冷却 180 s;每档 short ×3、longctx ×3 单流 + c=4 聚合吞吐(8 个请求 × 128 token)。脚本 `scripts/prebatch_check.py threads`,原始数据 `results/raw/prebatch/threads.json`,决策与剔除记录 `threads_decision.json`。
| `-t` | short (s) | longctx (s) | c=4 聚合 (tok/s) |
|---|---|---|---|
| 8 | 3.06–3.59(5 个) | 25.4–26.8(6 个) | 39.7 / 40.7 |
| 4 | 3.90–4.07(6 个) | 33.4–35.3(6 个) | 31.1 / 30.6 |
| 10 | 2.93–3.10(6 个) | 24.3–26.0(6 个) | 36.7(1 个) |
- 相对 `-t 8` 的几何平均加速比:`-t 4` 0.78,`-t 10` 1.006。`-t 10` 没有比 `-t 8` 快 10% → 按规则保持 `-t 8`;`-t 4` 明显更慢(单流 +30%、吞吐 −20%)。
- **剔除了两个被系统睡眠污染的样本**(没有改原始文件):第 4 档 `-t 8` 的第一个 short(647 s)和第 6 档 `-t 10` 的 c=4(1.0 tok/s)。证据:`pmset -g log` 显示 21:32:46 空闲睡眠到 21:43:30(644 s)、21:55:37 到 22:10:37(900 s);server 日志里 261 token 的 prompt eval 用了 644.6 s,4 个并发请求各 943 s。**插电也会空闲睡眠**,此前没有用 `caffeinate`。`prebatch_check.py` 现在启动时自带 `caffeinate -is -w <pid>`,后面的阶段和正式批跑都要如此。
- `-t 10` 的 c=4 只有 1 个有效样本,所以它与 `-t 8` 的吞吐比较(36.7 vs 40.2)证据较弱;但单流两者差 <3%,结论不依赖它。
- 第一轮(`threads_run1_confounded.*`)同配置前后差近一倍,原因没查清(可能是后台进程或降频),不用于决策;第二轮同配置前后一致(`-t 8` longctx 26.1 vs 26.4,`-t 4` 34.5 vs 34.5)。
- 冻结:`configs/server.json` = `{"threads": 8}`。
