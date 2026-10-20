# llama-server 行为核实(源码 + 官方文档)

- 调查的 commit:`1b29621bacc6a446bd523fc790316a24a06e886f`(b11741)
- `git rev-parse HEAD`(在 `~/src/llama.cpp`):`1b29621bacc6a446bd523fc790316a24a06e886f`,与目标 commit 一致;`git status --short` 为空(工作区干净),所以下文所有行号都是直接读工作区得到的,等价于固定 commit 的内容。
- 日期:2026-10-19
- 方法:只读源码和 `tools/server/README.md`,没有运行实验,没有修改 `~/src/llama.cpp`,没有 checkout/pull。路径均相对 `~/src/llama.cpp`。实测对照来自 `docs/lab-notebook.md`、`docs/phase1-handoff.md`、`results/raw/`。
- 标注约定:**一致** / **矛盾** / **源码无法确认**。"推断"表示由代码阅读得出、但没有实测验证的结论。

## 总表

| # | 问题 | 结论(一句话) | 与实测 |
|---|---|---|---|
| 1 | `-c`/`-np`/`-kvu` 如何决定每 slot `n_ctx`;池耗尽行为 | 非 unified:`n_ctx_seq = pad256(n_ctx/np)`;`-kvu`:每 slot 都等于整个 `n_ctx`,slot 之间没有准入控制。池耗尽时不是排队也不是截断:先清空空闲 slot 的缓存并缩小 batch 重试,仍失败则给**所有在跑请求**发 `"Context size has been exceeded."`(HTTP 500 / 流中 SSE error),并清空这些 slot 的 KV | **一致**(n_ctx 部分);池耗尽行为实测未做,只有源码结论 |
| 2 | KV buffer 何时分配、是否清零 | 在 `llama_context` 构造时(服务器报告 listening 之前)一次性 `vm_allocate`(macOS)并 `memset(0)` 整块,所以页在启动时就全部被写过,之后负载只覆盖已驻留的页 | **一致**(启动即占满、footprint 平);负载下 +29 MB 的来源与 macOS 压缩内存对 RSS 的影响**源码无法确认** |
| 3 | slot 选择与 `id_slot` | 先按 LCP/新 prompt 长度 > `-sps`(默认 0.10,严格大于)选最高相似度的空闲 slot;没有则选空 slot,再选 LRU;OAI chat 的响应、流式 chunk、`/slots` 都不含 `id_slot` | **一致** |
| 4 | 断连是否取消生成 | 会。HTTP 线程用 `is_connection_closed`(socket 探测)检测断连,析构 `server_response_reader` 时发 `CANCEL` 任务;主循环要等当前 `llama_process` 返回后才处理它。非流式检测靠 1 s 轮询,流式靠每个 chunk 检查 | **一致**(0.7 s ≈ ≤1 s 轮询 + 约 1 个 decode 步);测试 7 是流式还是非流式**源码无法确认** |
| 5 | `ignore_eos` 与流式 `timings` | `ignore_eos` 经"剩余字段透传"进入 schema 并给所有 EOG token 加 `-inf` logit bias,在 `/v1/chat/completions` 生效;流式时 `timings` 挂在最后一个 delta 数组的最后一个元素上(无 `include_usage` 时就是 `finish_reason` 那个 chunk),字段见下 | **一致** |

---

## Q1. 每 slot 的 `n_ctx`,以及共享池(`-kvu`)耗尽时的行为

### 1.1 `-c` 与 `-np` 如何决定 `n_ctx`

- `-np N` 直接成为 `cparams.n_seq_max`:`common/common.cpp:1715`(`cparams.n_seq_max = params.n_parallel`)。`-kvu`/`--no-kv-unified` 设置 `params.kv_unified`:`common/arg.cpp:1728-1733`;`common/common.h:580` 默认 `false`;只有 `-np` 为 auto(-1)时 server 才自动改成 `n_parallel = 4` 且 `kv_unified = true`:`tools/server/server.cpp:168-173`。我们显式给了 `-np`,所以不加 `--no-kv-unified` 也是非 unified,该选项只是显式化。
- `llama_context` 构造里决定每序列上下文(`src/llama-context.cpp:377-396`):
  - 先 `n_ctx = GGML_PAD(n_ctx, 256)`(`:377`)。
  - `kv_unified` 时 `n_ctx_seq = n_ctx`(`:379-380`)。
  - 否则 `n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)`,并把 `n_ctx` 改写成 `n_ctx_seq * n_seq_max`(`:382-391`)。
  - 因此 `-np 4 -c 16384 --no-kv-unified` → `n_ctx_seq = 4096`;`-np 8 -c 16384 -kvu` → `n_ctx_seq = 16384`。
- slot 的 `n_ctx` = `n_ctx_slot()`:`tools/server/server-context.cpp:4655-4662`(`llama_n_ctx_seq(ctx)`,再被 `--kv-unified-per-slot` 和模型训练上下文 `min` 截断);每个 slot 在 `:1437` 取该值;`/slots` 的 `n_ctx` 字段就是 `slot.n_ctx`(`:738`);`/props` 与 `/v1/models` 用 `meta.slot_n_ctx`(`:4829`)。所以 `-kvu` 下 `/slots` 每个 slot 都报 16384 是预期行为:报告的是"单序列可用上限",不是独占份额。
- KV 张量形状:`llama_kv_cache` 构造里 `n_stream = unified ? 1 : n_seq_max`(`src/llama-kv-cache.cpp:84`),K/V 张量是 `[n_embd_gqa, kv_size, n_stream]`(`:233-234`),`kv_size` 由 `src/llama-model.cpp` 传入 `cparams.n_ctx_seq`(默认分支见 `:3295-3302`)。两种模式总单元数都等于 `n_ctx`,所以 KV buffer 都是 1792 MiB,与 `results/raw/step5/*.server.log` 中 `llama_kv_cache: size = 1792.00 MiB ( 4096 cells, ..., 4/4 seqs)`(非 unified)和 `( 16384 cells, ..., 8/1 seqs)`(`-kvu`)一致(`src/llama-kv-cache.cpp:301` 是该日志行)。
- 一个会影响口径的细节:`n_ctx / n_seq_max` 不是 256 的倍数时,`n_ctx_seq` 向**上**取整到 256 的倍数,`n_ctx` 随之被改大(日志文案写的是 "rounding down",但 `:382-391` 的代码实际可能变大)。例如 `-np 3 -c 16384` → `n_ctx_seq = 5632`,`n_ctx = 16896`;`-np 6` 同理。系列 S 的 c∈{1,2,4,8} 都能整除成 256 的倍数(16384/8 = 2048),不受影响;但若用到 np∈{3,5,6,7},内存预算 B 就不再等于 `-c`。这是我从代码推出的,没有实测。
- 默认不做 context shift:`common/common.h:578`(`ctx_shift = false`),`common/arg.cpp:1744-1748`。所以 slot 满了不会滑窗丢前文:
  - prompt 本身太长 → 在分到 slot 后被拒,`ERROR_TYPE_EXCEED_CONTEXT_SIZE`,HTTP 400,消息 `request (%d tokens) exceeds the available context size (%d tokens), try increasing it`:`tools/server/server-context.cpp:3676-3684`(不可 split 的任务是 `:3665-3673`,文案为 `input (%d tokens) is larger than the max context size`);错误 JSON 带 `n_prompt_tokens`、`n_ctx`:`tools/server/server-task.cpp:1504-1510`;状态码 400:`tools/server/server-common.cpp:68-71`。
  - 生成过程中 `prompt.n_tokens() + 1 >= slot.n_ctx` → `STOP_TYPE_LIMIT`、`truncated = true`:`tools/server/server-context.cpp:2027-2035`,响应的 `finish_reason` 为 `"length"`(`tools/server/server-task.cpp:407` 及 `:412-414`)。
  - 注意:这些检查比的是 `slot.n_ctx`(`-kvu` 下是 16384),不是池的剩余量。

### 1.2 `-kvu` 共享池耗尽时的实际行为

**没有准入控制。** 任务到达时只看有没有空闲 slot(`tools/server/server-context.cpp:2851-2855`:没有空闲 slot 才 `queue_tasks.defer`,slot 释放时由 `callback_on_release` → `pop_deferred_task` 重新入队:`:1444-1445`、`tools/server/server-queue.cpp:90-110`;这就是 `/metrics` 里 `requests_deferred` 的来源,`:2962`)。这个 defer 是"slot 用尽"而不是"池用尽";8 个 slot 各自允许到 16384,但只有 16384 个共享单元,池可以被超额预订。

池耗尽只会在 `llama_decode` 找不到连续 KV 单元时暴露:

1. `llama_context::decode` 里 `memory->init_batch` 返回 `FAILED_PREPARE` → 先尝试一次 `memory_update(true)`(碎片整理),仍失败则打印 `failed to find a memory slot for batch of size %d` 并 `return 1`:`src/llama-context.cpp:1940-1954`(`init_batch`、`prepare`、`find_slot` 分别在 `src/llama-kv-cache.cpp:724`、`:771`、`:918`)。
2. server 的 `decode()` 收到 `ret != 0`:`tools/server/server-context.cpp:4294`。
   - `ret == 1` 且 `n_batch > 1`:不报错,先 `try_clear_idle_slots()`(**仅在 `kv_unified` 时生效**:`:1813-1815`;清掉第一个有缓存 token 的空闲 slot,日志 `purging slot %d with %zu tokens`:`:1823`),没有可清的才把 `n_batch /= 2`;日志 `failed to find free space in the KV cache, retrying with smaller batch size`:`:4335-4341`,然后重试。
   - `n_batch == 1 && ret == 1`:`err = "Context size has been exceeded."`(`:4299-4302`)。随后对**所有正在处理的 slot**(不只是触发失败的那个)`send_error(slot, err)` → `slot.release()` → `slot.prompt_clear()`(`:4318-4326`),再 `throw std::runtime_error(err)`(`:4330`),被 `update_slots` 捕获后 `abort_all_slots`(`:3353-3356`)。server 进程不退出。
3. 错误类型是默认的 `ERROR_TYPE_SERVER`(`send_error` 默认参数,`:2170`),所以:
   - 非流式:HTTP **500**,体为 `{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}`:`tools/server/server-common.cpp:54`、`tools/server/server-context.cpp:4872-4876`(`error()` 取 JSON 里的 `code`)、`:4981`。
   - 流式:如果错误是**第一个**结果,同样返回 HTTP 500 的普通 JSON(`:5009-5016`);如果已经发过 chunk,HTTP 状态已是 200,错误以 SSE `data: {"error":{...}}` 事件发出并终止流(`:5109-5117`)。
4. 不是截断:`ctx_shift` 默认关,且池耗尽路径上没有任何丢 token 的逻辑。

**含义:** `-kvu` + `--cache-ram 0` 时,空闲 slot 缓存的前缀一直留在池里(`--cache-idle-slots` 在 `--cache-ram 0` 下被禁用:`:1569-1573`,与我们日志 `requires --cache-ram, disabling` 一致;`:2883-2897` 的"启动新任务时存入缓存并清空空闲 slot"也因此不执行)。池快满时,新请求的 decode 失败 → 第一反应是 `try_clear_idle_slots()` 清空闲 slot 的前缀缓存,所以先表现为"缓存命中率下降",只有活跃序列本身就把池占满、清无可清时才出现 500。

### 1.3 `--kv-unified-per-slot N`

- 定义:`common/arg.cpp:1652-1659`(仅 server;`common/common.h:640`;`tools/server/README.md:165`:"context limit per parallel slot (default: unset...) when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N")。
- 作用 1,**给池定容量**:`tools/server/server.cpp:176-186`——仅当 `kv_unified_per_slot > 0` 且 `-c` 为 0(没设)且不是 `-c 0` 显式要求最大上下文时,`n_ctx = n_parallel * N`。
- 作用 2,**给每个 slot 的上限封顶**:`n_ctx_slot()` 取 `min(n_ctx_seq, N)`(`tools/server/server-context.cpp:4655-4662`),因此 `slot.n_ctx`、`/slots` 的 `n_ctx`、上面的超长拒绝和生成 `length` 停止都按封顶后的值走;启动时会打印 `capping per-slot context ...`,若 N 大于池里每序列容量则警告"cap has no effect"(`:1353-1373`)。
- 它**不预留内存**:封顶只是软上限。若同时给 `-c`,池大小仍由 `-c` 决定,各 slot 仍共享;只有"不给 `-c`、只给 N"时池恰好等于各 slot 上限之和。推断:此时因为每个 slot(含空闲 slot 缓存的前缀)占用都 ≤ N(`:2027` 保证活跃序列 ≤ N-1),总占用 ≤ `n_parallel * N`,池耗尽路径不应被触发。这是代码推断,没有实测。

### 与实测对照

- 非 unified 每 slot 4096、`-kvu` 每 slot 16384:**一致**(`results/raw/step5/A_mmap.server.log:221`、`C_kvu_np8.server.log:221`:`n_ctx_slot = 4096, kv_unified = 'false'` / `n_ctx_slot = 16384, kv_unified = 'true'`;`C_kvu_np8.server.log:223-230` 各 slot `n_ctx = 16384`)。
- 池耗尽行为:实测没做(handoff 里"尚未验证"),**只有源码结论**,标记为待系列 U 验证。

**对我们实验的影响:** 系列 U(`-kvu -np 8 -c 16384`)里观察"池耗尽"时,预期先看到 `purging slot` / `retrying with smaller batch size` 警告导致 `cache_n` 下降,极端时才有 HTTP 500 `Context size has been exceeded.`(此时会同时让所有在跑请求失败,不是只失败一个),runner 要把"500 + 该消息"单独归类(不要混进 `error_ctx_exceeded`,那个对应的是 400 `exceed_context_size_error`);系列 S 的 c=8(每 slot 2048)里 3035 token 的 longctx 会在分到 slot 后立即得到 400。

---

## Q2. KV buffer 的分配时机、清零与"footprint 为什么是平的"

- **分配时机:** `llama_context` 构造函数里创建 memory:`src/llama-context.cpp:484`(`memory.reset(model.create_memory(...))`)。这发生在 `common_init_from_model` 内,早于 server 日志 `initializing, n_slots` 与 `listening`(日志顺序:`llama_kv_cache: CPU KV buffer size` 在 `results/raw/step5/A_mmap.server.log:206`,`initializing` 在 `:221`)。
- **一次性分配全部 `n_ctx` 单元:** `src/llama-kv-cache.cpp:233-234` 按 `[n_embd_gqa, kv_size, n_stream]` 建 K/V 张量,`:276-291`(`ggml_backend_alloc_ctx_tensors_from_buft`)分配真实 buffer,并在 `:291` 打印 `... KV buffer size = ... MiB`。大小 = 2 × 28 层 × 1024 × 16384 × 2 B(f16) = 1792 MiB,与实测一致;之后 KV 不会按需增长。
- **CPU 后端的分配方式:** `ggml_backend_cpu_buffer_type_alloc_buffer` 调用 `ggml_aligned_malloc`(`ggml/src/ggml-backend.cpp:2561`);在 macOS(`TARGET_OS_OSX`)上 `ggml_aligned_malloc` 用 `vm_allocate` 而不是 `posix_memalign`(`ggml/src/ggml.c:360-362`)。
- **显式清零:** KV 分配后立刻 `ggml_backend_buffer_clear(buf, 0)`(`src/llama-kv-cache.cpp:293`;注释 `:276` "initialize the buffers to avoid NaNs in the padding"),CPU 后端的 `clear` 就是对整块 `memset`(`ggml/src/ggml-backend.cpp:2517-2520`)。所以整块 1792 MiB 在**启动阶段**就被写过一遍,所有页都已被触碰;`vm_allocate` 本身给的是按需零页,但 `memset` 之后它们都变成了驻留(或被 OS 压缩)的脏页。
- **之后 slot 复用 KV 不再清零:** `slot.prompt_clear()` 只调 `mem.seq_rm(id, -1, -1)` 并清 prompt 元数据(`tools/server/server-context.cpp:343-349`),`llama_kv_cache::seq_rm` 只改 cell 元数据(`src/llama-kv-cache.cpp:404-`);只有 `llama_kv_cache::clear(true)` 才会再次 `memset`(`:391-402`)。负载下新 token 的 K/V 写进的是已经驻留的页,不会产生新的页占用。
- **结论:** footprint 启动后平,**取决于**"KV 在启动时一次性分配并整块写入"这一事实,这是源码直接支持的(`:293` + `ggml-backend.cpp:2519`)。同理,A→B 的 footprint 差 ≈ KV 差(3584 vs 1792 MiB)、A→C 不变,都符合 KV 大小只由 `n_ctx` 决定。
- **源码无法确认的部分:**
  1. 负载下 footprint 额外 +29 MB 的具体来源(候选:compute buffer 之外的输出/logits 缓冲、token 向量、线程栈、sampler、HTTP 缓冲等),我没有在源码里找到可以直接对应的分配点。
  2. macOS 如何统计这些脏页(为什么 RSS 剧烈起伏而 footprint 平、`mapped file` 不计入 footprint)是 OS 行为,不是 llama.cpp 源码能回答的。
  3. 权重走 mmap 与 KV 的区别:权重不是这条路径(`-lm` 控制),本节只涉及 KV。

与实测对照:KV 日志 1792 MiB = 16384 × 114,688 B、启动后 footprint 基本平、负载 +29 MB:**一致**(+29 MB 的来源见上,源码无法确认)。

**对我们实验的影响:** "内存由启动配置决定、与负载无关"在源码层面成立(预分配 + 预清零),所以 E1 的内存轴不要期待 footprint 随并发变化,KV 利用率(live_tokens / B)才是随负载变化的量;`-c`(B)就是 footprint 里 KV 的全部,公式 B × 114,688 B 可以直接用。

---

## Q3. slot 选择:LCP 相似度阈值与 `id_slot`

### 3.1 选择逻辑(`get_available_slot`,`tools/server/server-context.cpp:1696-1790`)

参数:`-sps, --slot-prompt-similarity SIMILARITY`,默认 **0.10**,`0.0` 表示关闭(`common/arg.cpp:3817-3822`;`common/common.h:706`;`tools/server/README.md:243`)。server 启动时读入 `slot_prompt_similarity`(`:1349`)。没有请求级参数能改它。

按顺序:

1. 如果请求带 `id_slot`(解析在 `:4948`,默认 -1;README `id_slot` 说明在 `tools/server/README.md:592`),直接用该 slot(`:1702-1711`,id 对 slot 数取模);该 slot 忙则任务被 defer(`:2858-2862`)。
2. 若 `slot_prompt_similarity != 0`(`:1715`):遍历**空闲且已缓存 token 的** slot(忙的跳过:`:1723-1726`;空的跳过:`:1730-1734`),计算 `f_sim_cur = LCP长度 / 新请求 prompt 的 token 数`(`:1738-1739`,分母是**新 prompt** 长度,不是 slot 里缓存的长度),取 `f_sim_cur > f_sim_best && f_sim_cur > thold` 的最大者(`:1744`,**严格大于**,所以相似度相同时前面的、即 id 小的 slot 赢)。命中则日志 `selected slot by LCP similarity, f_sim_best = ... (> ... thold), f_keep = ...`(`:1755`);`f_keep < 0.5` 时标记要把旧内容存入 prompt cache(`:1760`,`--cache-ram 0` 时 `prompt_cache` 为空,此步无效:`:1508`、`:1778`)。
3. 否则(没有 slot 超过阈值)走 LRU:`get_free_slots(1, -1)`(`:2473-2491`)按"空 slot 优先,然后 `t_last_used` 最小"排序,取第一个;日志 `selected slot by LRU, n_tokens = ..., t_last = ...`(`:1773`)。
4. 都没有空闲 slot → `ret == nullptr` → 任务 defer(`:2851-2855`)。

### 3.2 对我们日志的解读(来自 `results/raw/step55/server.log`)

- 前 4 个请求是空 slot,走 LRU(`:256`、`:279`、`:308`、`:337`),之后几乎全是 LCP:该日志里 10 次 LRU、39 次 LCP。
- 注意阈值很容易被**chat template 前缀**越过:`step55/server.log:366` 起大量 `f_sim_best = 0.103 (> 0.100 thold)`。0.103 ≈ 30/290,即 30 token 的 template 前缀(与 `cached_tokens = 30` 一致)除以约 290 token 的 short prompt。也就是说对 ~290 token 的短请求,"仅共享 template 前缀"就已经够 LCP 命中,而对 3000 token 的长请求 30/3000 = 0.01 < 0.1,走 LRU。
- 并列时 id 小的 slot 赢(`:1744` 严格大于),所以多个空闲 slot 都只有 template 前缀时,请求倾向于落到最小 id 的、有缓存的空闲 slot(推断,日志里 id 0/2 反复被选中与此相容)。
- 重要副作用(推断):agent 任务的续步前缀(如上一步 1123 token 的 prompt)在原 slot 上相似度很高(962/1123),但如果这个 slot 在 tool delay 期间被别的短请求用 LCP=0.103 抢走并覆盖,续步就只能走 LRU、`cache_n` 回落到 30。`--cache-ram 0` 下没有 RAM 缓存兜底。

### 3.3 响应与 `/slots` 是否带 `id_slot`

- 非 OAI 的 `/completion`:最终响应和流式 partial 都带 `id_slot`(`tools/server/server-task.cpp:337`、`:1048`;最终响应对象赋值 `tools/server/server-context.cpp:2234`)。
- OAI `/v1/chat/completions`:非流式最终响应(`server-task.cpp:406-452`)、流式最终 delta 数组(`:454-518`)、流式 partial chunk(`:1103-1156`)**都不含 `id_slot`**,键只有 `choices/created/id/model/object/system_fingerprint`(+ `usage`/`timings`),与 `results/raw/step4/stream_sample.txt` 一致。
- `/slots`:`server_slot::to_json`(`server-context.cpp:733-`)的键是 `id`(**就是 slot 编号**,不叫 `id_slot`)、`n_ctx`、`speculative`、`is_processing`;一旦该 slot 处理过任务,还会多出 `id_task`、`n_prompt_tokens`、`params`、`next_token` 等(`:743-749`,取 `task` 或上一个 `task_prev`)。所以 `/slots` 其实有 slot 编号(`id`),只是没有叫 `id_slot` 的键;`id_task` 可与日志里的 task 号对应。
- 可用的补救办法(源码层面,未实测):请求加 `"verbose": true` 会在响应里附 `__verbose`,内容是 `to_json_non_oaicompat()`(含 `id_slot`):非流式在 `server-task.cpp:445`,流式挂在最终 delta 数组的**第一个**元素 `:514`;字段定义见 `tools/server/server-schema.cpp:17-18`;全局默认由 `verbosity > 9` 决定(`server-schema.cpp:485`)。代价是响应体变大。另一种是靠日志 `launch_slot_: id N | task T` 事后关联。

与实测对照:LCP 阈值 0.100、`by LCP` / `by LRU` 日志、响应和 `/slots` 无 `id_slot`:**一致**(`/slots` 的 `id` 字段本身即 slot 编号,这是对"没有 id_slot"的补充)。

**对我们实验的影响:** E3/E4(跨步前缀复用)里,续步能否命中原 slot 取决于"原 slot 是否空闲且没被 template-LCP 的短请求覆盖",而不是"空闲优先";混合负载下 `cache_n` 的波动应按这个机制解释。`requests.jsonl` 的 `id_slot` 只能事后从日志或 `/slots` 的 `id_task` 对应。

---

## Q4. 客户端断连:取消、释放与 0.7 s 的来源

### 4.1 检测断连

- 每个 HTTP 请求的 `server_http_req::should_stop` 被绑定到 httplib 的 `req.is_connection_closed`(`tools/server/server-http.cpp:736`;POST 同样),后者在 httplib 里定义为 `!is_socket_alive(sock)`(`vendor/cpp-httplib/httplib.cpp:10346-10348`),`is_socket_alive` 是 0 超时的 `select` + `MSG_PEEK` 读探测(`:1956-1965`)。
- 非流式:`wait_for_all(req.should_stop)`(`tools/server/server-context.cpp:4977`)→ `server_response_reader::next`(`tools/server/server-queue.cpp:550-577`)在 `recv_with_timeout` **超时**时调用 `should_stop()`(`:555`),超时间隔 = `HTTP_POLLING_SECONDS = 1`(`server-context.cpp:40`、`:4861`)。
- 流式:每次 `set_next` 的回调开头先判 `effective_should_stop()`(`:5056`),收到下一个结果后再判(`:5090-5093`);`effective_should_stop` → `server_res_spipe::should_stop` → `!conn_alive()` → `req->should_stop()`(`tools/server/server-stream.cpp:617-629`)。仅当请求带 `X-Conversation-Id` 头时才启用"断连后继续生成并可重放"的 spipe(`server-stream.cpp:597-615`),我们不带该头,所以不适用。另外 `sink.write` 失败也会终止流(`server-http.cpp:650-652`)。

### 4.2 取消与释放

- 终止 HTTP 处理后,`server_res_generator` 析构 → `~server_response_reader()` 调 `stop()`(`tools/server/server-queue.h:218-220`)→ 若任务未完成,打印 `cancel task, id_task = %d`(`server-queue.cpp:610`,这就是日志里 `W srv stop: cancel task` 那一行)并把 `SERVER_TASK_TYPE_CANCEL` **放到队首**(`:617`)。`post` 时还会 `cleanup_pending_task`,把还排在 `queue_tasks`、`queue_tasks_deferred`、`queue_tasks_unhandled` 里的该任务删除(`server-queue.cpp:30-34`、`:368-`),所以"还没分到 slot 的排队请求"被取消不会再占 slot。
- 主循环处理 `CANCEL`:找到 `slot.task->id == id_target` 的 slot 调 `slot.release()`(`server-context.cpp:2901-2909`)。`release()` 打印 `stop processing: n_tokens = ..., truncated = ...`(`:586-590`,即日志里的 `release` 行),置 `SLOT_STATE_IDLE`,但**不清 KV/prompt**(只对 child/stateless 任务清,`:598-600`),被取消请求已处理的前缀仍可被后续请求按 LCP 复用。

### 4.3 延迟来源

1. **检测延迟**:非流式最多约 `HTTP_POLLING_SECONDS = 1 s`(断连发生在两次轮询之间的任意时刻);流式约一个 chunk 间隔(下一个 token)。
2. **等当前 decode 结束**:`decode()` 在 `queue_tasks.yield_to_queue(...)` 里执行 `llama_process`(`server-context.cpp:4287-4291`);yield 期间主线程不处理 `CANCEL`——`process_single_task` 在 `is_yielding` 时只放行 `METRICS` 和 `SLOT_GET`,其余被"拒收"并放回(`server-context.cpp:2823-2826`,`server-queue.cpp:138-160`、`:222-263`)。所以 `CANCEL` 要等当前这次 `llama_process` 返回才被处理。
3. 我们的日志直接给出了第 2 项的量级:`results/raw/step4/server_smoke_run1.log:41-42`,`cancel task` 在 1.19.256,`release` 在 1.19.382,相隔 **0.126 s**,而该时段(低电量模式)单 token decode ≈ 137 ms(同文件 `:34` 的 `eval time ... 137.12 ms per token`)——即约一个 decode 步。
4. 因此 0.7 s 可以分解为:约 ≤1 s 的轮询检测延迟(0.7 s 落在 0~1 s 内)+ 约 0.13 s 的 decode 步。这与源码相容。

### 4.4 需要注意的边界(推断,未实测)

- **prefill 阶段取消延迟可能大得多**:一次 `llama_process` 处理的是整个 batch(`n_batch` 默认 2048,`n_ubatch` 512:`common/common.h:458-459`,日志 `llama_context: n_batch = 2048`),prefill 阶段单次调用可含最多 2048 个 token。按我们校准的 prefill 约 112–123 tok/s,一次 2048-token 的 batch 可达十几秒,期间 `CANCEL` 不会被处理。长上下文请求超时断连后,slot 可能还要被占用到当前 batch 结束。
- **非流式 + 服务器有别的流量时检测可能被推迟**:`recv_with_timeout` 用 `condition_results.wait_for`,**任何**带结果的 `send()` 都会 `notify_all`(`server-queue.cpp:489`),被唤醒但没有自己的结果时会重新开始一个完整的 1 s 等待(`:450-470`),只有在整整 1 s 内没有任何唤醒时才返回 `nullptr` 并检查 `should_stop`(`:462-468`)。当别的请求(尤其是流式)持续产生结果时,一个非流式请求(或在排队/prefill 中还没有自己第一个结果的流式请求)的断连可能要等到它自己得到结果才被发现。我读到的就是这样,但没有实测。
- 这个"排队中的请求断连也要等到被调度才发现"的情况,在我们的设计里(`c ≤ np`,runner 在客户端限并发,服务器端不排队)基本不会出现。

### 与实测对照

断连后约 0.7 s 内 `is_processing` 回 false、日志 `cancel task` → `release`:**一致**。测试 7 当时用的是流式还是非流式,手册给出的命令是非流式 curl(`docs/phase1-steps.md:228`),但没有保存实际命令,**源码无法确认**走的是哪条检测路径;两条路径都与 0.7 s 相容。

**对我们实验的影响:** runner 超时直接断连是可行的,按 ~1 s(非流式)或一个 token 间隔(流式)+ 一个 decode 步估计释放延迟;但 longctx 的 prefill 阶段超时可能让 slot 在断连后继续被占用十几秒,污染下一个请求的 TTFT,所以这类任务的超时统计要把"断连后 slot 仍忙"的窗口记入(例如断连后轮询 `/slots` 的 `is_processing` 再发下一个请求);runner 用流式请求(本来也需要 TTFT)检测更及时。

---

## Q5. `/v1/chat/completions` 上的 `ignore_eos` 与流式 `timings`

### 5.1 `ignore_eos`

- OAI 处理函数 `oaicompat_chat_params_parse` 把 body 里未被特殊处理的字段**原样复制**到 `llama_params`("Copy remaining properties ... This allows user to use llama.cpp-specific params like mirostat via OAI endpoint"):`tools/server/server-common.cpp:1469-1477`。所以 `ignore_eos` 会透传;`max_tokens` 通过 schema 别名映射到 `n_predict`(`tools/server/server-schema.cpp:46-47`)。
- 入口 `post_chat_completions`(`tools/server/server-context.cpp:5560-`)→ `handle_completions_impl` → `eval_llama_cmpl_schema`(`:4943`)。schema 字段 `ignore_eos`:`tools/server/server-schema.cpp:423-433`——为真时把预先算好的 EOG logit bias(每个 EOG token 为 `-INFINITY`)追加进 `sampling.logit_bias`。这份 `logit_bias_eog` 在 `common/common.cpp:1343-1349` 对词表中所有 `llama_vocab_is_eog` 的 token 生成(Llama 3.2 有多个 EOG,`<|eot_id|>`、`<|end_of_text|>` 等都被屏蔽),并经 `meta->logit_bias_eog` 传给 schema(`server-context.cpp:4844`、`:4943`)。
- 生效效果:EOG token 概率为 0,不会采样到,`STOP_TYPE_EOS` 分支(`server-context.cpp:2093-2096`)不会触发,只剩 `max_tokens`/stop 词/上下文上限三种停止方式,`finish_reason` 为 `"length"`。README 把它列为通用采样参数:`tools/server/README.md:582`("Ignore end of stream token and continue generating. Default: false")。

与实测对照(`completion_tokens = 200`、`finish_reason = length`):**一致**。

### 5.2 流式 `timings` 出现在哪个 chunk

- 流式时,最终结果由 `server_task_result_cmpl_final::to_json_oaicompat_chat_stream` 生成一个 **delta 数组**(`tools/server/server-task.cpp:454-518`):先是剩余内容 delta,然后一个 `finish_reason` 的 chunk(`:476-488`),如果 `stream_options.include_usage` 为真再追加一个 `choices: []` 的 usage chunk(`:494-505`;字段 `stream_options.include_usage` 见 `server-schema.cpp:26-27`);最后 `if (stats.is_set()) deltas.back()["timings"] = stats.to_json()`(`:508-510`)——`timings` 挂在数组**最后一个元素**上。
  - 默认(不带 `include_usage`):最后一个元素就是带 `finish_reason` 的 chunk,所以 `timings` 在带 `finish_reason` 的最后一个数据 chunk 里,之后才是 `data: [DONE]`(`server-context.cpp:5080` 的 `output = "data: [DONE]\n\n"`)。这与 `results/raw/step4/stream_sample.txt` 结尾一致。
  - 带 `include_usage: true`:`timings` 会在末尾那个 `choices: []`(含 `usage`)的 chunk 上,而不是 `finish_reason` 的 chunk。README 的示例(`tools/server/README.md:1426-1445`)就是 `"choices": []` + `timings`。**如果 runner 以后开启 `include_usage`,解析要改。**
- 中间 chunk(`server_task_result_cmpl_partial::to_json_oaicompat_chat`,`server-task.cpp:1103-1156`)只有在 `timings_per_token: true`(`tools/server/README.md:600`)时才带 `timings`(`server-task.cpp:1147-1149` 判断 `stats.is_set()`;`stats` 仅在 `slot.stop != STOP_TYPE_NONE || timings_per_token` 时才赋值:`server-context.cpp:2222-2225`)。
- 非流式:`timings` 在响应顶层(`server-task.cpp:447-449`),另有 `usage`(`:357-364`:`completion_tokens`、`prompt_tokens`、`total_tokens`、`prompt_tokens_details.cached_tokens`)。

### 5.3 `timings` 包含的字段(`server_slot_stats::to_json`,`tools/server/server-common.cpp:84-107`)

`cache_n`(复用的 prompt token 数)、`prompt_n`(本次实际处理的 prompt token 数,不含缓存部分)、`prompt_ms`、`prompt_per_token_ms`、`prompt_per_second`、`predicted_n`、`predicted_ms`、`predicted_per_token_ms`、`predicted_per_second`;只有使用推测解码时才额外有 `draft_n`、`draft_n_accepted`(`:99-102`)。README 的说明与示例见 `tools/server/README.md:1426-1456`("The total number of tokens in context is equal to `prompt_n + cache_n + predicted_n`")。我们 `stream_sample.txt` 最后一个 chunk 的键集合与上述完全一致(`cache_n: 30, prompt_n: 10, predicted_n: 15`)。

与实测对照:**一致**。

**对我们实验的影响:** runner 可以放心用流式 + `ignore_eos` 强制固定输出长度;prefill 耗时/命中量从最后一个 chunk 的 `timings.prompt_ms / prompt_n / cache_n` 取(流式时 TTFT 仍要自己在客户端计时);解析逻辑应读"最后一个含 `timings` 的数据 chunk"而不是写死 `finish_reason` chunk,以免日后开启 `include_usage` 时漏掉。

---

## 未能确认 / 需要后续实测的事项汇总

1. `-kvu` 池耗尽的真实表现(500 与否、`purging slot` 日志、`cache_n` 变化):只有源码路径,系列 U 需要实测。
2. 测试 7 的断连是流式还是非流式,以及 prefill 阶段断连的释放延迟:源码给出机制,未实测。
3. 负载下 footprint +29 MB 的来源、macOS 对脏页的统计方式:源码无法确认。
4. `--kv-unified-per-slot`(不给 `-c`)是否确实使池耗尽不可达:代码推断,未实测。
5. 非流式请求在有其它流量时断连检测被 `notify_all` 推迟:代码阅读推断,未实测。
