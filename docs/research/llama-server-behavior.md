# llama-server behavior verification (source code + official documentation)

- Commit investigated: `1b29621bacc6a446bd523fc790316a24a06e886f` (b11741)
- `git rev-parse HEAD` (in `~/src/llama.cpp`): `1b29621bacc6a446bd523fc790316a24a06e886f`, consistent with the target commit; `git status --short` is empty (clean working tree), so all line numbers below were obtained by reading the working tree directly, equivalent to the content of the pinned commit.
- Date: 2026-10-19
- Method: only read the source and `tools/server/README.md`; no experiments were run, `~/src/llama.cpp` was not modified, and no checkout/pull was done. Paths are all relative to `~/src/llama.cpp`. The measured comparisons come from `docs/lab-notebook.md`, `docs/phase1-handoff.md` and `results/raw/`.
- Labeling convention: **consistent** / **contradictory** / **cannot be confirmed from the source**. "Inferred" means a conclusion derived from reading the code but not verified by measurement.

## Summary table

| # | Question | Conclusion (one sentence) | vs. measurement |
|---|---|---|---|
| 1 | How `-c`/`-np`/`-kvu` determine per-slot `n_ctx`; pool exhaustion behavior | Non-unified: `n_ctx_seq = pad256(n_ctx/np)`; `-kvu`: every slot equals the whole `n_ctx`, with no admission control between slots. When the pool is exhausted it is neither queueing nor truncation: first the caches of idle slots are cleared and the batch is shrunk for a retry, and if it still fails **all running requests** get `"Context size has been exceeded."` (HTTP 500 / SSE error in the stream), and the KV of those slots is cleared | **Consistent** (the n_ctx part); pool exhaustion behavior was not measured, only a source-code conclusion |
| 2 | When the KV buffer is allocated, whether it is zeroed | Allocated in one go (`vm_allocate` on macOS) and the whole block `memset(0)` when `llama_context` is constructed (before the server reports listening), so all pages have been written at startup, and afterwards load only overwrites already-resident pages | **Consistent** (fully occupied at startup, flat footprint); the origin of the +29 MB under load and the effect of macOS compressed memory on RSS **cannot be confirmed from the source** |
| 3 | Slot selection and `id_slot` | First select the idle slot with the highest similarity when LCP / new prompt length > `-sps` (default 0.10, strictly greater); otherwise pick an empty slot, then LRU; the OAI chat response, streaming chunks and `/slots` all contain no `id_slot` | **Consistent** |
| 4 | Whether a disconnect cancels generation | Yes. The HTTP thread detects a disconnect via `is_connection_closed` (socket probe), and sends a `CANCEL` task when `server_response_reader` is destructed; the main loop only handles it after the current `llama_process` returns. Non-streaming detection relies on 1 s polling, streaming checks on every chunk | **Consistent** (0.7 s ≈ ≤1 s polling + about 1 decode step); whether test 7 was streaming or non-streaming **cannot be confirmed from the source** |
| 5 | `ignore_eos` and streaming `timings` | `ignore_eos` enters the schema through "remaining-field passthrough" and adds a `-inf` logit bias to all EOG tokens, taking effect on `/v1/chat/completions`; when streaming, `timings` hangs on the last element of the last delta array (without `include_usage` it is the `finish_reason` chunk), fields below | **Consistent** |

---

## Q1. Per-slot `n_ctx`, and behavior when the shared pool (`-kvu`) is exhausted

### 1.1 How `-c` and `-np` determine `n_ctx`

- `-np N` directly becomes `cparams.n_seq_max`: `common/common.cpp:1715` (`cparams.n_seq_max = params.n_parallel`). `-kvu`/`--no-kv-unified` set `params.kv_unified`: `common/arg.cpp:1728-1733`; `common/common.h:580` defaults to `false`; only when `-np` is auto (-1) does the server automatically change to `n_parallel = 4` and `kv_unified = true`: `tools/server/server.cpp:168-173`. We pass `-np` explicitly, so even without `--no-kv-unified` it is non-unified, and that option merely makes it explicit.
- The per-sequence context is decided in the `llama_context` constructor (`src/llama-context.cpp:377-396`):
  - First `n_ctx = GGML_PAD(n_ctx, 256)` (`:377`).
  - With `kv_unified`, `n_ctx_seq = n_ctx` (`:379-380`).
  - Otherwise `n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)`, and `n_ctx` is rewritten as `n_ctx_seq * n_seq_max` (`:382-391`).
  - Hence `-np 4 -c 16384 --no-kv-unified` → `n_ctx_seq = 4096`; `-np 8 -c 16384 -kvu` → `n_ctx_seq = 16384`.
- A slot's `n_ctx` = `n_ctx_slot()`: `tools/server/server-context.cpp:4655-4662` (`llama_n_ctx_seq(ctx)`, then capped by `min` with `--kv-unified-per-slot` and the model's training context); each slot takes this value at `:1437`; the `n_ctx` field of `/slots` is `slot.n_ctx` (`:738`); `/props` and `/v1/models` use `meta.slot_n_ctx` (`:4829`). So under `-kvu` every slot in `/slots` reporting 16384 is expected: it reports the "usable upper limit of a single sequence", not an exclusive share.
- KV tensor shape: in the `llama_kv_cache` constructor `n_stream = unified ? 1 : n_seq_max` (`src/llama-kv-cache.cpp:84`), and the K/V tensors are `[n_embd_gqa, kv_size, n_stream]` (`:233-234`), with `kv_size` passed in from `cparams.n_ctx_seq` by `src/llama-model.cpp` (default branch at `:3295-3302`). In both modes the total number of cells equals `n_ctx`, so the KV buffer is 1792 MiB in both, consistent with `llama_kv_cache: size = 1792.00 MiB ( 4096 cells, ..., 4/4 seqs)` (non-unified) and `( 16384 cells, ..., 8/1 seqs)` (`-kvu`) in `results/raw/step5/*.server.log` (`src/llama-kv-cache.cpp:301` is that log line).
- A detail that affects the definitions: when `n_ctx / n_seq_max` is not a multiple of 256, `n_ctx_seq` is rounded **up** to a multiple of 256 and `n_ctx` is enlarged accordingly (the log text says "rounding down", but the code at `:382-391` can actually make it larger). For example `-np 3 -c 16384` → `n_ctx_seq = 5632`, `n_ctx = 16896`; `-np 6` likewise. c∈{1,2,4,8} of series S are all divisible into multiples of 256 (16384/8 = 2048) and unaffected; but with np∈{3,5,6,7}, the memory budget B no longer equals `-c`. This is deduced by me from the code and not measured.
- No context shift by default: `common/common.h:578` (`ctx_shift = false`), `common/arg.cpp:1744-1748`. So when a slot is full it does not slide a window and drop earlier text:
  - If the prompt itself is too long → it is rejected after being assigned a slot, `ERROR_TYPE_EXCEED_CONTEXT_SIZE`, HTTP 400, message `request (%d tokens) exceeds the available context size (%d tokens), try increasing it`: `tools/server/server-context.cpp:3676-3684` (for non-splittable tasks `:3665-3673`, with the text `input (%d tokens) is larger than the max context size`); the error JSON carries `n_prompt_tokens`, `n_ctx`: `tools/server/server-task.cpp:1504-1510`; status code 400: `tools/server/server-common.cpp:68-71`.
  - During generation, `prompt.n_tokens() + 1 >= slot.n_ctx` → `STOP_TYPE_LIMIT`, `truncated = true`: `tools/server/server-context.cpp:2027-2035`, and the response's `finish_reason` is `"length"` (`tools/server/server-task.cpp:407` and `:412-414`).
  - Note: these checks compare against `slot.n_ctx` (16384 under `-kvu`), not the remaining amount in the pool.

### 1.2 Actual behavior when the `-kvu` shared pool is exhausted

**There is no admission control.** When a task arrives it only checks whether there is an idle slot (`tools/server/server-context.cpp:2851-2855`: only if there is no idle slot is `queue_tasks.defer` called, and when a slot is released `callback_on_release` → `pop_deferred_task` re-enqueues: `:1444-1445`, `tools/server/server-queue.cpp:90-110`; this is the origin of `requests_deferred` in `/metrics`, `:2962`). This defer is "slots used up", not "pool used up"; each of the 8 slots is allowed up to 16384, but there are only 16384 shared cells, so the pool can be oversubscribed.

Pool exhaustion is exposed only when `llama_decode` cannot find contiguous KV cells:

1. In `llama_context::decode`, `memory->init_batch` returns `FAILED_PREPARE` → first try one `memory_update(true)` (defragmentation), and if it still fails print `failed to find a memory slot for batch of size %d` and `return 1`: `src/llama-context.cpp:1940-1954` (`init_batch`, `prepare`, `find_slot` are at `src/llama-kv-cache.cpp:724`, `:771`, `:918` respectively).
2. The server's `decode()` receives `ret != 0`: `tools/server/server-context.cpp:4294`.
   - `ret == 1` and `n_batch > 1`: no error; first `try_clear_idle_slots()` (**effective only with `kv_unified`**: `:1813-1815`; it clears the first idle slot that has cached tokens, log `purging slot %d with %zu tokens`: `:1823`), and only if nothing can be cleared is `n_batch /= 2`; log `failed to find free space in the KV cache, retrying with smaller batch size`: `:4335-4341`, then retry.
   - `n_batch == 1 && ret == 1`: `err = "Context size has been exceeded."` (`:4299-4302`). Then for **all slots being processed** (not only the one that triggered the failure) `send_error(slot, err)` → `slot.release()` → `slot.prompt_clear()` (`:4318-4326`), then `throw std::runtime_error(err)` (`:4330`), which is caught by `update_slots` and followed by `abort_all_slots` (`:3353-3356`). The server process does not exit.
3. The error type is the default `ERROR_TYPE_SERVER` (the default argument of `send_error`, `:2170`), so:
   - Non-streaming: HTTP **500**, with body `{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}`: `tools/server/server-common.cpp:54`, `tools/server/server-context.cpp:4872-4876` (`error()` takes the `code` from the JSON), `:4981`.
   - Streaming: if the error is the **first** result, it likewise returns an ordinary JSON with HTTP 500 (`:5009-5016`); if chunks have already been sent, the HTTP status is already 200, and the error is emitted as an SSE `data: {"error":{...}}` event and terminates the stream (`:5109-5117`).
4. It is not truncation: `ctx_shift` is off by default, and there is no logic that drops tokens on the pool exhaustion path.

**Implication:** with `-kvu` + `--cache-ram 0`, the prefixes cached by idle slots stay in the pool (`--cache-idle-slots` is disabled under `--cache-ram 0`: `:1569-1573`, consistent with `requires --cache-ram, disabling` in our log; the "store into the cache and clear the idle slot when starting a new task" at `:2883-2897` therefore does not execute either). When the pool is nearly full, a new request's decode fails → the first reaction is `try_clear_idle_slots()` to clear the idle slots' prefix caches, so it first appears as "cache hit rate drops", and the 500 only appears when active sequences alone fill the pool and there is nothing left to clear.

### 1.3 `--kv-unified-per-slot N`

- Definition: `common/arg.cpp:1652-1659` (server only; `common/common.h:640`; `tools/server/README.md:165`: "context limit per parallel slot (default: unset...) when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N").
- Effect 1, **sizing the pool**: `tools/server/server.cpp:176-186` — only when `kv_unified_per_slot > 0` and `-c` is 0 (not set) and `-c 0` was not explicitly requested for maximum context, `n_ctx = n_parallel * N`.
- Effect 2, **capping each slot's limit**: `n_ctx_slot()` takes `min(n_ctx_seq, N)` (`tools/server/server-context.cpp:4655-4662`), so `slot.n_ctx`, the `n_ctx` of `/slots`, the overlong rejection above and the generation `length` stop all go by the capped value; at startup it prints `capping per-slot context ...`, and warns "cap has no effect" if N is larger than the per-sequence capacity in the pool (`:1353-1373`).
- It **does not reserve memory**: the cap is only a soft upper limit. If `-c` is also given, the pool size is still decided by `-c` and the slots still share it; only with "no `-c`, only N" does the pool equal exactly the sum of the slots' limits. Inferred: in that case, since every slot's usage (including the prefix cached by an idle slot) is ≤ N (`:2027` guarantees active sequences ≤ N-1), the total usage ≤ `n_parallel * N` and the pool exhaustion path should not be triggered. This is a code inference and was not measured.

### Comparison with measurement

- Non-unified 4096 per slot, `-kvu` 16384 per slot: **consistent** (`results/raw/step5/A_mmap.server.log:221`, `C_kvu_np8.server.log:221`: `n_ctx_slot = 4096, kv_unified = 'false'` / `n_ctx_slot = 16384, kv_unified = 'true'`; `C_kvu_np8.server.log:223-230` each slot `n_ctx = 16384`).
- Pool exhaustion behavior: not measured (listed as "not yet verified" in the handoff), **only a source-code conclusion**, marked as to be verified in series U.

**Impact on our experiments:** when observing "pool exhaustion" in series U (`-kvu -np 8 -c 16384`), expect to first see `purging slot` / `retrying with smaller batch size` warnings that cause `cache_n` to drop, and only in extreme cases an HTTP 500 `Context size has been exceeded.` (which makes all running requests fail at the same time, not just one); the runner must classify "500 + that message" separately (don't mix it into `error_ctx_exceeded`, which corresponds to the 400 `exceed_context_size_error`); in c=8 of series S (2048 per slot), the 3035-token longctx will immediately get a 400 after being assigned a slot.

---

## Q2. When the KV buffer is allocated, zeroing, and "why footprint is flat"

- **Allocation time:** the memory is created in the `llama_context` constructor: `src/llama-context.cpp:484` (`memory.reset(model.create_memory(...))`). This happens inside `common_init_from_model`, earlier than the server log's `initializing, n_slots` and `listening` (log order: `llama_kv_cache: CPU KV buffer size` at `results/raw/step5/A_mmap.server.log:206`, `initializing` at `:221`).
- **All `n_ctx` cells allocated at once:** `src/llama-kv-cache.cpp:233-234` builds the K/V tensors as `[n_embd_gqa, kv_size, n_stream]`, `:276-291` (`ggml_backend_alloc_ctx_tensors_from_buft`) allocates the real buffer, and `:291` prints `... KV buffer size = ... MiB`. Size = 2 × 28 layers × 1024 × 16384 × 2 B (f16) = 1792 MiB, consistent with measurement; afterwards KV does not grow on demand.
- **How the CPU backend allocates:** `ggml_backend_cpu_buffer_type_alloc_buffer` calls `ggml_aligned_malloc` (`ggml/src/ggml-backend.cpp:2561`); on macOS (`TARGET_OS_OSX`), `ggml_aligned_malloc` uses `vm_allocate` rather than `posix_memalign` (`ggml/src/ggml.c:360-362`).
- **Explicit zeroing:** right after the KV allocation `ggml_backend_buffer_clear(buf, 0)` is called (`src/llama-kv-cache.cpp:293`; the comment at `:276` "initialize the buffers to avoid NaNs in the padding"), and the CPU backend's `clear` is a `memset` over the whole block (`ggml/src/ggml-backend.cpp:2517-2520`). So the whole 1792 MiB is written once at the **startup stage**, and every page has been touched; `vm_allocate` itself gives demand-zero pages, but after the `memset` they all become resident (or OS-compressed) dirty pages.
- **Slots reusing KV afterwards is not zeroed again:** `slot.prompt_clear()` only calls `mem.seq_rm(id, -1, -1)` and clears prompt metadata (`tools/server/server-context.cpp:343-349`), and `llama_kv_cache::seq_rm` only changes cell metadata (`src/llama-kv-cache.cpp:404-`); only `llama_kv_cache::clear(true)` does `memset` again (`:391-402`). Under load, the K/V of new tokens is written into already-resident pages and produces no new page usage.
- **Conclusion:** footprint being flat after startup **depends on** the fact that "KV is allocated at startup in one go and written as a whole", which is directly supported by the source (`:293` + `ggml-backend.cpp:2519`). Likewise, the A→B footprint difference ≈ KV difference (3584 vs 1792 MiB) and A→C unchanged are both consistent with KV size being determined only by `n_ctx`.
- **Parts that cannot be confirmed from the source:**
  1. The exact origin of the extra +29 MB of footprint under load (candidates: output/logits buffers other than the compute buffer, token vectors, thread stacks, sampler, HTTP buffers, etc.); I did not find an allocation point in the source that corresponds directly.
  2. How macOS accounts for these dirty pages (why RSS fluctuates wildly while footprint is flat, and why `mapped file` is not counted in footprint) is OS behavior and cannot be answered from the llama.cpp source.
  3. The difference between weights going through mmap and KV: weights are not on this path (controlled by `-lm`); this section covers only KV.

Comparison with measurement: KV log 1792 MiB = 16384 × 114,688 B, footprint basically flat after startup, +29 MB under load: **consistent** (for the origin of the +29 MB see above, cannot be confirmed from the source).

**Impact on our experiments:** "memory is determined by the startup configuration and is independent of load" holds at the source level (preallocation + prezeroing), so for the memory axis of E1 do not expect footprint to change with concurrency; KV utilization (live_tokens / B) is the quantity that changes with load; `-c` (B) is all of the KV in footprint, and the formula B × 114,688 B can be used directly.

---

## Q3. Slot selection: LCP similarity threshold and `id_slot`

### 3.1 Selection logic (`get_available_slot`, `tools/server/server-context.cpp:1696-1790`)

Parameter: `-sps, --slot-prompt-similarity SIMILARITY`, default **0.10**, with `0.0` meaning off (`common/arg.cpp:3817-3822`; `common/common.h:706`; `tools/server/README.md:243`). The server reads `slot_prompt_similarity` at startup (`:1349`). No request-level parameter can change it.

In order:

1. If the request carries `id_slot` (parsed at `:4948`, default -1; the README description of `id_slot` is at `tools/server/README.md:592`), that slot is used directly (`:1702-1711`, the id taken modulo the number of slots); if that slot is busy the task is deferred (`:2858-2862`).
2. If `slot_prompt_similarity != 0` (`:1715`): iterate over the slots that are **idle and have cached tokens** (busy ones are skipped: `:1723-1726`; empty ones are skipped: `:1730-1734`), compute `f_sim_cur = LCP length / number of tokens of the new request's prompt` (`:1738-1739`, the denominator is the length of the **new prompt**, not the cached length in the slot), and take the largest with `f_sim_cur > f_sim_best && f_sim_cur > thold` (`:1744`, **strictly greater**, so on equal similarity the earlier one, i.e. the slot with the smaller id, wins). On a hit the log prints `selected slot by LCP similarity, f_sim_best = ... (> ... thold), f_keep = ...` (`:1755`); when `f_keep < 0.5` the old content is marked to be saved into the prompt cache (`:1760`; with `--cache-ram 0` the `prompt_cache` is empty and this step has no effect: `:1508`, `:1778`).
3. Otherwise (no slot exceeds the threshold) go to LRU: `get_free_slots(1, -1)` (`:2473-2491`) sorts by "empty slots first, then smallest `t_last_used`" and takes the first; log `selected slot by LRU, n_tokens = ..., t_last = ...` (`:1773`).
4. If there is no idle slot → `ret == nullptr` → the task is deferred (`:2851-2855`).

### 3.2 Interpreting our logs (from `results/raw/step55/server.log`)

- The first 4 requests got empty slots and went through LRU (`:256`, `:279`, `:308`, `:337`), and afterwards it was almost all LCP: in that log 10 LRU and 39 LCP.
- Note that the threshold is easily crossed by the **chat template prefix**: from `step55/server.log:366` on there are many `f_sim_best = 0.103 (> 0.100 thold)`. 0.103 ≈ 30/290, i.e. the 30-token template prefix (consistent with `cached_tokens = 30`) divided by a short prompt of about 290 tokens. In other words, for short requests of ~290 tokens, "sharing only the template prefix" is enough for an LCP hit, whereas for a 3000-token long request 30/3000 = 0.01 < 0.1 and it goes through LRU.
- On ties the slot with the smaller id wins (`:1744` strictly greater), so when several idle slots have only the template prefix, requests tend to land in the cached idle slot with the smallest id (inferred; ids 0/2 being selected repeatedly in the log is compatible with this).
- Important side effect (inferred): the continuation-step prefix of an agent task (e.g. the 1123-token prompt of the previous step) has high similarity (962/1123) on its original slot, but if this slot is taken and overwritten by another short request via LCP=0.103 during the tool delay, the continuation step can only go through LRU and `cache_n` falls back to 30. With `--cache-ram 0` there is no RAM cache as a safety net.

### 3.3 Whether the response and `/slots` carry `id_slot`

- Non-OAI `/completion`: both the final response and streaming partials carry `id_slot` (`tools/server/server-task.cpp:337`, `:1048`; the final response object is assigned at `tools/server/server-context.cpp:2234`).
- OAI `/v1/chat/completions`: the non-streaming final response (`server-task.cpp:406-452`), the streaming final delta array (`:454-518`) and the streaming partial chunk (`:1103-1156`) **all contain no `id_slot`**, with only the keys `choices/created/id/model/object/system_fingerprint` (+ `usage`/`timings`), consistent with `results/raw/step4/stream_sample.txt`.
- `/slots`: the keys of `server_slot::to_json` (`server-context.cpp:733-`) are `id` (**which is the slot number**, not named `id_slot`), `n_ctx`, `speculative`, `is_processing`; once the slot has processed a task, there are also `id_task`, `n_prompt_tokens`, `params`, `next_token`, etc. (`:743-749`, taken from `task` or the previous `task_prev`). So `/slots` does have the slot number (`id`), just no key named `id_slot`; `id_task` can be matched with the task number in the log.
- A remedy available (at source level, not measured): adding `"verbose": true` to the request attaches `__verbose` to the response, whose content is `to_json_non_oaicompat()` (containing `id_slot`): for non-streaming at `server-task.cpp:445`, and for streaming it hangs on the **first** element of the final delta array `:514`; the field definition is at `tools/server/server-schema.cpp:17-18`; the global default is decided by `verbosity > 9` (`server-schema.cpp:485`). The cost is a larger response body. Another way is to correlate afterwards using the log `launch_slot_: id N | task T`.

Comparison with measurement: the LCP threshold 0.100, the `by LCP` / `by LRU` logs, and no `id_slot` in the response and `/slots`: **consistent** (the `id` field of `/slots` is itself the slot number, which supplements "there is no id_slot").

**Impact on our experiments:** in E3/E4 (cross-step prefix reuse), whether a continuation step hits its original slot depends on "whether the original slot is idle and not overwritten by a short request via template LCP", not on "idle first"; fluctuations of `cache_n` under a mixed workload should be explained by this mechanism. The `id_slot` of `requests.jsonl` can only be matched afterwards from the log or the `id_task` of `/slots`.

---

## Q4. Client disconnect: cancellation, release, and the origin of 0.7 s

### 4.1 Detecting the disconnect

- The `server_http_req::should_stop` of each HTTP request is bound to httplib's `req.is_connection_closed` (`tools/server/server-http.cpp:736`; likewise for POST), which in httplib is defined as `!is_socket_alive(sock)` (`vendor/cpp-httplib/httplib.cpp:10346-10348`), and `is_socket_alive` is a zero-timeout `select` + `MSG_PEEK` read probe (`:1956-1965`).
- Non-streaming: `wait_for_all(req.should_stop)` (`tools/server/server-context.cpp:4977`) → `server_response_reader::next` (`tools/server/server-queue.cpp:550-577`) calls `should_stop()` when `recv_with_timeout` **times out** (`:555`), with the timeout interval = `HTTP_POLLING_SECONDS = 1` (`server-context.cpp:40`, `:4861`).
- Streaming: the callback of every `set_next` first checks `effective_should_stop()` (`:5056`), and checks again after receiving the next result (`:5090-5093`); `effective_should_stop` → `server_res_spipe::should_stop` → `!conn_alive()` → `req->should_stop()` (`tools/server/server-stream.cpp:617-629`). The spipe that "continues generating after a disconnect and can replay" is enabled only when the request carries the `X-Conversation-Id` header (`server-stream.cpp:597-615`); we do not send that header, so it does not apply. Also, a failing `sink.write` terminates the stream (`server-http.cpp:650-652`).

### 4.2 Cancellation and release

- After HTTP processing is terminated, the `server_res_generator` destructs → `~server_response_reader()` calls `stop()` (`tools/server/server-queue.h:218-220`) → if the task is not finished, it prints `cancel task, id_task = %d` (`server-queue.cpp:610`, which is the `W srv stop: cancel task` line in the log) and puts `SERVER_TASK_TYPE_CANCEL` **at the head of the queue** (`:617`). On `post` it also runs `cleanup_pending_task`, deleting that task if it is still in `queue_tasks`, `queue_tasks_deferred` or `queue_tasks_unhandled` (`server-queue.cpp:30-34`, `:368-`), so cancelling a "queued request not yet assigned a slot" no longer takes up a slot.
- The main loop handles `CANCEL`: it finds the slot with `slot.task->id == id_target` and calls `slot.release()` (`server-context.cpp:2901-2909`). `release()` prints `stop processing: n_tokens = ..., truncated = ...` (`:586-590`, i.e. the `release` line in the log), sets `SLOT_STATE_IDLE`, but **does not clear KV/prompt** (it only does so for child/stateless tasks, `:598-600`), so the prefix already processed by the cancelled request can still be reused by later requests via LCP.

### 4.3 Sources of latency

1. **Detection latency**: for non-streaming up to about `HTTP_POLLING_SECONDS = 1 s` (the disconnect happens at any moment between two polls); for streaming about one chunk interval (the next token).
2. **Waiting for the current decode to finish**: `decode()` executes `llama_process` inside `queue_tasks.yield_to_queue(...)` (`server-context.cpp:4287-4291`); during the yield the main thread does not handle `CANCEL` — when `is_yielding`, `process_single_task` lets only `METRICS` and `SLOT_GET` through, and the rest are "rejected" and put back (`server-context.cpp:2823-2826`, `server-queue.cpp:138-160`, `:222-263`). So `CANCEL` is handled only after the current `llama_process` returns.
3. Our log directly gives the order of magnitude for item 2: `results/raw/step4/server_smoke_run1.log:41-42`, `cancel task` at 1.19.256 and `release` at 1.19.382, separated by **0.126 s**, while single-token decode in that period (low-power mode) was ≈ 137 ms (`eval time ... 137.12 ms per token` at `:34` of the same file) — about one decode step.
4. Therefore 0.7 s can be decomposed as: a polling detection delay of about ≤1 s (0.7 s falls within 0~1 s) + a decode step of about 0.13 s. This is compatible with the source.

### 4.4 Boundaries to note (inferred, not measured)

- **Cancellation latency during the prefill phase may be much larger**: one `llama_process` handles a whole batch (`n_batch` defaults to 2048, `n_ubatch` 512: `common/common.h:458-459`, log `llama_context: n_batch = 2048`), and during prefill a single call can contain up to 2048 tokens. At our calibrated prefill of about 112–123 tok/s, one 2048-token batch can take over ten seconds, during which `CANCEL` is not handled. After a long-context request times out and disconnects, the slot may remain occupied until the current batch finishes.
- **For non-streaming with other traffic on the server, detection may be delayed**: `recv_with_timeout` uses `condition_results.wait_for`, and **any** `send()` carrying a result does `notify_all` (`server-queue.cpp:489`); when woken without a result of its own it restarts a full 1 s wait (`:450-470`), and only when no wake-up happens for a whole 1 s does it return `nullptr` and check `should_stop` (`:462-468`). When other requests (especially streaming ones) keep producing results, the disconnect of a non-streaming request (or of a streaming request that does not yet have its own first result while queued/in prefill) may not be noticed until it gets a result of its own. That is what I read, but it was not measured.
- This case, "a disconnected queued request is noticed only when it gets scheduled", basically does not occur in our design (`c ≤ np`, the runner limits concurrency on the client side, and there is no server-side queueing).

### Comparison with measurement

`is_processing` returning to false within about 0.7 s after the disconnect, and the log `cancel task` → `release`: **consistent**. Whether test 7 used streaming or non-streaming: the command the handbook gives is a non-streaming curl (`docs/phase1-steps.md:228`), but the actual command was not saved, so **which detection path was taken cannot be confirmed from the source**; both paths are compatible with 0.7 s.

**Impact on our experiments:** having the runner disconnect directly on timeout is feasible; estimate the release latency as ~1 s (non-streaming) or one token interval (streaming) + one decode step; but a timeout during the prefill phase of longctx may keep the slot occupied for over ten seconds after the disconnect, polluting the TTFT of the next request, so the timeout statistics for such tasks must account for the window "slot still busy after disconnect" (e.g. after disconnecting, poll `is_processing` in `/slots` before sending the next request); the runner uses streaming requests (needed for TTFT anyway), which detect more promptly.

---

## Q5. `ignore_eos` and streaming `timings` on `/v1/chat/completions`

### 5.1 `ignore_eos`

- The OAI handler `oaicompat_chat_params_parse` **copies verbatim** the fields in the body that receive no special handling into `llama_params` ("Copy remaining properties ... This allows user to use llama.cpp-specific params like mirostat via OAI endpoint"): `tools/server/server-common.cpp:1469-1477`. So `ignore_eos` is passed through; `max_tokens` is mapped to `n_predict` through a schema alias (`tools/server/server-schema.cpp:46-47`).
- The entry `post_chat_completions` (`tools/server/server-context.cpp:5560-`) → `handle_completions_impl` → `eval_llama_cmpl_schema` (`:4943`). The schema field `ignore_eos`: `tools/server/server-schema.cpp:423-433` — when true, it appends the precomputed EOG logit bias (`-INFINITY` for each EOG token) to `sampling.logit_bias`. This `logit_bias_eog` is generated in `common/common.cpp:1343-1349` for all tokens in the vocabulary for which `llama_vocab_is_eog` holds (Llama 3.2 has several EOG tokens, and `<|eot_id|>`, `<|end_of_text|>`, etc. are all masked), and is passed to the schema via `meta->logit_bias_eog` (`server-context.cpp:4844`, `:4943`).
- Effect: the probability of EOG tokens is 0 and they are never sampled, the `STOP_TYPE_EOS` branch (`server-context.cpp:2093-2096`) is never triggered, and only the three stop modes `max_tokens`/stop words/context limit remain, with `finish_reason` being `"length"`. The README lists it as a general sampling parameter: `tools/server/README.md:582` ("Ignore end of stream token and continue generating. Default: false").

Comparison with measurement (`completion_tokens = 200`, `finish_reason = length`): **consistent**.

### 5.2 Which chunk `timings` appears in when streaming

- When streaming, the final result is produced by `server_task_result_cmpl_final::to_json_oaicompat_chat_stream` as a **delta array** (`tools/server/server-task.cpp:454-518`): first the remaining content delta, then a `finish_reason` chunk (`:476-488`), and if `stream_options.include_usage` is true, a further usage chunk with `choices: []` (`:494-505`; the field `stream_options.include_usage` is in `server-schema.cpp:26-27`); finally `if (stats.is_set()) deltas.back()["timings"] = stats.to_json()` (`:508-510`) — `timings` hangs on the **last element** of the array.
  - Default (without `include_usage`): the last element is the chunk carrying `finish_reason`, so `timings` is in the last data chunk that carries `finish_reason`, and only after it comes `data: [DONE]` (`output = "data: [DONE]\n\n"` at `server-context.cpp:5080`). This is consistent with the end of `results/raw/step4/stream_sample.txt`.
  - With `include_usage: true`: `timings` will be on the trailing chunk with `choices: []` (containing `usage`), not the `finish_reason` chunk. The README example (`tools/server/README.md:1426-1445`) is exactly `"choices": []` + `timings`. **If the runner enables `include_usage` in future, the parsing must change.**
- Intermediate chunks (`server_task_result_cmpl_partial::to_json_oaicompat_chat`, `server-task.cpp:1103-1156`) carry `timings` only when `timings_per_token: true` (`tools/server/README.md:600`) (`server-task.cpp:1147-1149` checks `stats.is_set()`; `stats` is assigned only when `slot.stop != STOP_TYPE_NONE || timings_per_token`: `server-context.cpp:2222-2225`).
- Non-streaming: `timings` is at the top level of the response (`server-task.cpp:447-449`), with `usage` as well (`:357-364`: `completion_tokens`, `prompt_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens`).

### 5.3 Fields contained in `timings` (`server_slot_stats::to_json`, `tools/server/server-common.cpp:84-107`)

`cache_n` (number of reused prompt tokens), `prompt_n` (number of prompt tokens actually processed this time, excluding the cached part), `prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_token_ms`, `predicted_per_second`; only with speculative decoding are there additionally `draft_n`, `draft_n_accepted` (`:99-102`). The README description and example are at `tools/server/README.md:1426-1456` ("The total number of tokens in context is equal to `prompt_n + cache_n + predicted_n`"). The key set of the last chunk of our `stream_sample.txt` matches the above exactly (`cache_n: 30, prompt_n: 10, predicted_n: 15`).

Comparison with measurement: **consistent**.

**Impact on our experiments:** the runner can safely use streaming + `ignore_eos` to force a fixed output length; prefill time / hit amount are taken from `timings.prompt_ms / prompt_n / cache_n` of the last chunk (when streaming, TTFT still has to be timed on the client); the parsing logic should read "the last data chunk containing `timings`" rather than hard-coding the `finish_reason` chunk, so as not to miss it if `include_usage` is enabled later.

---

## Summary of unconfirmed items / items needing later measurement

1. The real behavior of `-kvu` pool exhaustion (500 or not, the `purging slot` log, the change of `cache_n`): only the source-code path exists; series U needs measurement.
2. Whether the disconnect in test 7 was streaming or non-streaming, and the release latency of a disconnect during prefill: the source gives the mechanism, not measured.
3. The origin of the +29 MB footprint under load, and how macOS accounts for dirty pages: cannot be confirmed from the source.
4. Whether `--kv-unified-per-slot` (without `-c`) really makes pool exhaustion unreachable: code inference, not measured.
5. Disconnect detection of non-streaming requests being delayed by `notify_all` when other traffic exists: inferred from reading the code, not measured.
