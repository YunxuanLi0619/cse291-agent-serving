# Lab notebook (append for every operation; do not delete or rewrite)

## Environment
- Date: 2026-10-19
- Machine: **MacBook Air** (fanless, sustained load causes thermal throttling, so record `pmset -g therm` for every run), Apple M4, 16GB, 10 cores, macOS (uname -a: Darwin Yunxuans-MacBook-Air.local 25.6.0 Darwin Kernel Version 25.6.0: Fri Jul 31 19:11:03 PDT 2026; root:xnu-12377.161.14~5/RELEASE_ARM64_T8132 arm64)
- llama.cpp commit: 1b29621bacc6a446bd523fc790316a24a06e886f (tag b11741, 2026-10-19 12:30 -0700; `llama-server --version`: 0.7.0-dev build 11741, AppleClang 21.0.0)
- Build command and time: `cmake -B build -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release` + `cmake --build build --config Release -t llama-server -j 10`; time real 2:24.91 (user 641.86s). CPU-only.
- Model file and sha256: Llama-3.2-3B-Instruct-Q4_K_M.gguf, 2019377696 B (≈1.88 GiB), sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff; source https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf; compared with lfs.oid and size from the HF API (tree/main), and they match.
- Launch parameters:

## Log

### 2026-10-19 Step 0 — Sample reports
(The Drive preview can only be browsed as screenshots; I only skimmed the first few pages of each and did not read them through.)
1. High-scoring (Edge Computer-Use Agent): isolates the bottleneck first (studies grounding alone first, then builds the full system); strict experimental control in comparisons (same 100-sample subset, same prompt/parsing pipeline, only one factor changed); results quantified with tables + bar charts (Acc@5/10/20/50px, latency, quantization-bit-width trade-offs), and states the hardware (Ryzen 7 9700X / 32GB).
2. Low-scoring (Jamie Food Recommender): most of the space goes to architecture/deployment/isolation design (LangGraph, Cloud Run, IAM, GCS prefixes), with a small share of quantitative controlled experiments; the argument for "improvements" is descriptive rather than controlled measurement. → We should spend space on measurement methodology and controls rather than system implementation details.
3. Common points / takeaways: the high-scoring report starts from an "observed failure", then does controlled ablations and gives trade-off curves (accuracy vs latency); our counterpart is the curve of concurrency c vs throughput/p99/failure rate + the dominant bottleneck conclusion.
(The link of the low-scoring report in the course PDF is ...EOz3cgTi; the actual ID is the `cgTi` of ...FOEz3cgTi; extracting links with pdftotext is more reliable than reading by eye.)

### 2026-10-19 Step 1 — Environment
- xcode-select -p: /Library/Developer/CommandLineTools
- cmake 4.4.4 (brew); Python 3.12.13 (uv venv .venv); dependencies locked in requirements.lock.txt (25 lines)
- git init done (branch main, no commits yet); gh logged in as YunxuanLi0619. Whether to `gh repo create` is up to the user (not created).
- About 110 GiB of disk left.

### 2026-10-19 Step 2 — Build
- See the environment above. Confirmed in `--help` that -np / -kvu --no-kv-unified / --cache-ram / --metrics / --cache-idle-slots exist.
- New finding: this commit has `--kv-unified-per-slot N` (per-slot context cap, unset by default); consider it as a comparison in Step 5.

### 2026-10-19 Step 3 — Model
- See the environment above. Download took about 6 minutes (in parallel with build/clone, sharing bandwidth).

### 2026-10-19 Step 4 — Launch and smoke test (config A: `-np 4 -c 16384 --no-kv-unified -t 8 --cache-ram 0 --metrics`)
For the meaning of the parameters see the Step 4 table in phase1-steps.md. Server log: `server-startup.log` (gitignored; the first time without `-lv`, the second time with `-lv 4` only to get the hyperparameter/KV buffer log, which does not affect memory).
| Test | Result |
|---|---|
| 1 /health | `{"status":"ok"}`; cold start (page cache already warm) 5–18 s |
| 2 chat | has `choices`, `usage.prompt_tokens_details.cached_tokens`, `timings.{cache_n,prompt_n,...}` |
| 3 /slots | **4 entries, each n_ctx=4096** (the even-split assumption for non-unified holds); server log `n_ctx_slot = 4096, kv_unified = 'false'` |
| 4 /metrics | has requests_processing / requests_deferred / n_tokens_max / n_busy_slots_per_decode |
| 5 streaming | ✅ `timings` is on the last data chunk (the one with `finish_reason:"stop"`, before `[DONE]`) |
| 6 ignore_eos | ✅ takes effect on the OpenAI-compatible interface: `completion_tokens=200`, `finish_reason=length` |
| 7 cancel | ✅ `is_processing` returns to false within ≈0.7 s after the client disconnects; server log `cancel task` → `release` |
Additional findings (llama.cpp behavior, from the logs, to be verified by /research or the source):
- Slot selection: `selected slot by LCP similarity (f_sim_best > 0.100 thold)`; with no match, `by LRU`. → Which slot a continuation step lands in depends on prefix similarity, not "idle first". (The basis for explaining E3/E4)
- With `--cache-ram 0`, `--cache-idle-slots` is disabled (log `requires --cache-ram, disabling`); but KV inside a slot is still reused: the second request has `cached_tokens=30` (the chat template prefix).
- `--no-mmap` **no longer exists** in this commit; it is replaced by `-lm none` (`-lm mmap|none|mlock|mmap+mlock`). For `--no-mmap` in handbook 5.3, use `-lm none`.
- The default log level does not print `n_layer`/KV buffer; `-lv 4` is needed.
- Whether the response carries `id_slot`: not checked (confirmed in 5.5).

### 2026-10-19 Step 5 — Memory baseline
**5.1 KV per token**: n_layer=28, n_head_kv=8, n_embd_head_k=n_embd_head_v=128 (n_head=24, n_embd_k_gqa=1024). 2×28×8×128×2 B = **114,688 B = 112 KiB/token**. 16384 tokens → 1792 MiB, exactly matching the log `CPU KV buffer size = 1792.00 MiB` (K 896 + V 896, f16). compute buffer 82 MiB.
**5.2/5.3 Memory when idle and under load** (`results/raw/step5/`, sampler `scripts/rss_footprint_sampler.sh`):
- Config A idle: `footprint` 3789 MB; `ps` RSS 692 MB (this time).
- Load: 4 concurrent requests (300 tokens each, ignore_eos, finishing in 85.8 s in total), sampled 639 points / 411 s: **footprint 3789→3818 MB (only +29 MB)**; **RSS fluctuates wildly 592→2536→170 MB** (it rises when weight pages are touched, then macOS reclaims/compresses them).
- Conclusions: Q1 idle memory ≈ weights + **the whole KV** (KV is allocated in one go at startup); Q2 footprint is almost independent of concurrency/load and is **determined by the startup configuration**; Q3 RSS is untrustworthy on this machine (idle RSS of the same configuration over different launches is 689 / 4678 / 3577 MB), and **macOS compressed memory is not counted in RSS**. → 5.5-C: **primary peak-memory measure = footprint** (RSS is still recorded as the secondary measure, and the report explains why it is unreliable).
- ⚠️ The assumption of handbook 5.3 is confirmed: "memory is determined by the startup configuration, not by load". The assumption in `docs/action-guide.md` §3.3 that "RSS grows with load" should be void.
**5.4 Configuration table** (idle_t0 = when /health is ready; footprint is the first-line Footprint of `footprint <pid>`):
| Config | Parameters | n_ctx per slot | KV buffer | idle footprint | idle RSS (t0 → after 20s) | cold start |
|---|---|---|---|---|---|---|
| A (mmap) | -np 4 -c 16384 --no-kv-unified | 4096 | 1792 MiB | 3789 MB | 3577 → 1210 MB | 5 s |
| A (-lm none) | same as above + `-lm none` | 4096 | 1792 MiB | **4083 MB** | 3712 → 985 MB | 8 s |
| B | -np 8 -c 32768 --no-kv-unified | 4096 | 3584 MiB | 5584 MB | 2763 → 1634 MB | 13 s |
| C | -np 8 -c 16384 -kvu | **16384** (8 slots share one pool) | 1792 MiB | 3791 MB | 5028 → 3229 MB | 9 s |
- The footprint difference between mmap and `-lm none` ≈ 294 MB: with mmap about 300 MiB of weights are file-mapped and not counted in footprint (footprint prints "mapped file 677 MB clean"). → footprint (mmap) is a lower bound and `-lm none` is an upper bound that includes all weights; put it in the report.
- Relation to KV: A→B footprint difference 1795 MB ≈ KV difference 1792 MiB ✓; A→C footprint basically unchanged ✓.
- /slots in config C reports n_ctx = 16384 per slot (the pool capacity, not exclusive capacity); 8 slots share a pool of 16384 tokens.
- The cold-start numbers are all for the case where the model is already in the page cache; a true cold-disk start was not measured.
### ⚠️ Environment warning (2026-10-19 13:5x)
- **The Mac is on battery and `lowpowermode=1` (low-power mode)**. The smoke test and memory experiments (looking only at footprint) are unaffected, but **speed data is unusable**: single-stream decode is only ≈7 tok/s (an M4 should be clearly higher), and 4 concurrent total ≈14 tok/s. **The 5.5 calibration must be done after plugging in and turning off low-power mode**, otherwise the λ grid, deadline and T_solo will all be off.

### 2026-10-19 Step 5.5 — Capacity calibration and finalizing definitions
**Conditions:** config A (`-np 4 -c 16384 --no-kv-unified -t 8 --cache-ram 0`), low-power mode off, **but done on battery power** (55%→49%, no thermal throttling recorded). ⚠️ Before the Step 9 batch the laptop must be plugged in, and T_solo must first be re-measured with 3 tasks per type; if it deviates from the table below by >15%, recalibrate. Earlier, in low-power mode, decode was only ~7 tok/s and after turning it off ~35 tok/s, showing that speed is extremely sensitive to the power mode.
Data: `results/raw/step55/calibration.json`, `calibrate.stdout`, `server.log`; script `scripts/step55_calibrate.py` (a rough prototype; prompts are seeded random words, with different content for every request; **not** the Step 7 generator).

**A.1 Single stream (3000-token prompt + 256 outputs, 3 runs):** prefill 112–123 tok/s (≈26 s), decode 12.9–16.4 tok/s, total time 40–47 s.
**A.2 Aggregate throughput (8 requests of 256in/128out):**
| c | wall-clock s | aggregate tok/s | per-request decode tok/s | mean TTFT s |
|---|---|---|---|---|
| 1 | 46.9 | 21.8 | 33.0 | 2.0 |
| 2 | 39.0 | 26.2 | 19.8 | 3.2 |
| 4 | 31.2 | 32.8 | 15.0 | 6.5 |
Aggregate throughput grows only gently with concurrency (c=4 is only 1.5× c=1), with no obvious knee, so CPU compute is the main constraint; c=8 was not measured (assumed ≤ c=4).
**A.3 T_solo (c=1, 3 runs each; prompt_tokens is the measured value including the chat template):**
| Type | prompt_tokens | 3-run times s | Mean |
|---|---|---|---|
| short (64 out) | 290 (target 256, +13%, includes ~32 template tokens) | 4.0 / 4.6 / 4.1 | 4.3 |
| longctx (64 out) | 3035 (target 3000) | 33.5 / 34.2 / 37.3 | 35.0 |
| longgen (512 out) | 290 | 21.9 / 23.0 / 22.7 | 22.5 |
| agent (4 steps, tool delay 1.0 s, tool result 150 tok) | 835→1123→1411→1698 | 35.5 / 34.4 / 35.8 | 35.2 (including idle 3.0 s) |
- Decode speed drops with context: 31 tok/s (290 ctx) → 26 (longgen) → 13 (3000 ctx); per step of the agent 27.8→21.5 tok/s.
- Prefix reuse for agent continuation steps in the same slot works: `cached_tokens` = 30 → 962 → 1250 → 1538 (c=1, still there after 1 s idle).
- **Neither streaming chunks nor /slots has id_slot** (chunk keys: choices, created, id, model, object, system_fingerprint, timings; slot keys: id, is_processing, n_ctx, speculative). → The `id_slot` field of requests.jsonl can only be left empty, or correlated afterwards from the server log (`slot launch_slot_: id N`).
- Fix for the Step 7 token self-check: short with target 256 measures 290, the template adds about +32; the generator should calibrate (±5%) based on `usage.prompt_tokens`.

**B. Budget (default mix 40/20/20/20 = short/longctx/longgen/agent):**
- S̄ = 0.4×4.3 + 0.2×35.0 + 0.2×22.5 + 0.2×35.2 = **20.26 s**
- Effective parallelism P_eff = G(4)/G(1) = 1.5 (c=2: 1.2); μ ≈ 1.5/20.26 = **0.074 task/s (4.45 task/min)**
- λ grid {0.3, 0.6, 0.9, 1.2}×μ = **{0.022, 0.044, 0.067, 0.089} task/s** (mean inter-arrival times 45 / 22.5 / 15 / 11.3 s). E1/E3/E4 use λ = 0.9μ = **0.067** by default.
- n_tasks = 100 (lower bound, not reduced further), 3 repetitions; 30 s per restart. Duration of one run = max(n/λ, n·S̄/P_eff(c)): at λ=0.9μ, c=1/2/4/8 is about 34.3 / 28.6 / 25.5 / 25.5 min.
- Total machine time (n=100, 3 repetitions): E1 series S 5.7 h + E1 series U (c=4,8) 2.5 h + E3 (tool delay 0 and 3 s) 2.5 h + E4 (only cache_prompt=false, c=1 and c=4) 3.0 h + E5 (λ=0.3/0.6/1.2μ, c=4) 6.8 h = **20.6 h**.
  - **This number assumes reuse:** E4's cache_prompt=true uses the corresponding E1 run of the same configuration, and E5's λ=0.9μ uses the E1 c=4 run, with no rerun; without reuse E5 is 8.1 h and E4 another +3.0 h, about 25 h in total.
  - Uncertainty of the estimate: P_eff(8) not measured; c=8 (series S) has only 2048 tokens per slot, so longctx will directly exceed ctx and fail early, which makes that configuration actually shorter; calibration was done on battery.
  - If the actual time exceeds this, first cut E1 series U (keep only one of c=4,8) and the second delay point of E3, without reducing the number of repetitions.
- Runs at low λ have their duration determined by the arrival process (one run at 0.3μ takes 75 min, three take 3.75 h) — E5 is the most expensive experiment.

**C. Finalized definitions (not to be changed afterwards):**
| Item | Definition |
|---|---|
| Memory budget B | Total KV token count at server start (`-c`); bytes = B × **114,688 B/token** (B=16384 → 1792 MiB) |
| Peak memory | **Primary measure = `footprint`** (max during sampling); the secondary measure RSS is also recorded but marked unreliable in the report (macOS compressed memory is not counted in RSS, and idle RSS of the same configuration fluctuates between 689–4678 MB, Step 5). The report gives both footprints for `-lm none` and the default mmap (difference ≈294 MB: file-mapped weights are not counted in footprint under mmap); the main experiments use the default mmap and state that footprint is a lower bound. Also report the time curve and peak of KV utilization = live_tokens / B (footprint itself is basically flat; KV utilization is what reflects load) |
| Concurrency c and np | Invariant c ≤ np; at startup the runner verifies `/slots` count ≥ c, otherwise it refuses to run |
| Timeouts | Request-level 120 s; task-level (including queueing) = min(300, 10×T_solo): short 43 s, longctx 300 s, longgen 225 s, agent 300 s. On timeout the connection is closed actively (Step 4 test 7: ≈0.7 s to release the slot ✓) and `status=timeout` is recorded |
| deadline / goodput | `deadline_s = 5 × T_solo(type)`: short 21 s, longctx 175 s, longgen 113 s; agents are computed from each task's own number of steps/delays: ≈ 5 × (8.05 s × n_steps + Σ tool delay) (8.05 s = (35.2−3.0)/4 steps). Fixed on the dev set and then frozen; goodput = number of tasks completed within the deadline with quality_pass ÷ total duration |
| Cold start | Record seconds from server start → /health ok (5–14 s with a warm page cache, cold disk not measured) and the first request's latency; the first 3 tasks of each run are marked `warmup=true` and excluded from steady-state statistics (raw logs kept) |
| Failure classification | ok / timeout / error_ctx_exceeded / error_http / error_other; denominator = all submitted tasks |

**D. Design of the memory-axis experiments:** Step 5 confirmed that footprint is determined by the startup configuration (independent of load), so both series from the handbook are kept:
- Series S (B=16384 fixed, np=c, c∈{1,2,4,8}, capacity per slot = 16384/c): c=8 → 2048 tokens/slot, the 3035-token longctx necessarily exceeds the limit (a deterministic failure, which is a measured phenomenon); c=4 → 4096 just fits the last agent step 1698+128.
- Series U (`-kvu -np 8 -c 16384`, c∈{4,8}): observe behavior when the pool is exhausted (defer/error/truncate). Note that under `-kvu`, /slots reports n_ctx=16384 (the pool capacity) for every slot.
- If time is tight, do S first, then c∈{4,8} of U (consistent with the budget above).

### 2026-10-19 #3 addendum — Slot release time after a disconnect during prefill (real machine)
Config `-np 1 -c 4096` (`scripts/start_server.sh 1 4096`), a single slot so no attribution is needed. prompt = longctx 3000 tokens (a random prefix is added each time to avoid the whole prompt hitting KV in the slot). Script `scripts/measure_slot_release.py`, raw data `results/raw/slot_release/`. Disconnect = the client closes the streaming connection; release = `is_processing` in `/slots` returns to false (50 ms polling).
| Disconnect time (after sending) | Disconnect → release | Send → release |
|---|---|---|
| prefill 1 s (4 times) | 16.3 / 17.9 / 18.0 / 18.6 s | about 17–19 s |
| prefill 5 s | 31.8 s | 36.8 s |
| prefill 10 s | 22.1 s | 32.1 s |
| prefill 15 s | 11.8 s | 26.8 s |
| prefill 20 s | 7.9 s | 27.9 s |
| prefill 24 s | 6.6 s | 30.6 s |
| decode (1 s after the first token) | 0.11 s | — |
- **Disconnect during decode:** released immediately (0.11 s), consistent with Step 4 test 7. The statement above that "active disconnect on timeout releases the slot in ≈0.7 s ✓" **holds only for the decode phase**.
- **Disconnect during prefill:** the slot stays busy until prefill ends. For a 1 s disconnect the server logs `cancel task` on the spot, but `release` only happens after the first 2048-token batch finishes (`n_tokens ≈ 2078`), about 17–19 s; for 5–24 s disconnects, `cancel task` appears only after the whole prompt has been processed (`n_tokens ≈ 3022` at `release`), send → release ≈ 27–37 s ≈ the full prefill time.
- **Not understood:** why a 1 s disconnect is noticed immediately while those at ≥5 s are noticed only after prefill ends (only the phenomenon, the mechanism was not verified). The sample is very small (a single prompt, 1 run per moment, 4 repetitions at 1 s), and it was measured on battery power.
- **Implication for the runner:** after a timeout disconnect of a long prompt (longctx), the server's slot may stay busy for another 10–30 s, while the runner released its own concurrency slot at the moment of disconnect, so the next request may hit server-internal queueing (the situation ADR 0003 worries about). The runner currently does not detect this.

### 2026-10-30 #7 Thread count comparison — freeze `-t 8`
Config `-np 4 -c 16384 --no-kv-unified --cache-ram 0`, on AC power, low-power mode off, no thermal throttling recorded. Interleaved order `8 4 10 8 4 10`, 180 s cooldown between levels; for each level short ×3, longctx ×3 single-stream + c=4 aggregate throughput (8 requests × 128 tokens). Script `scripts/prebatch_check.py threads`, raw data `results/raw/prebatch/threads.json`, decision and exclusion record `threads_decision.json`.
| `-t` | short (s) | longctx (s) | c=4 aggregate (tok/s) |
|---|---|---|---|
| 8 | 3.06–3.59 (5) | 25.4–26.8 (6) | 39.7 / 40.7 |
| 4 | 3.90–4.07 (6) | 33.4–35.3 (6) | 31.1 / 30.6 |
| 10 | 2.93–3.10 (6) | 24.3–26.0 (6) | 36.7 (1) |
- Geometric-mean speedup relative to `-t 8`: `-t 4` 0.78, `-t 10` 1.006. `-t 10` is not 10% faster than `-t 8` → per the rule keep `-t 8`; `-t 4` is clearly slower (single stream +30%, throughput −20%).
- **Two samples contaminated by system sleep were excluded** (the raw files were not modified): the first short of `-t 8` in the 4th level (647 s) and c=4 of `-t 10` in the 6th level (1.0 tok/s). Evidence: `pmset -g log` shows idle sleep from 21:32:46 to 21:43:30 (644 s) and from 21:55:37 to 22:10:37 (900 s); in the server log prompt eval of a 261-token prompt took 644.6 s, and 4 concurrent requests took 943 s each. **Idle sleep also happens on AC power**, and `caffeinate` had not been used before. `prebatch_check.py` now starts with its own `caffeinate -is -w <pid>`, and later stages and the official batch must do the same.
- The c=4 of `-t 10` has only 1 valid sample, so its throughput comparison with `-t 8` (36.7 vs 40.2) is weak evidence; but the two differ by <3% in single stream, so the conclusion does not depend on it.
- In the first round (`threads_run1_confounded.*`) the same configuration differed by nearly 2× between before and after, the cause was not found (possibly background processes or throttling), and it is not used for decisions; in the second round the same configuration was consistent before and after (`-t 8` longctx 26.1 vs 26.4, `-t 4` 34.5 vs 34.5).
- Frozen: `configs/server.json` = `{"threads": 8}`.

### 2026-10-30 #7 T_solo re-measurement (`-t 8`, AC power) — all types 23–29% faster than Step 5.5, recomputed
Same configuration as above (`-np 4 -c 16384 --no-kv-unified -t 8`), run under `caffeinate`, no thermal throttling recorded; 3 single-stream runs per type. Script `scripts/prebatch_check.py tsolo`, raw data `results/raw/prebatch/tsolo.json`.
| Type | Step 5.5 (battery) | Re-measurement, 3 runs (s) | Mean | Deviation |
|---|---|---|---|---|
| short | 4.3 | 3.37 / 3.13 / 3.06 | 3.19 | −26% |
| longctx | 35.0 | 26.1 / 26.3 / 26.7 | 26.38 | −25% |
| longgen | 22.5 | 16.4 / 15.5 / 15.9 | 15.95 | −29% |
| agent (4 steps, tool delay 1.0 s) | 35.2 | 28.0 / 27.7 / 27.8 | 27.8 (6.20 s per step) | −23% (per step) |
- All four types exceed 15% → recompute per the acceptance criterion. **The root cause is basically the power source:** Step 5.5 was measured on battery and now we are plugged in; consistent with the earlier observation in this notebook that "speed is extremely sensitive to power state".
- `configs/t_solo.json` has been updated to the new table (`agent_per_step_s` = 6.2), and `configs/server.json` = `{"threads": 8}`.
- New budget (default mix 40/20/20/20, still assuming P_eff(4)=1.5): S̄ = 15.30 s, μ ≈ **0.098 task/s**, λ grid {0.3, 0.6, 0.9, 1.2}×μ = **{0.029, 0.059, 0.088, 0.118} task/s** (old: S̄ 20.26 s, μ 0.074, λ 0.022–0.089).
- New deadline = 5 × T_solo: short 16.0 s, longctx 131.9 s, longgen 79.8 s, agent 5 × (6.2 × steps + Σ tool delay); the task-level timeout formula min(300, 10 × T_solo) gives short 32 s, longctx 264 s, longgen 160 s. **The deadlines in the already generated `workloads/dev/default_s1.json` are the old ones and must be regenerated when #8 freezes deadlines.**
- Total machine time must be re-estimated with the new S̄ and the P_eff of c=8 (do it after c=8 is measured).
- Test: `test_deadline_is_five_times_t_solo_from_input_table` used to read the shared `configs/t_solo.json` and hard-code the old values, so recalibration would inevitably fail it → changed to bring its own fixed table.

### 2026-10-30 #7 c=8 aggregate throughput
`-np 8 -c 16384 --no-kv-unified -t 8`, AC power, `caffeinate`, no thermal throttling recorded; alternately measured c=4 / c=8, 16 requests × 128 tokens each, 2 rounds. Script `prebatch_check.py c8`, raw data `results/raw/prebatch/c8.json`.
| Round | c=4 | c=8 |
|---|---|---|
| 1 | 42.3 tok/s | 42.9 tok/s |
| 2 | 34.3 tok/s | 42.6 tok/s |
| Mean | 38.3 | 42.8 |
- The paired ratio in round 1 is 1.014; the mean ratio of 1.116 is caused by c=4 being accidentally slow in round 2 (59.7 s vs 48.4 s), and **we cannot conclude from it that c=8 is faster**. Conclusion: c=8 throughput is no lower than c=4, and the budget assumption "P_eff(8) = P_eff(4)" holds.
- There are only 2 rounds, and the two rounds of c=4 differ by 20%. P_eff(4) = 1.5 (c=4 vs c=1) is still the old battery data and was not re-measured on AC power.

### 2026-10-30 #7 Cold-disk start
Right after `sudo purge`, run `prebatch_check.py cold` (executed back to back in a system terminal), `-np 4 -t 8`, 2.0 GB model (mmap). Raw data `results/raw/prebatch/cold_start.json`.
| | `/health` ready | server log "model loaded" |
|---|---|---|
| Cold (after purge) | 2.64 s | 2.18 s |
| Warm (restart right after) | 1.60 s | 1.18 s |
- A cold start is only about 1 s slower than a warm one, consistent with reading a 2 GB file from the SSD once, so purge is inferred to have worked (no other direct evidence). Sample 1 vs 1.
- Inconsistent with the "5–14 s with a warm page cache" recorded in Step 5.5: a warm start now takes only 1.6 s. The cause was not investigated (different power state/load when the old value was measured?); only the fact is recorded. The restart overhead before each Run is therefore much smaller than the 30 s assumed in the budget.

### 2026-10-30 #7 Summary
- Frozen: `-t 8` (`configs/server.json`), the new T_solo table (`configs/t_solo.json`), μ ≈ 0.098 task/s, λ ∈ {0.029, 0.059, 0.088, 0.118}.
- Total machine time scaled approximately in proportion to S̄ (15.30 / 20.26 = 0.755): about 20.6 h → **≈ 15.6 h** (rough estimate, not recomputed item by item; P_eff(4) and P_eff(8) both keep 1.5). About 19 h without reusing the data of E4 / E5.
- #8 must: regenerate the Dev workload with the new T_solo and freeze deadlines (`workloads/dev/default_s1.json` has the old values).

### 2026-10-31 #8 Smoke test (real llama-server, 30 Tasks, c=2)
`-np 4 -c 16384 --no-kv-unified -t 8`, on AC power, low-power mode off, run under `caffeinate`. `runner --c 2 --lambda 0.06 --n-tasks 30`, raw data `results/raw/smoke_8/run/` (server log `smoke_8/server.log`, one shared copy for the whole of #8).
- **Three kinds of logs + manifest are complete; `tasks.jsonl` has exactly 30 lines** (`requests.jsonl` 43 lines). `errors` in `manifest.json` is empty.
- **`--timeout 1`** (`smoke_8/timeout1`, 8 Tasks, λ=0.5): all 8 lines of `tasks.jsonl` are `status=timeout`, 8 lines of `requests.jsonl` are `timeout/request_timeout`, no Task dropped; 20 s later all four in `/slots` are idle.
- **The sampler against the real server:** `footprint_mb` 3792–3872 (1244 samples, not a single `error`), `live_tokens` 0–6121, `requests_processing` 0–3, per-slot `is_processing` all present; the `/slots` count check before startup passed.
- **16 of the 30 Tasks timed out, which is not a bug:** λ=0.06 is overload for c=2 (rough capacity of c=2 about 0.07 task/s, not measured; the first 30 have many heavy Tasks), queueing 100–250 s hit the task-level timeout (short 32 s, others 10×T_solo capped at 300). The smoke test is only used to verify that logs do not lose rows.
- **`requests_processing` reached 3 > c=2:** if a request disconnected on timeout is still in prefill, the server's slot stays busy until prefill ends (see the 2026-10-19 #3 addendum); the runner has already released the concurrency slot, and the next request overlaps with it. Consistent with existing observations, no contradiction.
- **Root cause of the needle failures (found in the smoke test):** in the smoke test 3 of 9 longctx had `quality_pass=false`. Replaying the same prompts directly (temperature 0): Llama-3.2-3B replied "I can't provide information or guidance on illegal or harmful activities ..." — **"secret key" triggers a safety refusal**. After replacing "The secret key is" / "What is the secret key ..." with "The reference code is" / "What is the reference code ... Reply with the code only.", all 3 failing prompts were answered correctly. `src/runner/gen_workload.py` (`RECALL_QUESTION`, the needle sentence) was changed and the Dev workload regenerated; the wording in `tests/test_workload.py` was updated accordingly. The smoke test used the workload from before the change (sha `df59750…`).

### 2026-10-31 #8 Was prefix reuse of agent continuation steps interrupted by other short requests (research addendum)
Tool: `python -m src.analysis.slot_trace` (new, with unit tests): maps each request's `selected slot by LRU/LCP` + `launch_slot_` in the server log one-to-one back onto `requests.jsonl` by `t_start` (OAI responses have no `id_slot`). It refuses to map if the counts do not match, and uses `--skip` when the log is shared. Self-check of the mapping: the tool has a built-in check of each ok request's prompt+output token count against the `n_tokens` when the server releases the slot (it errors on mismatch; smoke 35, baseline 137, rerun 42 all had 0 inconsistencies, including the c=2 smoke test); in addition, from step 3 on `cached_tokens` ≈ the complete context of the previous step and 100% return to the same slot. Limitation: only for ok requests that reached the server; requests cancelled before reaching the server make the counts mismatch, and the tool refuses rather than guesses.
- **Step 2 of the baseline (c=1, 21 agents, including steps already sent before timeout):** in 23 of 27 cases it returned to the original slot (`cached_tokens` ≈ 927/928); 4 changed slot, with `cached_tokens=30` (only the chat template prefix left). Steps 3–6 returned to the original slot 23/23, 13/13, 8/8, 3/3.
- **A clear instance of interruption by a short request:** `t0075` (agent, status ok) step 1 was in slot 1; during the Tool delay `t0076` (short) arrived and **selected slot 1 by LCP, `f_sim_best = 0.125` (> 0.100 thold)**, overwriting the agent's prefix; step 2 could only select slot 2 by LRU, `cached_tokens=30`. This is exactly the mechanism predicted by the research, **observed directly for the first time here**.
- **longgen (256-token prompt) triggers it as well:** in the baseline, longgen such as `t0011/t0037` preempt other slots by LCP at f_sim 0.117–0.125, not just short.
- **Another 3 cases (`t0009/t0010/t0034`, all in queueing bursts):** step 2 of the agent was re-queued at the tail of the FIFO, and by the time it was served there had been 4–5 other launches in between (including LCP overwrites and LRU rotation), so the 4 slots had basically all been rotated once. In the smoke test (c=2, heavily overloaded), 6 of the 7 agent step-2s were like this (4–9 launches in between), and only `t0029` hit. **The original slot can survive only when fewer than 3 launches occur in between** (`t0029`: 3 launches landed on the other 3 slots, and step 2 hit by LCP).
- **Implication for E3/E4:** in a mixed workload whether `cached_tokens` of an agent continuation step hits depends on how other requests occupy slots during the Tool delay / queueing, not on the `cache_prompt` switch itself; the analysis must explain it with this mechanism, and not treat every `cached_tokens=30` as an effect of `cache_prompt=false`.
- The sample is small (a single seed, a c=1 baseline + one overloaded smoke test) and no statistics were done; it is only a mechanism check.

### 2026-10-31 #8 Single-stream (c=1) quality baseline — every type ≥ 90%, deadlines frozen
Dev workload (`workloads/dev/default_s1.json`, seed 1, 100 Tasks: short 39 / longctx 21 / longgen 19 / agent 21). `runner --c 1 --lambda 0.029 --policy fifo`, ran for 3616 s, on AC power, no sampling gap ≥5 s, `manifest.errors` empty. Raw data `results/raw/baseline_8/run/`.
- **In the first pass 18 Tasks timed out (queueing, not quality):** Poisson bursts caused queueing at c=1; shorts were all stuck at the 32 s task-level timeout (11), agent at 300 s (6), longgen at 160 s (1); not a single request in `requests.jsonl` reported an error itself. Among completed Tasks, pass: short 28/28, longctx 20/21, longgen 18/18, agent 14/15.
- **Re-measurement:** extracted these 18 Tasks into a separate subset (`baseline_8/rerun_workload/timeouts_subset.json`, `unit_gap_s` all set to 1, λ=1/70 i.e. a fixed 70 s interval to guarantee no queueing; sha `8eb6432…`, **not** the frozen workload), `baseline_8/rerun/`, all 18 ok, `requests_processing` max 1, no sleep gap. The result of each Task is "use the first pass if it was ok, otherwise use the re-measurement".
- **Single-stream quality pass rate (Phase 2's quality baseline):**
  | Type | Count | pass | Rate |
  |---|---|---|---|
  | short | 39 | 39 | 100.0% |
  | longctx | 21 | 20 | 95.2% |
  | longgen | 19 | 19 | 100.0% |
  | agent | 21 | 20 | 95.2% |
  2 failures: `t0071` (longctx) — the model answered `6y85e5u8`, **dropping the `KEY-` prefix**, which counts as a failure under the substring check although the content is actually right ("Reply with the code only" easily makes the model strip the prefix; the check criterion stays a strict substring, consistent across all conditions, so it does not affect comparisons and only makes the absolute value slightly conservative; if changed to check only the 8-character suffix, longctx would be 100%, but that would require re-measuring the baseline); `t0098` (agent, 5 steps) cause not investigated.
- **Single-stream times of the re-measurement:** short mean 2.9 s (11), longgen 14.0 s (1), agent 35.5 s (6, different steps/delay), the same order of magnitude as T_solo (3.19 / 15.95 / 6.2×steps+Σdelay).
- **Kept running with the lid closed (incidental observation):** during the first-pass Run the lid was closed from 23:46:53 (`Clamshell Sleep` → DarkWake in `pmset -g log`), computation did not stop (no sampling gap), and after closing the lid the short latency was 2.9 s vs 3.35–3.49 s before, not slower. With AC power + `caffeinate -is`, this machine can keep running with the lid closed; `pmset disablesleep` was not set.
- **Frozen:** the Dev workload's `deadline_s = 5 × T_solo` (using the new table from #7: short 15.95 s, longctx 131.9 s, longgen 79.75 s, agent 5×(6.2×steps+Σdelay)) has been written. **`workloads/dev/default_s1.json` sha256 = `4eefd673a414c9ff86a6b9d5c440e6dc57851d1477826027604de43017611d2a`**, consistent with `manifest.json:workload_sha256` of the baseline Run. It must not be changed afterwards; changing T_solo or the needle wording requires regenerating and re-measuring the baseline.
- **Discrepancies with existing observations:** none contradicts. Two points to add: ① "active disconnect on timeout releases the slot in ≈0.7 s" holds only for the decode phase; in the smoke test, requests that timed out during prefill still held the slot (`requests_processing` reached 3 above); ② the LCP mechanism in Step 5.5 went from a "source-code conclusion" to "has one measured instance" (see above).
- **Note:** the single-stream baseline was measured only under `-np 4 -c 16384 --no-kv-unified -t 8`; quality after changing c / moving to series U must be measured separately.

### 2026-11-01 #9 Batch E1–E5 results (all 45 official Runs completed)
Conditions: HEAD `d23e96d` (working tree clean at start), AC power 100%, low-power mode off, `caffeinate -is`, `-t 8`; started 2026-10-31 11:53, ended 2026-11-01 04:50, ≈ 17 h in total (estimate 18.9 h). Raw data `results/raw/batch/` (45 directories, 47 MB), summary `results/batch/summary.json`, analysis `results/analysis/batch/`.
- **Completion:** planned 45 / done 45; `failed_runs`, `blocked_stages`, `retried_runs`, `throttled_runs`, `missing_reused` all empty; `invalid/` empty — every Run passed validation on the first attempt, there was no throttling in the whole batch (`pmset -g therm` never showed a warning), and no unplugging. Cold start 1.25–3.29 s (mean 1.95 s). The 5 30-Task pilots (E1-S/E4/E5/E3/E1-U) all passed before the official batch and were not redone.
- **Composition:** E1-S 12 (c∈{1,2,4,8}, λ=0.088), E4 6 new (`cache_prompt=off`, c∈{1,4}), E5 9 new (c=4, λ∈{0.029,0.059,0.118}), E3 12 (c=4, λ=0.088, Tool delay {0,0.5,1,3} s), E1-U 6 (np=8 `-kvu`, c∈{4,8}); E4's on side and E5's λ=0.088 point reuse E1-S; E2 is sliced by Step from E1 data and was not run separately.
- **Most important finding: λ=0.9μ is heavily overloaded for c≤4, so μ ≈ 0.098 is an overestimate.** Average of the three seeds of E1-S:

  | c | goodput (task/s) | p99 (completion/deadline) | fail rate |
  |---|---|---|---|
  | 1 | 0.0049 | 1.82 | 63.6% |
  | 2 | 0.0063 | 1.90 | 63.6% |
  | 4 | 0.0149 | 1.91 | 58.8% |
  | 8 | 0.0520 | 1.40 | 29.6% |

  Failures are almost all task-level timeouts (`timeout_rate` ≈ `fail_rate`). E5 sweeps λ at c=4: 0.029 → fail 3.4%, p50 25 s; 0.059 → 51.5%; 0.088 → 58.8%; 0.118 → 82.1%. So the true saturation point of c=4 is **between 0.3μ and 0.6μ** (≈ 0.03–0.05 task/s), not 0.098. The root cause is consistent with the judgment at the 10-31 pilot: P_eff(4)=1.5 was an estimate from before plugging in and was never re-measured on AC power; it is not a bug in the runner or server (`requests_processing` max ≤ c, no validation failure). The λ grid is frozen and was not changed.
- **Fixed-concurrency baseline c (by the rule in `src/batch/baseline.py`): c = 8, but `p99_acceptable = False`.** None of the four c satisfy "p99(completion/deadline) ≤ 1 and fail ≤ 5%", and the rule falls back to the c with the largest goodput, c=8. **This c=8 is the relative optimum within the overload region, not selected under an acceptable p99; before Phase 2 uses it as the baseline, a decision on how to handle it is needed** (options: measure E1 additionally between 0.3–0.6μ; or explicitly accept this fallback result). `complete=true` in `baseline_c.json` (4 c × 3 seeds all present).
- **E4 (`cache_prompt` on − off, paired by seed, 3 pairs):** c=1: p50 −27.3 s (std 14.3), goodput +0.0027, fail −9.6 percentage points; c=4: p50 −35.9 s (std 51.2), goodput +0.0105, fail −18.6 percentage points. The direction is consistent (cache on is better), but both configurations are in the overload region, the fail-rate difference contains queueing amplification and the std is large, so it shows only the direction and cannot be taken as a precise effect size.
- **E3 (c=4, λ=0.088, Tool delay 0/0.5/1/3 s):** fail 59–63%, p50 103–114 s, no discernible trend; the effect of Tool delay is drowned in overload. The cause is the same as above, not corrupted data.
- **E1-U (np=8 `-kvu`, shared pool):** c=4 fail 65%, c=8 fail 76%, **worse than E1-S c=8 (30%)**. The queue_time of c=8 is only 17.6 s (0.45 s for series S c=8), which indicates that under the shared pool concurrent requests contend for the 16384-token pool rather than queue; there were no pool-exhaustion errors.
- **Memory (`footprint_mb` peak):** for series S all c are about 3.84–3.87 GB (c=1/2/4/8 differ by <1%), series U about 3.90 GB — with the total KV budget fixed, peak memory is basically determined by B and independent of c, in line with the expectations of ADR 0002/0004; RSS peak 5.5–5.8 GB.
- **Problems during the run and fixes:**
  1. The summary analysis at the end of the batch crashed (`TypeError`): `with_line_label` ran into a missing `tool_delay_s` when assembling legend names (only E3 Runs have this key). Fixed (`src/analysis/report.py`) with a regression test added, and after rerunning `results/analysis/batch/` was generated completely. The data itself is unaffected.
  2. The full pytest once hung for more than 16 minutes at `test_runner.py::test_failures_are_classified_and_never_drop_a_task` and could not be reproduced (the single file passed in 12.8 s, and all 138 passed in 76 s); recorded as an occasional flake.
- **Not done:** the "shared / not shared system prompt" variant of E4; c∈{1,2} of E1-U (only {4,8} were run in the plan).
- **Background interference:** XProtect occasionally used CPU on the machine, which I cannot rule out and cannot identify separately in the data; among the three seeds of the same configuration no obviously outlying Run was seen, but no systematic outlier test was done.

### 2026-11-01 #9 Correction: E1-S c=8 looks best only by "rejecting longctx", so baseline c=8 cannot be used directly
The statement in the previous entry that "c=8 is the relative optimum within the overload region" was incomplete; after counting by type from the status in `tasks.jsonl` (E1-S, steady-state Tasks, 3 seeds combined), it is corrected as follows:
| c | capacity per slot | longctx | agent | longgen | short |
|---|---|---|---|---|---|
| 1 | 16384 | 62 ok | 9 ok / 44 timeout | 27 ok / 33 timeout | 8 ok / 108 timeout |
| 2 | 8192 | 62 ok | 12 / 41 | 23 / 37 | 9 / 107 |
| 4 | 4096 | 62 ok | 17 / 36 | 19 / 41 | 22 / 94 |
| 8 | **2048** | **0 ok / 62 `error_ctx_exceeded`** | 42 ok / 11 `error_ctx_exceeded` | 52 / 8 timeout | 111 / 5 timeout |
- **A slot at c=8 has only 2048 tokens, so all longctx and 11 agents directly hit `error_ctx_exceeded`** (failing within seconds, neither holding a slot nor queueing). The heaviest load (longctx, T_solo 26.4 s, 20% of Tasks) was rejected wholesale, and only then did the remaining short/longgen not time out. So the goodput of 0.052 and fail 29.6% at c=8 are not "concurrency 8 is faster" but "concurrency 8 rejected a class of Tasks". The `p99_deadline_ratio=1.40` of c=8 in `baseline_c.json` also counts only ok Tasks, so this cannot be seen.
- **Failures at c=1/2/4 are mainly short timeouts (the task-level timeout of short is only 32 s):** longctx is all ok, but each occupies a slot for about 26 s, and under FIFO the shorts queued behind time out once they wait more than 32 s. At the same λ, the larger c is, the fewer short timeouts (108 → 107 → 94), but the improvement is small.
- **c=1 is overloaded at λ=0.088 by arithmetic, not by accident:** mean service time S̄ ≈ 15.3 s → single-stream capacity ≈ 1/15.3 = 0.065 task/s < 0.088. The capacity of c=2 and c=4 depends on P_eff (not re-measured on AC power), and E5 shows c=4 saturates at λ between 0.03–0.05.
- **Implication for the baseline:** the current rule (`src/batch/baseline.py`) counts the p99 of only ok Tasks, and falls back to the largest goodput when no c is acceptable, which selects "c=8 that rejects longctx". **This c=8 should not be used as the Fixed-concurrency baseline.** Until the problems below are solved, the baseline c is regarded as **undetermined**: ① measure E1 additionally at λ ≈ 0.03–0.05 (only c=1/2/4 could satisfy the p99 and fail rate); ② in the rule, treat `error_ctx_exceeded` as unacceptable on its own (any c with ctx_exceeded is disqualified).
- **The comparison of E1-U with E1-S c=8 is likewise void:** the claim in the previous entry that "E1-U is worse than E1-S c=8" compared against a c=8 with no longctx and does not hold. E1-U (shared pool) has no single-slot cap of 2048 and longctx runs normally, so it should be compared with E1-S c=4 (which also runs longctx): U c=4 fail 65% vs S c=4 58.8%, U c=8 fail 76%.

### 2026-11-01 #9 Baseline rule relaxed (post-hoc adjustment) and baseline c recomputed
**This rule was changed after looking at the E1–E5 data, not decided in advance; the report must say so.** The original rule (p99/deadline ≤ 1 and failure rate ≤ 5%) has two problems on the existing data: ① the p99 of 100 Tasks is basically the slowest one or two, and even c=4, λ=0.029 which is not overloaded gives 1.72 / 1.91 / 1.84 (7–13 of the completed Tasks exceeded the deadline); ② when no c qualifies, it falls back to the highest goodput, selecting c=8 which rejects longctx.
- **New rule (`rule="relaxed"`, `src/batch/baseline.py`):** any c with `error_ctx_exceeded` is disqualified and cannot be chosen in the fallback either; otherwise only the mean failure rate ≤ 10% is required; p99/deadline is only recorded and is no longer a threshold. The compared quantity and the rule that 5% ties go to the smaller c are unchanged. The original rule is kept as `rule="strict"`, and its result is written to `strict_rule_result` in `baseline_c.json`.
- **Recomputation (`scripts/recompute_baseline.py`, E1-S λ=0.088, 12 Runs, no rerun):** goodput of c=1 / 2 / 4 / 8 is 0.0049 / 0.0063 / 0.0149 / 0.0520, failure rate 63.6% / 63.6% / 58.8% / 29.6%; the `ctx_exceeded_rate` of c=8 = 25.1%, so it is excluded. **baseline c = 4, but `acceptable=False`**: all c are overloaded at λ=0.088, and c=4 is the fallback choice "highest goodput among the c that do not reject tasks". The original rule still gives c=8.
- **Supplementary evidence (not part of the selection):** the 3 Runs of c=4, λ=0.029 (E5) qualify under the new rule (failure rate 3.4%, goodput 0.0253). **c=1 and 2 were not measured at this λ**, so we cannot say c=4 is optimal at λ=0.029, only that it qualifies.
- **Still unresolved:** to formally compare c=1/2/4 at a non-overloaded λ, E1 needs to be measured additionally (λ≈0.045, c∈{1,2,4}, 3 seeds, ≈6–7 h).
- Tests: the old cases in `tests/test_batch_baseline.py` were changed to `rule="strict"`, and 3 relaxed cases were added; the full suite passes.
