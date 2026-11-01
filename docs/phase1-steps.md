# Phase 1 step-by-step handbook (10/19 → 10/28)

> Usage notes for beginners
> - Every step has: **goal → commands → how to tell it worked → what to record**. Do them in order and tick each step when done.
> - All commands are run in a **macOS terminal (zsh)**. The project directory name contains a space, so paths need quotes: `cd ~/"CSE 291"`.
> - Difficulty and metrics follow the course requirements exactly, with no simplification. The course's "reproducible" means: **the version, parameters and duration of every step must be written down** (see the lab notebook in Step 0).
> - Your machine: Apple M4, 16 GB memory, 10 cores, 98 GB of disk left. Already has git / uv / jq / curl / gh / python3 (3.14); **no cmake**.
> - 📌 Commands marked ✅ were checked by me against the current documentation; places marked ⚠️ are ones I cannot 100% guarantee and you need to confirm by measurement.
> - 🔧 **v2 review revision (2026-10-19):** a batch of gaps was filled by checking against the original course PDF and the llama.cpp documentation: added Step 5.5 (calibration + finalizing definitions), the three smoke tests of Step 4 on streaming/`ignore_eos`/cancellation, the c ≤ np invariant, deadline and timeout semantics, a manifest for every run, `/health` polling instead of `sleep`, and experiment priorities. Changes are marked 🆕; a new "How to carry it out (skills and methods)" section is added at the end.

---

## Overall progress table

| Step | Content | Estimated time | Suggested completion day |
|---|---|---|---|
| 0 | Read sample reports + create the lab notebook | 1 h | 10/19 |
| 1 | Install tools, create the Python environment | 20 min | 10/19 |
| 2 | Build llama.cpp | 15 min | 10/19 |
| 3 | Download and verify the model | 15 min | 10/19 |
| 4 | Start the server and do smoke tests | 30 min | 10/20 |
| 5 | **Memory baseline experiment** (decides the whole later design) | 1–2 h | 10/20 |
| 5.5 | 🆕 **Capacity calibration + finalizing definitions** (memory budget, λ grid, deadline, total experiment duration) | 1–2 h | 10/20 |
| 6 | Create the project skeleton and configs | 30 min | 10/21 |
| 7 | Write the workload | half a day | 10/21–22 |
| 8 | Write the runner + FIFO scheduler + monitoring | 1 day | 10/22–23 |
| 9 | Run experiments E1–E5 | 1.5 days | 10/24–25 |
| 10 | Analysis and plotting | half a day | 10/26 |
| 11 | Write the writeup, package, submit | 1 day | 10/27–28 |

> **About the code of Steps 7–10:** these steps require writing Python code. In this handbook I give the **goal, file interfaces and run commands** of each step; I will write and test the code after you finish Step 5 (I can verify the logic with a fake server first), so that you do not debug a pile of code that has never been run.

---

## Step 0 — Read sample reports + create the lab notebook (10/19)

**Goal:** Know what "high-scoring" looks like; record from the very first minute, to meet the "reproducible" requirement.

1. Open the two Google Drive links on page 2 of the course PDF and read each one: the **high-scoring report** and the **low-scoring report**. Compare where they differ (figures? controls? handling of failures?).
2. Create the notebook:

```bash
cd ~/"CSE 291"
mkdir -p docs
cat > docs/lab-notebook.md <<'EOF'
# Lab notebook (append for every operation; do not delete or rewrite)

## Environment
- Date:
- Machine: Apple M4, 16GB, 10 cores, macOS (uname -a: )
- llama.cpp commit:
- Build command and time:
- Model file and sha256:
- Launch parameters:

## Log
EOF
```

**Record:** The 3 points you learned from the two sample reports (write them into the notebook).

---

## Step 1 — Install tools, create the Python environment (10/19)

**Goal:** Have a compiler, cmake, and a Python environment with a fixed version.

```bash
# 1) Confirm the command-line developer tools (should output /Library/Developer/CommandLineTools)
xcode-select -p

# 2) Install cmake
brew install cmake
cmake --version            # should show a version number

# 3) Create the Python environment in the project
#    Don't use the system 3.14, use 3.12 to avoid some libraries having no matching packages
cd ~/"CSE 291"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install httpx psutil pandas numpy matplotlib pyyaml pytest
python --version           # should show 3.12.x
```

> Every time you open a new terminal window afterwards, first `cd ~/"CSE 291" && source .venv/bin/activate`.

**How to tell it worked:** both `cmake --version` and `python --version` produce output.
**Record:** the cmake version and Python version (`uv pip freeze > requirements.lock.txt` saves the dependency versions).

🆕 **At the same time turn the project into a git repo** (the current directory is not yet a repo; the later issue workflow and "reproducibility" both depend on it, so do not put it off to Step 6):

```bash
cd ~/"CSE 291" && git init
gh auth status        # confirm gh is logged in. Whether to `gh repo create` (private recommended) is your decision
```

---

## Step 2 — Build llama.cpp (10/19)

**Goal:** Get a `llama-server` at a fixed commit.

**Important design decision:** I recommend **turning off Metal (GPU) and building CPU-only**. Reasons:

1. The course will most likely measure on an AWS **CPU instance** in the end, and the CPU build behaves more like it;
2. With Metal the model and KV live in GPU unified memory, process RSS cannot accurately reflect memory usage, and the core metric "peak memory" would be distorted.

```bash
mkdir -p ~/src ~/models
cd ~/src
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Record the commit (all later experiments are based on it; don't git pull again)
git rev-parse HEAD

# Build, and time it (the course requires build time to be counted)
cmake -B build -DGGML_METAL=OFF -DCMAKE_BUILD_TYPE=Release
time cmake --build build --config Release -t llama-server -j 10
```

✅ The command to build the server and the `-DGGML_METAL=OFF` option come from the official documentation.

**How to tell it worked:**

```bash
~/src/llama.cpp/build/bin/llama-server --version
~/src/llama.cpp/build/bin/llama-server --help | grep -E "parallel|kv-unified|cache-ram|metrics"
```

If you can see the version and these parameters, it worked.

**Record:** the commit hash and the duration from the `time` output.

---

## Step 3 — Download and verify the model (10/19)

**Goal:** Choose a model and **fix it from now on, never changing it** (the course requires model and precision to be fixed).

Choice: **Llama-3.2-3B-Instruct, Q4_K_M quantization (about 2.0 GB)**. Reason: its KV cache is larger than the 1B model's (memory pressure appears more easily), but it is still comfortable on a 16GB machine. (I have confirmed the download URL is valid.)

```bash
cd ~/models
curl -L -o Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

ls -lh Llama-3.2-3B-Instruct-Q4_K_M.gguf     # about 1.9G
shasum -a 256 Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Record:** file name, size, sha256, and the download source URL.

🆕 **Verify rather than just record:** open that file's page on Hugging Face and compare the SHA256 shown on the page with the local `shasum` result; if they differ, download again (an interrupted `curl -L` gives a corrupted file, and the server may start but the results are untrustworthy).

---

## Step 4 — Start the server and do smoke tests (10/20)

**Goal:** Confirm the server runs and all four endpoints work.

**Terminal A (start the server and keep it running):**

```bash
~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -np 4 -c 16384 --no-kv-unified \
  -t 8 --cache-ram 0 --metrics \
  2>&1 | tee ~/"CSE 291"/server-startup.log
```

Meaning of the parameters (all must be written into the notebook):

| Parameter | Meaning |
|---|---|
| `-np 4` | 4 concurrent slots |
| `-c 16384` | 16384 tokens of total context |
| `--no-kv-unified` | each slot has its own KV (each slot ≈ 16384/4 = 4096) |
| `-t 8` | 8 threads, leaving 2 cores for your runner and monitoring |
| `--cache-ram 0` | turns off the extra prompt cache in host memory, so it does not get mixed into memory measurements |
| `--metrics` | enables the `/metrics` endpoint |

> ⚠️ For stable experiments: **plug in the power, close big programs such as browsers, and don't let the Mac sleep**. You can run `caffeinate -dimsu` in another terminal to prevent sleep.

**Terminal B (testing):**

```bash
cd ~/"CSE 291" && source .venv/bin/activate

# 1) Health check
curl -s http://127.0.0.1:8080/health

# 2) One chat
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi in 5 words."}],"max_tokens":32,"temperature":0}' | jq

# 3) Look at the number of slots and each slot's context size
curl -s http://127.0.0.1:8080/slots | jq '.[] | {id, n_ctx, is_processing}'

# 4) Look at metrics
curl -s http://127.0.0.1:8080/metrics | grep -E "requests_|n_tokens_max|n_busy"
```

**How to tell it worked:**

- `/health` returns ok;
- the chat has `choices` and the `timings` and `usage` fields (✅ the documentation confirms `timings.cache_n`, `timings.prompt_n`, `usage.prompt_tokens_details.cached_tokens`);
- `/slots` returns 4 entries, **check whether each `n_ctx` is 4096** (⚠️ verifies the assumption "non-unified splits evenly by slot");
- `/metrics` has `requests_processing` and `requests_deferred`.

**Record:** the full launch command and the measured `n_ctx`.

> ✅ The documentation confirms: `--cache-idle-slots` depends on `--cache-ram`. This handbook sets `--cache-ram 0`, so it is disabled as well, and **prompt reuse comes only from the KV the slot itself still holds**. This must be stated in the report's environment description; the explanations of E3/E4 all depend on it.

🆕 **Three additional smoke tests that must pass** (the later measurement methods depend on them; if not verified now, Step 8 will hit trouble):

```bash
# 5) Streaming: t_first_token can only be measured with streaming. Confirm the last chunk carries timings
curl -sN http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count to 5."}],"max_tokens":32,"stream":true}' | tail -n 6

# 6) Whether ignore_eos takes effect on the OpenAI-compatible interface: require usage.completion_tokens to be exactly 200
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":200,"ignore_eos":true,"temperature":0}' | jq .usage

# 7) Cancellation: first send a request with max_tokens=512, disconnect with Ctrl+C after a few seconds; in another window run the following repeatedly
curl -s http://127.0.0.1:8080/slots | jq '.[] | {id, is_processing}'
```

- Test 5: the last data chunk has `timings` (containing `prompt_n`, `cache_n`); if not, another way to get the prefill time is needed.
- Test 6: if it is not 200 → switch to the native `/completion` interface or record the limitation "output length cannot be forced".
- Test 7: after the disconnect `is_processing` returns to `false` within ≈1–2 s → the runner can release the slot by simply disconnecting on timeout; **if not, timed-out requests keep occupying the slot and pollute later measurements**, which must be stated in the report and the design changed.

---

## Step 5 — Memory baseline experiment (10/20, **the most important step**)

**Goal:** Figure out "what exactly determines memory", which directly decides the direction of your Phase 2.

### 5.1 Compute the KV memory per token

```bash
grep -E "n_layer|n_head_kv|n_embd_head_k|n_embd_head_v" ~/"CSE 291"/server-startup.log | head
grep -iE "kv.*(size|buffer)|KV self" ~/"CSE 291"/server-startup.log | head
```

Apply the formula: `KV bytes per token = 2 × n_layer × n_head_kv × head_dim × 2` (f16).
For Llama-3.2-3B I estimate about **112 KB/token**, but **go by the numbers in your log**. Then cross-check `-c 16384 × bytes per token` against the total KV size printed in the log; the two should be close.

### 5.2 Measure "process memory at idle/under load"

```bash
PID=$(pgrep -x llama-server | head -1)     # -x matches the process name exactly, avoiding tee / the current shell
echo "Idle RSS (MB):  $(( $(ps -o rss= -p $PID) / 1024 ))"
footprint $PID | head -20                   # 🆕 macOS physical-usage measure (⚠️ please confirm the command's output format by measurement)
```

Then open another window to sample continuously, and send a few concurrent requests from terminal B at the same time:

```bash
# Sampling window: record RSS every 0.5 seconds into a file
while true; do echo "$(date +%s.%N),$(ps -o rss= -p $PID)"; sleep 0.5; done > /tmp/rss_probe.csv
```

```bash
# Terminal B: send 4 requests at the same time (concurrent)
for i in 1 2 3 4; do
  curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"Write a 300 word story."}],"max_tokens":300,"temperature":0}' \
    -o /dev/null &
done; wait
```

Stop the sampling window (Ctrl+C) and see whether the RSS in `/tmp/rss_probe.csv` changes noticeably before and after the load.

### 5.3 Questions you must answer (write into the notebook)

1. Is idle RSS ≈ model weights + **the whole KV**, or only the weights?
2. Does RSS under load **rise with concurrency**, or stay almost unchanged?
3. 🆕 Are RSS and `footprint` consistent? Model weights are mmap'd by default, and RSS grows as pages are touched (it may differ before and after the first batch of requests following a cold start). Measure the idle value once more with `--no-mmap`, see how much it differs, then **choose one measure and write it into Step 5.5-C** (recommended: report both).

> ⚠️ **This is my assumption and you need to verify it:** at startup llama.cpp allocates and zeroes KV for `-c` tokens in one go. If this holds, RSS is basically **flat** after startup — memory is determined by the "startup configuration" (`-np × per-slot context`), not by load.
> This matters a great deal for the project: "peak memory" cannot rely on RSS alone, and must also record **"tokens actually in use / KV capacity"** (counted by your runner itself). If RSS is indeed flat, memory pressure in Phase 1 has to be shown through **(a) the memory corresponding to different `-np`/`-c` configurations, and (b) requests failing/queueing because they exceed slot capacity** — which is exactly the course's "bounded memory budget".
> The `docs/action-guide.md` §3.3 I wrote earlier assumed RSS grows with load; **go by the measurement of this step**.

### 5.4 Try another set of configurations (record RSS and `/slots` for each)

Shut down the server (Ctrl+C in terminal A), switch to the following two, and measure idle RSS after restarting each:

```bash
# Config B: more slots
... -np 8 -c 32768 --no-kv-unified ...

# Config C: shared KV pool (all sequences share one block, long tasks can take more)
... -np 8 -c 16384 -kvu ...
```

**Record:** idle RSS and per-slot `n_ctx` for the three configurations. This table will be the first piece of evidence in the Phase 1 report for "what determines memory".

---

## 🆕 Step 5.5 — Capacity calibration and finalizing definitions (10/20, right after Step 5)

**Why it is needed:** In the original plan, λ, n_tasks, number of repetitions and total batch duration had no basis, and the definition of "peak memory" was not pinned down. Without this step, Step 9 will likely either not finish, or be overloaded throughout, or show no memory phenomenon at all.

### A. Speed calibration (start the server with `-np 4 -c 16384 --no-kv-unified`)

1. **Single-stream speed:** one ~3000-token prompt + 256 output tokens (`ignore_eos`); record `timings.prompt_per_second` and `predicted_per_second`.
2. **Aggregate throughput:** send 4 identical requests each at concurrency 1 / 2 / 4, and record "total output tokens ÷ wall-clock time". See roughly where the knee is.
3. **Run each task type 3 times on its own (c=1, no contention), and record the uncontended completion time `T_solo(type)`.** The deadline and slowdown both need it.

### B. Compute the experiment budget

- Mean service time under the default mix `S̄ = Σ share × T_solo`; capacity `μ ≈ (parallelism converted from the best aggregate throughput in A.2) / S̄`.
- Take the λ grid as `{0.3, 0.6, 0.9, 1.2} × μ` (including overload). Duration of one run ≈ `n_tasks / λ` (under overload, look at the drain time).
- Total machine time = number of runs × (run duration + about 30 s restart). **Keep all Step 9 experiments to about 20 h in total**; if over, reduce `n_tasks` (lower bound about 100) or shrink the E2 grid; **do not reduce the number of repetitions** (< 3 does not qualify).

### C. Finalize definitions (write each into the notebook, not to be changed afterwards; the table below is the suggested default)

| Item | Definition |
|---|---|
| Memory budget B | The **total KV token count** at server start (`-c`); bytes = B × KV bytes per token (Step 5.1) |
| Peak memory | During sampling, report both measures `max(RSS)` and `max(footprint)` of the server process; also report the time curve and peak of **KV utilization = live_tokens / B** |
| Concurrency c and np | **Invariant: c ≤ np**, otherwise queueing happens inside the server and the "FIFO we control ourselves" is defeated. At startup the runner verifies `/slots` count ≥ c, otherwise it refuses to run |
| Timeouts | Two levels: request-level (a single HTTP request, suggested 120 s) and task-level (including queueing, suggested `10 × T_solo`, capped at 300 s). On timeout the connection must be **closed actively** (verified by Step 4 test 7) and `status=timeout` recorded |
| deadline / goodput | Task deadline = `5 × T_solo(type)` (fixed on the dev set and then frozen); goodput = number of tasks completed within the deadline **and** with `quality_pass` ÷ total duration |
| Cold start | Record seconds from server start → `/health` returning ok, and the first request's latency; the first 3 tasks of each run are marked `warmup=true` and excluded from steady-state statistics (raw logs still kept) |
| Failure classification | `ok / timeout / error_ctx_exceeded / error_http / error_other`; the denominator is always **all submitted** tasks |

### D. Design of the memory-axis experiments (choose here based on the Step 5 results)

If Step 5 confirms that RSS is determined by the startup configuration, then an E1 that "fixes `-np 8 -c 32768` and changes only c" gives a flat line, and **the relation between memory and concurrency cannot be seen**. Two series are suggested:

- **Series S (fixed budget, split slots):** total budget B=16384 fixed, `np=c`, c ∈ {1,2,4,8} → capacity per slot = 16384/c. When c is large, long tasks fail because they exceed slot capacity → naturally yielding a "concurrency ↑ → failure rate ↑" curve.
- **Series U (shared pool):** `-kvu -np 8 -c 16384`, c ∈ {1,2,4,8}. Observe **the server's actual behavior when the pool is exhausted** (defer / error / truncate) — this is exactly the situation Phase 2's KV admission control targets.

Keep both series; if time is tight, do S first, then c ∈ {4,8} of U.

---

## Step 6 — Create the project skeleton and configs (10/21)

```bash
cd ~/"CSE 291"
git init                      # harmless if already run in Step 1
mkdir -p workloads/dev workloads/eval src/scheduler src/runner src/monitor \
         experiments configs tests results/raw analysis report/tables scripts
touch src/__init__.py src/scheduler/__init__.py src/runner/__init__.py src/monitor/__init__.py   # 🆕 otherwise python -m src.runner... fails

cat > .gitignore <<'EOF'
.venv/
__pycache__/
*.gguf
server-startup.log
results/raw/*.tmp
EOF

# Launch script: all server parameters in one place, ensuring every start is identical
cat > scripts/start_server.sh <<'EOF'
#!/usr/bin/env bash
# Usage: scripts/start_server.sh <np> <ctx> [extra flags...]
set -euo pipefail
NP=${1:?np}; CTX=${2:?ctx}; shift 2
exec ~/src/llama.cpp/build/bin/llama-server \
  -m ~/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -np "$NP" -c "$CTX" -t 8 --cache-ram 0 --metrics "$@"
EOF
chmod +x scripts/start_server.sh

# 🆕 Poll /health instead of a fixed sleep (returns 503 while loading, -f makes curl fail, so it keeps waiting)
cat > scripts/wait_health.sh <<'EOF'
#!/usr/bin/env bash
# Usage: scripts/wait_health.sh [timeout_s=180]; on success prints the cold-start wait in seconds to stdout
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

After this, always start the server with: `scripts/start_server.sh 4 16384 --no-kv-unified`.

---

## Step 7 — Write the workload (10/21–22)

**Goal:** A **fixed-seed, reproducible** workload definition.

**File conventions:**

- `workloads/dev/*.yaml`: the development set (used for design and tuning)
- `workloads/eval/*.yaml`: the evaluation set (used only in Phase 2; **for now just create the directory and stipulate that its seeds/ratios differ from the development set**)

**Four task types (course requirement: short/long context, short/long output, multi-step + controlled tool delay):**

| Type | Initial context | Output per step | Steps | tool delay |
|---|---|---|---|---|
| `short` | ~256 tok | 64 | 1 | — |
| `longctx` | ~3000 tok (leaving room for output, not exceeding slot 4096) | 64 | 1 | — |
| `longgen` | ~256 tok | 512 | 1 | — |
| `agent` | ~800 tok, each step appends the previous output and tool result | 128 | 3–6 | 0.2–3 s (controlled by the seed) |

**Designs that must be built in:**

1. **Arrival process:** open-loop Poisson arrivals with adjustable rate λ (not "finish one, then start the next").
2. **Controlled output length:** fixed `max_tokens`, `temperature=0`; `ignore_eos` when needed.
3. **Quality check (add it now, Phase 2 will use it):** the last step of every task carries a "needle" — a random key buried in the context that the model is asked to repeat, and the program checks whether the output contains that key, yielding `quality_pass`.
4. **The file is pure data** (YAML), generated by a script + seed; the same seed must produce the same workload.
5. 🆕 **Token counts must be precisely controllable:** build prompts with the server's `POST /tokenize`, so that the measured `usage.prompt_tokens` (including chat template overhead) of "~256 / ~800 / ~3000 tokens" falls within the target ±5%; put a self-check in the generator.
6. 🆕 **Work out the agent context budget:** the step-k prompt ≈ `800 + (k−1) × (128 + tool_result_tokens)`. With tool_result ≈ 150 tokens and at most 6 steps → the last step is about 2.5k < 4096, so the baseline does not exceed the limit; E2 then deliberately enlarges tool_result/steps to exceed slot capacity (the failures are then a measured phenomenon, not a bug).
7. 🆕 **Needle checks cover multiple steps:** the key is buried in the step-1 context and repeated in the last step; `quality_pass` is decided only by the program, not by eye.
8. 🆕 **Make dev / eval separation a rule:** dev uses seeds 1–5, eval uses seeds 101–105, and the mix ratios differ (e.g. dev 40/20/20/20, eval 30/30/10/30); the sha256 of the workload file is written into the manifest of every run.
9. 🆕 **deadline field:** every task in the file carries `deadline_s` (from `5 × T_solo` in Step 5.5), for Phase 2 to use directly.

**How to run (acceptance commands of Step 7; I will write and test the code then):**

```bash
python -m src.runner.gen_workload --mix configs/mix_default.yaml --seed 1 --n-tasks 200 \
  --out workloads/dev/default_s1.yaml
# Run twice in a row; the diff should be empty:
python -m src.runner.gen_workload ... --seed 1 --out /tmp/a.yaml
diff workloads/dev/default_s1.yaml /tmp/a.yaml && echo "reproducible ✔"
```

---

## Step 8 — runner + FIFO scheduler + monitoring (10/22–23)

**Goal:** A FIFO queue that you **control yourself** (not relying on the server's built-in queueing), with the concurrency limit fixed at `c`; while outputting three kinds of logs.

**Three kinds of logs (JSONL, one record per line, **failures and timeouts must also be written**):**

| File | Fields |
|---|---|
| `requests.jsonl` | task_id, step, t_submit, t_start, t_first_token, t_end, prompt_tokens, cached_tokens, output_tokens, status, error_type, warmup, id_slot (if the response provides it) |
| `tasks.jsonl` | task_id, type, t_arrive, t_first_dispatch, t_done, queue_time_total, idle_time_total (sum of tool delays), n_steps, status (ok / timeout / error_*), quality_pass, deadline_s, warmup |
| `system.jsonl` | t, rss_mb, requests_processing, requests_deferred, live_tokens (in-flight token count counted by the runner), is_processing of each slot |

🆕 **Design constraints that must be satisfied** (each is written as a test, see "How to carry it out" at the end):

1. The runner owns the FIFO queue; the number of in-flight requests is ≤ c ≤ np at all times (`/slots` is verified at startup).
2. Requests use `stream: true`, and `t_first_token` is the arrival time of the first token chunk; `timings` is taken from the last chunk.
3. Timeout = close the connection actively and record `status=timeout`; every task, whether it succeeds or fails, writes **exactly one line** of `tasks.jsonl`.
4. Each run produces `manifest.json`: repo git sha (+ whether dirty), llama.cpp commit, model sha256, full server command line, workload file sha256, seed, c, λ, timeout, start/end times, `pmset -g therm` output.
5. The system sampler runs in a separate thread, once every 0.5 s; if a sample fails, a line (with an error field) must still be written and it must not be silently skipped.
6. First verify 1–5 with a **fake server** (a local HTTP service that sleeps, and can inject failures / slow responses / disconnects), then connect the real llama-server.

**How to run (acceptance commands):**

```bash
# Terminal A: start the server
scripts/start_server.sh 4 16384 --no-kv-unified
# Terminal B: run a small-scale trial (30 tasks, concurrency 2)
python -m src.runner.run --workload workloads/dev/default_s1.yaml \
  --policy fifo --concurrency 2 --lambda 0.5 --timeout 120 \
  --out results/raw/smoke_$(date +%Y%m%d_%H%M%S)
```

**How to tell it worked:** there are three JSONL files under `results/raw/smoke_*/`; `tasks.jsonl` has 30 lines; with `--timeout 1` on purpose you can see records with `status=timeout` (proving that tasks are not silently dropped).

---

## Step 9 — Run experiments E1–E5 (10/24–25)

**General rules (course requirements):**

- **Repeat each configuration ≥ 3 times** (different random seeds or repeats of the same seed, state which);
- **Restart the server** before each experiment (to keep cold start consistent), send 1–2 warm-up requests first, **mark the warm-up segment separately** and compute steady-state data separately;
- **Keep failures/timeouts**;
- Use `caffeinate` + plugged in; do nothing else during a batch of experiments.

**Batch template (taking E1 as an example):**

```bash
# 🆕 Save as scripts/run_e1.sh and then run it (don't paste it directly into an interactive shell). Series S: fixed budget B=16384, np=c (see Step 5.5-D).
# λ is taken from the grid in Step 5.5-B; don't use a made-up 1.0
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

| Experiment | Variable | Purpose |
|---|---|---|
| **E1 concurrency sweep** | c = 1, 2, 4, 8 (16 if necessary) | How throughput knee, p50/p99, memory and failure rate change with concurrency → **find the dominant bottleneck** |
| **E2 context growth** | initial context and number of steps | How context growth pushes up memory/latency |
| **E3 idle periods** | tool delay 0 / 0.5 / 1 / 3 s | Whether idle time between steps lets slots be held by idle context, causing queueing |
| **E4 prefix reuse** | `"cache_prompt": false` vs `true` in the request; shared/not shared system prompt | Whether prefix reuse **materially** affects this runtime (look at `timings.cache_n` and prefill time) |
| **E5 arrival rate** | λ sweep (low → overload) | Find the system saturation point and decide Phase 2's working range |

🆕 **Priority (cut in this order when time is short):** must do E1 (series S), E4, E5 → then E3 (at least the two points tool delay = 0 and 3 s) → E1 series U → E2. E2 can be merged: use the agent task's "k-th step" directly as the context-length independent variable, sliced from E1 data, without necessarily running it separately. **Run each experiment first as a trial with `n_tasks=30` and 1 repetition**, confirm the logs and analysis scripts are correct, then open up the full run.

🆕 **Notes for E4:** prefix reuse only happens when a request lands in "a slot that still holds that prefix" (in this configuration `--cache-idle-slots` is disabled). At c=1 the continuation step necessarily returns to the same slot; at c=4 with a tool delay, the continuation step may land in another slot and lose reuse — which is exactly the phenomenon E3 measures. So measure E4 at least once each at c=1 and c=4, and record `cache_n` for every request.

**The result "best fixed concurrency":** in E1, select on the development set the c that gives **the highest throughput (or goodput) with an acceptable p99**. This is the **tuned baseline** that Phase 2 must beat; write it into the notebook now.

---

## Step 10 — Analysis and plotting (10/26)

**Goal:** All figures are regenerated by one command from `results/raw/`.

```bash
python analysis/make_figures.py --raw results/raw --out report/figs
```

**Figures that must at least exist (the course requires "performance curves"):**

1. concurrency c vs **throughput** (with error bars from repeated experiments)
2. concurrency c vs **p50 / p99 task completion time**
3. concurrency c vs **queueing time**
4. concurrency c vs **failure / timeout rate**
5. **Memory**: RSS of different server configurations (bar chart) + time curve of `live_tokens / KV capacity` during a run
6. E3: tool delay vs throughput/queueing
7. E4: prefill time and `cached_tokens` share with prefix reuse on/off
8. E5: λ vs p99 (the "knee" should be visible)

🆕 **The analysis script needs tests too:** use a hand-written fixture of ~10 JSONL lines, assert the **hand-computed values** of p50/p99, throughput and failure rate, and assert that "failed / timed-out tasks count in the denominator". The data of each figure is also output as CSV (`report/tables/`), and the numbers in the report all cite the CSV, to avoid figures and text disagreeing.

**Questions the analysis must answer (a one-sentence conclusion in the report):** is the dominant bottleneck **KV capacity**, **compute (decode speed)**, or **idle occupancy**? Each conclusion is backed by a figure as evidence.

---

## Step 11 — Phase 1 writeup and submission (10/27–28)

**What to submit (course requirement):** source code, dataset/workload, experiment instructions, results and logs, a concise report.

```bash
# 1) Reproduce once from a clean state (the course grading values reproducibility)
git status                               # should be clean
rm -rf /tmp/repro && git clone . /tmp/repro && cd /tmp/repro
cat README.md                            # follow the README steps once

# 2) Tag to fix the submitted version
git tag phase1-submission
```

**The README must contain:** environment versions (llama.cpp commit, model sha256, Python dependencies), build and launch commands, one command to run the smoke test, one command to reproduce the figures, and the commands of each experiment.

**Report structure (3–5 pages):**

1. Research question and environment (all versions pinned)
2. Workload definition and metric definitions
3. Measurement method and instrumentation
4. Results: performance curves
5. **Dominant bottleneck conclusion**
6. Effect of prefix / retained context (E4)
7. **Phase 2 direction and rationale** (must cite your data)
8. Limitations (laptop, CPU-only, single machine, model size, etc.)

---

## Appendix: What to do when you hit problems

| Symptom | Check first |
|---|---|
| `cmake: command not found` | Whether `brew install cmake` in Step 1 succeeded |
| Build errors | Paste the full error to me, don't guess |
| Server won't start / exits immediately | Look at the error in terminal A; common causes are a wrong model path or insufficient memory |
| `/metrics` returns 404 | `--metrics` was left out at startup |
| `n_ctx` in `/slots` is not 4096 | Record the measured value and design the workload around it (this is itself a finding) |
| Experiment results jitter a lot | Whether plugged in, whether background programs exist, whether thermal throttling; if there are too few repetitions, add more |
| Request reports context exceeded | The task context exceeds slot capacity; this is a real phenomenon to **record as a failure** |

## To confirm with the TA (does not block starting now)

- Is it acceptable to run the Phase 1 baseline on a laptop (the task document says "or your laptops")? Must the final (Phase 2) measurement be on AWS?
- The course documents are marked **TENTATIVE**; watch announcements for changes to deadlines and options.
- 🆕 Team situation (1–5 people) and whether the topic needs to be registered in the course channel.

---

## 🆕 How to carry it out (skills and methods)

There are two kinds of work in Phase 1 with different natures, handled with different methods:

| Work | Step | Method |
|---|---|---|
| Setting up the environment / build / download / measurement / calibration | 0–5.5 | **Do not go through the spec workflow.** Let Claude execute directly in the terminal and append the results to `docs/lab-notebook.md`. (`/wizard` is only for steps "only a human can do"; don't use it for what the agent can do itself.) |
| Establishing facts about llama.cpp behavior | After Step 2 pins the commit | `/research`: a background agent reads the source/docs of that commit and produces a md with citations |
| Writing the runner / scheduler / monitor / analysis scripts | 6–10 | Main flow: `/grill-with-docs` → `/to-spec` → `/to-tickets` → for each ticket `/clear` and then `/implement` (which includes `/tdd` and `/code-review`) |
| Running batch experiments | 9 | Script-driven (`scripts/run_*.sh`), not a skill |
| Retrospective | After Phase 1 ends | `/retro`, carrying the lessons into Phase 2's environment |

**Timeline**

1. **10/19:** Steps 0–3 (including `git init`). After Step 2 finishes, start `/research` (questions below).
2. **10/20:** Steps 4, 5, 5.5, with results written into the notebook.
3. **Evening of 10/20:** open a new session, `/grill-with-docs` (don't clear midway until `/to-tickets` ends): write the definitions of 5.5-C into `GLOSSARY.md`; record "CPU-only build", "the total KV token count B as the memory budget" and "c ≤ np" as ADRs. Then `/to-spec` → `/to-tickets`.
4. **10/21–23:** for each ticket `/clear` → `/implement`.
5. **10/24–26:** run experiments and produce figures (30-task trial first, then the full run).
6. **10/27–28:** write the report → `/retro`. Start filling in the report skeleton from 10/25; don't pile the whole report onto the last day.

**Questions `/research` must answer (all marked "unverified" in the original document)**

1. Under `--no-kv-unified` and `-kvu`, how `-c` and `-np` determine each slot's `n_ctx`; the server's behavior when the shared pool is exhausted.
2. When the KV buffer is allocated on the CPU backend and whether it is zeroed (why RSS is flat / not flat).
3. How a request chooses a slot (prefix similarity? idle first?); whether the response carries `id_slot`.
4. Whether a client disconnect cancels the ongoing generation and releases the slot.
5. Whether `ignore_eos` and streaming `timings` take effect on the OpenAI-compatible interface / in which chunk they appear.

**Seams for `/tdd` (to be confirmed by you during `/implement`)** — test only external behavior, not llama-server itself:

1. The `gen_workload` command: same seed → byte-identical file; the token self-check falls within ±5%.
2. The `run` command + fake server → JSONL: in-flight requests ≤ c; timeout disconnects and records `timeout`; exactly one line per task; failures are counted.
3. Analysis functions: hand-written JSONL fixture → hand-computed p50/p99, throughput, failure rate.

**Suggested tracer-bullet tickets (the starting point for `/to-tickets`)**

| # | Ticket | Blocked by |
|---|---|---|
| 1 | Deterministic workload generation + single-task runner working against a fake server, writing one line each of requests/tasks | none |
| 2 | FIFO + concurrency limit + open-loop Poisson + persist timeouts/failures | 1 |
| 3 | Multi-step agent task + tool delay + needle check | 1 |
| 4 | System sampler + manifest | 1 |
| 5 | Connect to the real llama-server, 30-task smoke test | 2, 3, 4 |
| 6 | Analysis script: metrics + figures + CSV (using fixtures, independent of real data) | 1 |
| 7 | Batch script E1–E5 | 5, 6 |

For hard-to-reproduce problems such as "results jitter / the server hangs for no reason / timeouts don't release the slot", use `/diagnosing-bugs` and don't rely on guesses.
