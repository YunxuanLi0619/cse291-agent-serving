# CSE 291A Project Action Guide: Memory-Aware Agent Edge Serving

> Generated: 2026-10-19 (Monday). Based on: `Memory Aware Agent Edge Serving.pdf`, `CSE 291A Fall 2026 Course Projects.pdf`, and the current llama.cpp `llama-server` documentation.
> The course documents are marked **TENTATIVE**; for details, follow the course Piazza/announcements.

---

## 0. One-page overview

**Core question (original wording):** On a memory-constrained machine, how can admission and scheduling decisions increase "completed agent work" while keeping memory usage bounded?

**Key dates**

| Milestone | Date | Time left |
|---|---|---|
| Phase 1: implementation + measurement + Phase 1 writeup | **Wed 2026-10-28** | 9 days |
| Phase 2: implementation + evaluation + final report | **Thu 2026-12-10** | 52 days |

**Resources:** $50 AWS credits per person (can be pooled within a team); for LLM APIs use TritonAI (this project runs a local model, so it is barely needed); the list of available GPU instances is on the course "Useful Resources" tab (I cannot see it, you need to confirm it yourself). A laptop can be used for development. Teams of 1–5, **the whole team gets the same grade**, and no individual contribution statement is required.

**Grading dimensions (summarized from the course document)**

1. Baseline reproduction and measurement quality
2. Soundness of the Phase 2 design (it must be derived from measurement data) and implementation quality
3. Experimental control and quality of comparisons
4. Depth of analysis
5. Reproducibility and report clarity

**One-sentence strategy:** The task statement says "implementing many policies is worse than understanding one policy thoroughly". So take the path of **one focused mechanism + a tuned simple baseline + strict controls + an honest report (including cases with no gain)**.

---

## 1. Understand the task first: the deliverables, item by item

### Phase 1 (due 10/28)

| Requirement | What exactly to deliver |
|---|---|
| FIFO queue + fixed concurrency | A scheduler you control yourself (not the server's built-in queueing) |
| Mixed load: short/long context, short/long output, multi-step tasks (controlled tool delay) | A **workload definition file** (YAML/JSON + fixed random seed) |
| Instrument both **a single model request** and **the complete agent task** | Two levels of logs: request-level, task-level |
| Metrics: task completion time, queueing time, tail latency, throughput, peak memory, failures/timeouts | Raw JSONL logs + analysis scripts |
| Figure out how concurrency, context growth and idle periods between steps affect memory pressure and completed work | 3+ sweep experiments (see §4) |
| Judge whether retained context / reusable prefix materially affects the chosen runtime | prefix cache on/off comparison |
| **Reproducible baseline + performance curves + explanation of the dominant bottleneck** | A clear "bottleneck conclusion" in the writeup |
| Phase 2 direction | A **data-backed** Phase 2 plan at the end of the writeup |

### Phase 2 (due 12/10)

- Pick **one** focused mechanism and compare **FIFO** and **the best fixed-concurrency baseline selected on the development workload**.
- If you do a cache-aware design, you must also compare against a **simple fixed cache policy**.
- Evaluate under **varying arrival rates and workload mixes**, and **it must include cases where the policy gains little**.
- Check that the gain does not come from: skipping tasks, shortening outputs, or lowering task quality.
- Account for the policy's own overhead; check whether long tasks are unfairly delayed.
- Submit: scheduler, workload runner, raw measurements, analysis scripts, report.

### Hard course-level rules (where points are easily lost)

- **Separate the development workload from the evaluation workload**: design/tune on the development workload, and score on a different evaluation workload.
- **Pin versions**: model, runtime (llama.cpp commit), dataset and code version must all be pinned.
- Account for **setup / build time** (if it affects the use case); distinguish **cold-start** from **steady-state**.
- **Failures and timeouts must be recorded, not silently dropped**.
- Randomized experiments must be **repeated** and **variance reported**.
- Claiming "better" requires controlled experiments; "showing that the simple baseline is sufficient" is also a valid conclusion.

---

## 2. Research direction suggestions

### 2.1 First build the right mental model of the problem

In a runtime like llama.cpp, "memory" consists mainly of two parts:

- **Model weights**: constant, independent of concurrency.
- **KV cache**: grows linearly with the "number of tokens resident at the same time". The KV bytes per token are approximately

  `2 × n_layer × n_kv_head × head_dim × bytes_per_elem` (bytes=2 for f16)

  This formula lets you estimate memory **before a request arrives**, and is the basis of an admission policy.

The difference between agent tasks and ordinary requests is that **there is a tool delay between steps**. During this idle period the context has two possible fates:

- It keeps occupying the slot / KV → memory is eaten up by "idle context" and new tasks cannot get in;
- It is overwritten by other requests → the next step must prefill again, wasting compute and lengthening latency.

This **"hold without using vs. release and recompute" trade-off** is the core tension of this task, and Phase 1 must quantify it with data.

### 2.2 A risk that must be answered first: is memory really the bottleneck?

On a small CPU machine, decode often hits **compute / memory bandwidth** first, throughput saturates at concurrency 2–4, and adding more concurrency only lengthens latency. In that case "memory pressure" may not appear.

How to handle it (the task statement allows it): **create the memory constraint deliberately**, for example

- choose an instance with little memory, or limit available memory with cgroup / Docker `--memory` / `systemd-run -p MemoryMax=` (an OOM kill is recorded as a failure);
- or set a **KV token budget B** (determined by `--ctx-size` and the number of slots), treat it as the "memory budget", and explain the definition in the report.

The Phase 1 "dominant bottleneck" conclusion can be "KV capacity under budget X" or "compute under budget Y"; both are acceptable, and **the key is to prove it with curves**.

### 2.3 Phase 2 candidate mechanisms (ordered by recommendation)

| Priority | Mechanism | Idea | Difficulty | Risk |
|---|---|---|---|---|
| **★ Recommended** | **KV-budget admission + continuation-first** | Use the estimated KV token count for admission (admit only if `committed + new task estimate ≤ B`); the "next step" of a task already running takes priority over new tasks (finish-what-you-start), with aging to prevent starvation of long tasks | Medium | Inaccurate estimates; need to prove it does not win by rejecting tasks |
| Option A | Adaptive concurrency (AIMD) | Increase/decrease concurrency based on observed pressure signals (`requests_deferred`, RSS, latency) | Low–medium | Noisy signals; easily tied by a "tuned fixed concurrency" |
| Option B | Cache-aware retention policy | Decide whether to keep or release a task's prefix during the tool delay | High | Depends on controls exposed by the runtime; needs an extra comparison against a fixed cache policy |
| Not recommended | Doing several at once | The task statement explicitly says several are worse than one in depth | — | The analysis becomes shallow |

**My recommendation:** Make "KV-budget admission + continuation-first" the main line. It is a mechanism that **can be split into two ablations** (budget only / continuation-first only / both), which naturally provides material for "deep analysis", and it is implemented entirely in your own scheduler, **without depending on any control exposed by the runtime**, so the risk is lowest. Cache-aware is an extension only if the Phase 1 data show "a significant gain from prefix reuse".

> The final basis for choosing a mechanism must be the Phase 1 data (the course requires "justify the policy from measurements"). The table above is a prior; the 10/28 writeup must confirm or refute it with your curves.

---

## 3. Technical plan (llama.cpp essentials)

The following comes from the current `llama-server` documentation (queried via Context7); **please double-check it on your pinned commit**, since flags change between versions.

### 3.1 Startup and concurrency

```bash
llama-server -m model.gguf --host 127.0.0.1 --port 8080 \
  -np 4 -c 16384 --metrics
```

- `-np, --parallel N`: number of slots, i.e. the server-side concurrency limit. Default `-1` = auto (in which case `n_parallel=4` and `kv_unified=true` is forced).
- `-c, --ctx-size`: total context. **With non-unified KV it is divided evenly among slots** (16384/4 = 4096/slot, historical behavior, to be verified); with unified KV all sequences share one KV buffer. **Always confirm with the `n_ctx` field in `GET /slots` how much each slot actually gets**, and write `-kvu / --no-kv-unified` explicitly into the experiment configuration instead of relying on defaults.
- `-cb, --cont-batching`: on by default (continuous batching).
- A single batch is shared by all active slots, and `llama_decode` is the main compute bottleneck.

### 3.2 Cache controls (use only what the runtime actually exposes)

| Control | Effect |
|---|---|
| `--cache-prompt` / `--no-cache-prompt` (on by default) | Reuse the common prefix across requests |
| `--cache-reuse N` | Minimum chunk size to reuse via KV shifting (default 0) |
| `-cram, --cache-ram N` | Maximum MiB of the prompt cache (default 8192) |
| `--cache-idle-slots` | Save idle slots into the prompt cache when a new task arrives (on by default, depends on cache-ram) |
| `--slot-save-path` + `POST /slots/{id}?action=save\|restore` | Save/restore a slot's KV to disk, returning `n_restored`, `restore_ms` |

From the task statement: *"Any cache-aware extension must use cache controls and measurements that the runtime actually exposes."* So a cache-aware plan can only be built on the controls in the table above.

### 3.3 Observability

- `GET /metrics` (needs `--metrics`): `llamacpp:requests_processing`, `requests_deferred`, `n_tokens_max`, `n_busy_slots_per_decode`, `prompt_tokens_seconds`, `predicted_tokens_seconds`, etc.
- `GET /slots`: each slot's `is_processing`, `n_ctx`, `id_task`, etc.
- **The documentation has no direct "KV usage in bytes" metric** → peak memory has to rely on **sampling the server process RSS externally** (`psutil`, once every 100 ms–1 s).
  ⚠️ **Note:** llama.cpp very likely allocates and zeroes the whole KV in one go at startup, in which case RSS is basically flat (determined by `-np × per-slot context`, not by load). This assumption must be verified by measurement in Step 5 of `docs/phase1-steps.md`; if it holds, you must also record the runner's own "in-flight tokens / KV capacity" to represent memory pressure, and cannot look only at RSS increments.
- The `timings` field in each response (`prompt_n`, `cache_n`, etc.; I did not verify it in the documentation, please confirm it yourself) can be used to compute "how many tokens the prefix hit".

### 3.4 Model and machine choice

- Keep the model and precision **fixed throughout**. Choose a small model that fits "comfortably" on the chosen machine (e.g. a 1B–3B instruct, Q4_K_M quantization), and create pressure through concurrency/context.
- **Your laptop is macOS (Darwin)**: Apple Silicon has unified memory, so a hard memory cap is hard to impose, and Metal behaves very differently from AWS CPU. Recommended: **develop on the laptop, measure finally on a small AWS CPU instance**, and limit memory with cgroup. The usable instances are per the course Useful Resources.
- **Spending discipline:** get the scripts working locally first, then start a cloud instance to run batches; stop it immediately when done. Work out "hours for one full sweep on AWS × unit price" to make sure you can rerun ≥3 times.

---

## 4. Phase 1 action plan (10/19 → 10/28)

### Day 1–2 (10/19–10/20): environment and "it runs"

- [ ] **Read the two sample reports provided by the course** (one high-scoring, one low-scoring; the links are on page 2 of the course PDF). This is the fastest way to understand the grading criteria.
- [ ] Confirm the team situation (1–5 people) and confirm the topic through the course channel.
- [ ] Fetch and **record the llama.cpp commit hash**, build and **record the build time**; download the model and record its SHA256.
- [ ] Start `llama-server --metrics` locally and manually curl `/v1/chat/completions`, `/metrics`, `/slots`.
- [ ] Confirm the RSS sampling script works on the laptop.

### Day 3–5 (10/21–10/23): workload and runner

**Workload design (development set):**

| Task type | Context | Output | Steps | Purpose |
|---|---|---|---|---|
| Short | ~256 tok | ~64 tok | 1 | Control |
| LongCtx | 3–4k tok | short | 1 | Test prefill/KV pressure |
| LongGen | short | ~512 tok | 1 | Test decode occupancy time |
| Agent | start ~1k tok, each step appends the previous output + tool result | medium | 3–6 | Test context growth + idle periods |

Key design choices:

- **Open-loop arrivals (Poisson, rate λ)**, not closed-loop. Only open-loop can measure queueing time and tail latency and sweep out the "knee".
- **Tool delay uses a controlled `sleep`** (e.g. a fixed distribution of 0.2–3 s + seed), without calling real tools.
- **Output length must be controlled**: fixed `max_tokens`, `temperature=0`, and `ignore_eos` if necessary to make lengths reproducible (and explain it in the documentation).
- **Add a programmatically checkable "quality check" to tasks**: e.g. needle-in-context (bury a key in the context and ask for it to be repeated in the final step). Only this lets Phase 2 prove "the gain does not come from lowering quality". **Design it now, don't wait for Phase 2.**
- Fix the seeds and put the workload files under version control.

**Runner / logs:**

- request-level: `task_id, step, t_submit, t_start, t_first_token, t_end, prompt_tokens, cached_tokens, output_tokens, status`
- task-level: `task_id, type, t_arrive, t_first_dispatch, t_done, queue_time_total, n_steps, status (ok/timeout/oom/error), quality_pass`
- system-level: `t, rss_mb, requests_processing, requests_deferred, n_busy_slots_per_decode`
- All failures/timeouts must be persisted, and the analysis script's denominators must include them.
- The scheduler is your own FIFO + fixed concurrency limit (e.g. a semaphore); do not rely on the server's queueing.

### Day 6–8 (10/24–10/26): experiments

| Experiment | Variable | Question answered |
|---|---|---|
| E1 concurrency sweep | concurrency c ∈ {1,2,4,8,…} | Where is the throughput knee? How do p50/p99 and peak memory change with c? When do failures start? |
| E2 context growth | starting context / number of steps | How does context growth push up memory and latency? |
| E3 idle periods | tool delay size | Does idle time between steps let the slot be held by "idle context" and cause queueing? |
| E4 prefix reuse | `--cache-prompt` on/off, shared/not shared system prompt | Does prefix reuse **materially** affect this runtime? How much prefill is saved? |
| E5 arrival rate | λ sweep | Find the saturation point and choose the working range for Phase 2 |

- Each configuration is **repeated ≥3 times**, reporting the mean and variance/interval;
- **Report cold start and steady-state separately** (discard or list the warm-up segment separately);
- In E1 select the "**best fixed concurrency on the development set**", which is the baseline Phase 2 must beat.

### Day 9 (10/27–10/28): writeup and packaging

**Phase 1 writeup structure (suggested 3–5 pages):**

1. Research question and environment (machine, model, commit, flags, all pinned)
2. Workload definition and metric definitions
3. Measurement method and instrumentation
4. Results: performance curves (c vs throughput/p99/peak memory/failure rate)
5. **Dominant bottleneck conclusion** (one sentence + evidence curves)
6. Effect of prefix / retained context
7. **Phase 2 direction and rationale** (must cite the data above)
8. Limitations

**Submission package:** source code, workload definition, raw logs, analysis scripts, **instructions to reproduce the main figures with one command**, report.

---

## 5. Phase 2 action plan (10/29 → 12/10)

| Time | Task |
|---|---|
| 10/29–11/05 | **Freeze the design** based on Phase 1 feedback and data; write down the policy pseudocode, estimation method, hyperparameters and tuning protocol |
| 11/06–11/19 | Implement the scheduler; unit tests (estimation, admission, aging); tune on the development set |
| 11/20–12/01 | After **freezing parameters**, run all comparisons on the **independent evaluation set** (avoid a concentrated sprint in Thanksgiving week 11/26) |
| 12/02–12/08 | Ablations, fairness, overhead analysis; produce figures; write the report |
| 12/09 | Reproduction drill (rerun from a clean environment following the README) |
| 12/10 | Submit |

### 5.1 Concrete design of the recommended policy (adjustable according to Phase 1 data)

```
State: committed_tokens = Σ KV token estimates of in-flight tasks
B = KV token budget (derived from the memory limit / ctx)

on new task arrival:
    est = estimate(task)           # by task type + historical moving average
    if committed + est ≤ B: admit  else: enqueue in the waiting queue
on a task's next step becoming ready:
    enqueue in the "continuation queue"   # takes priority over the new-task queue
dispatch:
    take from the continuation queue first (with an aging cap), then from the new-task queue
on task completion: committed -= actual usage
```

- The estimator must report its **error** (estimated vs actual tokens), which is part of "justify from measurements".
- Optional extension: correct B online using RSS feedback.

### 5.2 Comparison matrix (mandatory by the course)

| Configuration | Description |
|---|---|
| FIFO | Phase 1 baseline |
| Best-fixed-concurrency | The optimal c tuned on the **development set** (the tuned simple baseline) |
| Proposed policy | Run on the evaluation set after freezing parameters |
| Ablation | budget only / continuation-first only / both |
| (If cache-aware) simple fixed cache policy | e.g. "always release" / "always keep" / fixed TTL |

**Varied dimensions:** arrival rate λ (low/medium/high/overload) × workload mix (short-heavy, long-heavy, multi-step-heavy, uniform), **must include a no-gain region** (low λ, homogeneous short tasks, ample memory), and report it truthfully.

### 5.3 Anti-"cheating" checklist (explicitly required by the course)

- [ ] The **number of completed tasks** is the same across policies, or the difference is explicitly reported; timed-out/rejected tasks count in the denominator
- [ ] The **output token count** per step is consistent with the baseline (distribution comparison)
- [ ] The **quality_pass rate** does not drop
- [ ] For the main metric use **goodput = number of tasks completed within the deadline and passing the check / second**, to avoid comparing only mean latency
- [ ] **Policy overhead**: compare scheduler CPU time and per-decision time against a "null-policy pass-through"
- [ ] **Long-task fairness**: report p50/p99 and slowdown (completion time / uncontended time) per task type, to see whether the longest tasks are starved
- [ ] The evaluation set and the development set differ in **both seeds and mix ratios**

---

## 6. Report structure (final, suggested)

1. Problem and motivation
2. System and environment (all versions pinned)
3. Workload and metrics
4. Phase 1: baseline and bottleneck
5. Phase 2: policy design and rationale (citing Phase 1)
6. Experimental setup (controlled variables, number of repetitions, cold/warm)
7. Results: main comparison → ablations → fairness → overhead → **no-gain region**
8. Statistics of failures/timeouts
9. Limitations and threats to validity
10. Reproduction instructions

Grading values "**connecting observed behavior with memory-management decisions**". Every figure must answer "what does this show about memory behavior, and therefore why is the policy designed this way".

---

## 7. Suggested repository structure

```
/
├── CLAUDE.md
├── docs/ (agents/, action-guide.md)
├── workloads/      # workload definitions + seeds (dev / eval separate)
├── src/
│   ├── scheduler/  # FIFO, fixed concurrency, proposed policy
│   ├── runner/     # open-loop arrivals, agent task execution, tool delay simulation
│   └── monitor/    # RSS and /metrics sampling
├── experiments/    # one config file per experiment
├── results/raw/    # raw JSONL (do not edit by hand)
├── analysis/       # scripts that generate all figures from raw
├── report/
└── README.md       # one-command reproduction
```

---

## 8. Risks and countermeasures

| Risk | Countermeasure |
|---|---|
| Memory is not the bottleneck, the policy has no gain | Impose a memory budget deliberately (§2.2); "under what conditions there is no gain" is itself a valid conclusion |
| The baseline is tied by "tuned fixed concurrency" | The course allows this conclusion; use ablations and the λ sweep to show the boundary |
| High experimental noise | Fixed seeds, repeat ≥3 times, report intervals; do not mix different instance types in the cloud |
| AWS budget runs out | Get things working locally before going to the cloud; stop immediately after running; estimate the cost of each round in advance |
| Uncontrolled output length distorts comparisons | Fixed `max_tokens`, `temperature=0`, `ignore_eos` if necessary |
| Flags/behavior change with the llama.cpp version | Pin the commit; write all flags explicitly in the experiment configuration (especially `-np`, `-c`, `-kvu`, `--cache-*`) |
| Discovering only in Phase 2 that quality checks are missing | Write needle-style checks into the workload already in Phase 1 |
| Parallel mechanisms make the analysis shallow | Stick to one main mechanism + ablations |

---

## 9. Things you need to confirm/decide soon

1. **Team**: how many people? (Affects total credits and division of work; the whole team gets the same grade.)
2. **Machine**: laptop model (Apple Silicon? how much memory?) and the available AWS instances in Useful Resources.
3. **Where the final measurement is done**: AWS small CPU instance + memory limit is recommended; please confirm with the TA whether laptop results can serve as final evidence.
4. **Model choice**: pick one and fix it until the end.
5. The course documents are **TENTATIVE**; watch for announcements changing option numbers or deadlines (the option name in the document is "Memory-Aware Agent Serving", and your topic file says "Option 1").

---

## 10. What could not be verified (honest statement)

- I did not have access to the course's **Useful Resources tab** or the sample reports (Google Drive links); for the AWS instance list and grading details, follow those.
- The `timings.cache_n` field of `llama-server`, the `id_slot` parameter in requests, and the exact behavior of even ctx division in non-unified mode were **not directly confirmed in the documentation query just done**; please verify them by measurement on your pinned commit.
- The best Phase 2 mechanism depends on the Phase 1 measurements; the recommendation above is a prior judgment.
