# Memory-Aware Agent Edge Serving

Serve multi-step agent Tasks with llama.cpp on a memory-constrained machine; measure how concurrency, context growth and inter-step idle time affect latency, memory and failures, and provide a baseline for Phase 2's memory-aware scheduling.

## Language

### Work

**Task**:
A complete unit of work, made up of one or more Steps, belonging to one of the four Task types.
_Avoid_: job, session

**Task type**:
The category of a Task: `short`, `longctx`, `longgen`, `agent`; it determines the initial context length, output length and number of steps.
_Avoid_: category, class

**Step**:
One model call of a Task, plus the Tool delay that follows (the last step has no delay). The unit of FIFO queueing.
_Avoid_: turn, round, stage

**Request**:
The HTTP call to llama-server made within a Step; logs are recorded per Request, while domain discussion uses Step.
_Avoid_: call, query

**Tool delay**:
The controlled wait between two adjacent Steps of an agent Task, simulating tool execution; it does not hold a concurrency slot.
_Avoid_: think time, sleep

**Needle**:
A random key buried in the first-step context of a Task, which the model is asked to repeat in the last step; the program uses it to decide quality pass.
_Avoid_: canary, probe

**Quality pass**:
The final output of a Task contains the correct Needle.
_Avoid_: correctness, accuracy

### Load

**Workload**:
A pure-data file generated from a fixed seed: the Task list, unit inter-arrival times at λ=1, tool text, Needles and deadlines. Byte-identical for the same seed.
_Avoid_: dataset, trace

**Dev workload / Eval workload**:
Dev is used for design and tuning (seeds 1–5), Eval for the final comparison (seeds 101–105); their mix ratios differ.
_Avoid_: train/test set

**Arrival rate (λ)**:
The number of Tasks arriving per second under an open-loop Poisson process; the runner scales with `unit interval / λ`.
_Avoid_: load, QPS

**Capacity (μ)**:
The estimated rate at which the system can sustainably complete Tasks under the default mix (≈ effective parallelism / mean service time). λ is expressed as a multiple of μ.
_Avoid_: max throughput

**Warm-up task**:
The first 3 Tasks to arrive in each run; the raw logs keep them, steady-state statistics exclude them.
_Avoid_: burn-in

### Memory

**Slot**:
The context position of one parallel sequence inside llama-server, holding its own KV.
_Avoid_: lane, worker

**Memory budget (B)**:
The total number of KV tokens at server start (`-c`); bytes = B × 114,688.
_Avoid_: memory limit, RAM budget

**KV utilization**:
The number of tokens occupied by in-flight Requests during a run, divided by B.
_Avoid_: memory usage

**Retained context**:
The KV prefix that stays in a Slot after a Step completes; it may be reused by the continuation step, or taken over by another Request.
_Avoid_: cache, session state

**Prefix reuse**:
A Request hits an existing prefix in the Slot and only needs to process the remaining tokens (`cached_tokens` > 0).
_Avoid_: cache hit

**Series S**:
The experiment series with B fixed and np = c, where each Slot's capacity = B / c.
_Avoid_: split config

**Series U**:
The experiment series where all Slots share one KV pool of capacity B (`-kvu`).
_Avoid_: shared config

### Scheduling

**Concurrency (c)**:
The upper bound on the number of Requests the runner allows in flight at once; invariant c ≤ np.
_Avoid_: parallelism, workers

**FIFO queue**:
The queue owned by the runner, first-in-first-out by "Step ready time".
_Avoid_: backlog

**Scheduling policy**:
The rule that decides which ready Step to dispatch next; FIFO is the first implementation, and Phase 2's memory-aware policy is another.
_Avoid_: scheduler (the scheduler is the whole that owns the queue and the concurrency limit; the Policy is just the decision within it)

**Fixed-concurrency baseline**:
The c selected on the Dev workload with the highest throughput or goodput and an acceptable p99; the tuned baseline that Phase 2 must beat.
_Avoid_: best config

### Measurement

**T_solo**:
The completion time of a Task type at c=1 without contention.
_Avoid_: baseline latency

**Deadline**:
The completion deadline of a Task = 5 × T_solo (agents compute it from their own number of steps and delays), frozen on the Dev workload.
_Avoid_: SLO, SLA

**Goodput**:
The number of Tasks that complete within the deadline and with quality pass, divided by the total duration.
_Avoid_: throughput (throughput does not require a deadline or quality pass)

**Run**:
A complete experiment with a fixed configuration (server parameters, c, λ, workload, seed), producing three kinds of logs and one manifest.
_Avoid_: trial, experiment (Experiment refers to a series such as E1–E5)
