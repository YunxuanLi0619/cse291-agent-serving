# The memory budget B is defined as the total KV token count at server start (`-c`)

Memory budget B = llama-server's `-c`; converted to bytes = B × 114,688 B/token (Llama-3.2-3B, f16 KV: 2 × 28 layers × 8 kv heads × 128 × 2 bytes; B=16384 → 1792 MiB, consistent with `CPU KV buffer size` in the log).

Why this choice: Step 5 measured that KV is allocated in one go at startup and `footprint` barely changes under load (3789 → 3818 MB), so memory during a run does not "grow", and memory pressure can only show up as "requests exceed slot capacity and fail/queue". Therefore the KV capacity at startup is used as the budget, and `live_tokens / B` as counted by the runner expresses utilization during a run.

## Consequences

- B = `-c` only when `n_ctx / np` is a multiple of 256; otherwise llama-server rewrites `n_ctx` upward (a source-code conclusion, not measured: np=3 → 16896). Series S with c ∈ {1,2,4,8} and B=16384 is unaffected; with other np, rely on the actual `n_ctx` in the log.

- With `--no-kv-unified`, each slot's capacity = B / np; with `-kvu`, all slots share one pool of capacity B (`/slots` reports n_ctx = B for every slot).
- For a given configuration, memory is determined by the launch parameters and is independent of load (an experimental finding, see lab-notebook Step 5). Phase 2's admission control targets the KV pool, not growth of process memory.
