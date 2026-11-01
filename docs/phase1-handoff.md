# Phase 1 handoff: after Steps 0–5.5 (2026-10-19)

A new session (`/grill-with-docs` etc.) starts reading from here. For detailed data see `docs/lab-notebook.md`; for raw data see `results/raw/`.

## Established facts (can be written directly into GLOSSARY / ADR)
- **ADR candidates:** ① CPU-only build (`-DGGML_METAL=OFF`); ② memory budget B = total KV token count (`-c`), bytes = B × 114,688; ③ invariant c ≤ np; ④ primary peak-memory measure = `footprint`, RSS only secondary (compressed memory on macOS makes RSS untrustworthy); ⑤ for a given configuration, memory is determined by launch parameters and is independent of load.
- **Definitions table:** timeouts, deadline, goodput, cold start, failure classification; see lab-notebook "5.5-C".
- **Pinned versions:** llama.cpp `1b29621bacc6a446bd523fc790316a24a06e886f` (b11741), model `Llama-3.2-3B-Instruct-Q4_K_M.gguf` sha256 `6c1a2b41…28ff`, binary sha256 in `results/raw/env/env_snapshot.txt`.
- **Calibration numbers (measured on battery):** T_solo short 4.3 / longctx 35.0 / longgen 22.5 / agent (4 steps, 1 s delay) 35.2 s; S̄=20.26 s; μ≈0.074 task/s; λ grid {0.022, 0.044, 0.067, 0.089}; total machine time ≈20.6 h (relies on E4-on and E5's 0.9μ reusing E1 data).

## Where this differs from the handbook (phase1-steps.md); follow this when implementing
1. `--no-mmap` does not exist in this commit; use `-lm none` instead (`-lm mmap|none|mlock|mmap+mlock`).
2. The default log does not contain hyperparameters/KV buffer; `-lv 4` is needed.
3. `docs/action-guide.md` §3.3 assumes "RSS grows with load" — this has been refuted by measurement and is void.
4. Streaming chunks and `/slots` **have no `id_slot`**: leave that field of `requests.jsonl` empty, or correlate afterwards using the server log `launch_slot_: id N | task T`.
5. Token self-check: the chat template adds about +32 tokens, and short (target 256) measures 290; the generator calibrates to ±5% based on `usage.prompt_tokens`.
6. `timings` is in the last data chunk (the one with `finish_reason`, before `[DONE]`). `ignore_eos` takes effect on the OpenAI-compatible interface. A disconnect releases the slot in ≈0.7 s.
7. Newly found parameter: `--kv-unified-per-slot N` (per-slot context cap), not yet tried.

## To do before Step 9 / not yet verified
- [ ] **Re-measure T_solo on AC power** (3 tasks per type); if it deviates from 5.5 by >15%, recalibrate the λ grid, deadlines and budget. Calibration was done on battery power with low-power mode off.
- [ ] **Thread-count sensitivity:** this machine has 4 performance cores + 6 efficiency cores, and `-t 8` = 4P+4E. The speed and stability of `-t 4` / `-t 8` / `-t 10` have not been compared; if it is to change, it must be decided before recalibration (changing it means re-measuring every speed number).
- [ ] The c=8 aggregate throughput has not been measured (the budget assumes = c=4).
- [ ] Cold-disk start has not been measured (all cold-start numbers have the model already in the page cache).
- [ ] The server's actual behavior when the shared pool (`-kvu`) is exhausted has not been measured (series U should observe it); `/research` is yet to be run.
- [ ] The 5 questions of `/research` (see "Questions /research must answer" in phase1-steps.md) — of which slot selection (LCP similarity), disconnect cancellation and the missing `id_slot` already have log evidence but have not been verified against the source.
- [ ] System sampling must use `footprint` (about 0.2–0.5 s per call); do not make the sampling interval lower than 0.5 s; `rss_footprint_sampler.sh` already does this.

## Environment reminders
- Before a batch: plug in, turn off low-power mode, `caffeinate -dimsu`, record `pmset -g therm`. This machine is a MacBook Air (fanless) and will throttle under prolonged full load.
- Currently there are two redundant `caffeinate -i -t 300` (5-minute timers not started by this session) and one `caffeinate -dimsu` (started by this session, to be killed manually after the batch ends).
- The git author email is `851489600@qq.com`, which differs from the GitHub account email; the course PDFs are in `.gitignore` and are not checked in.
