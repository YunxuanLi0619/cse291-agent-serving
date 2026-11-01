# Memory-Aware Agent Edge Serving — Phase 1 baseline

For terminology see `GLOSSARY.md`; for decisions see `docs/adr/`. Environment: `uv venv .venv --python 3.12 && uv pip install -r requirements.lock.txt`.

## Tests

```bash
.venv/bin/python -m pytest          # all tests (no real llama-server needed)
```

## Pipeline (currently available: Workload generation + runner: FIFO, concurrency limit, open-loop Poisson, timeouts/failures persisted, system sampler and manifest + analysis scripts)

```bash
scripts/start_server.sh 4 16384 --no-kv-unified    # terminal A: start llama-server (parameters are kept in the script)
scripts/wait_health.sh                              # poll /health; on success prints the cold-start wait in seconds

# Generate a Workload (token lengths are calibrated and self-checked to ±5% against the running llama-server); two runs with the same seed give byte-identical output
python -m src.runner.gen_workload --mix configs/mix_dev.json --t-solo configs/t_solo.json \
  --seed 1 --n-tasks 100 --out workloads/dev/default_s1.json

# Execute a Workload: concurrency limit c=4, arrival rate λ=0.05 per second, FIFO; writes requests.jsonl, tasks.jsonl, system.jsonl (sampled every 0.5 s) and manifest.json
# (pass --server-pid, the llama-server process id, to sample footprint/RSS and record the launch command line; otherwise those fields are null and an error is written)
# (before starting, /slots count ≥ c is verified; --timeout overrides the request-level timeout, default 120 s; --n-tasks submits only the first N)
python -m src.runner.run --workload workloads/dev/default_s1.json --c 4 --lambda 0.05 \
  --policy fifo --out results/raw/try1 --server-pid $(pgrep -f llama-server)

# Responses of the OAI interface carry no id_slot: map the per-request slot selection in the server log (LRU / LCP and similarity) back onto requests.jsonl in order,
# to explain the cached_tokens of agent continuation steps (--skip = number of launches already in the log before this Run; required when the log is shared with other requests)
python -m src.analysis.slot_trace --run results/raw/try1 --server-log server.log --skip 0

# Analysis: one command regenerates all CSVs and plots from the raw log directory (completed Run directories are found recursively)
python -m src.analysis --raw results/raw/e1 --out results/analysis
```

## Batch E1–E5 (#9)

```bash
.venv/bin/python scripts/run_batch.py --dry-run               # plan and estimated duration (about 19 h); AC power not required
.venv/bin/python scripts/run_batch.py --pilots-only           # only the 30-Task pilot of each experiment
caffeinate -is scripts/run_batch.sh                           # official batch; after an interruption, rerun as-is to continue from the next Run
```

- The matrix and ordering are in `src/batch/plan.py` (parameters in `configs/batch.json`): E1-S (c=1/2/4/8, λ=0.9μ) → E4 (cache_prompt off, c=1/4) → E5 (λ=0.3/0.6/1.2μ, c=4) → E3 (Tool delay 0/0.5/1/3 s, c=4) → E1-U (`-kvu`, np=8, c=4/8); each configuration is repeated 3 times, repetition i uses seed i; E4's "on" side and E5's 0.9μ reuse the Runs of E1-S, and E2 is sliced by Step from E1 data.
- Each experiment first does a 30-Task pilot and the full run is opened up only if it passes; an experiment whose pilot fails is recorded in `blocked_stages` of `results/batch/state.json`, and the other experiments continue.
- The server is restarted before every Run and the cold-start seconds are recorded (`cold_start_s` in the manifest); the first 3 Tasks are marked `warmup`. Before each batch starts, the power state, low-power mode and `pmset -g therm` are written to `results/batch/batch_start_*.json`; the batch refuses to start when not on AC power (exit code 3); if the charger is unplugged mid-batch, it pauses and waits.
- A Run that passes validation lives in `results/raw/batch/<run>/` (the presence of `batch_meta.json` marks it as complete); attempts that fail validation are moved to `results/batch/invalid/` and retried (at most 3 times). Throttling is only recorded (`throttled` in `batch_meta.json`), not rejected.
- At the end it writes `results/batch/summary.json` and `baseline_c.json` (the Fixed-concurrency baseline on the Dev workload; rules in `src/batch/baseline.py`) and runs the analysis into `results/analysis/batch`.

- Analysis output: `runs.csv` (one row per Run); each figure = a same-named `.csv` + `.png`, and the report cites the CSV: `perf_by_{c,lambda,tool_delay_s}_<platform>` (throughput/goodput, p50/p99, queueing, failure/timeout rate, lines split by the remaining configuration, error bars = sample standard deviation across repetitions), `paired_cache_prompt_<platform>` (on − off within the same seed), `step_slice_<platform>` (k-th step of agents), `memory_peak_<platform>`, `kv_util_<platform>`. A separate set is produced for each platform; they are not plotted together.
- The metric definitions are at the top of `src/analysis/metrics.py`: steady state excludes `warmup=true`; p50/p99 count only ok Tasks; failures/timeouts are in the denominator; throughput and goodput are divided by the steady-state window (earliest arrival → latest completion).
- The analysis reads its configuration from the manifest: `c`, `lambda`, `seed`, `server_cmdline` (parsing `-np`, `-c`, series S/U). Grouping also uses the optional fields `tool_delay_s` and `cache_prompt`, written into the manifest by the batch script; when absent they are empty and do not affect other analyses. A Run whose manifest has no `t_end` is considered incomplete and skipped.

- `configs/mix_dev.json` and `mix_eval.json`: seed ranges (1–5 / 101–105) and mix ratios differ; generation is refused if the seed is not in its range.
- `configs/t_solo.json`: the T_solo table (an input config, not hard-coded); `deadline_s = 5 × T_solo`, and agents compute it from their own number of steps and Tool delays.
- A Workload is JSON (`sort_keys`, stable bytes); each Task's `steps` already hold the complete user messages, `max_tokens` and the Tool delay after the Step.
