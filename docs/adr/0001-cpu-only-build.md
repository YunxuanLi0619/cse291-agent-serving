# llama.cpp is built CPU-only (`-DGGML_METAL=OFF`)

The Phase 1 baseline is measured on the M4 laptop with the pure CPU backend. The course document allows "a small AWS instance or a laptop" and does not require a GPU (whether the final measurement must be on AWS has not yet been confirmed with the TA). With the CPU backend, model weights and KV both live in process memory, which is closer to a CPU cloud instance; under Metal they go into GPU unified memory, and whether `RSS`/`footprint` faithfully reflect KV usage is unverified (a concern from the handbook, not measured).

## Considered Options

- **Metal (GPU):** expected to be faster (especially prefill), and concurrency scaling may also be better; not measured. Rejected because: all numbers from Steps 4–5.5 (memory baseline, T_solo, λ grid, deadline, 20.6 h budget) would have to be redone, the memory measure would need re-verification, and the Phase 1 deadline is 10/28.

## Consequences

- CPU compute is the dominant constraint (calibration: c=4 aggregate throughput is only 1.5× that of c=1), which may make memory pressure look less prominent on this platform than on a GPU platform; this must be stated in the report's limitations.
- All speed numbers are valid only for "this M4 + fixed commit + fixed thread count"; changing platform (e.g. AWS) requires redoing Step 5.5.
