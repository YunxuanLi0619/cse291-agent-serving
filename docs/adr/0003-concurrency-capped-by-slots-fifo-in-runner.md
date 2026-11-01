# Concurrency c ≤ np, with queueing handled by the runner's own FIFO

The runner owns the FIFO queue, and the number of in-flight requests is ≤ c ≤ np at all times. At startup the runner verifies that the `/slots` count ≥ c and refuses to run otherwise.

If c > np, the extra requests queue inside llama-server (`requests_deferred`), the queueing time is no longer controlled by our scheduler, both the "FIFO that we control ourselves" and Phase 2's scheduling policy become ineffective, and queueing time cannot be attributed.
