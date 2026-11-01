# The FIFO queueing unit is the step (request), not the task

A concurrency slot (c) is held only for the duration of one HTTP request. After a step of an agent task completes, the slot is not held during the tool delay; the moment the delay ends, the next step enters the tail of the FIFO at its "ready time".

Why this choice: one of Phase 2's candidate mechanisms is "prioritizing ready agent steps", so the baseline queue must queue steps for there to be something comparable. The phenomenon E3/E4 must measure is: while a step is idle, the KV in its slot gets taken over by other requests via LCP/LRU, and the continuation step loses prefix reuse; under task-level FIFO (a task holds its slot until it finishes) this phenomenon cannot be measured.

## Consequences

- Task latency includes the time a continuation step spends re-queueing at the tail; the task-level timeout includes all queueing time.
- Which slot a continuation step lands in is decided by llama-server's slot selection (LCP similarity, otherwise LRU); the runner does not specify it.
