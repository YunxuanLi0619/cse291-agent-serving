"""Scheduler: owns the queue of ready Steps and the concurrency limit c; the Policy only decides "which to dispatch next"."""
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass(eq=False)
class ReadyStep:
    task_id: str
    step: int
    ready_at: float
    seq: int  # enqueue sequence number, used to keep first-in-first-out when ready times tie
    started_at: float | None = None
    held: bool = False  # currently holding a concurrency slot
    granted: asyncio.Event = field(default_factory=asyncio.Event)

    def queue_time(self, now: float) -> float:
        """Time spent queued; if not yet dispatched (e.g. timed out while queued), counted up to now."""
        return (self.started_at if self.started_at is not None else now) - self.ready_at


class Policy(Protocol):
    def pick(self, ready: Sequence[ReadyStep]) -> ReadyStep: ...


class FifoPolicy:
    def pick(self, ready: Sequence[ReadyStep]) -> ReadyStep:
        return min(ready, key=lambda s: (s.ready_at, s.seq))


POLICIES: dict[str, Callable[[], Policy]] = {"fifo": FifoPolicy}


class Scheduler:
    def __init__(self, policy: Policy, c: int, now: Callable[[], float]):
        self._policy, self._c, self._now = policy, c, now
        self._ready: list[ReadyStep] = []
        self._in_flight = 0
        self._seq = 0

    def submit(self, task_id: str, step: int) -> ReadyStep:
        entry = ReadyStep(task_id, step, self._now(), self._seq)
        self._seq += 1
        self._ready.append(entry)
        self._dispatch()
        return entry

    def finish(self, entry: ReadyStep) -> None:
        """A Step finishing, being cancelled, or timing out while queued all go through here; idempotently releases the queue position or concurrency slot it holds."""
        if entry in self._ready:
            self._ready.remove(entry)
        elif entry.held:
            entry.held = False
            self._in_flight -= 1
            self._dispatch()

    def _dispatch(self) -> None:
        while self._ready and self._in_flight < self._c:
            entry = self._policy.pick(self._ready)
            self._ready.remove(entry)
            entry.started_at = self._now()
            entry.held = True
            self._in_flight += 1
            entry.granted.set()
