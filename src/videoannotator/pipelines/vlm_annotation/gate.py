"""Tries go ahead of runs on the same model (spec 027 R5, FR-017a).

One gate per (base URL, model), in this process. Every model call holds it:
a run's next sample waits while any try is waiting, and a try waits only
for the call in progress, so a try waits at most one run sample. A call in
progress is never interrupted.

A job run by a separate worker process doesn't share the gate (documented
limitation): a try then queues behind it at the model server.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

Priority = Literal["try", "run"]


class ModelGate:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._held = False
        self._tries_waiting = 0
        # Runs are served in arrival order, so two runs on one model take turns.
        self._next_run_ticket = 0
        self._run_serving = 0

    @contextmanager
    def acquire(self, priority: Priority) -> Iterator[None]:
        with self._cond:
            if priority == "try":
                self._tries_waiting += 1
                try:
                    while self._held:
                        self._cond.wait()
                finally:
                    self._tries_waiting -= 1
            else:
                ticket = self._next_run_ticket
                self._next_run_ticket += 1
                while self._held or self._tries_waiting or ticket != self._run_serving:
                    self._cond.wait()
                self._run_serving += 1
            self._held = True
        try:
            yield
        finally:
            with self._cond:
                self._held = False
                self._cond.notify_all()


_gates: dict[tuple[str, str], ModelGate] = {}
_gates_lock = threading.Lock()


def gate_for(base_url: str, model: str) -> ModelGate:
    key = (base_url.rstrip("/").lower(), model)
    with _gates_lock:
        gate = _gates.get(key)
        if gate is None:
            gate = _gates[key] = ModelGate()
        return gate
