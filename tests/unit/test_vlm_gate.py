"""The per-model gate: a try is served before a run's next call (spec 027 R5)."""

import threading
import time
from itertools import pairwise

import pytest

from videoannotator.pipelines.vlm_annotation.gate import ModelGate, gate_for


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.005)


def test_a_try_queued_during_a_run_call_goes_before_the_runs_next_call():
    gate = ModelGate()
    order: list[str] = []
    run_in_call = threading.Event()
    release_run_call = threading.Event()

    def run():
        with gate.acquire("run"):
            order.append("run-1")
            run_in_call.set()
            release_run_call.wait(5)
        with gate.acquire("run"):
            order.append("run-2")

    def try_():
        with gate.acquire("try"):
            order.append("try")

    runner = threading.Thread(target=run)
    runner.start()
    assert run_in_call.wait(5)
    trier = threading.Thread(target=try_)
    trier.start()
    _wait_for(lambda: gate._tries_waiting == 1)
    release_run_call.set()
    runner.join(5)
    trier.join(5)
    assert order == ["run-1", "try", "run-2"]


def test_two_runs_take_turns():
    gate = ModelGate()
    order: list[str] = []
    barrier = threading.Barrier(2)

    def run(name):
        barrier.wait()
        for i in range(3):
            with gate.acquire("run"):
                order.append(f"{name}{i}")
                time.sleep(0.02)

    threads = [threading.Thread(target=run, args=(n,)) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(order) == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert all(x[0] != y[0] for x, y in pairwise(order))


def test_no_deadlock_when_a_call_raises():
    gate = ModelGate()
    with pytest.raises(RuntimeError):
        with gate.acquire("run"):
            raise RuntimeError("model fell over")
    done = threading.Event()

    def try_():
        with gate.acquire("try"):
            done.set()

    threading.Thread(target=try_).start()
    assert done.wait(2)
    with gate.acquire("run"):
        pass


def test_one_gate_per_model_and_server():
    assert gate_for("http://localhost:11434/", "m") is gate_for(
        "http://LOCALHOST:11434", "m"
    )
    assert gate_for("http://localhost:11434", "m") is not gate_for(
        "http://localhost:11434", "other"
    )
