from __future__ import annotations

import subprocess
import sys
import time

import pytest

from ico.core.operator import IcoOperator
from ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
)
from ico.core.runtime.event import (
    IcoHeartbeatEvent,
)
from ico.core.runtime.exceptions import IcoRuntimeError
from ico.core.runtime.runtime import IcoRuntime
from ico.runtime.worker.mp.mp_worker import (
    MPWorker,
)
from tests.apriori.ico.core.runtime.runtime_node.test_utils import RecordingRuntimeNode

# ───────────────────────────────────────────────
# Helper flows
# ───────────────────────────────────────────────

FloatOperator = IcoOperator[float, float]


def op_identity() -> FloatOperator:
    """Simple identity operator."""
    return FloatOperator(lambda x: x)


def op_double() -> FloatOperator:
    """Double transformation."""
    return FloatOperator(lambda x: x * 2)


def op_fail_factory() -> FloatOperator:
    """Fail when receiving the value three."""
    return FloatOperator(op_fail_on_three)


def op_fail_on_three(x: float) -> float:
    """Fail when receiving the value three."""
    if x == 3:
        raise IcoRuntimeError("boom")
    return x


# ───────────────────────────────────────────────
# Test: Worker processes items via flow
# ───────────────────────────────────────────────


def test_worker_basic_roundtrip() -> None:
    """Worker spawned by MPProcessRunnerHost should process data using remote flow."""

    mp_process = MPWorker(op_double)
    mp_process.activate()

    result = mp_process(21)
    assert result == 42
    mp_process.deactivate()  # must stop remote worker


# ───────────────────────────────────────────────
# Test: Multiple sequential items
# ───────────────────────────────────────────────


def test_worker_multiple_items() -> None:
    """Worker should correctly process multiple items in sequence."""

    mp_process = MPWorker(op_double)
    mp_process.activate()

    try:
        results = [mp_process(i) for i in range(5)]
        assert results == [i * 2 for i in range(5)]
    finally:
        mp_process.deactivate()


# ───────────────────────────────────────────────
# Test: Exception propagation from worker
# ───────────────────────────────────────────────


def test_worker_exception_propagation() -> None:
    """If worker raises IcoRuntimeError, host should receive corresponding runtime event."""

    worker = MPWorker(op_fail_factory).activate()

    # normal items ok
    assert worker(1) == 1
    assert worker(2) == 2

    with pytest.raises(IcoRuntimeError) as exc:
        worker(3)
    assert "boom" in str(exc.value)

    worker.deactivate()


# ───────────────────────────────────────────────
# Test: Commands propagate to worker runtime
# ───────────────────────────────────────────────


def test_worker_command_propagation() -> None:
    """activate/pause/resume/deactivate should propagate to worker runtime."""

    # Attach recording runtime to channel
    end_runtime = RecordingRuntimeNode()

    worker = MPWorker(op_identity)
    worker.add_runtime_children(end_runtime)

    # channel is a parent runtime of the worker host runtime
    worker.activate().deactivate()

    # Worker receives commands in same order
    assert end_runtime.recorded_commands == [
        IcoActivateCommand,
        IcoDeactivateCommand,
    ]


# ───────────────────────────────────────────────
# Test: Events bubble back to host
# ───────────────────────────────────────────────


def test_worker_event_propagation() -> None:
    """Worker should bubble events back to host runtime."""

    main_runtime = RecordingRuntimeNode()
    mp_process = MPWorker(op_identity)
    main_runtime.add_runtime_children(mp_process)

    main_runtime.activate()
    # flow = host.channel.send | host.channel.receive

    # Worker echoes heartbeat event manually
    mp_process.bubble_event(IcoHeartbeatEvent.create())

    assert main_runtime.recorded_events == [IcoHeartbeatEvent]

    main_runtime.deactivate()


# ───────────────────────────────────────────────
# Test: mp_process api end-to-end
# ───────────────────────────────────────────────


def test_worker_mp_process_end_to_end() -> None:
    """Portal wiring must allow transparent remote computation."""

    collected: list[str] = []

    def collect_output(x: float) -> None:
        collected.append(f"Received {x}")

    src = IcoOperator[None, float](lambda _: 7)
    dst = IcoOperator[float, None](collect_output)

    flow = src | MPWorker(op_double) | dst

    runtime = IcoRuntime(flow)
    runtime.activate().run().deactivate()

    assert collected == ["Received 14"]


# ───────────────────────────────────────────────
# Test: Worker shuts down cleanly on deactivate
# ───────────────────────────────────────────────


def test_worker_clean_shutdown() -> None:
    """Worker spawned by host should terminate on deactivate."""

    mp_process = MPWorker(op_identity)
    mp_process.activate()

    assert mp_process.is_alive

    mp_process.deactivate()

    # give process time to shut down
    time.sleep(0.1)

    assert not mp_process.is_alive, "Worker process did not exit cleanly"


# ───────────────────────────────────────────────
# Test: A child of the runner does not delay deactivation
# ───────────────────────────────────────────────

GRANDCHILD_SECONDS = 3.0


def start_grandchild(x: float) -> float:
    """Start a process that inherits the runner's fds and outlives it.

    Like ``torch_shm_manager``, it keeps the write end of the runner's
    process sentinel open.
    """
    subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({GRANDCHILD_SECONDS})"],
        close_fds=False,
    )
    return x


def op_start_grandchild() -> FloatOperator:
    return FloatOperator(start_grandchild)


def test_a_child_of_the_runner_does_not_delay_deactivation() -> None:
    mp_process = MPWorker(op_start_grandchild)
    mp_process.activate()
    assert mp_process(1.0) == 1.0

    start = time.perf_counter()
    mp_process.deactivate()
    elapsed = time.perf_counter() - start

    assert not mp_process.is_alive
    assert elapsed < GRANDCHILD_SECONDS / 2


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
