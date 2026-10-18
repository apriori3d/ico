"""A runner process that dies ends the wait for it instead of hanging.

Native crashes (e.g. SIGSEGV in a C extension) killed a worker process and
left its ``async_stream`` waiting for a result that never came.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from ico.core.async_stream import IcoAsyncStream, IcoStreamErrors
from ico.core.operator import IcoOperator
from ico.core.runtime.channel.channel import (
    IcoChannel,
    IcoReceiveEndpoint,
    IcoSendEndpoint,
)
from ico.core.runtime.channel.messages import DataMessage, RuntimeMessage
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.worker import IcoWorkerDied
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.runtime.worker.mp.mp_worker import MPWorker

SRC = Path(__file__).resolve().parents[5]

# ─── Worker flows (module level: spawned processes import them) ───


def crash_on_three(n: int) -> int:
    if n == 3:
        os.kill(os.getpid(), signal.SIGSEGV)
    time.sleep(0.05)
    return n * 10


def exit_on_three(n: int) -> int:
    if n == 3:
        os._exit(1)
    time.sleep(0.05)
    return n * 10


def always_exit(n: int) -> int:
    os._exit(2)


def crash_flow() -> IcoOperator[int, int]:
    return IcoOperator(crash_on_three)


def exit_flow() -> IcoOperator[int, int]:
    return IcoOperator(exit_on_three)


def doomed_flow() -> IcoOperator[int, int]:
    return IcoOperator(always_exit)


def slow_flow() -> IcoOperator[int, int]:
    return IcoOperator(lambda n: n)


def numbers() -> Iterator[int]:
    yield from range(8)


def run_pool(
    factory: Callable[[], IcoOperator[int, int]], on_error: Any
) -> tuple[list[int], float]:
    outputs: list[int] = []
    pool = IcoAsyncStream(lambda: MPWorker(factory), pool_size=2, on_error=on_error)
    runtime = IcoRuntime(IcoSource(numbers) | pool | IcoSink(outputs.append))
    runtime.activate()
    started = time.monotonic()
    try:
        runtime.run()
    finally:
        runtime.deactivate()
        elapsed = time.monotonic() - started
    return outputs, elapsed


# ─── Worker side ───


def test_a_crash_fails_the_run_at_once_with_raise() -> None:
    started = time.monotonic()
    with pytest.raises(IcoWorkerDied) as info:
        run_pool(crash_flow, "raise")

    assert time.monotonic() - started < 10
    assert (info.value.exitcode, info.value.signal) == (-signal.SIGSEGV, "SIGSEGV")
    assert "died: SIGSEGV" in str(info.value)


def test_other_workers_finish_the_items_with_collect() -> None:
    with pytest.raises(IcoStreamErrors) as info:
        run_pool(exit_flow, "collect")

    [error] = info.value.errors
    assert error.index == 3
    assert isinstance(error.error, IcoWorkerDied)
    assert error.error.exitcode == 1
    assert "exited with code 1" in str(error.error)


def test_items_fail_fast_when_every_worker_is_dead() -> None:
    started = time.monotonic()
    with pytest.raises(IcoStreamErrors) as info:
        run_pool(doomed_flow, "collect")

    assert time.monotonic() - started < 10
    assert sorted(e.index for e in info.value.errors) == list(range(8))
    assert all(isinstance(e.error, IcoWorkerDied) for e in info.value.errors)


# ─── Runner side ───

ORPHAN = """
import sys
from ico.core.runtime.runtime import IcoRuntime
from ico.core.source import IcoSource
from ico.core.sink import IcoSink
from ico.runtime.worker.mp.mp_worker import MPWorker
from tests.apriori.ico.runtime.mp.test_dead_worker import slow_flow

worker = MPWorker(slow_flow)
runtime = IcoRuntime(IcoSource(lambda: iter([1])) | worker.stream() | IcoSink(print))
runtime.activate()
print(worker._runner_process.pid, flush=True)
import os
os._exit(0)  # the parent dies without deactivating its worker
"""


def test_an_orphaned_runner_exits() -> None:
    done = subprocess.run(
        [sys.executable, "-c", ORPHAN],
        cwd=SRC,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    pid = int(done.stdout.split()[-1])

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pytest.fail("the orphaned runner kept running")


# ─── Channel ───


class _Silent(IcoReceiveEndpoint[int]):
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []

    def receive(
        self, timeout: float | None = None
    ) -> DataMessage[int] | RuntimeMessage:
        self.timeouts.append(timeout)
        raise Empty

    def close(self) -> None:
        pass


class _Nowhere(IcoSendEndpoint[int]):
    def send(self, message: DataMessage[int] | RuntimeMessage) -> None:
        pass

    def close(self) -> None:
        pass


class _Channel(IcoChannel[int, int]):
    def invert(self) -> IcoChannel[int, int]:
        return self


def test_waiting_polls_the_peer_check() -> None:
    receiver = _Silent()
    calls: list[int] = []

    def check() -> None:
        calls.append(1)
        if len(calls) == 3:
            raise IcoWorkerDied(-signal.SIGKILL)

    channel = _Channel(_Nowhere(), receiver, check_peer=check, poll_interval=0.01)
    with pytest.raises(IcoWorkerDied, match="SIGKILL"):
        channel.wait_for_item()

    assert len(calls) == 3
    assert receiver.timeouts == [0.01, 0.01, 0.01]


def test_without_a_peer_check_receive_blocks() -> None:
    receiver = _Silent()
    channel = _Channel(_Nowhere(), receiver, ignore_receive_timeouts=False)

    with pytest.raises(Empty):
        channel.wait_for_item()
    assert receiver.timeouts == [None]
