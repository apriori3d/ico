"""A runner that does not start in time is stopped instead of hanging.

A runner that hangs while it imports modules or builds its flow (e.g. a
stuck CUDA initialization) used to block activation forever: only a dead
runner ended the wait.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.worker import IcoWorkerDied, IcoWorkerStartTimeout
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.registry import IcoRegistry, register
from ico.runtime.worker.mp.mp_worker import DEFAULT_START_TIMEOUT, MPWorker
from ico.spec import NodeSpec, SpecError, build, run, to_spec

# Module level: spawned runner processes import this module.
REGISTRY = IcoRegistry()


def in_runner() -> bool:
    return multiprocessing.parent_process() is not None


@register("start.hang", 1, registry=REGISTRY)
class Hang(IcoOperator[int, int]):
    """Hangs while the runner builds its flow (only in a runner process)."""

    def __init__(self) -> None:
        if in_runner():
            time.sleep(60)
        super().__init__(fn=self._same)

    def _same(self, n: int) -> int:
        return n


@register("start.record", 1, registry=REGISTRY)
class Record(IcoOperator[int, int]):
    """Writes the runner's pid to ``path`` when built in a runner."""

    def __init__(self, path: str = "") -> None:
        self.path = path
        if in_runner() and path:
            Path(path).write_text(str(os.getpid()))
        super().__init__(fn=self._same)

    def _same(self, n: int) -> int:
        return n


def hang_flow() -> IcoOperatorProtocol[int, int]:
    time.sleep(60)
    return IcoOperator(slow_double)


def slow_double(n: int) -> int:
    time.sleep(0.5)
    return n * 2


def double_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(slow_double)


def numbers() -> Iterator[int]:
    yield from range(4)


def ignore(x: Any) -> None:
    pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_hanging_runner_is_stopped_after_its_start_timeout() -> None:
    worker = MPWorker(hang_flow, name="stuck", start_timeout=1)
    runtime = IcoRuntime(IcoSource(numbers) | IcoStream(worker) | IcoSink(ignore))
    clock = time.monotonic()
    with pytest.raises(IcoWorkerStartTimeout) as error:
        runtime.activate()
    elapsed = time.monotonic() - clock

    assert 1 <= elapsed < 10
    assert isinstance(error.value, IcoWorkerDied)
    assert "worker 'stuck'" in str(error.value) and "1 s" in str(error.value)
    assert not worker.is_alive
    runtime.deactivate()  # does not hang on the stopped runner


def test_the_timeout_limits_the_start_only() -> None:
    worker = MPWorker(double_flow, start_timeout=1)
    out: list[int] = []
    flow = IcoSource(numbers) | IcoStream(worker) | IcoSink(out.append)
    runtime = IcoRuntime(flow)
    runtime.activate().run().deactivate()  # 4 items × 0.5 s > 1 s

    assert out == [0, 2, 4, 6]


def test_spec_run_stops_the_workers_that_started(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    started = MPWorker(lambda: Record(path=str(pid_file)), start_timeout=30)
    stuck = MPWorker(lambda: Hang(), start_timeout=1)
    spec = to_spec(IcoAsyncStream([started, stuck]))

    with pytest.raises(IcoWorkerStartTimeout):
        run(spec, [1, 2], registry=REGISTRY)

    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive(pid)


def test_start_timeout_in_specs() -> None:
    def worker_spec(**kwargs: Any) -> dict[str, Any]:
        worker = MPWorker(lambda: Record(), **kwargs)
        return to_spec(worker).root.to_dict()

    default, custom, unlimited = (
        worker_spec(),
        worker_spec(start_timeout=30),
        worker_spec(start_timeout=None),
    )
    assert "start_timeout" not in default["params"]  # specs stay as they were
    assert custom["params"]["start_timeout"] == 30
    assert unlimited["params"]["start_timeout"] is None

    def rebuilt(**params: Any) -> MPWorker[Any, Any]:
        """The worker built from ``custom`` with its ``start_timeout`` set
        to ``params`` (absent when empty)."""
        data = dict(custom)
        data["params"] = {"type": custom["params"]["type"], **params}
        return cast(
            MPWorker[Any, Any], build(NodeSpec.from_dict(data), registry=REGISTRY)
        )

    assert rebuilt(start_timeout=30).start_timeout == 30.0
    assert rebuilt(start_timeout=None).start_timeout is None
    assert rebuilt().start_timeout == DEFAULT_START_TIMEOUT
    with pytest.raises(SpecError, match="start_timeout"):
        rebuilt(start_timeout=0)
