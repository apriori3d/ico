"""Samples: previews of an operator's results on request, also during a run
and in worker processes; the probe a profiler set is kept."""

import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.plan_keys import plan_nodes, remote_flow_key
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.sample import SampleProbe, preview
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.profiler import ProfilerTool, instrument
from ico.tools.profiler.instrument import label
from ico.tools.profiler.probe import NodeProbe
from ico.tools.sample import JsonSampleTool, SampleSnapshot

# Module level: spawned runner processes import this module.
N = 40


def numbers() -> Iterator[int]:
    yield from range(N)


def double(x: int) -> int:
    return 2 * x


def slow_double(x: int) -> int:
    time.sleep(0.01)
    return 2 * x


def ignore(_: int) -> None:
    pass


def key_of(flow: Any, node: object) -> str:
    [key] = [p.key for p in plan_nodes(flow) if p.node is node]
    return key


class Samples:
    def __init__(self) -> None:
        self.items: list[SampleSnapshot] = []
        self.lock = threading.Lock()
        self.tool = JsonSampleTool(self.add)

    def add(self, sample: SampleSnapshot) -> None:
        with self.lock:
            self.items.append(sample)


def test_samples_before_a_run_are_the_first_results() -> None:
    op = IcoOperator(double)
    flow = IcoSource(numbers) | IcoStream(op) | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()

    runtime.sample(key_of(flow, op), count=2)
    runtime.run().deactivate()

    assert samples.items == [
        {"node": key_of(flow, op), "index": 0, "type": "int", "preview": "0"},
        {"node": key_of(flow, op), "index": 1, "type": "int", "preview": "2"},
    ]
    assert op.probe is None  # gone after the samples


def test_samples_during_a_run() -> None:
    op = IcoOperator(slow_double)
    flow = IcoSource(numbers) | IcoStream(op) | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()
    threading.Timer(0.1, runtime.sample, args=(key_of(flow, op), 3)).start()

    runtime.run().deactivate()

    values = [int(s["preview"]) for s in samples.items]
    assert len(values) == 3 and values[0] > 0  # taken mid-run
    assert values == list(range(values[0], values[0] + 6, 2))  # consecutive


def remote_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(slow_double)


def test_samples_from_a_worker_process_during_a_run() -> None:
    worker = MPWorker(remote_flow)
    flow = IcoSource(numbers) | IcoStream(worker) | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()
    # The remote flow is a single operator: its key is the remote flow's.
    key = remote_flow_key(key_of(flow, worker), worker)
    try:
        threading.Timer(0.2, runtime.sample, args=(key, 3)).start()
        runtime.run()
    finally:
        runtime.deactivate()

    assert [s["node"] for s in samples.items] == [key] * 3
    assert [s["index"] for s in samples.items] == [0, 1, 2]


def test_a_profiler_keeps_measuring() -> None:
    op = IcoOperator(double)
    profiler = ProfilerTool(interval=None)
    flow = instrument(IcoSource(numbers) | IcoStream(op) | IcoSink(ignore), profiler)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[profiler, samples.tool]).activate()

    runtime.sample(key_of(flow, op), count=2)
    runtime.run().deactivate()

    assert len(samples.items) == 2
    assert isinstance(op.probe, NodeProbe)  # given back
    [node] = [n for n in profiler.snapshot()["nodes"] if n["label"] == "double"]
    startup = node["startup"]
    assert startup is not None
    calls = startup["wall"]["count"] + node["metrics"]["wall"]["count"]
    assert calls == N


def test_a_new_request_replaces_the_one_in_progress() -> None:
    op = IcoOperator(double)
    flow = IcoSource(numbers) | IcoStream(op) | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()

    runtime.sample(key_of(flow, op), count=5)
    runtime.sample(key_of(flow, op), count=1)
    probe = op.probe
    runtime.run().deactivate()

    assert isinstance(probe, SampleProbe) and probe.inner is None  # not stacked
    assert len(samples.items) == 1 and op.probe is None


def test_a_stream_previews_as_an_iterator() -> None:
    stream = IcoStream(IcoOperator(double))
    flow = IcoSource(numbers) | stream | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()

    runtime.sample(key_of(flow, stream))
    runtime.run().deactivate()

    [sample] = samples.items
    assert sample["preview"].startswith("<iterator")


def test_invalid_requests() -> None:
    op = IcoOperator(double)
    flow = IcoSource(numbers) | IcoStream(op) | IcoSink(ignore)
    runtime = IcoRuntime(flow)

    with pytest.raises(ValueError, match="at least 1"):
        runtime.sample(key_of(flow, op), count=0)
    with pytest.raises(ValueError, match="no node 0.9"):
        runtime.sample("0.9")
    assert op.probe is None


def test_previews_are_short() -> None:
    assert preview(np.zeros((2, 3))) == "ndarray(shape=(2, 3), dtype=float64)"
    assert preview(np.float64(1.5)) == repr(np.float64(1.5))  # a scalar
    assert preview(iter([1])) == "<iterator list_iterator>"
    assert preview(list(range(1000))) == "[0, 1, 2, 3, 4, 5, 6, 7, ...]"
    assert len(preview("x" * 10_000)) <= 200
    assert len(preview({"k": "v" * 500}, max_chars=50)) == 50

    class Broken:
        def __repr__(self) -> str:
            raise RuntimeError

    assert preview(Broken()).startswith("<Broken instance")


# A long item in a worker: a video whose frames stream inside the worker.


def videos() -> Iterator[int]:
    yield from range(3)


def frames(video: int) -> Iterator[int]:
    return iter(range(video * 100, video * 100 + 40))


def slow_frame(frame: int) -> int:
    time.sleep(0.05)  # 2 s per video
    return frame


def total(values: Iterator[int]) -> int:
    return sum(values)


def video_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(frames) | IcoStream(IcoOperator(slow_frame)) | IcoOperator(total)


def test_samples_reach_a_worker_during_a_long_item() -> None:
    worker = MPWorker(video_flow)
    flow = IcoSource(videos) | IcoStream(worker) | IcoSink(ignore)
    samples = Samples()
    runtime = IcoRuntime(flow, tools=[samples.tool]).activate()
    [key] = [
        p.key
        for p in plan_nodes(flow, expand_remote_flows=True)
        if label(p.node) == "slow_frame"
    ]
    try:
        threading.Timer(0.5, runtime.sample, args=(key, 2)).start()
        runtime.run()
    finally:
        runtime.deactivate()

    frames_seen = [int(s["preview"]) for s in samples.items]
    assert len(frames_seen) == 2
    assert all(f < 100 for f in frames_seen)  # from the first video, mid-item
