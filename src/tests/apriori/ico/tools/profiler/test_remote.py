import pickle
import time
from dataclasses import dataclass
from typing import Any

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.describe.plan.json_renderer import plan_view
from ico.describe.plan.options import PlanRendererOptions
from ico.resources import Resource, ResourcePool, resource, set_pool
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.profiler import (
    IcoProfileEvent,
    NodeProfile,
    ProfilerTool,
    ProfileSnapshot,
    instrument,
)
from ico.tools.profiler.remote import ProfileHost, RunnerProfiler
from ico.tools.profiler.table import Layout, NodeAggregate, NodeInfo, ProfileTable

# Module-level: spawned runner processes import this module.


def work(x: int) -> int:
    time.sleep(0.005)
    return x


def double(x: int) -> int:
    return x * 2


def make_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(work) | IcoOperator(double)


def make_worker() -> MPWorker[int, int]:
    return MPWorker(make_flow, name="worker")


@resource("test.profiled_table", 1)
@dataclass(frozen=True)
class Table(Resource[list[int]]):
    """A small table loaded once per process."""

    count: int = 3

    def load(self) -> list[int]:
        return list(range(self.count))


TABLE = Table()


def look_up(x: int) -> int:
    with TABLE.use() as table:
        return table[x % len(table)]


def make_table_flow() -> IcoOperatorProtocol[int, int]:
    return IcoOperator(look_up)


def ignore(x: Any) -> None:
    pass


def numbers(n: int) -> IcoSource[int]:
    def provide() -> list[int]:
        return list(range(n))

    return IcoSource(provide)


def profile(flow: Any, profiler: ProfilerTool) -> ProfileSnapshot:
    flow = instrument(flow | IcoSink(ignore), profiler)
    IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
    return profiler.snapshot()


def by_label(snapshot: ProfileSnapshot) -> dict[str, NodeProfile]:
    return {node["label"]: node for node in snapshot["nodes"]}


# ─── Runner processes ───


def test_a_worker_reports_its_resource_pool() -> None:
    worker = MPWorker(make_table_flow, name="tables")
    snapshot = profile(numbers(4) | IcoStream(worker), ProfilerTool(interval=None))

    [row] = snapshot["resources"]
    assert row["process"] == "tables" and row["ref"] == "test.profiled_table@1"
    assert row["label"] == "test.profiled_table@1"  # all fields at their defaults
    assert (row["uses"], row["loads"], row["reloads"]) == (4, 1, 0)
    assert row["devices"] == ["cpu"]


def test_a_worker_flow_is_profiled_in_its_runner() -> None:
    worker = make_worker()
    snapshot = profile(numbers(4) | IcoStream(worker), ProfilerTool(interval=None))
    nodes = by_label(snapshot)

    portal = nodes["worker"]
    assert portal["calls"] == 4
    for label in ("work", "double"):
        assert nodes[label]["calls"] == 4
        assert nodes[label]["key"].startswith(portal["key"] + ".0.")
    mean = nodes["work"]["metrics"]["wall"]["mean"]
    assert mean is not None and mean >= 0.005

    # Plan order: the runner's nodes follow the worker.
    keys = [n["key"] for n in snapshot["nodes"]]
    assert keys.index(portal["key"]) < keys.index(nodes["work"]["key"])

    # The portal's wall time not spent in the runner's flow.
    transfer = portal["transfer"]
    wall = portal["metrics"]["wall"]
    assert transfer is not None and 0 <= transfer < wall["total"]
    # The portal waits while the runner works: its self is the transfer,
    # also for its first call (the runner's first call is its own startup).
    assert wall["self"] == transfer
    startup = portal["startup"]
    assert startup is not None
    assert startup["wall"]["self"] < startup["wall"]["total"]

    [main, runner] = snapshot["processes"]
    assert main["source"] == "main" and runner["source"] != "main"
    assert runner["peak_rss"] is None or runner["peak_rss"] > 0


def test_a_factory_pool_of_workers_merges_its_runners() -> None:
    pool = IcoAsyncStream(make_worker, pool_size=2, name="pool")
    snapshot = profile(numbers(6) | pool, ProfilerTool(interval=None))
    nodes = by_label(snapshot)

    assert nodes["work"]["calls"] == 6  # from two runner processes
    assert len(snapshot["processes"]) == 3
    utilization = nodes["pool"]["utilization"]
    assert utilization is not None and 0 < utilization <= 1.05


# ─── Pools in the main process ───


def test_utilization_of_a_thread_pool() -> None:
    def slow(x: int) -> int:
        time.sleep(0.01)
        return x

    pool = IcoAsyncStream(lambda: IcoOperator(slow), pool_size=2, name="pool")
    nodes = by_label(profile(numbers(8) | pool, ProfilerTool(interval=None)))
    utilization = nodes["pool"]["utilization"]
    assert utilization is not None and 0.5 < utilization <= 1.05
    assert nodes["slow"]["utilization"] is None
    assert nodes["pool"]["transfer"] is None


def test_a_starved_pool_has_a_low_utilization() -> None:
    def slow_numbers() -> Any:
        for i in range(4):
            time.sleep(0.02)
            yield i

    pool = IcoAsyncStream(lambda: IcoOperator(double), pool_size=4, name="pool")
    nodes = by_label(
        profile(IcoSource(slow_numbers) | pool, ProfilerTool(interval=None))
    )
    utilization = nodes["pool"]["utilization"]
    assert utilization is not None and utilization < 0.2


# ─── The flow does not change ───


def test_worker_signature_and_plan_with_remote_flows_are_unchanged() -> None:
    worker: MPWorker[Any, Any] = MPWorker(make_flow)  # signature from the factory
    flow = numbers(2) | IcoStream(worker)
    options = PlanRendererOptions(query_iterable_size=False, show_remote_flows=True)
    before = (plan_view(flow, options), worker.signature)
    instrument(flow, ProfilerTool(interval=None))
    assert (plan_view(flow, options), worker.signature) == before
    assert worker.flow_factory is make_flow
    assert isinstance(worker.runner_probe, RunnerProfiler)


def test_instrumenting_twice_sets_the_runner_probe_once() -> None:
    worker = make_worker()
    profiler = ProfilerTool(interval=None)
    instrument(worker, profiler)
    probe = worker.runner_probe
    instrument(worker, profiler)
    assert worker.runner_probe is probe


def test_the_runner_probe_and_events_pickle() -> None:
    worker = make_worker()
    instrument(worker, ProfilerTool(interval=None))
    probe = pickle.loads(pickle.dumps(worker.runner_probe))
    host = probe.wrap(make_flow())
    assert isinstance(host, ProfileHost)
    assert host(3) == 6

    node = NodeAggregate(2)
    node.record(0, [5, 7], [5, 7], 0)
    layout = Layout(nodes={"0": NodeInfo("work", "IcoOperator")})
    event = pickle.loads(
        pickle.dumps(IcoProfileEvent.create(layout, {"0": node}, peak_rss=1))
    )
    assert event.nodes["0"].calls == 1
    assert event.nodes["0"].events[1].sum == 7
    assert event.layout.nodes["0"].label == "work"


class Collector(IcoRuntimeNode):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[IcoRuntimeEvent] = []

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        self.events.append(event)
        return None


def test_the_host_reports_after_items_at_most_every_interval() -> None:
    now = [0.0]
    table = ProfileTable(ProfilerTool().table.metrics)
    host = ProfileHost(IcoOperator(double), table, 1.0, clock=lambda: now[0])
    parent = Collector()
    parent.add_runtime_children(host)

    host(1)  # no time passed
    now[0] = 1.5
    host(2)
    host(3)  # within the interval of the previous report
    assert len(parent.events) == 1
    host.on_deactivate()
    assert len(parent.events) == 2  # the final one
    assert all(isinstance(e, IcoProfileEvent) for e in parent.events)


def test_the_host_reports_a_resource_load_during_the_item() -> None:
    old = set_pool(ResourcePool())
    try:
        table = ProfileTable(ProfilerTool().table.metrics)
        parent = Collector()
        reported: list[int] = []

        def long_item(x: int) -> int:
            with Table(count=5).use() as rows:
                # Loaded: reported before the item (a long model call) ends.
                reported.append(len(parent.events))
            return x + len(rows)

        host = ProfileHost(IcoOperator(long_item), table, None, name="gpu")
        parent.add_runtime_children(host)
        host.on_activate()
        host(1)
        host(2)  # a hit: nothing to report
        [event] = parent.events
        assert isinstance(event, IcoProfileEvent) and reported == [1, 1]
        [row] = event.resources
        assert (row["label"], row["loads"]) == ("test.profiled_table@1 (count=5)", 1)
        host.on_deactivate()
    finally:
        set_pool(old)
