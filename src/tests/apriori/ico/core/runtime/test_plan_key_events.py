"""Events carry the plan key of the node they come from (``node``), the
same key as the node's row in ``plan_view`` and in the profile."""

from collections.abc import Iterator
from typing import Any

from ico.core.async_stream import IcoAsyncStream
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.progress import IcoProgress, IcoProgressEvent
from ico.core.runtime.runtime import IcoRuntime
from ico.core.runtime.state import IcoStateEvent, IcoStateRequestCommand
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.describe.plan.json_renderer import plan_view
from ico.describe.plan.options import PlanRendererOptions
from ico.registry import IcoRegistry, register
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.spec import NodeSpec, build, run
from ico.spec.model import JsonValue
from ico.tools.profiler import ProfilerTool
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot

# Module level: spawned runner processes import this module.
REGISTRY = IcoRegistry()


@register("keys.double", 1, registry=REGISTRY)
def double(x: int) -> int:
    return x * 2


def numbers() -> Iterator[int]:
    yield from range(4)


def ignore(x: Any) -> None:
    pass


def worker_flow() -> IcoOperatorProtocol[int, int]:
    return IcoProgress[int](total=1, name="inner") | IcoOperator(double)


class Spy(IcoRuntimeTool):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[IcoRuntimeEvent] = []

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        self.events.append(event)


def row_keys(flow: Any, label: str, *, remote: bool = False) -> set[str]:
    """Plan keys of the rows whose flow cell mentions ``label``."""
    options = PlanRendererOptions(query_iterable_size=False, show_remote_flows=remote)
    view = plan_view(flow, options)
    return {
        key
        for key, row in zip(view["keys"], view["rows"], strict=True)
        if key is not None and label in "".join(s["text"] for s in row[0])
    }


def test_progress_events_carry_the_plan_key_of_their_row() -> None:
    progress: IcoProgress[int] = IcoProgress(total=4, name="items")
    flow = IcoSource(numbers) | progress.stream() | IcoSink(ignore)
    spy = Spy()
    IcoRuntime(flow, tools=[spy]).activate().run().deactivate()

    nodes = {e.node for e in spy.events if isinstance(e, IcoProgressEvent)}
    assert nodes == {progress.plan_key} == row_keys(flow, "IcoProgress")


def test_spec_run_keys_match_the_plan_and_the_profile() -> None:
    progress: JsonValue = {"kind": "progress", "name": "items", "params": {"total": 4}}
    body: JsonValue = {
        "kind": "seq",
        "children": [progress, {"kind": "op", "op": "keys.double@1"}],
    }
    spec: dict[str, JsonValue] = {"kind": "stream", "children": [body]}
    snapshots: list[ProgressSnapshot] = []
    profiler = ProfilerTool(interval=None)

    node_spec = NodeSpec.from_dict(spec)
    run(
        node_spec,
        [1, 2, 3, 4],
        registry=REGISTRY,
        tools=[JsonProgressTool(snapshots.append)],
        profiler=profiler,
    )

    flow = build(node_spec, registry=REGISTRY)
    expected = row_keys(flow, "IcoProgress")
    assert {s["node"] for s in snapshots} == expected
    profiled = {n["key"] for n in profiler.snapshot()["nodes"]}
    assert expected <= profiled


def make_worker() -> MPWorker[int, int]:
    return MPWorker(worker_flow)


def test_progress_inside_workers_has_the_keys_of_the_remote_rows() -> None:
    pool = IcoAsyncStream(make_worker, pool_size=2)
    flow = IcoSource(numbers) | pool | IcoSink(ignore)
    spy = Spy()
    IcoRuntime(flow, tools=[spy]).activate().run().deactivate()

    nodes = {e.node for e in spy.events if isinstance(e, IcoProgressEvent)}
    # Both runners of the factory pool report under the one key of the plan.
    assert nodes == row_keys(flow, "IcoProgress", remote=True)
    assert len(nodes) == 1


def test_answers_have_keys_inside_the_plan_and_none_outside() -> None:
    progress: IcoProgress[int] = IcoProgress(total=4, name="items")
    flow = IcoSource(numbers) | progress.stream() | IcoSink(ignore)
    runtime = IcoRuntime(flow, tools=[Spy()])

    answers = runtime.query(IcoStateRequestCommand.create())

    by_node = {a.node for a in answers if isinstance(a, IcoStateEvent)}
    assert progress.plan_key in by_node
    assert None in by_node  # the runtime, the toolbox and the tool
    assert runtime.plan_key is None and runtime.toolbox.plan_key is None


def test_plan_root_keys_a_wrapped_flow() -> None:
    progress: IcoProgress[int] = IcoProgress(total=4, name="items")
    flow = IcoSource(numbers) | progress.stream()
    closure = flow | IcoSink(ignore)
    runtime = IcoRuntime(closure, plan_root=flow)

    assert runtime.plan_root is flow
    assert {progress.plan_key} == row_keys(flow, "IcoProgress")
    # Keyed from the closure, the same node would have another key.
    assert row_keys(closure, "IcoProgress") != {progress.plan_key}
