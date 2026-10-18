from collections.abc import Iterable
from typing import Any

import pytest

from ico import IcoOperator, plan_view
from ico.core.async_stream import IcoAsyncStream
from ico.core.context_operator import IcoContextOperator
from ico.core.context_pipeline import IcoContextPipeline
from ico.core.epoch import IcoEpoch
from ico.core.operator import IcoOperatorProtocol
from ico.core.plan_keys import plan_nodes, remote_flow_key
from ico.core.route import route
from ico.core.runtime.runtime import IcoRuntime
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.describe.plan.json_renderer import PlanCell, PlanView
from ico.describe.plan.rich_renderer.plan_renderer import PlanRenderer
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.tools.profiler import ProfilerTool, instrument

# Module-level: spawned runner processes import this module.


def first(text: str) -> str:
    return text[:1]


def upper(text: str) -> str:
    return text.upper()


def strip(text: str) -> str:
    return text.strip()


def scale(x: float) -> float:
    return x * 2


def shift(x: float) -> float:
    return x + 1


def worker_flow() -> IcoOperatorProtocol[float, float]:
    return IcoOperator(scale) | IcoOperator(shift)


def make_worker() -> MPWorker[float, float]:
    return MPWorker(worker_flow)


def floats() -> Iterable[float]:
    return [1.0, 2.0, 3.0]


def text(cell: PlanCell) -> str:
    return "".join(segment["text"] for segment in cell).strip()


def keyed_rows(view: PlanView) -> list[tuple[str | None, str]]:
    return [
        (key, text(row[0])) for row, key in zip(view["rows"], view["keys"], strict=True)
    ]


# ─── Keys of plan rows ───


def test_route_rows_have_the_keys_of_their_nodes() -> None:
    routed = route(
        IcoOperator(first),
        {"a": IcoOperator(upper), "b": IcoOperator(strip) | IcoOperator(upper)},
        default=IcoOperator(strip),
    )
    assert keyed_rows(plan_view(routed)) == [
        ("0", "╭── route by 🚦IcoRoute(keys='a', 'b', default)"),
        (None, "├─▸ key"),  # a single operator: its own row has the key
        ("0.0", "│   first"),
        (None, "├─▸ case 'a'"),
        ("0.1", "│   upper"),
        ("0.2", "├─▸ case 'b'"),  # a chain has no row: its label has the key
        ("0.2.0", "│   strip"),
        ("0.2.1", "│   upper"),
        (None, "├─▸ default"),
        ("0.3", "│   strip"),
        (None, "╰─▸ merge"),
    ]


def test_epoch_parts_and_remote_flows_are_keyed() -> None:
    class Ctx:
        pass

    def train(x: float, c: Ctx) -> Ctx:
        return c

    def log(c: Ctx) -> Ctx:
        return c

    epoch = IcoEpoch(
        source=IcoSource(floats) | IcoAsyncStream(make_worker, pool_size=2),
        context_operator=IcoContextPipeline(IcoContextOperator(train), log),
    )
    keys = [key for key, _ in keyed_rows(plan_view(epoch))]
    assert keys == [
        "0",  # the epoch
        "0.0",  # for each in: the source chain
        "0.0.0",
        "0.0.1",  # parallel in
        "0.0.1.0",  # send to MPWorker
        "0.0.1.0.0.0",  # scale, in the worker's remote flow
        "0.0.1.0.0.1",  # shift
        None,  # receive
        None,  # yield
        "0.1",  # apply: the context pipeline
        "0.1.0",
        "0.1.1",
        None,  # emit
    ]


def test_keys_are_unique_and_match_plan_nodes() -> None:
    flow = IcoSource(floats) | IcoStream(IcoOperator(scale) | IcoOperator(shift))
    keys = [k for k in plan_view(flow)["keys"] if k is not None]
    assert len(keys) == len(set(keys))
    nodes = {p.key: p.node for p in plan_nodes(flow)}
    assert set(keys) <= set(nodes)


# ─── plan_nodes ───


def test_factory_pool_members_share_the_first_key() -> None:
    pool = IcoAsyncStream(lambda: IcoOperator(scale), pool_size=3)
    planned = [(p.key, p.node) for p in plan_nodes(pool)]
    assert [key for key, _ in planned] == ["0", "0.0", "0.0", "0.0"]
    assert len({id(node) for _, node in planned[1:]}) == 3


def test_root_key_and_remote_flow_key() -> None:
    worker = make_worker()
    assert remote_flow_key("0.4", worker) == "0.4.0"
    remote = [p.key for p in plan_nodes(worker, expand_remote_flows=True)]
    assert remote == ["0", "0.0", "0.0.0", "0.0.1"]
    assert [p.key for p in plan_nodes(worker_flow(), root_key="0.0")] == [
        "0.0",
        "0.0.0",
        "0.0.1",
    ]


def test_render_subtree_only_takes_children() -> None:
    renderer = PlanRenderer()
    op = IcoOperator(scale)
    renderer._rendering.append((op, "0"))  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="not a child"):
        renderer.render_subtree(IcoOperator(shift))


# ─── The profiler uses the same keys ───


def ignore(x: Any) -> None:
    pass


def test_profile_keys_are_plan_keys_also_in_runners() -> None:
    flow = (
        IcoSource(floats)
        | IcoAsyncStream(make_worker, pool_size=2)
        | IcoStream(IcoOperator(scale) | IcoOperator(shift))
        | IcoSink(ignore)
    )
    rows = keyed_rows(plan_view(flow))
    plan_keys = {k for k, _ in rows if k is not None}
    [worker] = [k for k, row in rows if "MPWorker" in row]
    profiler = ProfilerTool(interval=None)
    instrument(flow, profiler)
    IcoRuntime(flow, tools=[profiler]).activate().run().deactivate()
    profiled = {n["key"]: n for n in profiler.snapshot()["nodes"]}

    assert plan_keys <= set(profiled)
    # Rows of the worker's remote flow are measured in the runners.
    assert worker is not None
    remote = sorted(
        k
        for k in plan_keys
        if k.startswith(remote_flow_key(worker, make_worker()) + ".")
    )
    assert remote and all(profiled[k]["calls"] == 3 for k in remote)
