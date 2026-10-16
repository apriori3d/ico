import json
from collections.abc import Iterator
from typing import cast

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.fork import fork
from ico.core.operator import IcoOperator
from ico.core.runtime.progress import IcoProgress
from ico.core.stream import IcoStream
from ico.core.tree_utils import TreePathIndex
from ico.registry import IcoRegistry, register
from ico.runtime.worker.mp.mp_worker import MPWorker
from ico.spec import SpecError, run, to_spec
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot

# Module-level registrations: spawned MPWorker runner processes import this module.
REGISTRY = IcoRegistry()


@register("run.inc", 1, registry=REGISTRY)
def inc(x: int) -> int:
    return x + 1


@register("run.double", 1, registry=REGISTRY)
def double(x: int) -> int:
    return x * 2


@register("run.to_str", 1, registry=REGISTRY)
def to_str(x: int) -> str:
    return str(x)


@register("run.fail_odd", 1, registry=REGISTRY)
def fail_odd(x: int) -> int:
    if x % 2:
        raise ValueError(f"odd {x}")
    return x


@register("run.opaque", 1, registry=REGISTRY)
def opaque(x: int) -> object:
    return object()


@register("run.numbers", 1, registry=REGISTRY)
class Numbers(IcoOperator[None, Iterator[int]]):
    """Source of 0..n-1; its params live in the spec."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        super().__init__(fn=self._numbers)

    def _numbers(self, _: None) -> Iterator[int]:
        return iter(range(self.n))


def make_double_worker() -> IcoOperator[int, int]:
    return cast(IcoOperator[int, int], double)


def op(name: str) -> IcoOperator[int, int]:
    return cast(IcoOperator[int, int], REGISTRY.get(f"run.{name}@1").node)


def test_scalar_input_and_output() -> None:
    result = run(to_spec(inc | double), 3, registry=REGISTRY)
    assert (result.outputs, result.total, result.truncated) == ((8,), 1, False)


def test_source_without_input_and_iterator_output() -> None:
    flow = Numbers(n=5) | IcoStream(double)
    result = run(to_spec(flow), registry=REGISTRY)
    assert result.outputs == (0, 2, 4, 6, 8)
    assert result.total == 5 and not result.truncated


def test_tools_are_attached_to_the_run() -> None:
    progress: IcoProgress[int] = IcoProgress(total=3, name="items")
    flow = Numbers(n=3) | (progress | op("inc")).stream()
    snapshots: list[ProgressSnapshot] = []

    result = run(
        to_spec(flow), registry=REGISTRY, tools=[JsonProgressTool(snapshots.append)]
    )

    assert result.outputs == (1, 2, 3)
    assert snapshots[-1]["label"] == "items"
    assert (snapshots[-1]["done"], snapshots[-1]["total"]) == (3, 3)


def test_max_items_caps_outputs_but_counts_all() -> None:
    flow = Numbers(n=10) | IcoStream(inc)
    result = run(to_spec(flow), registry=REGISTRY, max_items=3)
    assert result.outputs == (1, 2, 3)
    assert result.total == 10
    assert result.truncated


def test_fork_tuple_becomes_list() -> None:
    result = run(to_spec(fork(inc, to_str)), 1, registry=REGISTRY)
    assert result.outputs == ([2, "1"],)


def test_non_json_output_is_an_error_with_hint() -> None:
    with pytest.raises(SpecError, match="writes results and returns paths"):
        run(to_spec(op("opaque")), 1, registry=REGISTRY)


def test_collected_item_errors_are_located() -> None:
    stream = IcoAsyncStream([op("fail_odd")], ordered=True, on_error="collect")
    flow = Numbers(n=4) | stream
    spec = to_spec(flow, strict=True)

    result = run(spec, registry=REGISTRY)

    assert result.outputs == (0, 2)
    assert [(str(e.path), e.index, e.item, e.error) for e in result.errors] == [
        ("1", 1, 1, "ValueError: odd 1"),
        ("1", 3, 3, "ValueError: odd 3"),
    ]
    path = result.errors[0].path
    assert path == TreePathIndex.create(1)
    assert path is not None
    assert spec.root.at(path).kind == "async_stream"


def test_build_errors_propagate() -> None:
    from ico.spec import NodeSpec, PipelineSpec

    with pytest.raises(SpecError, match="at root: Cannot resolve operator"):
        run(
            PipelineSpec(root=NodeSpec(kind="op", op="run.missing@1")),
            registry=REGISTRY,
        )


def test_mpworker_pool_runs_from_spec() -> None:
    pool = IcoAsyncStream(lambda: MPWorker(make_double_worker), pool_size=2)
    flow = Numbers(n=4) | pool
    result = run(to_spec(flow, strict=True), registry=REGISTRY)
    assert sorted(cast(list[int], list(result.outputs))) == [0, 2, 4, 6]


def test_to_dict_is_json() -> None:
    stream = IcoAsyncStream([op("fail_odd")], on_error="collect")
    result = run(to_spec(Numbers(n=3) | stream), registry=REGISTRY)
    data = json.loads(json.dumps(result.to_dict()))
    assert data["total"] == 2
    assert data["errors"][0]["path"] == "1"
