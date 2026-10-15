from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from ico.core.async_stream import IcoAsyncStream
from ico.core.batcher import IcoBatcher
from ico.core.chain import IcoChain
from ico.core.context_operator import IcoContextOperator
from ico.core.epoch import IcoEpoch
from ico.core.filter import keep
from ico.core.fork import fork
from ico.core.node import IcoNode
from ico.core.operator import IcoOperator, IcoOperatorProtocol, operator
from ico.core.partition import partition
from ico.core.process import IcoProcess
from ico.core.route import route
from ico.core.runtime.progress import IcoProgress
from ico.core.sink import IcoSink
from ico.core.source import IcoSource
from ico.core.stream import IcoStream
from ico.spec import (
    NodeSpec,
    PipelineSpec,
    SpecContext,
    SpecError,
    from_json,
    spec_serializer,
    to_json,
    to_spec,
)

# ────────────────────────────────────────────────
# Module-level callables (referenceable from a spec)
# ────────────────────────────────────────────────


def inc(x: int) -> int:
    return x + 1


def double(x: int) -> int:
    return x * 2


def to_str(x: int) -> str:
    return str(x)


@operator()
def square(x: int) -> int:
    return x * x


def numbers() -> list[int]:
    return [1, 2, 3]


def consume(x: int) -> None:
    pass


def accumulate(x: int, total: int) -> int:
    return total + x


def make_worker() -> IcoOperator[int, int]:
    return IcoOperator(double)


MOD = __name__


# ────────────────────────────────────────────────
# Leaf operators
# ────────────────────────────────────────────────


def test_leaf_operator_ref_and_signature() -> None:
    root = to_spec(IcoOperator(inc, name="inc")).root

    assert root.kind == "op"
    assert root.name == "inc"
    assert root.ref == f"{MOD}:inc"
    assert root.sig == "int → int"


def test_decorated_operator_ref_points_to_module_attribute() -> None:
    root = to_spec(square).root

    assert root.kind == "op"
    assert root.ref == f"{MOD}:square"


def test_lambda_is_opaque_in_lenient_mode() -> None:
    root = to_spec(IcoOperator[int, int](lambda x: x)).root

    assert root.kind == "op"
    assert root.ref is None


def test_lambda_raises_in_strict_mode() -> None:
    with pytest.raises(SpecError, match="Anonymous callable"):
        to_spec(IcoOperator[int, int](lambda x: x), strict=True)


def test_local_function_raises_in_strict_mode() -> None:
    def local(x: int) -> int:
        return x

    with pytest.raises(SpecError):
        to_spec(IcoOperator(local), strict=True)


# ────────────────────────────────────────────────
# Chains
# ────────────────────────────────────────────────


def test_chain_is_flattened_into_seq() -> None:
    flow = (
        IcoOperator(inc) | IcoOperator(double) | IcoOperator(inc) | IcoOperator(to_str)
    )
    root = to_spec(flow, strict=True).root

    assert root.kind == "seq"
    assert [c.ref for c in root.children] == [
        f"{MOD}:inc",
        f"{MOD}:double",
        f"{MOD}:inc",
        f"{MOD}:to_str",
    ]
    assert root.sig == "int → str"
    assert root.name is None


def test_fork_branches_in_order() -> None:
    flow = IcoOperator(inc) | fork(double, to_str, name="split")
    root = to_spec(flow, strict=True).root

    fork_spec = root.children[1]
    assert fork_spec.kind == "fork"
    assert fork_spec.name == "split"
    assert fork_spec.sig == "int → tuple[int, str]"
    assert [c.ref for c in fork_spec.children] == [f"{MOD}:double", f"{MOD}:to_str"]


def is_even(x: int) -> bool:
    return x % 2 == 0


def parity(x: int) -> str:
    return "even" if x % 2 == 0 else "odd"


def test_filter_and_partition() -> None:
    filter_spec = to_spec(keep(is_even), strict=True).root
    assert filter_spec.kind == "filter"
    assert filter_spec.sig == "Iterator[int] → Iterator[int]"
    assert [c.ref for c in filter_spec.children] == [f"{MOD}:is_even"]

    partition_spec = to_spec(partition(is_even), strict=True).root
    assert partition_spec.kind == "partition"
    assert [c.ref for c in partition_spec.children] == [f"{MOD}:is_even"]


def test_route_keys_and_children() -> None:
    flow = route(parity, {"even": double, "odd": inc}, default=inc, name="by_parity")
    root = to_spec(flow, strict=True).root

    assert root.kind == "route"
    assert root.name == "by_parity"
    assert root.sig == "int → int"
    assert root.params == {"keys": ["even", "odd"], "default": True}
    assert [c.ref for c in root.children] == [
        f"{MOD}:parity",
        f"{MOD}:double",
        f"{MOD}:inc",
        f"{MOD}:inc",
    ]


def test_route_non_json_key() -> None:
    flow = route(parity, {("even",): double})  # type: ignore[dict-item]

    assert to_spec(flow).root.params["keys"] == ["('even',)"]
    with pytest.raises(SpecError, match="not a JSON scalar"):
        to_spec(flow, strict=True)


def test_right_nested_chain_is_flattened() -> None:
    right = IcoChain(IcoOperator(double), IcoOperator(inc))
    flow = IcoChain(IcoOperator(inc), right)

    root = to_spec(flow).root

    assert [c.ref for c in root.children] == [
        f"{MOD}:inc",
        f"{MOD}:double",
        f"{MOD}:inc",
    ]


# ────────────────────────────────────────────────
# Composite nodes
# ────────────────────────────────────────────────


def test_source_stream_sink() -> None:
    flow = IcoSource(numbers) | IcoStream(inc) | IcoSink(consume)
    root = to_spec(flow, strict=True).root

    source, stream, sink = root.children
    assert source.kind == "source"
    assert source.ref == f"{MOD}:numbers"
    assert stream.kind == "stream"
    assert stream.children[0].ref == f"{MOD}:inc"
    assert sink.kind == "sink"
    assert sink.ref == f"{MOD}:consume"


def test_process_params() -> None:
    root = to_spec(IcoProcess(inc, num_iterations=5)).root

    assert root.kind == "process"
    assert root.params == {"num_iterations": 5}
    assert root.children[0].ref == f"{MOD}:inc"


def is_large(x: int) -> bool:
    return x > 100


def test_process_until_params() -> None:
    flow = IcoProcess(double, until=is_large, max_iterations=20, on_limit="raise")
    root = to_spec(flow, strict=True).root

    assert root.kind == "process"
    assert root.params == {"max_iterations": 20, "on_limit": "raise"}
    assert [c.ref for c in root.children] == [f"{MOD}:double", f"{MOD}:is_large"]


def test_epoch_children_order() -> None:
    epoch = IcoEpoch(IcoSource(numbers), IcoContextOperator(accumulate))
    root = to_spec(epoch).root

    assert root.kind == "epoch"
    assert [c.kind for c in root.children] == ["source", "context_op"]
    assert root.children[1].ref == f"{MOD}:accumulate"


def test_batcher_and_progress_params() -> None:
    flow = IcoBatcher[int](4, drop_last=True) | IcoProgress[Iterator[Iterator[int]]](10)
    batcher, progress = to_spec(flow).root.children

    assert batcher.params == {"batch_size": 4, "drop_last": True}
    assert progress.kind == "progress"
    assert progress.params == {"total": 10}


def test_async_stream_from_factory_emits_one_worker() -> None:
    stream = IcoAsyncStream(make_worker, pool_size=8, ordered=True)
    root = to_spec(stream).root

    assert root.kind == "async_stream"
    assert root.params == {"pool_size": 8, "ordered": True, "factory": True}
    assert len(root.children) == 1
    assert root.children[0].ref == f"{MOD}:double"


def test_async_stream_explicit_pool_lists_all_workers() -> None:
    stream = IcoAsyncStream([IcoOperator(inc), IcoOperator(double)])
    root = to_spec(stream).root

    assert root.params["factory"] is False
    assert [c.ref for c in root.children] == [f"{MOD}:inc", f"{MOD}:double"]


# ────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────


class CustomNode(IcoNode):
    def __init__(self, body: IcoOperator[int, int], factor: int) -> None:
        super().__init__(name="custom", children=[body])
        self.body = body
        self.factor = factor


class UnregisteredNode(IcoNode):
    pass


def test_unregistered_node_uses_generic_fallback() -> None:
    root = to_spec(UnregisteredNode(name="x")).root

    assert root.kind == "node"
    assert root.params == {"type": f"{MOD}:UnregisteredNode"}


def test_custom_serializer_overrides_fallback() -> None:
    @spec_serializer(CustomNode)
    def _custom(node: CustomNode, ctx: SpecContext) -> NodeSpec:  # pyright: ignore[reportUnusedFunction]
        return NodeSpec(
            kind="custom",
            name=node.name,
            params={"factor": node.factor},
            children=(ctx.walk(node.body),),
        )

    root = to_spec(CustomNode(IcoOperator(inc), factor=3)).root

    assert root.kind == "custom"
    assert root.params == {"factor": 3}
    assert root.children[0].ref == f"{MOD}:inc"


# ────────────────────────────────────────────────
# JSON round trip
# ────────────────────────────────────────────────


def _full_flow() -> IcoOperatorProtocol[None, None]:
    return (
        IcoSource(numbers)
        | IcoStream(IcoOperator(inc) | IcoOperator(double))
        | IcoAsyncStream(make_worker, pool_size=2)
        | IcoSink(consume)
    )


def test_json_round_trip_is_lossless() -> None:
    spec = to_spec(_full_flow(), strict=True)

    text = to_json(spec)
    restored = from_json(text)

    assert restored == spec
    assert to_json(restored) == text


def test_json_is_compact_and_versioned() -> None:
    data = json.loads(to_json(to_spec(IcoOperator(inc))))

    assert data["version"] == 1
    # Empty fields are omitted to keep specs short for agents.
    assert set(data["root"]) == {"kind", "sig", "ref"}


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        '{"version": 99, "root": {"kind": "op"}}',
        '{"version": 1}',
        '{"version": 1, "root": {"name": "no kind"}}',
        '{"version": 1, "root": {"kind": "seq", "children": {}}}',
    ],
)
def test_invalid_json_raises_spec_error(text: str) -> None:
    with pytest.raises(SpecError):
        from_json(text)


def test_pipeline_spec_default_version() -> None:
    assert PipelineSpec(root=NodeSpec(kind="op")).version == 1
