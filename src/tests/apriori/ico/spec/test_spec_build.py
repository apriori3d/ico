from typing import Any

import pytest

from ico.core.batcher import IcoBatcher
from ico.core.chain import IcoChain
from ico.core.filter import keep
from ico.core.fork import fork
from ico.core.identity import IcoIdentity
from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator
from ico.core.partition import partition
from ico.core.pipeline import IcoPipeline
from ico.core.process import IcoProcess
from ico.core.route import route
from ico.core.stream import IcoStream
from ico.registry import IcoRegistry, register
from ico.spec import (
    NodeSpec,
    PipelineSpec,
    SpecError,
    build,
    from_json,
    to_json,
    to_spec,
)

# ─── Registered catalog ───

REGISTRY = IcoRegistry()


def _reg(name: str) -> Any:
    return register(name, 1, registry=REGISTRY)


@_reg("inc")
def inc(x: int) -> int:
    """Add one."""
    return x + 1


@_reg("double")
def double(x: int) -> int:
    return x * 2


@_reg("to_str")
def to_str(x: int) -> str:
    return str(x)


@_reg("is_even")
def is_even(x: int) -> bool:
    return x % 2 == 0


@_reg("is_large")
def is_large(x: int) -> bool:
    return x > 50


@_reg("parity")
def parity(x: int) -> str:
    return "even" if x % 2 == 0 else "odd"


# Structural generic nodes are built unparametrized, as `build` does.
BATCHER: IcoBatcher[Any] = IcoBatcher(4, drop_last=True)
SMALL_BATCHER: IcoBatcher[Any] = IcoBatcher(2)
NOOP: IcoIdentity[Any] = IcoIdentity(name="noop")


def roundtrip(flow: IcoNodeProtocol) -> tuple[PipelineSpec, IcoNodeProtocol]:
    spec = to_spec(flow, strict=True)
    rebuilt = build(spec, REGISTRY)
    assert to_spec(rebuilt, strict=True) == spec
    # Through JSON as well.
    assert to_spec(build(from_json(to_json(spec)), REGISTRY), strict=True) == spec
    return spec, rebuilt


# ─── Export of registered operators ───


def test_registered_operator_exported_by_reference() -> None:
    root = to_spec(inc).root
    assert root == NodeSpec(kind="op", name=None, sig="int → int", op="inc@1")
    assert "op" in to_json(to_spec(inc))


def test_registered_composite_is_opaque() -> None:
    registry = IcoRegistry()
    flow = register("inc_twice", 1, registry=registry)(inc | inc)
    root = to_spec(inc | flow).root

    assert root.kind == "seq"
    assert [c.op for c in root.children] == ["inc@1", "inc_twice@1"]
    assert root.children[1].children == ()
    assert isinstance(build(to_spec(flow), registry), IcoChain)


def test_build_returns_registered_instances() -> None:
    rebuilt = build(to_spec(inc | double), REGISTRY)
    assert isinstance(rebuilt, IcoChain)
    assert list(rebuilt.children) == [inc, double]


# ─── Round trip per kind ───


@pytest.mark.parametrize(
    "flow",
    [
        pytest.param(inc, id="op"),
        pytest.param(inc | double | to_str, id="seq"),
        pytest.param(inc | (double | inc), id="seq-right-nested"),
        pytest.param(IcoStream(inc), id="stream"),
        pytest.param(IcoStream(inc, name="each"), id="stream-named"),
        pytest.param(IcoPipeline(inc, double), id="pipeline"),
        pytest.param(IcoProcess(double, num_iterations=3), id="process-fixed"),
        pytest.param(
            IcoProcess(double, until=is_large, max_iterations=10, on_limit="raise"),
            id="process-until",
        ),
        pytest.param(fork(inc, to_str, is_even, name="stats"), id="fork"),
        pytest.param(
            route(parity, {"even": double, "odd": inc}, default=inc, name="r"),
            id="route",
        ),
        pytest.param(route(parity, {"even": double}), id="route-no-default"),
        pytest.param(keep(is_even), id="filter"),
        pytest.param(partition(is_even, name="split"), id="partition"),
        pytest.param(BATCHER, id="batcher"),
        pytest.param(NOOP, id="identity"),
    ],
)
def test_roundtrip(flow: IcoNodeProtocol) -> None:
    roundtrip(flow)


def test_roundtrip_nested_combination() -> None:
    body = inc | fork(double, route(parity, {"even": to_str, "odd": to_str}))
    flow = keep(is_even) | IcoStream(body, name="per_item") | SMALL_BATCHER
    roundtrip(flow)


def test_rebuilt_flow_behaves_like_original() -> None:
    flow = (
        inc
        | IcoProcess(double, until=is_large, max_iterations=10)
        | fork(to_str, route(parity, {"even": inc}, default=double))
    )
    _, rebuilt = roundtrip(flow)
    assert flow(3) == cast_call(rebuilt, 3) == ("64", 65)


def cast_call(node: IcoNodeProtocol, value: Any) -> Any:
    return node(value)  # type: ignore[operator]


# ─── Errors ───


def spec_of(root: NodeSpec) -> PipelineSpec:
    return PipelineSpec(root=root)


def test_unknown_operator() -> None:
    with pytest.raises(SpecError, match="Cannot resolve operator 'missing@1'"):
        build(spec_of(NodeSpec(kind="op", op="missing@1")), REGISTRY)


def test_malformed_reference() -> None:
    with pytest.raises(SpecError, match="Cannot resolve operator 'inc'"):
        build(spec_of(NodeSpec(kind="op", op="inc")), REGISTRY)


def test_signature_drift() -> None:
    with pytest.raises(SpecError, match="Signature of 'inc@1' changed"):
        build(spec_of(NodeSpec(kind="op", sig="str → str", op="inc@1")), REGISTRY)


def test_unregistered_leaf_is_not_imported() -> None:
    def local(x: int) -> int:
        return x

    spec = to_spec(IcoOperator(local) | inc)
    with pytest.raises(SpecError, match="only registered operators"):
        build(spec, REGISTRY)

    importable = NodeSpec(kind="op", ref="os:system")
    with pytest.raises(SpecError, match="only registered operators"):
        build(spec_of(importable), REGISTRY)


@pytest.mark.parametrize("kind", ["async_stream", "epoch", "agent", "monitor", "node"])
def test_unsupported_kinds(kind: str) -> None:
    with pytest.raises(SpecError, match=f"kind '{kind}' yet"):
        build(spec_of(NodeSpec(kind=kind)), REGISTRY)


@pytest.mark.parametrize(
    ("root", "message"),
    [
        (NodeSpec(kind="stream"), "expects 1 children"),
        (
            NodeSpec(kind="seq", children=(NodeSpec(kind="op", op="inc@1"),)),
            "at least 2",
        ),
        (
            NodeSpec(kind="batcher", params={"batch_size": "4", "drop_last": False}),
            "must be int",
        ),
        (
            NodeSpec(kind="batcher", params={"batch_size": True, "drop_last": False}),
            "must be int",
        ),
        (
            NodeSpec(kind="batcher", params={"drop_last": False}),
            "requires param 'batch_size'",
        ),
        (
            NodeSpec(
                kind="process",
                params={"max_iterations": 3, "on_limit": "ignore"},
                children=(
                    NodeSpec(kind="op", op="inc@1"),
                    NodeSpec(kind="op", op="is_even@1"),
                ),
            ),
            "Invalid on_limit",
        ),
        (
            NodeSpec(
                kind="route",
                params={"keys": ["a", "b"], "default": False},
                children=(
                    NodeSpec(kind="op", op="parity@1"),
                    NodeSpec(kind="op", op="inc@1"),
                ),
            ),
            "expects 3 children",
        ),
    ],
)
def test_invalid_structure(root: NodeSpec, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        build(spec_of(root), REGISTRY)


def test_build_node_spec_and_default_registry() -> None:
    registry_node = register("tests.spec.build.neg", 1)(IcoOperator[int, int](abs))
    try:
        assert build(NodeSpec(kind="op", op="tests.spec.build.neg@1")) is registry_node
    finally:
        from ico.registry import default_registry

        default_registry.unregister("tests.spec.build.neg@1")
