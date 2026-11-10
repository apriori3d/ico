"""Registered operator classes whose constructor takes child operators."""

from collections.abc import Iterator

import pytest

from ico.core.node import IcoNodeProtocol
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.describe.plan.json_renderer import plan_view
from ico.registry import IcoRegistry, register
from ico.spec import NodeSpec, SpecError, build, from_json, to_json, to_spec


def _inc(x: float) -> float:
    return x + 1


def _double(x: float) -> float:
    return x * 2


class Repeat(IcoOperator[float, float]):
    """Apply a body a number of times."""

    def __init__(
        self,
        body: IcoOperatorProtocol[float, float],
        times: int = 2,
        name: str | None = None,
    ) -> None:
        self.body = body
        self.times = times
        super().__init__(fn=self._apply, name=name, children=[body])

    def _apply(self, x: float) -> float:
        for _ in range(self.times):
            x = self.body(x)
        return x


class Guarded(IcoOperator[float, float]):
    """Apply a body, then an optional check."""

    def __init__(
        self,
        body: IcoOperatorProtocol[float, float],
        check: IcoOperatorProtocol[float, float] | None = None,
    ) -> None:
        self.body = body
        self.check = check
        children: list[IcoNodeProtocol] = [body]
        if check is not None:
            children.append(check)
        super().__init__(fn=self._apply, children=children)

    def _apply(self, x: float) -> float:
        x = self.body(x)
        return self.check(x) if self.check is not None else x


class Untyped(IcoOperator[float, float]):
    def __init__(self, body: object) -> None:
        self.body = body
        super().__init__(fn=self._apply)

    def _apply(self, x: float) -> float:
        return x


@pytest.fixture
def registry() -> Iterator[IcoRegistry]:
    registry = IcoRegistry()
    register("inc", 1, registry=registry)(_inc)
    register("double", 1, registry=registry)(_double)
    register("repeat", 1, registry=registry)(Repeat)
    register("guarded", 1, registry=registry)(Guarded)
    register("untyped", 1, registry=registry)(Untyped)
    yield registry
    for ref in [e.ref for e in registry]:
        registry.unregister(ref)


def _op(registry: IcoRegistry, ref: str) -> IcoOperatorProtocol[float, float]:
    node = registry.get(ref).node
    assert node is not None
    return node  # pyright: ignore[reportReturnType]


def _round_trip(node: IcoNodeProtocol, registry: IcoRegistry) -> IcoNodeProtocol:
    spec = to_spec(node, strict=True)
    rebuilt = build(from_json(to_json(spec)), registry)
    assert to_spec(rebuilt, strict=True) == spec
    return rebuilt


def test_card_marks_child_params(registry: IcoRegistry) -> None:
    params = registry.get("guarded@1").params
    assert [(p.name, p.child) for p in params] == [("body", True), ("check", True)]
    assert [p.child for p in registry.get("repeat@1").params] == [True, False, False]
    card = registry.get("repeat@1").card()["params"]
    assert isinstance(card, list) and isinstance(card[0], dict)
    assert card[0]["child"] is True


def test_child_flow_and_literal_params_round_trip(registry: IcoRegistry) -> None:
    inc, double = _op(registry, "inc@1"), _op(registry, "double@1")
    node = Repeat(inc | double, times=3, name="r")

    spec = to_spec(node, strict=True).root
    assert spec.kind == "op" and spec.op == "repeat@1"
    assert spec.params == {"times": 3, "name": "r"}
    assert [c.kind for c in spec.children] == ["seq"]
    assert [c.op for c in spec.children[0].children] == ["inc@1", "double@1"]

    rebuilt = _round_trip(node, registry)
    assert isinstance(rebuilt, Repeat)
    assert rebuilt(1.0) == node(1.0) == 22.0  # ((1+1)*2+1)*2+1)*2


def test_nested_children_round_trip(registry: IcoRegistry) -> None:
    inc = _op(registry, "inc@1")
    node = Repeat(Guarded(Repeat(inc, times=2), check=_op(registry, "double@1")))

    spec = to_spec(node, strict=True).root
    guarded = spec.children[0]
    assert guarded.op == "guarded@1"
    assert [c.op for c in guarded.children] == ["repeat@1", "double@1"]
    assert guarded.children[0].children[0].op == "inc@1"

    rebuilt = _round_trip(node, registry)
    assert rebuilt(0.0) == node(0.0)  # type: ignore[operator]


def test_optional_child_set_to_none(registry: IcoRegistry) -> None:
    node = Guarded(_op(registry, "inc@1"))

    spec = to_spec(node, strict=True).root
    assert spec.params == {"check": None}
    assert len(spec.children) == 1

    rebuilt = _round_trip(node, registry)
    assert isinstance(rebuilt, Guarded) and rebuilt.check is None
    assert rebuilt(1.0) == 2.0


def test_fresh_instances_for_workers(registry: IcoRegistry) -> None:
    from ico.spec.builders import registry_context

    spec = to_spec(Repeat(Repeat(_op(registry, "inc@1")))).root
    a = registry_context(registry, fresh=True).build(spec)
    b = registry_context(registry, fresh=True).build(spec)
    assert isinstance(a, Repeat) and isinstance(b, Repeat)
    assert a is not b and a.body is not b.body
    assert a(0.0) == 4.0


def test_plan_shows_the_child(registry: IcoRegistry) -> None:
    node = Repeat(_op(registry, "inc@1"), name="looped")
    rebuilt = build(to_spec(node), registry)

    view = plan_view(rebuilt)
    text = "\n".join(
        "".join(s["text"] for s in cell) for row in view["rows"] for cell in row
    )
    assert "looped" in text and "_inc" in text
    assert view["rows"] == plan_view(node)["rows"]


def test_wrong_number_of_children(registry: IcoRegistry) -> None:
    spec = NodeSpec(kind="op", op="repeat@1", params={"times": 1})
    with pytest.raises(SpecError, match=r"expects 1 children \(body\), got 0"):
        build(spec, registry)

    inc = to_spec(_op(registry, "inc@1")).root
    spec = NodeSpec(
        kind="op", op="guarded@1", params={"check": None}, children=(inc, inc)
    )
    with pytest.raises(SpecError, match=r"expects 1 children \(body\), got 2"):
        build(spec, registry)


def test_child_errors_are_located(registry: IcoRegistry) -> None:
    bad = NodeSpec(kind="op", op="missing@1")
    spec = NodeSpec(kind="op", op="repeat@1", children=(bad,))
    with pytest.raises(SpecError) as info:
        build(spec, registry)
    assert str(info.value.path) == "0"


def test_child_param_value_in_params_is_rejected(registry: IcoRegistry) -> None:
    spec = NodeSpec(kind="op", op="repeat@1", params={"body": "inc@1"})
    with pytest.raises(SpecError, match="must be given as a child"):
        build(spec, registry)


def test_children_rejected_for_node_entries(registry: IcoRegistry) -> None:
    inc = to_spec(_op(registry, "inc@1")).root
    spec = NodeSpec(kind="op", op="double@1", children=(inc,))
    with pytest.raises(SpecError, match="takes no children"):
        build(spec, registry)


def test_unannotated_operator_param_hints_at_annotation(registry: IcoRegistry) -> None:
    with pytest.raises(SpecError, match="Annotate it with a node type"):
        to_spec(Untyped(_op(registry, "inc@1")))
