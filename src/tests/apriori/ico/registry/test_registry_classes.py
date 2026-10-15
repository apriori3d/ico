from collections.abc import Iterator
from typing import Any

import pytest

from ico.core.operator import IcoOperator
from ico.registry import IcoRegistry, get_meta, register
from ico.spec import (
    NodeSpec,
    PipelineSpec,
    SpecError,
    build,
    from_json,
    to_json,
    to_spec,
)


class Threshold(IcoOperator[float, bool]):
    """Compare a value with a cutoff.

    Details that are not part of the card.
    """

    def __init__(self, *, cutoff: float = 0.5, label: str = "high") -> None:
        self.cutoff = cutoff
        self.label = label
        super().__init__(fn=self._apply)

    def _apply(self, x: float) -> bool:
        return x > self.cutoff


class Scale(IcoOperator[float, float]):
    def __init__(self, factor: float) -> None:
        self.factor = factor
        super().__init__(fn=self._apply)

    def _apply(self, x: float) -> float:
        return x * self.factor


@pytest.fixture
def registry() -> Iterator[IcoRegistry]:
    registry = IcoRegistry()
    register("threshold", 1, tags={"math"}, registry=registry)(Threshold)
    register("scale", 1, registry=registry)(Scale)
    yield registry
    for ref in [e.ref for e in registry]:
        registry.unregister(ref)


# ─── Registration ───


def test_class_entry(registry: IcoRegistry) -> None:
    entry = registry.get("threshold@1")

    assert entry.cls is Threshold
    assert entry.node is None
    assert entry.signature.format() == "float → bool"
    assert entry.description == "Compare a value with a cutoff."
    assert [p.name for p in entry.params] == ["cutoff", "label"]


def test_class_card_lists_params(registry: IcoRegistry) -> None:
    assert registry.get("threshold@1").card()["params"] == [
        {"name": "cutoff", "required": False, "type": "float", "default": 0.5},
        {"name": "label", "required": False, "type": "str", "default": "high"},
    ]
    assert registry.get("scale@1").card()["params"] == [
        {"name": "factor", "required": True, "type": "float"}
    ]
    assert registry.get("scale@1").description == ""


def test_class_found_by_search(registry: IcoRegistry) -> None:
    assert [e.ref for e in registry.search(input=float, output=bool)] == ["threshold@1"]


def test_instances_are_registered_but_subclasses_are_not(registry: IcoRegistry) -> None:
    assert get_meta(Threshold(cutoff=1.0)) == get_meta(Threshold)

    class Strict(Threshold):
        pass

    assert get_meta(Strict) is None
    assert get_meta(Strict()) is None


def test_register_returns_class(registry: IcoRegistry) -> None:
    class Negate(IcoOperator[int, int]):
        def __init__(self) -> None:
            super().__init__(fn=self._apply)

        def _apply(self, x: int) -> int:
            return -x

    assert register("negate", 1, registry=registry)(Negate) is Negate
    assert Negate()(3) == -3


def test_reject_var_args(registry: IcoRegistry) -> None:
    class Flexible(IcoOperator[int, int]):
        def __init__(self, **options: Any) -> None:
            super().__init__(fn=abs)

    with pytest.raises(ValueError, match=r"\*args or \*\*kwargs \(found 'options'\)"):
        register("flexible", 1, registry=registry)(Flexible)


def test_reject_class_without_generic_signature(registry: IcoRegistry) -> None:
    class Untyped(IcoOperator[Any, Any]):
        def __init__(self) -> None:
            super().__init__(fn=abs)

    class Generic_(IcoOperator):  # type: ignore[type-arg]  # pyright: ignore[reportMissingTypeArgument]
        def __init__(self) -> None:
            super().__init__(fn=abs)  # pyright: ignore[reportUnknownMemberType]

    register("untyped", 1, registry=registry)(Untyped)  # Any → Any is a signature
    with pytest.raises(ValueError, match="cannot be inferred"):
        register("generic", 1, registry=registry)(Generic_)


def test_reject_non_operator_class(registry: IcoRegistry) -> None:
    class NotAnOperator:
        pass

    with pytest.raises(TypeError, match="only IcoOperator subclasses"):
        register("nope", 1, registry=registry)(NotAnOperator)  # type: ignore[call-overload,type-var]


def test_unregister_class(registry: IcoRegistry) -> None:
    registry.unregister("scale@1")
    assert get_meta(Scale) is None


# ─── Spec params ───


def test_spec_stores_constructor_kwargs(registry: IcoRegistry) -> None:
    root = to_spec(Threshold(cutoff=0.9)).root
    assert root == NodeSpec(
        kind="op",
        sig="float → bool",
        op="threshold@1",
        params={"cutoff": 0.9, "label": "high"},
    )


def test_build_creates_fresh_instance_per_node(registry: IcoRegistry) -> None:
    original = Scale(3.0) | Threshold(cutoff=10.0, label="big")
    spec = to_spec(original, strict=True)

    rebuilt = build(from_json(to_json(spec)), registry)
    assert to_spec(rebuilt, strict=True) == spec

    scale, threshold = rebuilt.children
    assert isinstance(scale, Scale) and scale.factor == 3.0
    assert isinstance(threshold, Threshold) and threshold.label == "big"
    assert threshold is not original.children[1]
    assert rebuilt(4.0) is True  # type: ignore[operator]
    assert rebuilt(3.0) is False  # type: ignore[operator]


def test_export_requires_same_named_attribute(registry: IcoRegistry) -> None:
    class Hidden(IcoOperator[int, int]):
        def __init__(self, offset: int) -> None:
            self._offset = offset
            super().__init__(fn=abs)

    register("hidden", 1, registry=registry)(Hidden)
    with pytest.raises(SpecError, match="attribute of the same name"):
        to_spec(Hidden(1))


def test_export_rejects_non_json_values(registry: IcoRegistry) -> None:
    with pytest.raises(SpecError, match="not JSON-compatible"):
        to_spec(Scale(complex(1, 2)))  # type: ignore[arg-type]


def test_build_with_invalid_params(registry: IcoRegistry) -> None:
    bad = NodeSpec(kind="op", op="scale@1", params={"facter": 2.0})
    with pytest.raises(SpecError, match="Cannot create 'scale@1' from params"):
        build(PipelineSpec(root=bad), registry)


def test_build_rejects_params_for_node_entries(registry: IcoRegistry) -> None:
    def inc(x: int) -> int:
        return x + 1

    register("inc", 1, registry=registry)(inc)
    bad = NodeSpec(kind="op", op="inc@1", params={"x": 1})
    with pytest.raises(SpecError, match="takes no params"):
        build(PipelineSpec(root=bad), registry)
