from collections.abc import Iterator
from typing import Any

import pytest

from ico.core.operator import IcoOperator
from ico.core.tree_utils import TreePathIndex
from ico.registry import IcoRegistry, register
from ico.spec import NodeSpec, PipelineSpec, SpecError, bind, build, run, spec_params

REGISTRY = IcoRegistry()


@register("scale", 1, registry=REGISTRY)
class Scale(IcoOperator[int, int]):
    """Multiply by a factor."""

    def __init__(self, factor: int) -> None:
        self.factor = factor
        super().__init__(fn=self._apply)

    def _apply(self, x: int) -> int:
        return x * self.factor


@register("label", 1, registry=REGISTRY)
class Label(IcoOperator[int, str]):
    """Prefix a number."""

    def __init__(self, prefix: str = "#") -> None:
        self.prefix = prefix
        super().__init__(fn=self._apply)

    def _apply(self, x: int) -> str:
        return f"{self.prefix}{x}"


@register("numbers", 1, registry=REGISTRY)
def numbers(n: int) -> Iterator[int]:
    """Count up to n."""
    return iter(range(n))


def ref(name: str, **declaration: Any) -> dict[str, Any]:
    return {"$param": name, **declaration}


def scale(factor: Any) -> NodeSpec:
    return NodeSpec(kind="op", op="scale@1", params={"factor": factor})


def label(prefix: Any) -> NodeSpec:
    return NodeSpec(kind="op", op="label@1", params={"prefix": prefix})


def seq(*children: NodeSpec) -> NodeSpec:
    return NodeSpec(kind="seq", children=children)


# ─── Declarations ───


def test_spec_params_lists_names_types_defaults_and_paths() -> None:
    spec = PipelineSpec(
        root=seq(
            scale(ref("factor")),
            scale(ref("factor")),
            label(ref("prefix", default="n=")),
        )
    )
    factor, prefix = spec_params(spec, REGISTRY)

    assert (factor.name, factor.type, factor.required) == ("factor", "int", True)
    assert factor.paths == (TreePathIndex((0,)), TreePathIndex((1,)))
    assert (prefix.name, prefix.required, prefix.default) == ("prefix", False, "n=")
    assert prefix.to_dict() == {
        "name": "prefix",
        "required": False,
        "type": "str",
        "default": "n=",
        "paths": ["2"],
    }


def test_structural_params_have_no_type_and_null_is_a_default() -> None:
    progress = NodeSpec(kind="progress", params={"total": ref("count", default=None)})
    [count] = spec_params(progress, REGISTRY)

    assert (count.type, count.required, count.default) == (None, False, None)


def test_a_parameter_must_be_declared_the_same_way() -> None:
    defaults = seq(scale(ref("x", default=2)), scale(ref("x", default=3)))
    types = seq(scale(ref("x")), label(ref("x")))

    with pytest.raises(SpecError, match="different defaults"):
        spec_params(defaults, REGISTRY)
    with pytest.raises(SpecError, match="different types: int and str") as info:
        spec_params(types, REGISTRY)
    assert info.value.path == TreePathIndex((1,))


def test_malformed_reference() -> None:
    with pytest.raises(SpecError, match="parameter reference"):
        spec_params(scale(ref("factor", value=2)), REGISTRY)
    with pytest.raises(SpecError, match="parameter reference"):
        spec_params(scale({"$param": 3}), REGISTRY)


# ─── Binding ───


def test_bind_substitutes_values_and_defaults() -> None:
    spec = PipelineSpec(
        root=seq(scale(ref("factor")), label(ref("prefix", default="#")))
    )
    bound = bind(spec, {"factor": 3})

    assert isinstance(bound, PipelineSpec)
    assert bound.root.children[0].params == {"factor": 3}
    assert bound.root.children[1].params == {"prefix": "#"}
    assert spec_params(bound, REGISTRY) == []
    assert spec.root.children[0].params == {"factor": ref("factor")}  # unchanged


def test_bind_errors_list_every_name() -> None:
    spec = seq(scale(ref("a")), scale(ref("b")), label(ref("p", default="#")))

    with pytest.raises(SpecError, match="missing value for parameter.s. a, b"):
        bind(spec, {})
    with pytest.raises(SpecError, match="unknown parameter.s. c; declared: a, b, p"):
        bind(spec, {"a": 1, "b": 2, "c": 3})


def test_spec_without_parameters() -> None:
    spec = seq(scale(2), label("#"))

    assert bind(spec) is spec
    with pytest.raises(SpecError, match="unknown parameter.s. x; declared: none"):
        bind(spec, {"x": 1})


# ─── Build and run ───


def test_build_and_run_take_params() -> None:
    spec = PipelineSpec(
        root=seq(
            NodeSpec(kind="op", op="numbers@1"),
            NodeSpec(
                kind="stream",
                children=(
                    seq(scale(ref("factor")), label(ref("prefix", default="#"))),
                ),
            ),
        )
    )

    built = build(spec, REGISTRY, params={"factor": 10})
    tripled = run(spec, 3, registry=REGISTRY, params={"factor": 3, "prefix": "x"})

    assert list(built(2)) == ["#0", "#10"]  # type: ignore[operator]
    assert list(tripled.outputs) == ["x0", "x3", "x6"]
    with pytest.raises(SpecError, match="missing value for parameter.s. factor"):
        run(spec, 3, registry=REGISTRY)
