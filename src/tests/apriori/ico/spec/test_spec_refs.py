"""Pipelines that use other pipelines: pipeline_ref nodes are inlined from a
library before a spec is bound and built."""

from typing import Any

import pytest

from ico.spec import (
    NodeSpec,
    PipelineSpec,
    SpecError,
    build,
    inline,
    pipeline_refs,
    run,
    spec_params,
)
from tests.apriori.ico.spec.test_spec_params import REGISTRY, label, ref, scale, seq


def pref(pipeline: str, name: str | None = None, **bind: Any) -> NodeSpec:
    params: dict[str, Any] = {"pipeline": pipeline}
    if bind:
        params["bind"] = bind
    return NodeSpec(kind="pipeline_ref", name=name, params=params)


LIBRARY: dict[str, NodeSpec] = {
    # Parametric: factor is required, prefix has a default.
    "scaled": seq(scale(ref("factor")), label(ref("prefix", default="#"))),
    "double": scale(2),
    "twice": seq(pref("double"), pref("double")),
    "loop_a": pref("loop_b"),
    "loop_b": pref("loop_a"),
}


def library(name: str) -> NodeSpec:
    return LIBRARY[name]


def test_a_reference_is_replaced_by_the_pipeline_and_named_after_it() -> None:
    spec = seq(scale(3), pref("double"))
    inlined = inline(spec, library)

    assert inlined.children[1] == NodeSpec(
        kind="op", op="scale@1", name="double", params={"factor": 2}
    )
    assert run(inlined, 5, registry=REGISTRY).outputs == (30,)


def test_bind_sets_inner_parameters_and_the_rest_are_lifted() -> None:
    spec = PipelineSpec(seq(scale(10), pref("scaled", name="score", factor=ref("k"))))
    inlined = inline(spec, library)

    assert inlined.root.children[1].name == "score"
    declared = {p.name: p for p in spec_params(inlined, REGISTRY)}
    assert set(declared) == {"k", "prefix"}  # factor bound to k; prefix lifted
    assert declared["k"].required and not declared["prefix"].required
    result = run(inlined, 1, registry=REGISTRY, params={"k": 4, "prefix": "n="})
    assert result.outputs == ("n=40",)


def test_constants_in_bind_and_nested_references() -> None:
    fixed = inline(pref("scaled", factor=3, prefix="x"), library)
    assert spec_params(fixed, REGISTRY) == []
    assert run(fixed, 2, registry=REGISTRY).outputs == ("x6",)
    nested = inline(pref("twice"), library)
    assert run(nested, 1, registry=REGISTRY).outputs == (4,)
    assert [c.name for c in nested.children] == ["double", "double"]


@pytest.mark.parametrize(
    ("spec", "message", "path"),
    [
        (seq(scale(1), pref("missing")), "unknown pipeline 'missing'", "1"),
        (
            seq(pref("scaled", size=2)),
            r"has no parameter\(s\) size; declared: factor, prefix",
            "0",
        ),
        (pref("loop_a"), "refer to each other: loop_a → loop_b → loop_a", ""),
        (NodeSpec(kind="pipeline_ref", params={"name": "x"}), "a pipeline_ref is", ""),
        (
            NodeSpec(kind="pipeline_ref", params={"pipeline": "double", "bind": 3}),
            "must be a mapping",
            "",
        ),
    ],
)
def test_invalid_references(spec: NodeSpec, message: str, path: str) -> None:
    with pytest.raises(SpecError, match=message) as info:
        inline(spec, library)
    assert str(info.value.path or "") == path


def test_errors_inside_a_pipeline_are_located_through_the_reference() -> None:
    def broken(name: str) -> NodeSpec:
        return seq(scale(1), pref("missing")) if name == "outer" else library(name)

    with pytest.raises(SpecError) as info:
        inline(seq(scale(1), pref("outer")), broken)
    assert str(info.value.path) == "1.1"


def test_an_uninlined_reference_does_not_build() -> None:
    with pytest.raises(SpecError, match="not inlined"):
        build(pref("double"), REGISTRY)


def test_pipeline_refs_lists_direct_references() -> None:
    assert pipeline_refs(seq(pref("scaled"), pref("twice"), pref("scaled"))) == [
        "scaled",
        "twice",
    ]
    assert pipeline_refs(scale(1)) == []
