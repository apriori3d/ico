"""Pipelines that use other pipelines: ``pipeline_ref`` nodes.

A spec node may stand for a whole saved pipeline::

    {"kind": "pipeline_ref", "name": "refine",
     "params": {"pipeline": "refine_masks",
                "bind": {"hold": 4, "work_dir": {"$param": "work_dir"}}}}

:func:`inline` replaces every such node by the root of the pipeline it
names, before the spec is bound and built, so building, workers,
validation, plans and tools see an ordinary spec. ``bind`` gives values to
parameters of the inner pipeline: constants, or references to parameters
of the outer one. The inner parameters it does not bind become parameters
of the outer pipeline under their own names. The inlined root is named
after the reference (its ``name``, else the pipeline's), so a plan shows
where it came from.

Where pipelines are stored is up to the caller: ``library(name)`` returns
the spec of a pipeline (``ico-mcp`` keeps them in files).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import overload

from ico.core.tree_utils import TreePathIndex
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError
from ico.spec.params import _reference  # pyright: ignore[reportPrivateUsage]

KIND = "pipeline_ref"
MAX_DEPTH = 8
"""Nesting limit of pipelines in pipelines."""

Library = Callable[[str], PipelineSpec | NodeSpec]
"""The spec of a saved pipeline by name; raises ``KeyError`` for an
unknown one."""


@overload
def inline(spec: PipelineSpec, library: Library) -> PipelineSpec: ...
@overload
def inline(spec: NodeSpec, library: Library) -> NodeSpec: ...
@overload
def inline(
    spec: PipelineSpec | NodeSpec, library: Library
) -> PipelineSpec | NodeSpec: ...
def inline(spec: PipelineSpec | NodeSpec, library: Library) -> PipelineSpec | NodeSpec:
    """Replace every ``pipeline_ref`` node by the pipeline it names,
    recursively.

    Raises:
        SpecError: At the reference, for an unknown pipeline, a malformed
            reference, a ``bind`` key the pipeline does not declare, a
            cycle (with its chain) or nesting deeper than ``MAX_DEPTH``.
    """
    root = spec.root if isinstance(spec, PipelineSpec) else spec
    inlined = _inline(root, TreePathIndex(), library, ())
    if isinstance(spec, PipelineSpec):
        return replace(spec, root=inlined)
    return inlined


def pipeline_refs(spec: PipelineSpec | NodeSpec) -> list[str]:
    """The pipelines a spec refers to directly, in order, without
    repeats (not those its pipelines refer to)."""
    root = spec.root if isinstance(spec, PipelineSpec) else spec
    found: list[str] = []
    for node in _walk(root):
        if node.kind == KIND:
            name = node.params.get("pipeline")
            if isinstance(name, str) and name not in found:
                found.append(name)
    return found


def _walk(node: NodeSpec) -> Iterator[NodeSpec]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _inline(
    node: NodeSpec, path: TreePathIndex, library: Library, stack: tuple[str, ...]
) -> NodeSpec:
    if node.kind != KIND:
        children = tuple(
            _inline(child, path.add_child(i), library, stack)
            for i, child in enumerate(node.children)
        )
        return replace(node, children=children)

    name, bound = _parse(node, path)
    if name in stack:
        chain = " → ".join((*stack, name))
        raise SpecError(f"pipelines refer to each other: {chain}", path=path)
    if len(stack) >= MAX_DEPTH:
        raise SpecError(
            f"pipelines nest deeper than {MAX_DEPTH}: {' → '.join(stack)}", path=path
        )
    try:
        found = library(name)
    except KeyError:
        raise SpecError(f"unknown pipeline {name!r}", path=path) from None
    inner = found.root if isinstance(found, PipelineSpec) else found
    try:
        inner = _inline(inner, TreePathIndex(), library, (*stack, name))
        declared = _declared(inner)
    except SpecError as e:
        raise e.at(_join(path, e.path)) from None
    unknown = sorted(set(bound) - declared)
    if unknown:
        listed = ", ".join(sorted(declared)) or "none"
        raise SpecError(
            f"pipeline {name!r} has no parameter(s) {', '.join(unknown)}; "
            f"declared: {listed}",
            path=path,
        )
    return replace(_rebind(inner, bound), name=node.name or name)


def _parse(node: NodeSpec, path: TreePathIndex) -> tuple[str, dict[str, JsonValue]]:
    """The pipeline name and the ``bind`` values of a reference."""
    extra = sorted(set(node.params) - {"pipeline", "bind"})
    name = node.params.get("pipeline")
    bound = node.params.get("bind", {})
    if extra or node.children or not isinstance(name, str) or not name:
        raise SpecError(
            f"a {KIND} is {{'params': {{'pipeline': name, 'bind': {{...}}}}}} "
            "without children",
            path=path,
        )
    if not isinstance(bound, dict):
        raise SpecError(f"'bind' of {KIND} {name!r} must be a mapping", path=path)
    return name, dict(bound)


def _declared(node: NodeSpec) -> set[str]:
    """The names of the parameters a spec uses."""
    from ico.spec.params import spec_params

    return {param.name for param in spec_params(node)}


def _rebind(node: NodeSpec, values: Mapping[str, JsonValue]) -> NodeSpec:
    """Replace the references to the parameters in ``values``; others stay
    (they become parameters of the outer pipeline)."""
    params = {
        key: values[found[0]]
        if (found := _reference(value, TreePathIndex())) is not None
        and found[0] in values
        else value
        for key, value in node.params.items()
    }
    children = tuple(_rebind(child, values) for child in node.children)
    return replace(node, params=params, children=children)


def _join(prefix: TreePathIndex, inner: TreePathIndex | None) -> TreePathIndex:
    if inner is None:
        return prefix
    return TreePathIndex.create(*prefix.path_index, *inner.path_index)
