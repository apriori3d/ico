"""The assets a spec needs (``spec_assets``), for validation and prefetch.

An operator needs an asset through an :data:`~ico.assets.AssetId`
constructor param (a file it reads itself) or through a resource param
whose resource class has ``AssetId`` fields (a model it loads). Values
come from the spec, else from the defaults.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, cast

from ico.core.tree_utils import TreePathIndex
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError
from ico.spec.params import bind
from ico.spec.refs import KIND as PIPELINE_REF

if TYPE_CHECKING:
    from ico.registry.registry import IcoRegistry


def spec_assets(
    spec: PipelineSpec | NodeSpec,
    params: Mapping[str, JsonValue] | None = None,
    *,
    registry: IcoRegistry | None = None,
) -> list[str]:
    """The asset ids a spec needs, in order of first use, without repeats.

    Args:
        spec: An inlined spec (``pipeline_ref`` nodes replaced, see
            :func:`ico.spec.inline`).
        params: Values of the pipeline parameters (as for ``build``).
        registry: Where operators are looked up; the default registry by
            default.

    Raises:
        SpecError: At the node, for an unknown operator, a malformed
            resource param, a ``pipeline_ref`` left, or parameter errors of
            :func:`ico.spec.bind`.
    """
    from ico.registry.registry import default_registry

    bound = bind(spec, params)
    root = bound.root if isinstance(bound, PipelineSpec) else bound
    lookup = registry if registry is not None else default_registry
    found: list[str] = []
    for node, path in _walk(root, TreePathIndex()):
        for asset in _node_assets(node, path, lookup):
            if asset not in found:
                found.append(asset)
    return found


def _node_assets(
    node: NodeSpec, path: TreePathIndex, registry: IcoRegistry
) -> list[str]:
    from ico.resources.base import Resource, from_spec

    if node.kind == PIPELINE_REF:
        raise SpecError("inline pipeline_ref nodes first (ico.spec.inline)", path=path)
    if node.kind != "op" or node.op is None:
        return []
    try:
        entry = registry.get(node.op)
    except KeyError as e:
        raise SpecError(
            f"Cannot resolve operator '{node.op}': {e.args[0]}", path=path
        ) from None
    out: list[str] = []
    for param in entry.params:
        given = param.name in node.params
        value = cast(object, node.params[param.name] if given else param.default)
        if not given and param.required:
            continue
        if param.asset and isinstance(value, str) and value:
            out.append(value)
        elif param.resource is not None:
            if given:
                try:
                    value = from_spec(cast(JsonValue, value), param.resource)
                except ValueError as e:
                    raise SpecError(
                        f"'{node.op}' param '{param.name}': {e}", path=path
                    ) from None
            if isinstance(value, Resource):
                out.extend(cast(Resource[Any], value).assets())
    return out


def _walk(
    node: NodeSpec, path: TreePathIndex
) -> Iterator[tuple[NodeSpec, TreePathIndex]]:
    yield node, path
    for i, child in enumerate(node.children):
        yield from _walk(child, path.add_child(i))
