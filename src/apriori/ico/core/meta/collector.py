from __future__ import annotations

from dataclasses import replace
from types import FunctionType
from typing import cast, overload

from apriori.ico.core.meta.ico_form_inference import infer_ico_form
from apriori.ico.core.meta.node_meta import IcoNodeMeta
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource


@overload
@staticmethod
def collect_meta(
    node: IcoNode,
    include_runtime: bool = False,
) -> IcoNodeMeta: ...


@overload
@staticmethod
def collect_meta(node: IcoRuntimeNode) -> IcoNodeMeta: ...


@staticmethod
def collect_meta(
    node: IcoNode | IcoRuntimeNode, include_runtime: bool = False
) -> IcoNodeMeta:
    """Recursively build an IcoFlow from an node tree."""

    if isinstance(node, IcoRuntimeNode):
        return _build_runtime_flow_meta(node)

    return _build_flow_meta(
        node,
        include_runtime=include_runtime,
    )


#  ──────────── Computation flow builder ────────────


def _build_flow_meta(
    node: IcoNode,
    include_runtime: bool = False,
) -> IcoNodeMeta:
    """Recursively build an IcoFlowMeta from a static node tree."""

    # Special case: runtime wrapper nodes
    if isinstance(node, IcoRuntimeWrapper) and not include_runtime:
        # Flatten wrapper nodes by promoting their single child
        assert (
            len(node.children) == 1
        ), "Runtime wrapper must have exactly one child: wrapped operator."

        wrapped_node = _build_flow_meta(
            node.children[0],
            include_runtime=False,
        )
        return wrapped_node

    # Build children recursively

    children: list[IcoNodeMeta] = [
        _build_flow_meta(
            c,
            include_runtime=include_runtime,
        )
        for c in node.children
    ]

    node = cast(IcoNode, node)
    ico_form = infer_ico_form(node)

    # Determine node name
    name = node.name
    name_origin = "user" if name is not None else None

    if (
        name is None
        and (isinstance(node, IcoOperator) and node.type_name == "Operator")
        or isinstance(node, IcoSource | IcoSink | IcoRuntimeWrapper)
    ):
        name = extract_fn_name(node.original_fn)  # type: ignore
        if name is not None:
            name_origin = "fn"

    if name is None:
        name = extract_class_name(node)  # type: ignore
        if name is not None:
            name_origin = "class"

    if name is None or name_origin is None:
        name = node.type_name
        name_origin = "type_name"

    node_meta = IcoNodeMeta(
        node_type_name=node.type_name,
        name=name,
        name_origin=name_origin,
        ico_form=ico_form,
        runtime_state=None,
        children=children,
    )
    if not include_runtime or not isinstance(node, IcoRuntimeNode):
        return node_meta

    runtime_node = cast(IcoRuntimeNode, node)
    runtime_meta = _build_runtime_flow_meta(runtime_node)
    updated = node_meta.add_runtime(
        runtime_name=runtime_meta.runtime_name,
        runtime_state=runtime_meta.runtime_state,
        runtime_children=runtime_meta.runtime_children,
    )
    return updated


# ──────────── Runtime builder ────────────


def _build_runtime_flow_meta(node: IcoRuntimeNode) -> IcoNodeMeta:
    children = [_build_runtime_flow_meta(c) for c in node.runtime_children]
    name = node.runtime_name
    name_origin = "user" if name is not None else None

    if name is None:
        name = extract_class_name(node)  # type: ignore
        if name is not None:
            name_origin = "class"

    if name is None or name_origin is None:
        name = node.type_name
        name_origin = "type_name"

    return IcoNodeMeta(
        node_type_name=node.type_name,
        name=name,
        name_origin=name_origin,
        runtime_name=name,
        runtime_state=replace(node.state),
        runtime_children=children,
    )


# ──────────── Helpers  ────────────


def extract_fn_name(fn: object) -> str | None:
    cls = getattr(fn, "__class__", None)
    if cls is FunctionType:
        return getattr(fn, "__name__", None)
    return None


def extract_class_name(obj: object) -> str | None:
    cls = getattr(obj, "__class__", None)
    if cls:
        return getattr(cls, "__name__", None)
    return None
