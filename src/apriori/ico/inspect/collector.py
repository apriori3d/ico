from __future__ import annotations

from dataclasses import replace
from typing import cast

from apriori.ico.core.meta.inspect.signature import infer_signature
from apriori.ico.core.meta.meta import IcoNodeMeta, IcoRuntimeNodeMeta
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource

# ──────────── Flow Nodes meta collector ────────────


def collect_meta(
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

        wrapped_node = collect_meta(
            node.children[0],
            include_runtime=False,
        )
        return wrapped_node

    # Build children recursively

    children: list[IcoNodeMeta] = [
        collect_meta(c, include_runtime=include_runtime) for c in node.children
    ]

    node = cast(IcoNode, node)
    ico_form = infer_signature(node)

    # Determine node name

    name = node.name
    name_origin = "direct" if name is not None else None

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
        type_name=node.type_name,
        name=name,
        name_origin=name_origin,
        ico_form=ico_form,
        children=children,
        runtime=None,
    )
    if not include_runtime or not isinstance(node, IcoRuntimeNode):
        return node_meta

    # Add runtime meta if applicable
    runtime_node = cast(IcoRuntimeNode, node)
    runtime_meta = collect_runtime_meta(runtime_node)
    updated = node_meta.add_runtime(runtime_meta)
    return updated


# ──────────── Runtime Nodes meta collector ────────────


def collect_runtime_meta(node: IcoRuntimeNode) -> IcoRuntimeNodeMeta:
    name = node.runtime_name
    name_origin = "direct" if name is not None else None

    if name is None:
        name = extract_class_name(node)  # type: ignore
        if name is not None:
            name_origin = "class"

    if name is None or name_origin is None:
        name = node.runtime_type_name
        name_origin = "type_name"

    children = [collect_runtime_meta(c) for c in node.runtime_children]

    return IcoRuntimeNodeMeta(
        name=name,
        name_origin=name_origin,
        type_name=node.runtime_type_name,
        state=replace(node.state),
        children=children,
    )
