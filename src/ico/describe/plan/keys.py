"""Keys of plan nodes: one definition of the plan's tree, shared by the
plan renderers (``plan_view(...)["keys"]``) and by tools that report per
node, such as the profiler.

The plan's tree is the flow's tree with two changes:

- an ``IcoAsyncStream`` whose pool was built from a factory shows its first
  member only: all members have the key of the first;
- a node with a remote flow (a worker) has that flow as its last child,
  when remote flows are expanded.

The root's key is ``0``; the ``i``-th child of ``k`` is ``k.i``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

from ico.core.async_stream import IcoAsyncStream
from ico.core.node import HasRemoteFlow, IcoNodeProtocol
from ico.core.tree_utils import TreePathIndex

ROOT_KEY = "0"


def child_key(key: str, index: int) -> str:
    return f"{key}.{index}"


def path_key(key: str, path: TreePathIndex) -> str:
    """The key of the node at ``path`` below the node keyed ``key``."""
    return "".join([key, *(f".{i}" for i in path.path_index)])


def is_factory_pool(node: IcoNodeProtocol) -> bool:
    return (
        isinstance(node, IcoAsyncStream)
        and cast(IcoAsyncStream[Any, Any], node).pool_from_factory
    )


def plan_children(
    node: IcoNodeProtocol, *, expand_remote_flows: bool
) -> Sequence[IcoNodeProtocol]:
    """The children of a node in the plan.

    Expanding a remote flow calls the node's flow factory, so every call
    builds a new remote flow.
    """
    if is_factory_pool(node):
        return [node.children[0]]
    children = list(node.children)
    if expand_remote_flows and isinstance(node, HasRemoteFlow):
        children.append(node.get_remote_flow_factory()())
    return children


def remote_flow_key(key: str, node: IcoNodeProtocol) -> str:
    """The key of the remote flow of a node keyed ``key`` (its last child in
    the plan with remote flows), without building the flow."""
    return child_key(key, len(plan_children(node, expand_remote_flows=False)))


@dataclass(frozen=True, slots=True)
class PlanNode:
    key: str
    node: IcoNodeProtocol
    parent: PlanNode | None


def plan_nodes(
    flow: IcoNodeProtocol,
    *,
    expand_remote_flows: bool = False,
    root_key: str = ROOT_KEY,
) -> Iterator[PlanNode]:
    """Every node of a flow with its plan key, in plan order (pre-order).

    Unlike the plan, all members of a factory pool are yielded, each with
    the key of the first member, so that a tool can reach every member. A
    node used at several places is yielded at each of them. ``root_key``
    keys a flow that is part of a larger plan (a worker's remote flow).
    """

    def walk(
        node: IcoNodeProtocol, key: str, parent: PlanNode | None
    ) -> Iterator[PlanNode]:
        here = PlanNode(key, node, parent)
        yield here
        if is_factory_pool(node):
            for member in node.children:
                yield from walk(member, child_key(key, 0), here)
            return
        children = plan_children(node, expand_remote_flows=expand_remote_flows)
        for i, child in enumerate(children):
            yield from walk(child, child_key(key, i), here)

    yield from walk(flow, root_key, None)
