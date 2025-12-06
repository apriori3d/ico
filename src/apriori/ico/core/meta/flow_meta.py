from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast, final, overload

from apriori.ico.core.chain import IcoChainOperator
from apriori.ico.core.iterate import IcoIterateOperator
from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form
from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper


@final
@dataclass(slots=True, frozen=True)
class IcoFlowMeta:
    """Meta-description of an ICO computational flow.

    Represents the **static structure** and **runtime state**
    of an ICO operator graph (without executing it).

    Each node captures:
      • its type (`NodeType`)
      • ICO form (I → C → O)
      • optional lifecycle and execution state
      • hierarchical composition of children
    """

    node_type: str
    name: str
    ico_form: IcoForm | None = None
    runtime_state: IcoRuntimeState | None = None
    children: list[IcoFlowMeta] = field(
        default_factory=lambda: cast(list[IcoFlowMeta], list)
    )

    def __str__(self) -> str:
        return self.name

    def traverse(self) -> Iterator[IcoFlowMeta]:
        yield self
        for c in self.children:
            yield from c.traverse()

    # ─── Factory helpers ───

    @overload
    @staticmethod
    def from_node(
        node: IcoNode,
        include_runtime: bool = False,
    ) -> IcoFlowMeta: ...

    @overload
    @staticmethod
    def from_node(
        node: IcoRuntimeNode,
    ) -> IcoFlowMeta: ...

    @staticmethod
    def from_node(
        node: IcoNode | IcoRuntimeNode,
        include_runtime: bool = False,
    ) -> IcoFlowMeta:
        """Recursively build an IcoFlow from an node tree."""

        if isinstance(node, IcoRuntimeNode):
            return _build_runtime_flow_meta(node)

        return _build_flow_meta(
            node,
            include_runtime=include_runtime,
        )


def _build_runtime_flow_meta(node: IcoRuntimeNode) -> IcoFlowMeta:
    return IcoFlowMeta(
        node_type=node.type_name,
        name=node.runtime_name,
        runtime_state=node.state,
        children=[_build_runtime_flow_meta(c) for c in node.runtime_children],
    )


def _build_flow_meta(
    node: IcoNode,
    include_runtime: bool = False,
) -> IcoFlowMeta:
    """Recursively build an IcoFlowMeta from a static node tree."""

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

    children: list[IcoFlowMeta] = [
        _build_flow_meta(
            c,
            include_runtime=include_runtime,
        )
        for c in node.children
    ]

    if isinstance(node, IcoChainOperator):
        name = " | ".join(c.name for c in children)
    elif isinstance(node, IcoIterateOperator):
        name = f"Iterate({children[0].name})"
    else:
        name = node.name

    if include_runtime and isinstance(node, IcoRuntimeNode):
        runtime_state = node.state
        children += [_build_runtime_flow_meta(c) for c in node.runtime_children]
    else:
        runtime_state = None

    node = cast(IcoNode, node)
    node_type = type(node).type_name

    return IcoFlowMeta(
        node_type=node_type,
        name=name,
        ico_form=infer_ico_form(node),
        runtime_state=runtime_state,
        children=children,
    )
