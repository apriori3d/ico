from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto
from typing import final, overload

from apriori.ico.core.async_operator import IcoAsyncOperator
from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChainOperator
from apriori.ico.core.context_operator import IcoContextOperator
from apriori.ico.core.context_pipeline import IcoContextPipeline
from apriori.ico.core.context_stream import IcoEpoch
from apriori.ico.core.iterate import IcoIterateOperator
from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream


class IcoNodeType(Enum):
    """Enumeration of ICO node types."""

    unknown = auto()
    operator = auto()
    context_operator = auto()
    async_operator = auto()
    iterator = auto()
    chain = auto()
    pipeline = auto()
    context_pipeline = auto()
    streamline = auto()
    process = auto()
    stream = auto()
    async_stream = auto()
    source = auto()
    sink = auto()
    epoch = auto()

    # Runtime node represents a node during execution
    runtime_node = auto()
    runtime_wrapper = auto()
    contour = auto()
    agent = auto()


CLASS_TO_NODE_TYPE: dict[type, IcoNodeType] = {
    IcoIterateOperator: IcoNodeType.iterator,
    IcoChainOperator: IcoNodeType.chain,
    IcoPipeline: IcoNodeType.pipeline,
    IcoContextPipeline: IcoNodeType.context_pipeline,
    IcoProcess: IcoNodeType.process,
    IcoStream: IcoNodeType.stream,
    IcoAsyncStream: IcoNodeType.async_stream,
    IcoSource: IcoNodeType.source,
    IcoSink: IcoNodeType.sink,
    IcoEpoch: IcoNodeType.epoch,
    IcoAsyncOperator: IcoNodeType.async_operator,
    IcoRuntimeContour: IcoNodeType.contour,
    IcoRuntimeWrapper: IcoNodeType.runtime_wrapper,
    IcoRuntimeNode: IcoNodeType.runtime_node,
    IcoOperator: IcoNodeType.operator,
    IcoContextOperator: IcoNodeType.context_operator,
}


@final
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

    __slots__ = (
        "node_type",
        "name",
        "ico_form",
        "runtime_state",
        "children",
    )

    node_type: IcoNodeType
    ico_form: IcoForm | None
    name: str
    runtime_state: IcoRuntimeState | None
    children: list[IcoFlowMeta]

    def __init__(
        self,
        *,
        node_type: IcoNodeType,
        name: str,
        ico_form: IcoForm | None = None,
        runtime_state: IcoRuntimeState | None = None,
        children: list[IcoFlowMeta] | None = None,
    ) -> None:
        self.node_type = node_type
        self.name = name
        self.ico_form = ico_form
        self.runtime_state = runtime_state
        self.children = children or []

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
        node_type=infer_node_type(node),
        name=node.runtime_name,
        runtime_state=node.state,
        children=[_build_runtime_flow_meta(c) for c in node.runtime_children],
    )


def _build_flow_meta(
    node: IcoNode,
    include_runtime: bool = False,
) -> IcoFlowMeta:
    """Recursively build an IcoFlowMeta from a static node tree."""

    node_type = infer_node_type(node)

    if node_type == IcoNodeType.runtime_wrapper and not include_runtime:
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

    if node_type == IcoNodeType.chain:
        name = " | ".join(c.name for c in children)
    elif node_type == IcoNodeType.iterator:
        name = f"iterate({children[0].name})"
    else:
        name = node.name

    if include_runtime and isinstance(node, IcoRuntimeNode):
        runtime_state = node.state
        children += [_build_runtime_flow_meta(c) for c in node.runtime_children]
    else:
        runtime_state = None

    return IcoFlowMeta(
        node_type=node_type,
        name=name,
        ico_form=infer_ico_form(node),
        runtime_state=runtime_state,
        children=children,
    )


def infer_node_type(node: IcoNode | IcoRuntimeNode) -> IcoNodeType:
    """Infer the node type based on its class."""
    for cls, node_type in CLASS_TO_NODE_TYPE.items():
        if isinstance(node, cls):
            return node_type
    return IcoNodeType.unknown
