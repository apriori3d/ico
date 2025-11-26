from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto
from typing import final

from apriori.ico.core.async_operator import IcoAsyncOperator
from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.chain import IcoChainOperator
from apriori.ico.core.iterate import IcoIterateOperator
from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form
from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.sink import IcoSink
from apriori.ico.core.source import IcoSource
from apriori.ico.core.stream import IcoStream


class IcoNodeType(Enum):
    """Enumeration of ICO node types."""

    unknown = auto()
    operator = auto()
    async_operator = auto()
    iterate = auto()
    chain = auto()
    pipeline = auto()
    process = auto()
    stream = auto()
    async_stream = auto()
    source = auto()
    sink = auto()


CLASS_TO_NODE_TYPE: dict[type[IcoNode], IcoNodeType] = {
    IcoOperator: IcoNodeType.operator,
    IcoAsyncOperator: IcoNodeType.async_operator,
    IcoIterateOperator: IcoNodeType.iterate,
    IcoChainOperator: IcoNodeType.chain,
    IcoPipeline: IcoNodeType.pipeline,
    IcoProcess: IcoNodeType.process,
    IcoStream: IcoNodeType.stream,
    IcoAsyncStream: IcoNodeType.async_stream,
    IcoSource: IcoNodeType.source,
    IcoSink: IcoNodeType.sink,
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
    ico_form: IcoForm
    name: str
    runtime_state: IcoRuntimeState | None
    children: list[IcoFlowMeta]

    def __init__(
        self,
        *,
        node_type: IcoNodeType,
        name: str,
        ico_form: IcoForm,
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

    @staticmethod
    def from_node(node: IcoNode) -> IcoFlowMeta:
        """Recursively build an IcoFlow from an node tree."""
        runtime_state = node.state if isinstance(node, IcoRuntimeNode) else None
        ico_form = infer_ico_form(node)

        return IcoFlowMeta(
            node_type=CLASS_TO_NODE_TYPE.get(type(node), IcoNodeType.unknown),
            name=node.name,
            ico_form=ico_form,
            runtime_state=runtime_state,
            children=[IcoFlowMeta.from_node(c) for c in node.children],
        )
