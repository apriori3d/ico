from __future__ import annotations

from collections.abc import Iterator
from typing import final

from apriori.ico.core.meta.ico_form import IcoForm, infer_ico_form
from apriori.ico.core.node import IcoNode
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState


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
        "name",
        "ico_form",
        "runtime_state",
        "children",
    )

    ico_form: IcoForm
    name: str
    runtime_state: IcoRuntimeState | None
    children: list[IcoFlowMeta]

    def __init__(
        self,
        *,
        name: str,
        ico_form: IcoForm,
        runtime_state: IcoRuntimeState | None = None,
        children: list[IcoFlowMeta] | None = None,
    ) -> None:
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
            name=node.name,
            ico_form=ico_form,
            runtime_state=runtime_state,
            children=[IcoFlowMeta.from_node(c) for c in node.children],
        )
