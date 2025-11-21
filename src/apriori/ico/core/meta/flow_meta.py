from __future__ import annotations

from collections.abc import Iterator
from typing import final

from apriori.ico.core.meta.ico_form import IcoForm
from apriori.ico.core.node import IcoNodeType, IcoOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeOperatorProtocol,
    IcoRuntimeStateType,
)


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
        "node_type",
        "ico_form",
        "runtime_state",
        "children",
    )

    node_type: IcoNodeType
    ico_form: IcoForm
    name: str
    runtime_state: IcoRuntimeStateType | None
    children: list[IcoFlowMeta]

    def __init__(
        self,
        *,
        name: str,
        node_type: IcoNodeType,
        ico_form: IcoForm,
        runtime_state: IcoRuntimeStateType | None = None,
        children: list[IcoFlowMeta] | None = None,
    ) -> None:
        self.name = name
        self.node_type = node_type
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
    def from_operator(operator: IcoNode) -> IcoFlowMeta:
        """Recursively build an IcoFlow from an operator tree."""
        runtime_state = (
            operator.state if isinstance(operator, IcoRuntimeOperatorProtocol) else None
        )
        ico_form = IcoForm.from_operator(operator)

        if not isinstance(operator, IcoOperator):
            return IcoFlowMeta(
                name=operator.name,
                node_type=IcoNodeType.unknown,
                ico_form=ico_form,
                runtime_state=runtime_state,
            )

        return IcoFlowMeta(
            name=operator.name,
            node_type=operator.node_type,
            ico_form=ico_form,
            runtime_state=runtime_state,
            children=[IcoFlowMeta.from_operator(c) for c in operator.children],
        )
