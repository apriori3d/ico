from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, final

from apriori.ico.core.meta.ico_form import IcoForm
from apriori.ico.core.runtime.execution import IcoExecutionState, SupportsIcoExecution
from apriori.ico.core.runtime.types import (
    IcoRuntimeOperatorProtocol,
    IcoRuntimeStateType,
)
from apriori.ico.core.types import IcoNodeType, IcoOperatorProtocol


@final
@dataclass(slots=True)
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

    node_type: IcoNodeType
    ico_form: IcoForm
    name: str
    state: IcoRuntimeStateType | None = None
    exec_state: IcoExecutionState | None = None
    children: list[IcoFlowMeta] = field(default_factory=list)

    def __str__(self) -> str:
        return self.name

    def traverse(self) -> Iterator[IcoFlowMeta]:
        yield self
        for c in self.children:
            yield from c.traverse()

    # ─── Factory helpers ───

    @staticmethod
    def from_operator(operator: IcoOperatorProtocol[Any, Any]) -> IcoFlowMeta:
        """Recursively build an IcoFlow from an operator tree."""
        state = (
            operator.state if isinstance(operator, IcoRuntimeOperatorProtocol) else None
        )
        exec_state = (
            operator.exec_state if isinstance(operator, SupportsIcoExecution) else None
        )
        return IcoFlowMeta(
            name=operator.name,
            node_type=operator.node_type,
            ico_form=IcoForm.from_operator(operator),
            state=state,
            exec_state=exec_state,
            children=[IcoFlowMeta.from_operator(c) for c in operator.children],
        )
