from __future__ import annotations

from typing import NamedTuple

from apriori.ico.core.meta.ico_form import IcoForm


class IcoFlowMeta(NamedTuple):
    """Meta-description of an ICO computational flow.

    Represents the **static structure** and **runtime state**
    of an ICO operator graph (without executing it).

    Each node captures:
      • its type (`NodeType`)
      • ICO form (I → C → O)
      • optional lifecycle and execution state
      • hierarchical composition of children
    """

    name: str
    name_origin: str
    node_type_name: str
    ico_form: IcoForm | None = None
    children: list[IcoFlowMeta] = list()

    runtime_name: str | None = None
    runtime_state: str | None = None
    runtime_children: list[IcoFlowMeta] = list()

    def update(self, *, ico_form: IcoForm, children: list[IcoFlowMeta]) -> IcoFlowMeta:
        return IcoFlowMeta(
            name=self.name,
            name_origin=self.name_origin,
            node_type_name=self.node_type_name,
            runtime_name=self.runtime_name,
            runtime_state=self.runtime_state,
            runtime_children=self.runtime_children,
            ico_form=ico_form,
            children=children,
        )

    # def __str__(self) -> str:
    #     return self.name

    # def traverse(self) -> Iterator[IcoFlowMeta]:
    #     yield self
    #     for c in self.children:
    #         yield from c.traverse()
