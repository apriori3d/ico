from __future__ import annotations

from typing import NamedTuple

from apriori.ico.core.meta.utils import format_ico_type
from apriori.ico.core.runtime.state import IcoRuntimeState


class IcoForm(NamedTuple):
    """
    A representation of the ICO form of an operator in DSL.
    Possible types are type[Any] or typing generics."""

    i: object
    c: object | None
    o: object

    def format(self) -> str:
        if self.c is None:
            return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"
        return f"{format_ico_type(self.i)}, {format_ico_type(self.c)} → {format_ico_type(self.o)}"

    @property
    def name(self) -> str:
        return self.format()


class IcoNodeMeta(NamedTuple):
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
    children: list[IcoNodeMeta] = list()

    runtime_name: str | None = None
    runtime_state: IcoRuntimeState | None = None
    runtime_children: list[IcoNodeMeta] = list()

    def add_runtime(
        self,
        *,
        runtime_name: str | None,
        runtime_state: IcoRuntimeState | None,
        runtime_children: list[IcoNodeMeta],
    ) -> IcoNodeMeta:
        return IcoNodeMeta(
            name=self.name,
            name_origin=self.name_origin,
            node_type_name=self.node_type_name,
            ico_form=self.ico_form,
            children=self.children,
            runtime_name=runtime_name,
            runtime_state=runtime_state,
            runtime_children=runtime_children,
        )

    # def update(self, *, ico_form: IcoForm, children: list[IcoNodeMeta]) -> IcoNodeMeta:
    #     return IcoNodeMeta(
    #         name=self.name,
    #         name_origin=self.name_origin,
    #         node_type_name=self.node_type_name,
    #         runtime_name=self.runtime_name,
    #         runtime_state=self.runtime_state,
    #         runtime_children=self.runtime_children,
    #         ico_form=ico_form,
    #         children=children,
    #     )

    # def __str__(self) -> str:
    #     return self.name

    # def traverse(self) -> Iterator[IcoFlowMeta]:
    #     yield self
    #     for c in self.children:
    #         yield from c.traverse()
