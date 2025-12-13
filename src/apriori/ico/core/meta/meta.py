from __future__ import annotations

from typing import NamedTuple

from apriori.ico.core.meta.utils import format_ico_type
from apriori.ico.core.runtime.state import IcoRuntimeState

# ────────────────────────────────────────────────
# Signature descriptions
# ────────────────────────────────────────────────


class IcoSignature(NamedTuple):
    """
    A representation of the ICO form of an operator in DSL.
    Possible types are type[Any] or typing generics."""

    i: object
    c: object | None
    o: object

    def format(self) -> str:
        if self.c is None or self.c is type(None):
            return f"{format_ico_type(self.i)} → {format_ico_type(self.o)}"
        return f"{format_ico_type(self.i)}, {format_ico_type(self.c)} → {format_ico_type(self.o)}"

    @property
    def name(self) -> str:
        return self.format()


# ────────────────────────────────────────────────
# Meta-descriptions
# ────────────────────────────────────────────────


class IcoRuntimeNodeMeta(NamedTuple):
    name: str
    name_origin: str
    type_name: str
    state: IcoRuntimeState
    children: list[IcoRuntimeNodeMeta]


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
    type_name: str
    ico_form: IcoSignature
    children: list[IcoNodeMeta] = list()

    runtime: IcoRuntimeNodeMeta | None = None

    def add_runtime(self, runtime_meta: IcoRuntimeNodeMeta) -> IcoNodeMeta:
        return IcoNodeMeta(
            name=self.name,
            name_origin=self.name_origin,
            type_name=self.type_name,
            ico_form=self.ico_form,
            children=self.children,
            runtime=runtime_meta,
        )
