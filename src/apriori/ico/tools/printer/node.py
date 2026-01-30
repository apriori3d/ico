from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from typing_extensions import final

from apriori.ico.core.operator import I, IcoOperator, O, wrap_operator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.tree_utils import TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoPrintEvent(IcoRuntimeEvent):
    message: str

    @staticmethod
    def create(message: str) -> IcoPrintEvent:
        return IcoPrintEvent(message=message, trace=TreePathIndex())


@final
class IcoPrinter(IcoRuntimeNode):
    def __call__(self, message: str) -> None:
        self.state_model.running()
        self.bubble_event(IcoPrintEvent.create(message=message))
        self.state_model.ready()


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


@overload
def printable(fn: IcoOperator[I, O], printer: IcoPrinter) -> IcoOperator[I, O]: ...


@overload
def printable(fn: Callable[[I], O], printer: IcoPrinter) -> IcoOperator[I, O]: ...


def printable(
    fn: Callable[[I], O] | IcoOperator[I, O], printer: IcoPrinter
) -> IcoOperator[I, O]:
    """
    Wrap raw callable into IcoOperator only when necessary.
    Ensures type inference for both mypy and pyright.
    """
    op = wrap_operator(fn)
    return IcoRuntimeWrapper[I, O](
        op,
        runtime_children=[printer],
    )
