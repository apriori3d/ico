from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from typing_extensions import final

from apriori.ico.core.operator import I, IcoOperator, O, wrap_operator
from apriori.ico.core.runtime.discovery import IcoDiscovarableNode
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper


@final
@dataclass(slots=True, frozen=True)
class IcoPrintEvent(IcoRuntimeEvent):
    node_id: int
    message: str


@final
class IcoPrinter(IcoDiscovarableNode):
    def __init__(self, *, name: str | None = None, strict: bool = False) -> None:
        super().__init__(runtime_name=name)
        self.strict = strict

    def __call__(self, message: str) -> None:
        if self.strict:
            self._ensure_is_ready()
            assert self.registered_id is not None
        elif self.registered_id is None:
            return

        self.bubble_event(IcoPrintEvent(node_id=self.registered_id, message=message))


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def printer(
    printer: IcoPrinter,
) -> Callable[[IcoOperator[I, O]], IcoOperator[I, O]]:
    def decorator(operator: IcoOperator[I, O]) -> IcoOperator[I, O]:
        return IcoRuntimeWrapper[I, O](
            operator,
            runtime_children=[printer],
        )

    return decorator


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
