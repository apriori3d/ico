from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from typing_extensions import final

from apriori.ico.core.operator import I, IcoOperator, O, wrap_operator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.runtime.tools.tool_node import IcoRegistrationEvent, IcoToolNode


@final
@dataclass(slots=True, frozen=True)
class IcoPrinterRegistrationEvent(IcoRegistrationEvent):
    pass


@final
@dataclass(slots=True, frozen=True)
class IcoPrintEvent(IcoRuntimeEvent):
    node_id: int
    message: str


@final
class IcoPrinter(IcoToolNode):
    def __call__(self, message: str) -> None:
        self.state_model.running()
        assert self.registered_id is not None

        self.bubble_event(IcoPrintEvent(node_id=self.registered_id, message=message))
        self.state_model.ready()


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def use_printer(
    printer: IcoPrinter,
) -> Callable[[IcoOperator[I, O]], IcoOperator[I, O]]:
    def decorator(operator: IcoOperator[I, O]) -> IcoOperator[I, O]:
        if isinstance(operator, IcoRuntimeWrapper):
            operator.runtime_children.append(printer)
            return operator

        return IcoRuntimeWrapper[I, O](
            operator,
            runtime_children=[printer],
            name=operator.name,
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
