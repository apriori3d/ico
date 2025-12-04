from collections.abc import Callable
from typing import overload

from typing_extensions import final

from apriori.ico.core.operator import I, IcoOperator, O, wrap_operator
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper


@final
class IcoPrinter(IcoRuntimeNode):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(runtime_name=name or "IcoPrinter")

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand | None:
        # The contract between LoggerTool and Logger is that LoggerTool send command to activate,
        # and logger accept that command and become ready to log messages.
        if (
            command.type == IcoRuntimeCommandType.activate
            and "IcoPrinter" not in command.targets
        ):
            return None

        return super().on_command(command)

    def __call__(self, message: str) -> None:
        if self._state != IcoRuntimeState.ready:
            raise RuntimeError(
                "Logger is not ready to log messages. Use activate command first."
            )
        self.bubble_event(
            IcoRuntimeEvent(
                type=IcoRuntimeEventType.print,
                meta={"message": message},
            )
        )


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


def printer(
    printer: IcoPrinter,
) -> Callable[[IcoOperator[I, O]], IcoOperator[I, O]]:
    def decorator(operator: IcoOperator[I, O]) -> IcoOperator[I, O]:
        return IcoRuntimeWrapper[I, O](
            operator,
            runtime_children=[printer],
        )

    return decorator
