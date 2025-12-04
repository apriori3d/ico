from collections.abc import Iterator
from typing import final

from rich.console import Console

from apriori.ico.core.operator import operator
from apriori.ico.core.runtime.command import IcoRuntimeCommand, IcoRuntimeCommandType
from apriori.ico.core.runtime.event import IcoRuntimeEvent, IcoRuntimeEventType
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.tools.printer.printer_node import (
    IcoPrinter,
    printer,
)


@final
class RichPrinterTool(IcoRuntimeNode):
    _console: Console

    def __init__(self, console: Console):
        super().__init__(runtime_name="RichPrinterTool")
        self._console = console

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand | None:
        # Establish contract between PrinterTool and Printer that informs the Printer to activate.
        if command.type == IcoRuntimeCommandType.activate:
            command = command.add_target("IcoPrinter")

        return super().on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if event.type == IcoRuntimeEventType.print:
            self._console.print(event.meta.get("message", ""))
            return None  # Stop propagation after handling log event

        return super().on_event(event)


if __name__ == "__main__":
    print = IcoPrinter()

    @source()
    def numbers() -> Iterator[int]:
        yield from range(3)

    @printer(print)
    @operator()
    def double(x: int) -> int:
        res = x * 2
        print(f"Doubling {x} to get {res}")
        return res

    @printer(print)
    @operator()
    def shift(x: int) -> int:
        res = x + 1
        print(f"Shifting {x} to get {res}")
        return res

    @printer(print)
    @sink()
    def print_result(x: int) -> None:
        print(f"Sink received: {x}")

    flow = numbers | (double | shift).iterate() | print_result
    flow.describe()

    console = Console()
    printer_tool = RichPrinterTool(console)

    runtime = flow.runtime().add_tool(printer_tool)
    runtime.activate().describe()

    runtime.run()

    runtime.deactivate().describe()
