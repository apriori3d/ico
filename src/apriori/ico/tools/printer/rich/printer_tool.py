from collections.abc import Iterator
from typing import final

from rich.console import Console

from apriori.ico.core.operator import operator
from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.discovery import IcoDiscoveryCommand, IcoDiscoveryEvent
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.node import IcoRuntimeNode, IcoRuntimeState
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.tools.printer.node import (
    IcoPrinter,
    IcoPrintEvent,
    printer,
)


@final
class RichPrinterTool(IcoRuntimeNode):
    _console: Console

    def __init__(self, console: Console):
        IcoRuntimeNode.__init__(self)
        self._console = console

    def on_command(self, command: IcoRuntimeCommand) -> IcoRuntimeCommand:
        if isinstance(command, IcoActivateCommand):
            if self.state == IcoRuntimeState.inactive:
                # Discover all printers to confirm activation
                self.broadcast_command(IcoDiscoveryCommand(node_types={IcoPrinter}))
            elif self.state != IcoRuntimeState.ready:
                raise RuntimeError(
                    "PrinterTool can only be activated from inactive state."
                )

        return super().on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoDiscoveryEvent):
            self._console.print(f"Discovered {event.node_name}, id={event.node_id}")
            # Stop propagation after handling log event
            return None

        if isinstance(event, IcoPrintEvent):
            self._console.print(event.message)
            # Stop propagation after handling log event
            return None

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
    flow.describe(include_runtime=True)

    console = Console()
    printer_tool = RichPrinterTool(console)

    runtime = flow.runtime().add_tool(printer_tool)
    runtime.describe()
    runtime.activate().describe()
    runtime.discover(IcoPrinter).describe()

    runtime.run()

    runtime.deactivate().describe()
