from collections.abc import Iterator
from typing import final

from rich.console import Console

from apriori.ico.core.operator import operator
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.tool import (
    IcoDiscovarableNode,
    IcoRegistrationEvent,
    IcoRuntimeTool,
)
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.tools.printer.node import (
    IcoPrinter,
    IcoPrinterRegistrationEvent,
    IcoPrintEvent,
    printer,
)


@final
class RichPrinterTool(IcoRuntimeTool):
    _console: Console

    def __init__(self, console: Console):
        IcoRuntimeNode.__init__(self)
        self._console = console

    def get_discoverable_node_types(self) -> set[type[IcoDiscovarableNode]]:
        return {IcoPrinter}

    def get_registration_event_types(self) -> set[type[IcoRegistrationEvent]]:
        return {IcoPrinterRegistrationEvent}

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
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

    flow = numbers | (double | shift).stream() | print_result
    flow.name = "Example Flow"

    flow.describe()
    flow.describe(include_runtime=True)

    console = Console()
    printer_tool = RichPrinterTool(console)

    runtime = flow.runtime().add_tool(printer_tool)
    runtime.describe()
    runtime.activate().describe()
    printer_tool.discover().describe()

    flow.describe(include_runtime=True)

    runtime.run()

    runtime.deactivate().describe()
