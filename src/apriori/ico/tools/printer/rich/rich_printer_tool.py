from collections.abc import Iterator
from typing import final

from rich.console import Console

from apriori.ico.core.operator import operator
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.shell import IcoShell
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.tools.printer.node import (
    IcoPrinter,
    IcoPrintEvent,
    use_printer,
)


@final
class RichPrinterTool(IcoRuntimeNode):
    _console: Console

    def __init__(self, console: Console):
        IcoRuntimeNode.__init__(self)
        self._console = console

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

    @use_printer(print)
    @operator()
    def double(x: int) -> int:
        res = x * 2
        print(f"Doubling {x} to get {res}")
        return res

    @use_printer(print)
    @operator()
    def shift(x: int) -> int:
        res = x + 1
        print(f"Shifting {x} to get {res}")
        return res

    @use_printer(print)
    @sink()
    def print_result(x: int) -> None:
        print(f"Sink received: {x}")

    flow = numbers | (double | shift).stream() | print_result
    flow.name = "Example Flow"

    flow.describe()

    console = Console()
    printer_tool = RichPrinterTool(console)

    shell = IcoShell(flow, tools=[printer_tool])
    shell.describe()

    # shell.activate()  # .describe()
    # printer_tool.register_nodes()  # .describe()

    # # flow.describe(show_runtime_nodes=True)

    # shell.run()

    # shell.deactivate()  # .describe()
