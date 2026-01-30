from collections.abc import Iterator
from typing import final

from rich.console import Console

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.runtime.runtime_wrapper import use_runtime
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
from apriori.ico.tools.printer.node import (
    IcoPrinter,
    IcoPrintEvent,
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
            return None  # Stop propagation after handling log event

        return super().on_event(event)


def create_agent_flow() -> IcoOperator[int, int]:
    # Create fresh instances to avoid sharing runtime connections with main flow
    printer = IcoPrinter()

    @use_runtime(printer)
    @operator()
    def agent_double(x: int) -> int:
        res = x * 2
        printer(f"Doubling {x} to get {res}")
        return res

    @use_runtime(printer)
    @operator()
    def agent_shift(x: int) -> int:
        res = x + 1
        printer(f"Shifting {x} to get {res}")
        return res

    return agent_double | agent_shift


if __name__ == "__main__":
    printer = IcoPrinter()

    @source()
    def numbers() -> Iterator[int]:
        yield from range(3)

    @use_runtime(printer)
    @sink()
    def print_result(x: int) -> None:
        printer(f"Sink received: {x}")

    flow = numbers | MPAgent(create_agent_flow).stream() | print_result
    flow.name = "Example Flow"

    flow.describe()

    console = Console()
    printer_tool = RichPrinterTool(console)

    runtime = IcoRuntime(flow, tools=[printer_tool])
    runtime.describe()
    runtime.activate().describe()

    runtime.run()

    runtime.deactivate().describe()
