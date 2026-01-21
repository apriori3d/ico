import time
from collections.abc import Iterator
from typing import ClassVar, final

from rich.progress import Progress, TaskID

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.progress import (
    IcoProgress,
    IcoProgressEvent,
)
from apriori.ico.core.runtime.tools.tool_node import (
    IcoTool,
)
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent


@final
class RichProgressTool(IcoTool):
    __slots__ = ("progress", "_tasks")
    node_types: ClassVar[set[type[IcoRuntimeNode]]] = {IcoProgress}

    progress: Progress
    _tasks: dict[int, TaskID]

    def __init__(self, progress: Progress):
        IcoTool.__init__(self)
        self.progress = progress
        self._tasks = {}

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoProgressEvent):
            task_id = self._tasks[event.trace.reverse()]
            task = self.progress.tasks[task_id]

            if task.finished:
                self.progress.reset(task_id)

            self.progress.advance(task_id, event.advance)

            # # Stop propagation after handling log event
            return None

        return super().on_event(event)


class WorkerFlow:
    name: str | None = None
    num_iters: int

    def __init__(self, num_iters: int = 10, name: str | None = None):
        self.name = name
        self.num_iters = num_iters

    def __call__(self) -> IcoOperator[int, int]:
        @operator()
        def double(x: int) -> int:
            time.sleep(0.1)
            res = x * 2
            return res

        @operator()
        def shift(x: int) -> int:
            res = x + 1
            return res

        progress = IcoProgress[int](total=self.num_iters, name=self.name)

        return IcoProcess[int](
            double | shift | progress,
            num_iterations=self.num_iters,
        )


if __name__ == "__main__":
    from apriori.ico.describe.describer import describe

    total = 10

    @source()
    def numbers() -> Iterator[int]:
        for i in range(total):
            time.sleep(1)
            yield i

    @sink()
    def print_result(x: int) -> None:
        pass

    progress = IcoProgress[int](total=total, name="Overall Progress")

    mp_process1 = MPAgent[int, int](WorkerFlow(name="Worker 1"))
    mp_process2 = MPAgent[int, int](WorkerFlow(name="Worker 2"))

    flow = numbers | (progress | mp_process1 | mp_process2).stream() | print_result
    flow.name = "Example Flow"

    with Progress() as progress:
        console = progress.console

        describe(flow, console=console)

        progress_tool = RichProgressTool(progress)

        shell = flow.shell().add_tool(progress_tool)
        shell.activate()
        describe(shell, console=console)

        # progress_tool.discover()
        # describe(flow, console=console)

        # runtime.run()

        # runtime.deactivate()
        # describe(flow, console=console)
