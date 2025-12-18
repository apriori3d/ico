import time
from collections.abc import Iterator
from typing import final

from rich.progress import Progress, TaskID

from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.process import IcoProcess
from apriori.ico.core.runtime.event import (
    IcoRuntimeEvent,
)
from apriori.ico.core.runtime.progress import (
    IcoProgress,
    IcoProgressEvent,
    IcoProgressRegistrationEvent,
)
from apriori.ico.core.runtime.tool import (
    IcoDiscovarableNode,
    IcoRegistrationEvent,
    IcoRuntimeTool,
)
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.describe.describe import describe
from apriori.ico.runtime.agent.mp_process.mp_agent import MPAgent


@final
class RichProgressTool(IcoRuntimeTool):
    _progress: Progress
    _tasks: dict[int, TaskID]

    def __init__(self, progress: Progress):
        IcoRuntimeTool.__init__(self)
        self._progress = progress
        self._tasks = {}

    def get_discoverable_node_types(self) -> set[type[IcoDiscovarableNode]]:
        return {IcoProgress}

    def get_registration_event_types(self) -> set[type[IcoRegistrationEvent]]:
        return {IcoProgressRegistrationEvent}

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        if isinstance(event, IcoProgressRegistrationEvent):
            node_task = self._progress.add_task(
                description=event.node_name or f"Progress {event.node_id}",
                total=event.total,
            )
            self._tasks[event.node_id] = node_task

            self._progress.print(
                f"Discovered progress node {event.node_name} with total={event.total}, id={event.node_id}"
            )
            # Stop propagation after handling log event
            return None

        if isinstance(event, IcoProgressEvent):
            task_id = self._tasks[event.node_id]
            task = self._progress.tasks[task_id]

            if task.finished:
                self._progress.reset(task_id)

            self._progress.advance(task_id, event.advance)

            # # Stop propagation after handling log event
            return None

        return super().on_event(event)


def create_item_flow() -> IcoOperator[int, int]:
    num_iters = 10

    @operator()
    def double(x: int) -> int:
        time.sleep(0.1)
        res = x * 2
        return res

    @operator()
    def shift(x: int) -> int:
        res = x + 1
        return res

    progress = IcoProgress[int](total=num_iters)

    return IcoProcess[int](
        double | shift | progress,
        num_iterations=num_iters,
    )


if __name__ == "__main__":
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

    mp_process1 = MPAgent[int, int](create_item_flow)
    mp_process2 = MPAgent[int, int](create_item_flow)

    # item_flow = create_item_flow()

    flow = numbers | (progress | mp_process1 | mp_process2).stream() | print_result
    flow.name = "Example Flow"
    describe(flow)
    describe(flow, show_runtime_nodes=True)

    with Progress() as progress:
        console = progress.console

        progress_tool = RichProgressTool(progress)

        runtime = flow.runtime().add_tool(progress_tool)
        runtime.activate()
        describe(flow, console=console)

        progress_tool.discover()
        describe(flow, console=console)

        runtime.run()

        runtime.deactivate()
        describe(flow, console=console)
