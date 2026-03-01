import time
from collections.abc import Iterator
from typing import final

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
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.runtime.tool import IcoTool
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.core.tree_utils import TreePathIndex
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent


@final
class RichProgressTool(IcoTool):
    """
    Rich Progress integration tool for ICO runtime system.

    Provides visual progress tracking using Rich library's Progress component.
    Automatically tracks IcoProgress nodes in the runtime tree and updates
    Rich progress bars based on IcoProgressEvent notifications.

    Architecture:
        - Registers IcoProgress nodes during runtime tree traversal
        - Creates Rich TaskID mappings for each progress node path
        - Updates progress bars on IcoProgressEvent reception
        - Handles task resets when progress nodes restart

    Runtime Integration:
        - Implements IcoTool interface for seamless runtime integration
        - Uses tree path indexing for progress node identification
        - Processes forward events to update visual progress indicators
        - Maintains progress state isolation between different runtime branches

    Example:
        ```python
        from rich.progress import Progress

        with Progress() as progress:
            progress_tool = RichProgressTool(progress)
            runtime = IcoRuntime(flow, tools=[progress_tool])
            runtime.run()  # Visual progress bars automatically appear
        ```

    Note:
        - Requires Rich library for progress visualization
        - Each IcoProgress node gets dedicated progress bar
        - Progress bars reset automatically on node restarts
        - Thread-safe through Rich Progress component
    """

    __slots__ = ("progress", "_tasks")

    progress: Progress
    _tasks: dict[TreePathIndex, TaskID]

    def __init__(self, progress: Progress):
        """
        Initialize Rich progress tool with existing Progress instance.

        Args:
            progress: Rich Progress instance for visual display
                     Must be active (within Progress context manager)

        State:
            - Creates empty task mapping dictionary
            - Stores reference to Rich Progress component
            - Inherits IcoTool runtime integration capabilities
        """
        super().__init__()
        self.progress = progress
        self._tasks = {}

    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None:
        """
        Register runtime node for progress tracking if applicable.

        Called during runtime tree traversal to identify IcoProgress nodes
        and create corresponding Rich progress bars.

        Args:
            node: Runtime node being registered
            path: Tree path index for node identification

        Behavior:
            - Only processes IcoProgress nodes (ignores others)
            - Creates Rich task with node name or auto-generated description
            - Uses node.total for progress bar maximum value
            - Stores TreePathIndex -> TaskID mapping for event handling

        Task Description:
            - Uses node.name if available
            - Falls back to "Progress {count}" format
            - Description appears in Rich progress bar display
        """
        if isinstance(node, IcoProgress):
            task_id = self.progress.add_task(
                description=node.name or f"Progress {len(self._tasks)}",
                total=node.total,
            )
            self._tasks[path] = task_id

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """
        Handle runtime events to update progress visualization.

        Processes IcoProgressEvent notifications to advance Rich progress bars.
        Maintains synchronization between ICO progress state and visual display.

        Args:
            event: Runtime event from ICO system

        Returns:
            None to stop event propagation after handling progress updates

        Process:
            1. Filters for IcoProgressEvent instances
            2. Resolves tree path from event trace
            3. Validates registered progress task exists
            4. Resets task if previously finished
            5. Advances progress by event.advance amount
            6. Stops event propagation

        Error Handling:
            - Raises RuntimeError for unregistered progress nodes
            - Indicates programming error in node registration

        Threading:
            - Safe through Rich Progress thread synchronization
            - Event processing serialized by ICO runtime
        """
        super().on_forward_event(event)

        if isinstance(event, IcoProgressEvent):
            path = event.trace.reverse()
            if path not in self._tasks:
                raise RuntimeError(
                    f"Received progress event for unregistered progress node at path {path}"
                )
            task_id = self._tasks[path]
            task = self.progress.tasks[task_id]

            if task.finished:
                self.progress.reset(task_id)

            self.progress.advance(task_id, event.advance)

            # # Stop propagation after handling log event
            return None


class WorkerFlow:
    """
    Example flow factory for main demo below.
    Must be defined at module level to avoid pickling issues with multiprocessing.
    """

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
    """
    Comprehensive demonstration of RichProgressTool with distributed computation.

    Example Architecture:
        - Multi-agent processing with independent progress tracking
        - Rich Progress visualization for real-time monitoring
        - ICO runtime coordination with tool integration
        - Stream processing with progress aggregation

    Flow Structure:
        numbers (source)
        ↓
        overall_progress
        ↓
        [mp_process1 | mp_process2].stream() (parallel workers)
        ↓
        print_result (sink)

    Progress Visualization:
        - "Overall Progress": tracks data through main flow
        - "Worker 1": tracks processing in first agent
        - "Worker 2": tracks processing in second agent
        - Rich console displays all progress bars simultaneously

    Execution Flow:
        1. Numbers generated with 1s intervals (simulated slow source)
        2. Overall progress tracks each number through main pipeline
        3. Stream distributes processing across two MPAgent workers
        4. Each worker shows independent progress for internal iterations
        5. Results collected and printed (discarded in demo)

    Key Features:
        - Distributed progress tracking across process boundaries
        - Rich console integration with flow description
        - Runtime state visualization before/during/after execution
        - Tool-based architecture for progress monitoring
    """
    from apriori.ico.describe.describer import describe

    total = 10

    @source()
    def numbers() -> Iterator[int]:
        """Generate sequence of integers with simulated delays.

        Produces numbers 0 through total-1 with 1-second intervals
        to simulate slow data source and demonstrate progress tracking.

        Returns:
            Iterator[int]: Sequential integers with timing delays

        Behavior:
            - Each number emission includes 1s sleep
            - Demonstrates progress tracking for slow sources
            - Total iterations known in advance for accurate progress bars
        """
        for i in range(total):
            time.sleep(1)
            yield i

    @sink()
    def print_result(x: int) -> None:
        """
        Terminal sink for processed results (discards output in demo).

        Args:
            x: Processed integer from worker pipeline

        Behavior:
            - Receives results from distributed worker processing
            - Discards output for demonstration purposes
            - Could be replaced with actual result handling
        """
        pass

    # Overall progress tracking for main data flow
    progress = IcoProgress[int](total=total, name="Overall Progress")

    # Distributed processing agents with individual progress tracking
    mp_process1 = MPAgent[int, int](WorkerFlow(name="Worker 1"))
    mp_process2 = MPAgent[int, int](WorkerFlow(name="Worker 2"))

    # Complete flow: source → progress → parallel workers → sink
    flow = numbers | (progress | mp_process1 | mp_process2).stream() | print_result
    flow.name = "Example Flow"

    with Progress() as progress:
        """Rich Progress context for visual progress monitoring."""
        console = progress.console

        # Display initial flow structure
        describe(flow, console=console)

        # Create and configure runtime with progress tool
        progress_tool = RichProgressTool(progress)
        runtime = IcoRuntime(flow, tools=[progress_tool])
        runtime.activate()

        # Display runtime tree with progress nodes
        describe(runtime, console=console)

        # Execute flow with live progress visualization
        runtime.run()

        # Cleanup and final state display
        runtime.deactivate()
        describe(flow, console=console)
