from __future__ import annotations

import contextlib
import itertools
import threading
from collections.abc import Sequence
from dataclasses import replace

from ico.core.node import HasRemoteFlow, IcoNodeProtocol
from ico.core.operator import IcoOperatorProtocol
from ico.core.plan_keys import ROOT_KEY, assign_plan_keys, plan_nodes, remote_flow_key
from ico.core.runtime.cancel import IcoCancelCommand, IcoCancelToken, cancellable
from ico.core.runtime.command import IcoRuntimeCommand
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import (
    IcoRuntimeNode,
)
from ico.core.runtime.sample import (
    RECENT_INTERVAL,
    IcoRecentCommand,
    IcoSampleCommand,
    keep_recent,
    start_sampling,
)
from ico.core.runtime.state import BaseStateModel
from ico.core.runtime.tool import IcoRuntimeTool
from ico.core.runtime.toolbox import IcoRuntimeToolBox
from ico.core.runtime.utils import discover_and_connect_runtime_nodes

# ────────────────────────────────────────────────
# Main runtime execution coordinator
# ────────────────────────────────────────────────


class IcoRuntime(IcoRuntimeNode):
    """Main runtime coordinator for complete ICO computation flows.

    IcoRuntime provides the top-level execution context for ICO computation flows,
    coordinating between computation logic and runtime infrastructure. It encapsulates
    complete flows (Source → Operators → Sink) and manages their execution lifecycle,
    tool integration, and runtime tree coordination.

    Runtime Coordination Architecture:
        • Flow Encapsulation: Wraps complete computation flows in runtime context
        • Lifecycle Management: Controls flow execution states and transitions
        • Tool Integration: Provides comprehensive tool management and event forwarding
        • Runtime Tree Root: Acts as root node for runtime hierarchy
        • Exception Handling: Centralized error handling with fault state management

    ICO Flow Integration:
        IcoRuntime bridges the separation between computation flows and runtime tree:

        **Computation Flow**: Source → [Operators] → Sink (data transformations)
        **Runtime Tree**: Runtime → ToolBox → Tools (execution monitoring)
        **Integration**: Runtime discovers and connects all computation nodes to runtime tree

    Flow Closure Requirements:
        • **Complete Flow**: Must start with Source (None → Iterable[O])
        • **Proper Termination**: Must end with Sink (Iterable[I] → None)
        • **Closed Loop**: Forms complete () → () execution closure
        • **Type Safety**: Maintains type safety through Generic flow composition

    Runtime Tree Hierarchy:
        ```
        IcoRuntime (root)
        ├── ToolBox (tool coordinator)
        │   ├── ProgressTool (progress monitoring)
        │   ├── LoggingTool (execution logging)
        │   └── DebugTool (runtime inspection)
        ├── Source (computation entry)
        ├── Operators (computation logic)
        └── Sink (computation exit)
        ```

    Tool Integration System:
        • **Managed Tools**: Tools registered through ToolBox for full runtime integration
        • **Event Distribution**: All runtime events forwarded to the tools
        • **Tool Lifecycle**: Automatic tool management through runtime lifecycle

    Execution Lifecycle:
        1. **Initialization**: Runtime tree discovery and connection
        2. **Activation**: State transition to ready (optional explicit activation)
        3. **Execution**: Flow execution with runtime coordination
        4. **Monitoring**: Real-time event forwarding to tools
        5. **Completion**: State transition back to ready or fault on exception

    State Management Integration:
        • **Ready State**: Runtime prepared for execution
        • **Running State**: Active flow execution in progress
        • **Fault State**: Exception condition requiring intervention
        • **Idle State**: Runtime inactive and available for cleanup

    Event Forwarding Architecture:
        All runtime events bubble through IcoRuntime and get forwarded to
        the ToolBox, which distributes them to all registered tools (a
        temporary subscriber is added and removed as a tool).
        This enables comprehensive monitoring without affecting computation flow.

    Runtime Discovery Process:
        IcoRuntime automatically discovers all computation nodes and integrates
        them into runtime tree using discover_and_connect_runtime_nodes():
        • Traverses computation flow structure
        • Identifies all IcoOperator instances
        • Establishes runtime parent-child relationships
        • Enables event bubbling from computation to runtime

    Example - Runtime with Progress Tracking:
        ```python
        from ico.core import IcoSource, IcoSink, IcoOperator
        from ico.core.runtime.progress import IcoProgress
        from ico.core.runtime.runtime import IcoRuntime
        from ico.tools.progress.rich_progress_tool import RichProgressTool
        import time

        # Build computation flow with embedded progress tracking
        def create_dataset():
            for i in range(100):
                yield f"item_{i}"

        def process_item(item: str) -> str:
            time.sleep(0.01)  # Simulate processing
            return item.upper()

        # Compose flow: Source → Progress → Operator → Sink
        source = IcoSource(create_dataset, name="data_source")
        progress = IcoProgress[str](total=100, name="processing_progress")
        processor = IcoOperator(process_item, name="processor")
        sink = IcoSink(print, name="output_sink")

        flow = source | progress | processor.stream() | sink

        # Create runtime with progress monitoring tool
        runtime = IcoRuntime(flow, tools=[RichProgressTool()], name="main_runtime")

        # Execute with real-time progress tracking
        runtime.run()
        # Output: [▓▓▓▓▓▓▓▓▓▓] 100/100 items processed
        ```

    Architecture Principles:
        IcoRuntime exemplifies ICO's core architectural principle: runtime tree
        infrastructure integrates with computation flows when execution coordination,
        monitoring, and management capabilities are needed, while maintaining
        clear separation between computation logic and runtime concerns.
    """

    closure: IcoOperatorProtocol[
        None, None
    ]  # Complete computation flow (Source → Sink)
    toolbox: IcoRuntimeToolBox  # Tool container for runtime monitoring
    plan_root: IcoNodeProtocol  # The flow whose plan keys events carry
    cancel_token: IcoCancelToken  # Stops the run between items when set

    def __init__(
        self,
        closure: IcoOperatorProtocol[None, None],
        *,
        name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        tools: Sequence[IcoRuntimeTool] | None = None,
        plan_root: IcoNodeProtocol | None = None,
        cancel: IcoCancelToken | None = None,
    ) -> None:
        """Initialize runtime coordinator with computation flow and tool integration.

        Args:
            closure: Complete computation flow (Source → Sink) forming () → () closure.
            name: Optional runtime identifier for debugging and monitoring.
            runtime_parent: Optional parent runtime node for hierarchical runtimes.
            runtime_children: Optional additional runtime children beyond automatic toolbox.
            state_model: Optional custom state model (default: BaseStateModel).
            tools: Optional sequence of runtime tools for monitoring and management.
            plan_root: The flow whose plan keys the runtime nodes and their
                events get (``node``), when ``closure`` wraps it (as
                ``ico.spec.run`` does); ``closure`` by default.
            cancel: The token that cancels the run (``IcoRuntime.cancel``
                sets it too); a new one by default.

        Initialization Process:
            1. **ToolBox Creation**: Create tool container with provided tools
            2. **Runtime Tree Setup**: Initialize as IcoRuntimeNode with toolbox as child
            3. **Flow Assignment**: Store computation flow for execution
            4. **Runtime Discovery**: Discover and connect all computation nodes to runtime tree

        Runtime Tree Integration:
            • Toolbox added as primary runtime child for tool management
            • Additional runtime children integrated if provided
            • Computation flow nodes discovered and connected to runtime hierarchy
            • Event flow established: Computation → Runtime → Tools

        Flow Requirements:
            The closure must form a complete computation flow:
            • **Entry Point**: Source (None → Iterable[O])
            • **Processing**: Optional operators for data transformation
            • **Exit Point**: Sink (Iterable[I] → None)
            • **Closure**: Overall signature () → () for complete execution
        """

        IcoRuntimeNode.__init__(
            self,
            runtime_name=name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
            state_model=state_model,
        )
        self.closure = closure

        # Tools in toolbox might need to use `IcoRuntimeToolRegistrationProtocol` to register for runtime events,
        # so we need to discover and connect runtime nodes before creating the toolbox and adding tools to it.

        discover_and_connect_runtime_nodes(self, closure)
        # Plan keys before the tools register the nodes.
        self.plan_root = plan_root if plan_root is not None else closure
        self.cancel_token = cancel if cancel is not None else IcoCancelToken()
        self.cancel_token.add_listener(self._cancel_runners)
        assign_plan_keys(self.plan_root)

        self.toolbox = IcoRuntimeToolBox(runtime=self, tools=tools)
        self.add_runtime_children(self.toolbox)
        self._requests = itertools.count(1)
        self._replies: dict[int, list[IcoRuntimeEvent]] = {}
        self._replies_lock = threading.Lock()

    # ────────────────────────────────────────────────
    # Event coordination and distribution
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """Handle and distribute runtime events to the tools.

        Overrides IcoRuntimeNode.on_event to forward every event to the
        toolbox, which distributes it to all tools.

        Args:
            event: Runtime event bubbled up from computation nodes.

        Returns:
            The processed event (standard runtime tree event flow).

        Event Distribution Process:
            1. **Parent Processing**: Call super().on_event() for standard runtime tree handling
            2. **Tool Distribution**: Forward event to toolbox for distribution to all tools
            3. **Event Return**: Return processed event for continued runtime tree flow

        Distribution Channels:
            • **ToolBox Channel**: event → toolbox.on_forward_event() → all tools

        Event Types Handled:
            • IcoProgressEvent: Progress updates from computation nodes
            • IcoStateChangeEvent: Runtime state transitions
            • IcoExceptionEvent: Exception conditions and error handling
            • Custom Events: Application-specific runtime events

        Multi-Channel Benefits:
            • **Tool Integration**: Full runtime tree integration with comprehensive tooling
        """
        super().on_event(event)

        if event.reply_to is not None:
            # An answer to a query goes to the one who asked, not to the tools.
            with self._replies_lock:
                replies = self._replies.get(event.reply_to)
                if replies is not None:
                    replies.append(event)
            return None

        self.toolbox.on_forward_event(event)

        return event

    def query(self, command: IcoRuntimeCommand) -> list[IcoRuntimeEvent]:
        """Broadcast ``command`` and return the events that answer it.

        The command gets a request number; the nodes that answer it bubble
        events with ``reply_to`` set to that number. They are collected
        here and not forwarded to the tools. Runners in worker processes
        answer through their channels before they acknowledge the command,
        so their answers are collected too.

        Example:
            >>> states = runtime.query(IcoStateRequestCommand.create())
            >>> {e.trace.reverse(): e.state for e in states}

        Returns:
            The answers, in the order they arrived; each one's ``trace`` is
            its path in the runtime tree.
        """
        request = next(self._requests)
        replies: list[IcoRuntimeEvent] = []
        with self._replies_lock:
            self._replies[request] = replies
        try:
            self.broadcast_command(replace(command, request=request))
        finally:
            with self._replies_lock:
                del self._replies[request]
        return replies

    # ────────────────────────────────────────────────
    # Flow execution coordination
    # ────────────────────────────────────────────────

    def cancel(self, reason: str = "cancelled") -> None:
        """Stop the run between items, from any thread: ``run`` raises
        ``IcoCancelled``. An item in a worker stops between its own inner
        items (its runner is cancelled too); an item in this process
        finishes. Deactivate as usual afterwards; workers stop cleanly."""
        self.cancel_token.cancel(reason)

    def sample(self, node: str, count: int = 1) -> None:
        """Preview the next ``count`` results of the node at plan key
        ``node``, from any thread, also during a run. Each one comes to the
        tools as an ``IcoSampleEvent`` (see :mod:`ico.core.runtime.sample`);
        workers take the request while they are active, at once.

        Raises:
            ValueError: ``count`` is below 1, or ``node`` is not in the plan.
        """
        if count < 1:
            raise ValueError(f"count must be at least 1, got {count}")
        command = IcoSampleCommand(node=node, count=count)
        if not start_sampling(self.plan_root, ROOT_KEY, command, self.bubble_event):
            self._check_remote_key(node)
        self.broadcast_command(command)

    def keep_recent(self, interval: float = RECENT_INTERVAL) -> None:
        """Preview a result of every operator at most every ``interval``
        seconds, also in workers (while they are active), as
        ``IcoSampleEvent`` with ``recent``: what already passed a node can
        be looked at later. Call it after ``activate``; it costs a clock
        read per call of an operator."""
        keep_recent(self.plan_root, ROOT_KEY, interval, self.bubble_event)
        self.broadcast_command(IcoRecentCommand(interval=interval))

    def _check_remote_key(self, key: str) -> None:
        """Accept a key below a worker's remote flow (its runner checks it,
        without building the flow here); reject any other."""
        for planned in plan_nodes(self.plan_root):
            if isinstance(planned.node, HasRemoteFlow):
                remote = remote_flow_key(planned.key, planned.node)
                if key == remote or key.startswith(remote + "."):
                    return
        raise ValueError(f"no node {key} in the plan")

    def _cancel_runners(self, reason: str) -> None:
        """Tell the runners of the workers to stop their items too. On a
        thread of its own: ``cancel`` may be called from a signal handler."""

        def send() -> None:
            with contextlib.suppress(Exception):  # best effort: workers may be gone
                self.broadcast_command(IcoCancelCommand(reason=reason))

        threading.Thread(target=send, name="ico-cancel", daemon=True).start()

    def run(self) -> IcoRuntime:
        """Execute complete computation flow with runtime coordination and monitoring.

        Provides coordinated execution of the encapsulated computation flow with
        comprehensive state management, exception handling, and runtime tree coordination.
        All runtime events generated during execution are captured and distributed
        to monitoring tools.

        Returns:
            Self reference for method chaining (fluent interface).

        Raises:
            Exception: Any exception from computation flow execution (re-raised after fault state).

        Execution Process:
            1. **State Transition**: Transition to running state (triggers runtime events)
            2. **Flow Execution**: Execute computation closure with None input (() → ())
            3. **Success Handling**: Transition to ready state on successful completion
            4. **Exception Handling**: Transition to fault state on any exception, then re-raise

        State Management Integration:
            • **running()**: Signals active execution start to runtime tree
            • **ready()**: Signals successful completion and availability for next execution
            • **fault()**: Signals exception condition requiring intervention

        Runtime Coordination:
            During execution, all computation nodes participate in runtime tree:
            • Progress events bubble up from IcoProgress nodes
            • State change events propagate through runtime hierarchy
            • Exception events captured and distributed to error handling tools
            • All events forwarded to tools for real-time monitoring

        Execution Safety:
            • **Exception Transparency**: All computation exceptions preserved and re-raised
            • **State Consistency**: Runtime state always reflects actual execution status
            • **Resource Cleanup**: Tools receive notifications for proper resource cleanup
            • **Error Visibility**: Exception conditions visible to all monitoring tools

        Usage Pattern:
            ```python
            runtime = IcoRuntime(flow, tools=[progress_tool, logger])

            try:
                runtime.run()  # Execute with monitoring
                print("Flow completed successfully")
            except Exception as e:
                print(f"Flow failed: {e}")
                # Runtime automatically in fault state
            ```

        Method Chaining:
            Returns self reference enabling fluent interface patterns:
            ```python
            runtime.activate().run().deactivate()
            ```
        """
        try:
            self.state_model.running()
            with cancellable(self.cancel_token):
                self.closure(None)
            self.state_model.ready()
            return self

        except Exception as e:
            self.state_model.fault()
            raise e
