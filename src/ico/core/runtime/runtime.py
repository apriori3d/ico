from __future__ import annotations

from collections.abc import Sequence

from ico.core.operator import IcoOperatorProtocol
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import (
    IcoRuntimeNode,
)
from ico.core.runtime.state import BaseStateModel
from ico.core.runtime.tool import IcoTool
from ico.core.runtime.toolbox import IcoToolBox
from ico.core.runtime.utils import discover_and_connect_runtime_nodes

# ────────────────────────────────────────────────
# Event forwarding protocol
# ────────────────────────────────────────────────


class OnForwardEventProtocol:
    """Protocol for objects that can receive forwarded runtime events.

    Simple protocol enabling external listeners to receive runtime events
    forwarded from IcoRuntime execution. Used for lightweight event monitoring
    without full IcoTool infrastructure.

    Integration:
        • Event Reception: Receives events via on_forward_event()
        • Lightweight Design: Simple protocol without runtime tree integration
        • External Monitoring: Enables monitoring from outside runtime hierarchy
    """

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """Handle forwarded runtime event.

        Args:
            event: Runtime event forwarded from IcoRuntime execution.
        """
        ...


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
        • **Event Listeners**: Lightweight OnForwardEventProtocol listeners for simple monitoring
        • **Event Distribution**: All runtime events forwarded to both systems
        • **Tool Lifecycle**: Automatic tool management through runtime lifecycle

    Execution Lifecycle:
        1. **Initialization**: Runtime tree discovery and connection
        2. **Activation**: State transition to ready (optional explicit activation)
        3. **Execution**: Flow execution with runtime coordination
        4. **Monitoring**: Real-time event forwarding to tools and listeners
        5. **Completion**: State transition back to ready or fault on exception

    State Management Integration:
        • **Ready State**: Runtime prepared for execution
        • **Running State**: Active flow execution in progress
        • **Fault State**: Exception condition requiring intervention
        • **Idle State**: Runtime inactive and available for cleanup

    Event Forwarding Architecture:
        All runtime events bubble through IcoRuntime and get forwarded to:
        1. **ToolBox**: Distributes to all registered tools
        2. **Event Listeners**: Direct forwarding to protocol listeners
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
    toolbox: IcoToolBox  # Tool container for runtime monitoring
    event_listeners: list[OnForwardEventProtocol] = []  # External event listeners

    def __init__(
        self,
        closure: IcoOperatorProtocol[None, None],
        *,
        name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
        tools: Sequence[IcoTool] | None = None,
    ) -> None:
        """Initialize runtime coordinator with computation flow and tool integration.

        Args:
            closure: Complete computation flow (Source → Sink) forming () → () closure.
            name: Optional runtime identifier for debugging and monitoring.
            runtime_parent: Optional parent runtime node for hierarchical runtimes.
            runtime_children: Optional additional runtime children beyond automatic toolbox.
            state_model: Optional custom state model (default: BaseStateModel).
            tools: Optional sequence of runtime tools for monitoring and management.

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

        # Tools in toolbox might need to use `IcoToolRegistrationProtocol` to register for runtime events,
        # so we need to discover and connect runtime nodes before creating the toolbox and adding tools to it.

        discover_and_connect_runtime_nodes(self, closure)

        self.toolbox = IcoToolBox(runtime=self, tools=tools)
        self.add_runtime_children(self.toolbox)

    # ────────────────────────────────────────────────
    # Event coordination and distribution
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """Handle and distribute runtime events to tools and listeners.

        Overrides IcoRuntimeNode.on_event to add comprehensive event distribution
        to both managed tools and external event listeners, enabling multi-channel
        runtime monitoring and management.

        Args:
            event: Runtime event bubbled up from computation nodes.

        Returns:
            The processed event (standard runtime tree event flow).

        Event Distribution Process:
            1. **Parent Processing**: Call super().on_event() for standard runtime tree handling
            2. **Tool Distribution**: Forward event to toolbox for distribution to all tools
            3. **Listener Notification**: Forward event to all registered external listeners
            4. **Event Return**: Return processed event for continued runtime tree flow

        Distribution Channels:
            • **ToolBox Channel**: event → toolbox.on_forward_event() → all tools
            • **Listener Channel**: event → listener.on_forward_event() (direct)

        Event Types Handled:
            • IcoProgressEvent: Progress updates from computation nodes
            • IcoStateChangeEvent: Runtime state transitions
            • IcoExceptionEvent: Exception conditions and error handling
            • Custom Events: Application-specific runtime events

        Multi-Channel Benefits:
            • **Tool Integration**: Full runtime tree integration with comprehensive tooling
            • **Lightweight Monitoring**: Simple protocol listeners for basic monitoring
            • **Event Redundancy**: Multiple monitoring channels for robustness
            • **Flexible Architecture**: Support for both integrated and external monitoring
        """
        super().on_event(event)

        self.toolbox.on_forward_event(event)

        for listener in self.event_listeners:
            listener.on_forward_event(event)

        return event

    # ────────────────────────────────────────────────
    # Flow execution coordination
    # ────────────────────────────────────────────────

    def run(self) -> IcoRuntime:
        """Execute complete computation flow with runtime coordination and monitoring.

        Provides coordinated execution of the encapsulated computation flow with
        comprehensive state management, exception handling, and runtime tree coordination.
        All runtime events generated during execution are captured and distributed
        to monitoring tools and listeners.

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
            • All events forwarded to tools and listeners for real-time monitoring

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
            self.closure(None)
            self.state_model.ready()
            return self

        except Exception as e:
            self.state_model.fault()
            raise e
