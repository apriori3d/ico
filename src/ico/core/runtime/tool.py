from __future__ import annotations

from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.node import IcoRuntimeNode

# ────────────────────────────────────────────────
# Runtime tool base architecture
# ────────────────────────────────────────────────


class IcoTool(IcoRuntimeNode):
    """Base class for ICO runtime tools.

    IcoTool provides the foundational architecture for runtime monitoring and
    management tools that integrate with the ICO runtime tree. Tools receive
    runtime events and provide various monitoring, debugging, visualization,
    and management capabilities during computation execution.

    Tool Architecture:
        • Runtime Integration: Full participation in runtime tree hierarchy
        • Event Reception: Receives forwarded events from runtime nodes
        • Passive Monitoring: Tools observe without affecting computation flow
        • Extensible Design: Base for specialized monitoring tools

    Runtime Tool System:
        The runtime tool system enables external observation and management
        of computation execution through standardized interfaces:

        1. **Event Flow**: Runtime events bubble up to tools via on_forward_event()
        2. **Tool Registration**: Tools register with runtime through IcoToolBox
        3. **Runtime Tree Integration**: Tools become part of runtime hierarchy
        4. **Passive Design**: Tools monitor without altering computation logic

    Tool Categories:
        • **Progress Tools**: Real-time progress tracking and visualization
        • **Debug Tools**: Runtime state inspection and debugging
        • **Logging Tools**: Event logging and execution tracing
        • **Performance Tools**: Timing, profiling, and resource monitoring
        • **Management Tools**: Runtime control and coordination

    Event Forwarding Architecture:
        Tools receive events through standardized forwarding mechanism:

        Runtime Node → Event Bubble → ToolBox → Tool.on_forward_event()

        This enables tools to:
        • React to runtime state changes
        • Track progress across distributed execution
        • Monitor exception conditions
        • Collect performance metrics
        • Provide real-time feedback

    Tool Integration with ToolBox:
        IcoTool instances are managed through IcoToolBox container,
        which provides:
        • Multi-tool coordination
        • Event distribution to multiple tools
        • Tool lifecycle management
        • Runtime tree registration
        • Dynamic tool addition/removal

    Base Implementation:
        IcoTool provides minimal base implementation with extension points
        for specialized tool functionality. Subclasses override methods
        to implement specific monitoring and management capabilities.

    Note:
        IcoTool is a base class requiring subclass implementation of
        on_forward_event() for specific tool functionality. Common tool
        implementations include RichProgressTool, LoggingTool, and DebugTool.
    """

    __slots__ = ()

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """Handle forwarded runtime event for tool-specific processing.

        This method receives all runtime events forwarded to the tool,
        enabling monitoring, debugging, visualization, and management
        capabilities based on runtime activity.

        Args:
            event: Runtime event forwarded from runtime tree execution.

        Event Processing:
            \u2022 Event Reception: Receives all runtime events via ToolBox forwarding
            \u2022 Event Filtering: Tools can filter events based on type or content
            \u2022 Tool Action: Implement monitoring, logging, visualization, etc.
            \u2022 Non-intrusive: Event processing should not affect computation flow

        Common Event Types:
            \u2022 IcoProgressEvent: Progress advancement from IcoProgress nodes
            \u2022 IcoStateChangeEvent: Runtime state transitions (ready/running/fault)
            \u2022 IcoExceptionEvent: Exception conditions and error handling
            \u2022 Custom Events: Application-specific runtime events

        Implementation Notes:
            Base implementation is no-op (pass). Subclasses override this method
            to provide specific tool functionality such as progress display,
            debug logging, performance monitoring, or runtime management.

        Tool Examples:
            \u2022 RichProgressTool: Displays progress bars for IcoProgressEvent
            \u2022 LoggingTool: Logs all events with timestamps and context
            \u2022 DebugTool: Inspects event details for debugging purposes

        Event Flow:
            Runtime → Event Bubble → ToolBox → on_forward_event() → Tool Action
        """
        pass
