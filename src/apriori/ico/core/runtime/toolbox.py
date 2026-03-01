from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode, create_runtime_walker
from apriori.ico.core.runtime.tool import IcoTool
from apriori.ico.core.tree_utils import TreePathIndex

# ────────────────────────────────────────────────
# Tool registration protocol
# ────────────────────────────────────────────────


@runtime_checkable
class IcoToolRegistrationProtocol(Protocol):
    """Protocol for tools that require runtime node registration.

    Tools implementing this protocol receive notifications about all nodes
    in the runtime tree, enabling comprehensive monitoring and management
    capabilities across the entire computation hierarchy.

    Registration Flow:
        1. Tool added to ToolBox via add_tool()
        2. ToolBox checks if tool implements IcoToolRegistrationProtocol
        3. If yes, ToolBox walks entire runtime tree
        4. For each node: calls tool.register_node(node, path)
        5. Tool can build internal tracking structures

    Use Cases:
        • Progress aggregation across multiple nodes
        • Performance profiling of specific node types
        • Debug inspection of runtime tree structure
        • Resource monitoring per node
    """

    def register_node(self, node: IcoRuntimeNode, path: TreePathIndex) -> None:
        """Register runtime node for tool-specific tracking.

        Args:
            node: Runtime node to register for monitoring.
            path: Tree path index identifying node position in runtime hierarchy.
        """
        ...


# ────────────────────────────────────────────────
# Tool container and management
# ────────────────────────────────────────────────


class IcoToolBox(IcoRuntimeNode):
    """Container and coordinator for multiple runtime tools.

    IcoToolBox manages collections of runtime tools, providing centralized
    event forwarding, tool lifecycle management, and runtime tree integration
    for comprehensive monitoring and management of ICO computation execution.

    Tool Container Architecture:
        • Multi-tool Management: Coordinates multiple tools simultaneously
        • Event Distribution: Forwards runtime events to all registered tools
        • Runtime Integration: Participates in runtime tree as tool coordinator
        • Dynamic Management: Supports runtime addition/removal of tools

    Tool Registration System:
        ToolBox provides two-tier tool registration:

        1. **Basic Registration**: All tools added as runtime children
        2. **Protocol Registration**: Tools implementing IcoToolRegistrationProtocol
           receive notifications about all runtime nodes for comprehensive tracking

    Event Flow Architecture:
        Runtime Tree → ToolBox.on_forward_event() → Tool.on_forward_event()

        ToolBox acts as event distributor:
        1. Receives events from runtime tree execution
        2. Forwards events to all registered tools
        3. Each tool processes events independently
        4. No coordination between tools required

    Tool Lifecycle Management:
        • **add_tool()**: Register tools and setup runtime tree relationships
        • **remove_tool()**: Unregister tools and cleanup runtime tree
        • **Runtime Integration**: Automatic runtime child management
        • **Protocol Handling**: Automatic runtime node registration for protocol tools

    Registration Protocol Support:
        Tools implementing IcoToolRegistrationProtocol receive comprehensive
        runtime tree visibility:

        1. Runtime tree traversal with expand_remote_runtimes=True
        2. Node registration for each runtime node
        3. Path-indexed access to runtime hierarchy
        4. Support for distributed runtime exploration

    Multi-Tool Coordination:
        ToolBox enables multiple specialized tools to operate simultaneously:
        • Progress tools for real-time progress tracking
        • Debug tools for runtime state inspection
        • Logging tools for execution tracing
        • Performance tools for resource monitoring
        • Management tools for runtime control

    Runtime Tree Position:
        ToolBox typically positioned as child of main runtime node,
        providing tools access to all runtime events from computation execution
        while maintaining separation from computation flow logic.

    Base Implementation:
        IcoToolBox provides complete tool container functionality with
        standardized interfaces for tool management, event distribution,
        and runtime tree integration.
    """

    __slots__ = ("runtime", "tools")

    runtime: IcoRuntimeNode  # Runtime node being monitored by tools
    tools: list[IcoTool]  # Collection of registered runtime tools

    def __init__(
        self,
        runtime: IcoRuntimeNode,
        tools: Sequence[IcoTool] | None = None,
    ) -> None:
        """Initialize tool container with runtime and optional initial tools.

        Args:
            runtime: Runtime node that tools will monitor.
            tools: Optional sequence of initial tools to register.

        Initialization:
            • Runtime assignment: Sets runtime node for tool monitoring
            • Tool collection: Initializes empty tool list
            • Automatic registration: Adds initial tools if provided
        """
        super().__init__()
        self.runtime = runtime
        self.tools = []
        if tools is not None:
            self.add_tool(*tools)

    def add_tool(self, *tools: IcoTool) -> None:
        """Add tools to container with full registration and runtime integration.

        Args:
            *tools: Variable number of tools to add to container.

        Registration Process:
            1. Add tools to internal collection
            2. Register tools as runtime children for event flow
            3. Check for IcoToolRegistrationProtocol implementation
            4. If protocol found: walk runtime tree and register all nodes
            5. Enable tools to receive runtime events and build tracking structures

        Protocol Registration:
            Tools implementing IcoToolRegistrationProtocol receive comprehensive
            runtime tree visibility through register_node() calls for each
            node in the runtime hierarchy.
        """
        self.tools.extend(tools)
        self.add_runtime_children(*tools)

        for tool in tools:
            if isinstance(tool, IcoToolRegistrationProtocol):
                runtime_walker = create_runtime_walker(expand_remote_runtimes=True)

                for node_info in runtime_walker.traverse(self.runtime):
                    tool.register_node(*node_info.node_path)

    def remove_tool(self, *tools: IcoTool) -> None:
        """Remove tools from container and cleanup runtime integration.

        Args:
            *tools: Variable number of tools to remove from container.

        Cleanup Process:
            1. Remove tools from internal collection
            2. Remove tools as runtime children to stop event flow
            3. Tool-specific cleanup handled by individual tools
        """
        for tool in tools:
            self.tools.remove(tool)
        self.remove_runtime_child(*tools)

    def on_forward_event(self, event: IcoRuntimeEvent) -> None:
        """Distribute runtime event to all registered tools.

        Args:
            event: Runtime event to forward to tools.

        Event Distribution:
            1. Receive runtime event from runtime tree
            2. Forward event to all registered tools via on_forward_event()
            3. Each tool processes event independently
            4. No coordination or ordering between tools

        Event Flow:
            Runtime → ToolBox → [Tool1, Tool2, ..., ToolN]
        """
        # Forward event to tools
        for tool in self.tools:
            tool.on_forward_event(event)
