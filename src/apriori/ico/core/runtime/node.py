from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from typing import TypeAlias

from typing_extensions import Protocol, Self, runtime_checkable

from apriori.ico.core.runtime.command import (
    IcoActivateCommand,
    IcoDeactivateCommand,
    IcoRunCommand,
    IcoRuntimeCommand,
)
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.state import (
    BaseStateModel,
    IcoRuntimeState,
    IcoStateEvent,
    IcoStateRequestCommand,
)
from apriori.ico.core.tree_utils import TraversalInfo, TreeWalker

# ────────────────────────────────────────────────
# Runtime node
# ────────────────────────────────────────────────


class IcoRuntimeNode(ABC):
    """Base class for nodes in the ICO runtime execution tree.

    IcoRuntimeNode represents execution management infrastructure that operates
    independently from the computation flow (ICO operators and types). The runtime
    tree embeds into computation flows when runtime management is needed.

    Runtime vs Computation Separation:
        • Runtime Tree: Manages HOW computation executes (lifecycle, resources, monitoring)
        • Computation Flow: Defines WHAT transformations happen (data processing logic)
        • The runtime tree provides execution infrastructure without affecting computation logic
        • Runtime nodes coordinate execution phases while remaining type-agnostic

    Core Architecture:
        • Commands: Broadcast lifecycle control (activate, run, deactivate) down the tree
        • Events: Bubble execution feedback (state changes, errors) up the tree
        • States: Track node lifecycle phases (inactive → ready → running → complete)
        • Tree Structure: Hierarchical parent/child relationships for command propagation and event bubbling

    Runtime Tree Embedding:
        The runtime tree embeds into computation flows in specific scenarios:
        • Agents: Distributed execution across processes requires runtime coordination
        • Progress Tracking: Monitoring execution progress needs runtime instrumentation
        • Resource Management: Complex resource allocation requires lifecycle control
        • Error Handling: Fault propagation and recovery needs runtime event system

    Command Flow (Top-Down):
        1. activate() → Allocate resources and prepare for execution
        2. run() → Begin computation execution
        3. deactivate() → Release resources and cleanup
        Commands broadcast to all children using broadcast_command()

    Event Flow (Bottom-Up):
        1. Runtime events bubble from children to parents using bubble_event()
        2. State changes, errors, and progress updates propagate upward
        3. Events carry tree path traces for debugging and monitoring

    State Management:
        • Each node maintains BaseStateModel for lifecycle tracking
        • States: Inactive, Ready, Running, Complete, Fault
        • State transitions triggered by commands and execution progress
        • State events automatically bubble to parent nodes

    Integration Examples:
        • IcoAgent: Embeds remote runtime node for multiprocess execution
        • IcoProgress: Embeds runtime node for progress tracking during data flow
        • IcoTool: Provides runtime services (logging, monitoring, visualization)
        • IcoRuntime: Root runtime node that orchestrates entire execution tree

    Note:
        Runtime nodes provide execution scaffolding while maintaining complete
        independence from computation types. The same runtime infrastructure
        works across all ICO operator types and data flows.
    """

    __slots__ = (
        "state_model",
        "runtime_name",
        "_runtime_children",
        "_runtime_parent",
    )

    # Runtime tree attributes (independent of computation types)
    state_model: BaseStateModel  # Execution lifecycle state management
    runtime_name: str | None  # Human-readable identifier for debugging
    _runtime_children: list[IcoRuntimeNode]  # Child nodes for command broadcast
    _runtime_parent: IcoRuntimeNode | None  # Parent node for event bubbling

    def __init__(
        self,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
        state_model: BaseStateModel | None = None,
    ) -> None:
        """Initialize runtime node with tree structure and state management.

        Args:
            runtime_name: Optional identifier for debugging and monitoring.
            runtime_parent: Parent node in runtime tree hierarchy.
            runtime_children: Initial child nodes for command propagation.
            state_model: State management model (defaults to BaseStateModel).

        Note:
            Automatically establishes bidirectional parent-child relationships
            for proper command broadcast and event bubbling behavior.
        """
        self.runtime_name = runtime_name
        self._runtime_parent = runtime_parent
        self._runtime_children = (
            list(runtime_children) if runtime_children is not None else []
        )
        self.state_model = state_model or BaseStateModel()

        for child in self._runtime_children:
            child._runtime_parent = self

    # ────────────────────────────────────────────────
    # Properties
    # ────────────────────────────────────────────────

    @property
    def runtime_parent(self) -> IcoRuntimeNode | None:
        """Parent node in the runtime tree for event bubbling.

        Returns:
            Parent node that receives events from this node, or None for root.
        """
        return self._runtime_parent

    @property
    def runtime_children(self) -> list[IcoRuntimeNode]:
        """Child nodes in the runtime tree for command broadcast.

        Returns:
            List of child nodes that receive commands from this node.
        """
        return self._runtime_children

    @property
    def state(self) -> IcoRuntimeState:
        """Current execution lifecycle state of the runtime node.

        Returns:
            Current state (inactive/ready/running/complete/fault) managing
            this node's position in the execution lifecycle.
        """
        return self.state_model.state

    # ────────────────────────────────────────────────
    # Commands
    # ────────────────────────────────────────────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """Process runtime command and update node state.

        Handles lifecycle commands (activate, run, deactivate) by updating
        the node's state model and responding to state requests with events.

        Args:
            command: Runtime command to process (activate/run/deactivate/state_request).

        Note:
            Subclasses can override to implement custom behavior like resource
            allocation, computation setup, or cleanup logic while maintaining
            the standard state management pattern.
        """
        self.state_model.update(command)

        if isinstance(command, IcoStateRequestCommand):
            # Respond with current state
            self.bubble_event(
                IcoStateEvent.create(state=self.state),
            )

    def broadcast_command(self, command: IcoRuntimeCommand) -> None:
        """Distribute runtime command to all nodes in the runtime subtree.

        Commands propagate downward through the runtime tree to coordinate
        lifecycle phases (activation, execution, deactivation) across all
        runtime nodes. Uses tree walker for efficient traversal.

        Args:
            command: Runtime command to broadcast (activate/run/deactivate).

        Note:
            Command broadcast respects the command's broadcast_order ('pre' or 'post')
            and excludes remote placeholder nodes from local propagation.
        """

        def visit_fn(node_info: BroadcastTraversalInfo) -> None:
            assert node_info.context is not None
            node_info.node.on_command(node_info.context)

        walker = create_broadcast_walker(command)
        walker.walk(
            self,
            visit_fn=visit_fn,
            order=command.broadcast_order,
        )

    # ────────────────────────────────────────────────
    # Events
    # ────────────────────────────────────────────────

    def on_event(self, event: IcoRuntimeEvent) -> IcoRuntimeEvent | None:
        """Process runtime event and determine propagation behavior.

        Args:
            event: Runtime event to handle (state changes, errors, heartbeats).

        Returns:
            Event to continue bubbling up the tree, or None to stop propagation.

        Note:
            Subclasses can override to implement custom event handling like
            logging, metrics collection, error recovery, or event filtering
            while controlling whether events continue bubbling upward.
        """
        return event

    def bubble_event(
        self,
        event: IcoRuntimeEvent,
        from_child: IcoRuntimeNode | None = None,
    ) -> None:
        """Propagate runtime event upward through the runtime tree.

        Events bubble from child nodes to parent nodes, carrying execution
        feedback like state changes, errors, and progress updates. Each node
        can process the event before deciding whether to continue propagation.

        Args:
            event: Runtime event to propagate upward (with tree path trace).
            from_child: Child node that originated this event (for trace tracking).

        Note:
            Automatically updates event trace with tree path index for debugging.
            Event propagation stops if on_event() returns None or at tree root.
        """
        if from_child is not None:
            child_index = self._runtime_children.index(from_child)
            event = replace(event, trace=event.trace.add_child(child_index))

        # If event is handled and should not propagate further, stop here
        if not self.on_event(event):
            return None

        if self._runtime_parent:
            self._runtime_parent.bubble_event(event, from_child=self)

    # ────────────────────────────────────────────────
    # Runtime Tree Management
    # ────────────────────────────────────────────────

    def add_runtime_children(self, *children: IcoRuntimeNode) -> Self:
        """Add child nodes to the runtime tree for command propagation.

        Establishes parent-child relationships in the runtime tree, enabling
        commands to broadcast from this node to children and events to bubble
        from children to this node.

        Args:
            *children: Runtime nodes to add as children.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any child already has a runtime parent.
        """
        for child in children:
            if child.runtime_parent is not None:
                raise ValueError(
                    f"Child node {child} already has a runtime parent "
                    f"{child.runtime_parent}"
                )
            if child not in self._runtime_children:
                self._runtime_children.append(child)
            child._runtime_parent = self
        return self

    def remove_runtime_child(self, *children: IcoRuntimeNode) -> Self:
        """Remove child nodes from the runtime tree.

        Breaks parent-child relationships in the runtime tree, disconnecting
        command propagation and event bubbling between nodes.

        Args:
            *children: Runtime nodes to remove as children.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If any child is not actually a child of this node.
        """
        for child in children:
            if child.runtime_parent is not self:
                raise ValueError(f"Child node {child} is not a runtime child of {self}")
            if child in self._runtime_children:
                self._runtime_children.remove(child)
            child._runtime_parent = None
        return self

    # ────────────────────────────────────────────────
    # Runtime Commands API
    # ────────────────────────────────────────────────

    def activate(self) -> Self:
        """Broadcast activation command through the runtime subtree.

        Initiates resource allocation and preparation phase across all runtime
        nodes in the subtree. Each node allocates required resources and
        transitions to ready state.

        Returns:
            Self for method chaining.
        """
        self.broadcast_command(IcoActivateCommand.create())
        return self

    def run(self) -> Self:
        """Broadcast run command through the runtime subtree.

        Initiates execution phase across all runtime nodes in the subtree.
        Nodes transition from ready state to running state and begin
        their execution logic.

        Returns:
            Self for method chaining.
        """
        self.broadcast_command(IcoRunCommand.create())
        return self

    def deactivate(self) -> Self:
        """Broadcast deactivation command through the runtime subtree.

        Initiates cleanup and resource release phase across all runtime
        nodes in the subtree. Each node releases resources and transitions
        to inactive state.

        Returns:
            Self for method chaining.
        """
        self.broadcast_command(IcoDeactivateCommand.create())
        return self

    # ────────────────────────────────────────────────
    # Describe util interface
    # ────────────────────────────────────────────────

    def describe(self) -> None:
        """Display visual representation of the runtime tree structure.

        Renders a formatted table showing the runtime tree hierarchy, node states,
        and relationships for debugging and monitoring purposes.
        """
        from apriori.ico.describe.describer import describe as describe_util

        describe_util(self)

    # ────────────────────────────────────────────────
    # Hirarchy Iteration
    # ────────────────────────────────────────────────

    def iterate_nodes(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all runtime nodes in the subtree (depth-first).

        Traverses the entire runtime subtree rooted at this node, yielding
        each node for inspection, monitoring, or batch operations.

        Yields:
            All runtime nodes in the subtree, starting with self.
        """
        yield self
        for c in self.runtime_children:
            yield from c.iterate_nodes()

    def iterate_parents(self) -> Iterator[IcoRuntimeNode]:
        """Recursively yield all parent runtime nodes up to the root.

        Traverses upward through the runtime tree hierarchy, yielding each
        parent node for debugging, event tracing, or tree analysis.

        Yields:
            All parent runtime nodes from immediate parent to root.
        """
        if self.runtime_parent is None:
            return

        yield self.runtime_parent
        yield from self.runtime_parent.iterate_parents()

    def __str__(self):
        """String representation showing runtime node class, name, and current state."""
        return (
            f"{self.__class__}(name={self.runtime_name}"
            f", state={self.state_model.state})"
        )


# ────────────────────────────────────────────────
# Remote runtime factory protocol
# ────────────────────────────────────────────────


class IcoRemotePlaceholderNode(IcoRuntimeNode):
    """Placeholder representing a remote runtime node in distributed execution.

    IcoRemotePlaceholderNode serves as a local proxy for runtime nodes executing
    in remote processes or machines. It maintains correct tree path indexing for
    command and event routing while the actual execution happens remotely.

    Remote Runtime Architecture:
        • Local runtime tree contains placeholder nodes for remote components
        • Commands sent to placeholders are routed to remote runtime processes
        • Events from remote nodes bubble through placeholders to local tree
        • Tree path indices remain consistent across local/remote boundaries

    Note:
        Placeholders enable seamless integration of distributed execution within
        the unified runtime tree model while maintaining command/event semantics.
    """

    pass


@runtime_checkable
class HasRemoteRuntime(Protocol):
    """Protocol for nodes that can create remote runtime instances.

    HasRemoteRuntime identifies runtime nodes capable of spawning remote
    execution contexts (processes, containers, network nodes) and providing
    factory functions to create corresponding runtime node instances.

    Remote Runtime Integration:
        • Enables distributed execution within unified runtime tree model
        • Factory creates actual runtime nodes for remote execution contexts
        • Supports dynamic runtime tree expansion across process boundaries

    Note:
        Used by agents and distributed execution components to integrate
        remote runtime nodes into the local runtime tree structure.
    """

    def get_remote_runtime_factory(self) -> Callable[[], IcoRuntimeNode]: ...


# ────────────────────────────────────────────────
# Tree walker api
# ────────────────────────────────────────────────

# ──────── Command broadcast walker ────────

BroadcastTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, IcoRuntimeCommand]
BroadcastTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, IcoRuntimeCommand]


def create_broadcast_walker(command: IcoRuntimeCommand) -> BroadcastTreeWalker:
    """Create tree walker for broadcasting runtime commands.

    Creates a specialized tree walker that propagates runtime commands through
    the runtime tree while excluding remote placeholder nodes from local traversal.

    Args:
        command: Runtime command to broadcast through the tree.

    Returns:
        Configured tree walker for command broadcast operations.

    Note:
        Remote placeholders are excluded as they handle command routing
        to remote processes through separate mechanisms.
    """

    def _get_children(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode]:
        return [
            c
            for c in node.runtime_children
            if not isinstance(c, IcoRemotePlaceholderNode)
        ]

    return BroadcastTreeWalker(
        get_children_fn=_get_children,
        initial_context=command,
    )


# ──────── Runtime tree walker ────────

RuntimeTreeWalker: TypeAlias = TreeWalker[IcoRuntimeNode, None]
RuntimeTraversalInfo: TypeAlias = TraversalInfo[IcoRuntimeNode, None]


def create_runtime_walker(expand_remote_runtimes: bool = False) -> RuntimeTreeWalker:
    """Create tree walker for runtime tree traversal and inspection.

    Creates a tree walker for navigating runtime node hierarchies with optional
    expansion of remote runtime nodes for comprehensive tree analysis.

    Args:
        expand_remote_runtimes: Whether to replace placeholder nodes with actual
                              remote runtime nodes during traversal.

    Returns:
        Configured tree walker for runtime tree inspection.

    Note:
        Remote expansion enables full tree visibility across distributed
        execution boundaries but may involve inter-process communication.
    """

    def _get_children(node: IcoRuntimeNode) -> Sequence[IcoRuntimeNode]:
        children = list(node.runtime_children)

        if expand_remote_runtimes and isinstance(node, HasRemoteRuntime):
            factory = node.get_remote_runtime_factory()
            placeholder_index = get_placeholder_index(children)
            children[placeholder_index] = factory()

        return children

    return RuntimeTreeWalker(get_children_fn=_get_children)


def get_placeholder_index(children: list[IcoRuntimeNode]):
    """Find the index of the remote placeholder node in children list.

    Args:
        children: List of runtime child nodes to search.

    Returns:
        Index of the IcoRemotePlaceholderNode in the children list.

    Raises:
        RuntimeError: If no placeholder found or multiple placeholders found.

    Note:
        Remote runtime integration expects exactly one placeholder per
        node with remote execution capability.
    """
    placeholder_indices = [
        i for i, c in enumerate(children) if isinstance(c, IcoRemotePlaceholderNode)
    ]
    if len(placeholder_indices) == 0:
        raise RuntimeError("No remote placeholder found among runtime children.")
    if len(placeholder_indices) > 1:
        raise RuntimeError("Multiple remote placeholders found among runtime children.")

    return placeholder_indices[0]
