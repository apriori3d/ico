from typing import Generic

from ico.core.identity import IcoIdentity
from ico.core.operator import I
from ico.core.runtime.node import IcoRuntimeNode


class IcoMonitor(
    Generic[I],
    IcoIdentity[I],
    IcoRuntimeNode,
):
    """Base runtime monitoring node for specialized monitoring implementations.

    IcoMonitor serves as a foundation class for creating runtime monitoring
    components that integrate into both computation flows and runtime trees.
    This dual inheritance enables monitoring functionality while maintaining
    runtime lifecycle management.

    Architecture Integration:
        • Computation Flow: Inherits IcoIdentity[I] for data processing
        • Runtime Tree: Inherits IcoRuntimeNode for lifecycle management
        • Generic Type: Parameterized by input/output data type I
        • State Management: Automatic running/ready state transitions

    Monitoring Lifecycle:
        1. Item arrives → _before_call() → state transitions to running
        2. Item processed → computation flow processing via IcoIdentity
        3. Item completed → _after_call() → state transitions to ready
        4. Runtime events bubble up for monitoring integration

    Specialized Implementations:
        • IcoProgress: Data flow progress tracking and reporting
        • Performance monitors: Execution timing and resource usage
        • Error monitors: Fault detection and recovery coordination
        • Custom monitors: Domain-specific monitoring requirements

    State Management:
        Automatically manages runtime states during processing:
        - Before processing: Transitions to RunningState
        - After processing: Returns to ReadyState
        - Enables runtime tree coordination and monitoring

    Usage Pattern:
        Subclass IcoMonitor to implement specialized monitoring:
        - Override _before_call() for pre-processing setup
        - Override _after_call() for post-processing cleanup
        - Leverage runtime tree integration for event coordination

    Note:
        IcoMonitor bridges computation flow and runtime infrastructure,
        enabling monitoring components that participate in both aspects
        of the ICO execution model.
    """

    def __init__(self, *, name: str | None = None) -> None:
        """Initialize monitor with dual inheritance setup.

        Initializes both computation flow identity and runtime node
        infrastructure, enabling the monitor to participate in both
        data processing and runtime tree management.

        Args:
            name: Optional name for both computation identity and runtime node.
        """
        IcoIdentity.__init__(self, name=name)  # pyright: ignore[reportUnknownMemberType]
        IcoRuntimeNode.__init__(self, runtime_name=name)

    def __call__(self, item: I) -> I:
        """Process item with automatic runtime state management.

        Orchestrates item processing with runtime state transitions,
        providing hooks for specialized monitoring implementations.

        Args:
            item: Data item to process through the monitor.

        Returns:
            Processed item (identity transformation by default).

        Processing Flow:
            1. _before_call() → runtime state management
            2. super().__call__() → computation flow processing
            3. _after_call() → runtime state cleanup
        """
        self._before_call(item)

        result = super().__call__(item)

        self._after_call(item)
        return result

    def _before_call(self, item: I) -> None:
        """Pre-processing hook with automatic state transition to running.

        Called before each item is processed. Automatically transitions
        runtime state to RunningState and provides extension point for
        specialized monitoring setup.

        Args:
            item: Item about to be processed.

        Override:
            Subclasses can override to implement custom pre-processing
            behavior while maintaining automatic state management.
        """
        self.state_model.running()

    def _after_call(self, item: I) -> None:
        """Post-processing hook with automatic state transition to ready.

        Called after each item is processed. Automatically transitions
        runtime state to ReadyState and provides extension point for
        specialized monitoring cleanup.

        Args:
            item: Item that was just processed.

        Override:
            Subclasses can override to implement custom post-processing
            behavior while maintaining automatic state management.
        """
        self.state_model.ready()
