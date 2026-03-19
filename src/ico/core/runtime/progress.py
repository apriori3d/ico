from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, final

from ico.core.operator import I
from ico.core.runtime.event import IcoRuntimeEvent
from ico.core.runtime.monitor import IcoMonitor
from ico.core.tree_utils import TreePathIndex

# ────────────────────────────────────────────────
# Progress tracking events
# ────────────────────────────────────────────────


@final
@dataclass(slots=True, frozen=True)
class IcoProgressEvent(IcoRuntimeEvent):
    """Runtime event carrying progress advancement information.

    IcoProgressEvent enables progress tracking through the runtime tree by
    bubbling progress updates from computation nodes to monitoring tools.
    Each event carries advancement information that can be aggregated
    for overall progress visualization.

    Progress Event Flow:
        1. IcoProgress processes data item
        2. Generates IcoProgressEvent with advancement value
        3. Event bubbles up through runtime tree
        4. Progress tools (RichProgressTool) receive and aggregate events
        5. Progress displays updated in real-time

    Event Integration:
        • Runtime tree bubbling: Events propagate to parent nodes
        • Tree path tracing: Events carry origin path for identification
        • Tool aggregation: Multiple progress nodes can report to same tool
        • Real-time updates: Immediate progress feedback during execution
    """

    advance: float

    @staticmethod
    def create(advance: float = 1) -> IcoProgressEvent:
        """Create progress event with advancement value.

        Args:
            advance: Amount of progress to advance (typically 1 per item).

        Returns:
            Progress event for bubbling through runtime tree.
        """
        return IcoProgressEvent(trace=TreePathIndex(), advance=advance)


# ────────────────────────────────────────────────
# Progress monitoring node
# ────────────────────────────────────────────────


class IcoProgress(Generic[I], IcoMonitor[I]):
    """Progress tracking node for data flow monitoring in ICO computation.

    IcoProgress provides seamless progress tracking by embedding into computation
    flows as a transparent monitoring node. It extends IcoMonitor to participate
    in both computation flow (data processing) and runtime tree (progress tracking),
    enabling real-time progress visualization during distributed execution.

    Dual Integration Architecture:
        • Computation Flow: Transparent identity operator (input = output)
        • Runtime Tree: Progress event generation and bubbling
        • Type Safety: Generic parameterization maintains data flow types
        • Non-intrusive: Zero impact on computation logic

    Runtime Tree Integration:
        IcoProgress embeds runtime tree functionality into computation flows
        when progress tracking is needed:

        1. **Computation Flow**: Source → [Progress] → Operator → Sink
        2. **Runtime Tree**: Runtime → Progress (child) → Tools
        3. **Event Bubbling**: Progress events bubble to monitoring tools
        4. **State Management**: Automatic running/ready state transitions

    Progress Tracking Flow:
        1. Data item enters IcoProgress node
        2. _before_call(): Transitions to running state, generates progress event
        3. super().__call__(): Identity processing (data passes through unchanged)
        4. _after_call(): Transitions back to ready state
        5. Progress event bubbles up runtime tree to tools

    Example - Basic Data Processing Progress:
        ```python
        from ico.core import IcoSource, IcoSink, IcoOperator
        from ico.core.runtime.progress import IcoProgress
        from ico.core.runtime.runtime import IcoRuntime
        from ico.tools.progress.rich_progress_tool import RichProgressTool

        # Computation flow with embedded progress tracking
        def create_dataset():
            for i in range(100):
                yield f"item_{i}"

        def process_item(item: str) -> str:
            time.sleep(0.1)  # Simulate processing
            return item.upper()

        # Build computation flow: Source → Progress → Operator → Sink
        source = IcoSource(create_dataset, name="data_source")
        progress = IcoProgress[str](total=100, name="processing_progress")
        processor = IcoOperator(process_item, name="item_processor")
        sink = IcoSink(print, name="output_sink")

        # Compose flow with progress tracking
        flow = source | progress | processor.map() | sink

        # Runtime with progress monitoring tool
        progress_tool = RichProgressTool()
        runtime = IcoRuntime(flow, tools=[progress_tool])

        # Execute with real-time progress display
        runtime.activate().run().deactivate()
        # Output: [▓▓▓▓▓▓▓▓▓▓] 100/100 items processed
        ```

    Runtime Tree Embedding:
        IcoProgress demonstrates how runtime tree functionality embeds into
        computation flows only when needed:

        • **Without Progress**: Pure computation flow, no runtime overhead
        • **With Progress**: Runtime tree embedded for monitoring functionality
        • **Transparent Integration**: Computation logic unchanged
        • **Event-Driven**: Progress updates via runtime event system

    Integration Benefits:
        • Real-time Monitoring: Live progress updates during execution
        • Distributed Progress: Works across agent/worker process boundaries
        • Multiple Tracking: Multiple progress points in complex flows
        • Tool Integration: Compatible with rich progress displays
        • Zero Overhead: No impact when progress monitoring not needed

    Generic Parameters:
        I: Type of data items flowing through the progress node (input = output)

    Note:
        IcoProgress exemplifies the ICO principle of embedding runtime tree
        functionality into computation flows when monitoring capabilities
        are required, maintaining separation while enabling integration.
    """

    __slots__ = ("total",)

    total: float  # Expected total number of items for progress calculation

    def __init__(
        self,
        total: float,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize progress tracking node with expected total count.

        Args:
            total: Expected total number of items that will flow through this node.
            name: Optional name for progress identification in monitoring tools.

        Initialization:
            • Dual inheritance: Both IcoMonitor and runtime node setup
            • Total tracking: Expected count for progress percentage calculation
            • Name identification: Used by progress tools for display labeling
        """
        IcoMonitor.__init__(self, name=name)  # pyright: ignore[reportUnknownMemberType]
        self.total = total

    def _before_call(self, item: I) -> None:
        """Process item entry with progress event generation.

        Overrides IcoMonitor._before_call to add progress tracking functionality
        while maintaining automatic state management and runtime coordination.

        Args:
            item: Data item about to be processed.

        Progress Tracking:
            1. Call parent _before_call(): Automatic transition to running state
            2. Generate IcoProgressEvent: Create progress advancement event
            3. Bubble event: Send progress event up runtime tree to tools

        Event Flow:
            IcoProgress → Runtime Tree → Progress Tools → Display Update

        Note:
            Each item advances progress by 1 unit. Progress tools calculate
            completion percentage based on item count vs total expected.
        """
        super()._before_call(item)
        self.bubble_event(IcoProgressEvent.create(advance=1))
