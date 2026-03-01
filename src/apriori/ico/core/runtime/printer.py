from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import overload

from typing_extensions import final

from apriori.ico.core.operator import I, IcoOperator, O, wrap_operator
from apriori.ico.core.runtime.event import IcoRuntimeEvent
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.runtime.runtime_wrapper import IcoRuntimeWrapper
from apriori.ico.core.tree_utils import TreePathIndex


@final
@dataclass(slots=True, frozen=True)
class IcoPrintEvent(IcoRuntimeEvent):
    """
    Runtime event for print message propagation through ICO tree.

    Carries print messages from IcoPrinter nodes up through the runtime
    hierarchy for handling by print-aware tools and consumers.

    Architecture:
        - Immutable dataclass following ICO event patterns
        - Contains string message for output rendering
        - Uses TreePathIndex trace for event routing
        - Extends IcoRuntimeEvent for runtime integration

    Event Flow:
        IcoPrinter.call() → bubble_event() → IcoTool.on_backward_event()

    Usage:
        ```python
        # Created automatically by IcoPrinter
        printer = IcoPrinter()
        printer("Hello World")  # Generates IcoPrintEvent

        # Manual creation (rare)
        event = IcoPrintEvent.create("Custom message")
        ```

    Note:
        - @final prevents inheritance (ICO event design)
        - slots=True optimizes memory usage
        - frozen=True ensures immutability
    """

    message: str

    @staticmethod
    def create(message: str) -> IcoPrintEvent:
        """
        Create print event with message and empty trace.

        Args:
            message: Text content for printing

        Returns:
            IcoPrintEvent: Ready for bubble_event() propagation

        Note:
            - TreePathIndex() creates empty trace (filled by runtime)
            - Static factory method follows ICO event conventions
        """
        return IcoPrintEvent(message=message, trace=TreePathIndex())


@final
class IcoPrinter(IcoRuntimeNode):
    """
    Runtime printing node for message output in ICO computation flows.

    Provides structured printing capability that integrates with ICO runtime
    system, enabling print output to be captured, formatted, and directed
    by runtime tools without disrupting computation flow.

    Architecture:
        - Inherits IcoRuntimeNode for runtime tree integration
        - Generates IcoPrintEvent for message propagation
        - Uses state model for execution lifecycle tracking
        - Bubbles events upward for tool consumption

    Runtime Integration:
        - Embeds in computation flows as runtime child
        - Events captured by print-aware ICO tools
        - State transitions: ready → running → ready
        - Compatible with distributed execution (events serialize)

    Usage Patterns:
        ```python
        # Direct usage
        printer = IcoPrinter()
        printer("Debug message")

        # Embedded in operators
        @operator()
        def debug_processor(x: int) -> int:
            printer(f"Processing: {x}")
            return x * 2

        # With printable() wrapper
        debug_op = printable(lambda x: x * 2, printer)
        ```

    Tool Integration:
        - RichPrinterTool: Rich console output
        - LoggerTool: Structured logging
        - FilePrinterTool: File output
        - Custom tools: Implement on_backward_event(IcoPrintEvent)

    Note:
        - @final prevents inheritance (use composition)
        - Thread-safe through ICO runtime event system
        - Print operations are synchronous within node
    """

    def __call__(self, message: str) -> None:
        """
        Print message through ICO runtime event system.

        Args:
            message: Text content to print/output

        Process:
            1. Transition to running state
            2. Create and bubble IcoPrintEvent
            3. Return to ready state

        State Changes:
            ready → running → ready

        Runtime Effects:
            - Event propagates up runtime tree
            - Tools receive event via on_backward_event()
            - Message content available to all print tools

        Example:
            ```python
            printer = IcoPrinter()
            printer("Hello World")  # Message sent to runtime tools
            ```
        """
        self.state_model.running()
        self.bubble_event(IcoPrintEvent.create(message=message))
        self.state_model.ready()


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


@overload
def printable(fn: IcoOperator[I, O], printer: IcoPrinter) -> IcoOperator[I, O]: ...


@overload
def printable(fn: Callable[[I], O], printer: IcoPrinter) -> IcoOperator[I, O]: ...


def printable(
    fn: Callable[[I], O] | IcoOperator[I, O], printer: IcoPrinter
) -> IcoOperator[I, O]:
    """
    Wrap operator with embedded printer for debug/monitoring output.

    Creates IcoRuntimeWrapper that combines computation with printing
    capability, enabling operators to generate print messages during
    execution without modifying operator logic.

    Args:
        fn: Callable or IcoOperator to wrap with printing
        printer: IcoPrinter instance for message output

    Returns:
        IcoOperator[I, O]: Wrapped operator with embedded printer

    Type Safety:
        - Preserves original operator type signature I → O
        - Compatible with both mypy and pyright type checkers
        - Handles raw callables and existing IcoOperators uniformly

    Architecture:
        - Uses IcoRuntimeWrapper for composition
        - Printer becomes runtime child of wrapped operator
        - Print events bubble from printer through wrapper
        - Original computation flow unchanged

    Usage:
        ```python
        printer = IcoPrinter()

        # Wrap raw function
        debug_op = printable(lambda x: x * 2, printer)

        # Wrap existing operator
        @operator()
        def process(x: int) -> int:
            return x + 1

        traced_op = printable(process, printer)

        # Use in flow (printer events handled by tools)
        flow = source | traced_op | sink
        ```

    Runtime Integration:
        - Wrapped operator executes normally
        - Printer available for manual calls within operator
        - Print events captured by runtime tools
        - Compatible with distributed execution

    Note:
        - wrap_operator() ensures IcoOperator conversion
        - Runtime children list enables printer event routing
        - Maintains operator composition patterns
    """
    op = wrap_operator(fn)
    return IcoRuntimeWrapper[I, O](
        op,
        runtime_children=[printer],
    )
