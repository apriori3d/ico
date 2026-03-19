from collections.abc import Callable, Sequence
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.runtime.command import IcoRuntimeCommand
from apriori.ico.core.runtime.node import IcoRuntimeNode
from apriori.ico.core.signature import IcoSignature


class IcoRuntimeWrapper(
    Generic[I, O],
    IcoOperator[I, O],
    IcoRuntimeNode,
):
    """
    Wrapper for embedding runtime capabilities into computational operators.

    Primary purpose: Transform pure computational operators into runtime-aware
    components by adding runtime properties like debugging, progress tracking,
    and monitoring without modifying the original computation logic.

    Core Use Case:
        Adding runtime features to computational operators:
        - Debug printing during operator execution
        - Progress tracking for long-running computations
        - Performance monitoring and profiling
        - Logging and audit trails
        - Custom runtime behaviors

    Architecture:
        - Multiple inheritance: IcoOperator[I, O] + IcoRuntimeNode
        - Delegates computation to wrapped operator (unchanged)
        - Manages runtime children (printers, progress, monitors)
        - Preserves operator composition and type safety

    Dual Nature:
        Computation Side (IcoOperator):
        - Maintains I → O type signature
        - Participates in operator pipelines (|)
        - Handles data flow and transformations

        Runtime Side (IcoRuntimeNode):
        - Embeds in runtime tree hierarchy
        - Manages runtime children lifecycle
        - Processes commands and events
        - Enables tool integration

    Common Example - Debug Printing:
        ```python
        # Original computational operator
        @operator()
        def compute(x: int) -> int:
            return x * x + 1

        # Add debug printing capability (concise)
        printer = IcoPrinter()
        debug_compute = printable(compute, printer)

        # Or using decorator syntax
        @wrap_runtime(printer)
        @operator()
        def debug_compute_v2(x: int) -> int:
            printer(f"Computing: {x}")
            return x * x + 1

        # Same computation, now with debug capabilities
        flow = source | debug_compute | sink
        ```

    Note:
        - Generic[I, O] preserves computational type safety
        - Signature delegation maintains operator semantics
        - Runtime children lifecycle managed automatically
        - Zero overhead when runtime features unused
    """

    operator: IcoOperator[I, O]

    def __init__(
        self,
        operator: IcoOperator[I, O],
        *,
        name: str | None = None,
        runtime_name: str | None = None,
        runtime_parent: IcoRuntimeNode | None = None,
        runtime_children: Sequence[IcoRuntimeNode] | None = None,
    ) -> None:
        """
        Initialize wrapper with operator and runtime configuration.

        Args:
            operator: IcoOperator to wrap with runtime capabilities
            name: Optional operator name (computation side)
            runtime_name: Optional runtime node name (runtime side)
            runtime_parent: Parent in runtime tree hierarchy
            runtime_children: Child runtime nodes (printers, progress, etc.)
        """
        # Note: pylance cannot infer IcoOperator.__init__ from Generic inheritance, but mypy can.
        IcoOperator.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            fn=self._wrapper_fn,
            name=name,
            children=[operator],
        )

        IcoRuntimeNode.__init__(
            self,
            runtime_name=runtime_name,
            runtime_parent=runtime_parent,
            runtime_children=runtime_children,
        )
        self.operator = operator

    def _wrapper_fn(self, item: I) -> O:
        """Delegate computation to wrapped operator."""
        return self.operator(item)

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """Handle runtime commands via standard IcoRuntimeNode processing."""
        return super().on_command(command)

    @property
    def signature(self) -> IcoSignature:
        """Delegate signature to wrapped operator when not explicitly inferred."""
        signature = super().signature
        return signature if signature.infered else self.operator.signature


# ─────────────────────────────────────────────
# Decorator Utilities
# ─────────────────────────────────────────────


def wrap_runtime(
    *nodes: IcoRuntimeNode,
) -> Callable[[IcoOperator[I, O]], IcoOperator[I, O]]:
    """
    Decorator factory for attaching runtime nodes to operators.

    Args:
        *nodes: Runtime nodes to attach (printers, progress, monitors)

    Usage:
        ```python
        @wrap_runtime(printer, progress)
        @operator()
        def debug_process(x: int) -> int:
            printer(f"Processing: {x}")
            return x * 2
        ```
    """

    def decorator(operator: IcoOperator[I, O]) -> IcoOperator[I, O]:
        return wrap_runtime_fn(operator, *nodes)

    return decorator


def wrap_runtime_fn(
    operator: IcoOperator[I, O],
    *nodes: IcoRuntimeNode,
) -> IcoOperator[I, O]:
    """
    Attach runtime nodes to operators using appropriate strategy.

    Args:
        operator: IcoOperator to enhance with runtime nodes
        *nodes: Runtime nodes to attach (must be unbound)

    Strategy:
        - Filter unbound nodes (runtime_parent is None)
        - If operator is IcoRuntimeNode: attach nodes directly
        - Otherwise: create IcoRuntimeWrapper with nodes as children

    Usage:
        ```python
        printer = IcoPrinter()
        wrapped = wrap_runtime_fn(process, printer)
        ```
    """
    unbind_nodes = [n for n in nodes if n.runtime_parent is None]
    if len(unbind_nodes) == 0:
        return operator

    if isinstance(operator, IcoRuntimeNode):
        # Attach only unbind nodes to runtime tree.
        operator.add_runtime_children(*unbind_nodes)
        return operator

    return IcoRuntimeWrapper[I, O](operator, runtime_children=unbind_nodes)
