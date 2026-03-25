from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar, final

from ico.core.operator import (
    IcoOperator,
    IcoOperatorProtocol,
    wrap_operator,
)
from ico.core.signature import IcoSignature

C = TypeVar("C")  # noqa: E741


@final
class IcoProcess(
    Generic[C],
    IcoOperator[C, C],
):
    """
    Repeatedly applies an operator to the same context for iterative computations.

    Represents a fixed update rule applied several times to the same state.
    Useful for iterative refinement, optimization, simulation steps, or modeling
    recursive algorithms with explicit iteration counts.

    Generic Parameters:
        C: Context type - the type of state being iteratively transformed.

    ICO signature:
        C → C → C   (repeated `num_iterations` times)

    Example:
        Fibonacci sequence as a process - iterative ICO process can
        model recursion or stateful computations:

        >>> def fib_step(state: tuple[int, int]) -> tuple[int, int]:
        ...     return (state[1], state[0] + state[1])
        >>>
        >>> fib_process = IcoProcess(fib_step, num_iterations=8)
        >>> result = fib_process((0, 1))
        >>> assert result == (21, 34)

    Attributes:
        num_iterations: Number of times to apply the body operator.
        body: The wrapped operator that performs each iteration step.
    """

    __slots__ = ("num_iterations", "body")

    num_iterations: int
    body: IcoOperatorProtocol[C, C]

    def __init__(
        self,
        body: Callable[[C], C],
        *,
        num_iterations: int,
        name: str | None = None,
    ):
        """Initialize a process with an iteration body and step count.

        Args:
            body: Callable that transforms C → C for each iteration step.
                  Will be automatically wrapped as an IcoOperator.
            num_iterations: Number of times to repeatedly apply the body.
                           Must be non-negative.
            name: Optional name for this process (defaults to "process").

        Note:
            The body callable is wrapped as an IcoOperator and stored as
            the single child of this process node.
        """
        # body_fn = body
        body = wrap_operator(body)

        super().__init__(
            fn=self._process_fn,
            name=name or "process",
            children=[body],
        )
        self.body = body
        self.num_iterations = num_iterations

    def _process_fn(self, context: C) -> C:
        """Execute the process by iteratively applying the body operator.

        Args:
            context: Initial context value of type C.

        Returns:
            The final context after applying the body operator
            num_iterations times sequentially.

        Note:
            This is the internal implementation used by __call__.
            Each iteration passes the output of the previous as input
            to the next iteration.
        """
        for _ in range(self.num_iterations):
            context = self.body(context)
        return context

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature of this process.

        Attempts to use the parent class signature inference first.
        If that fails, delegates to the body operator's signature.

        Returns:
            IcoSignature representing the type flow C → C of this process.

        Note:
            Since the process iteratively applies C → C transformations,
            the overall signature remains C → C, matching the body's signature.
        """
        signature = super().signature

        return signature if signature.infered else self.body.signature
