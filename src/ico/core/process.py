from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Literal, TypeAlias, TypeVar, final

from ico.core.operator import (
    IcoOperator,
    IcoOperatorProtocol,
    wrap_operator,
)
from ico.core.runtime.cancel import check_cancelled
from ico.core.signature import IcoSignature

C = TypeVar("C")  # noqa: E741

OnLimit: TypeAlias = Literal["return", "raise"]


class IcoProcessLimitError(RuntimeError):
    """Raised when ``until`` is still false after ``max_iterations`` steps
    and the process was created with ``on_limit="raise"``.

    Attributes:
        process_name: Name of the process node.
        iterations: Number of iterations performed (``max_iterations``).
        state: The last state, so that a caller can inspect or resume it.
    """

    def __init__(self, process_name: str, iterations: int, state: object) -> None:
        super().__init__(
            f"process '{process_name}': stop condition not met "
            f"after {iterations} iterations"
        )
        self.process_name = process_name
        self.iterations = iterations
        self.state = state


@final
class IcoProcess(
    Generic[C],
    IcoOperator[C, C],
):
    """
    Repeatedly applies an operator to the same context for iterative computations.

    Represents an update rule applied several times to the same state.
    Useful for iterative refinement, optimization, simulation steps, or modeling
    recursive algorithms. Two modes are supported:

    * fixed: ``num_iterations=N`` applies the body exactly N times;
    * conditional: ``until=pred, max_iterations=N`` applies the body while
      ``pred(state)`` is falsy, at most N times. The condition is checked
      before every iteration, so an initial state that already satisfies it
      is returned unchanged. When the limit is reached, ``on_limit`` decides
      whether to return the last state (default) or raise
      :class:`IcoProcessLimitError`.

    Generic Parameters:
        C: Context type - the type of state being iteratively transformed.

    ICO signature:
        C → C → C   (repeated `num_iterations` times, or until `until(C)`)

    Example:
        Fibonacci sequence as a process - iterative ICO process can
        model recursion or stateful computations:

        >>> def fib_step(state: tuple[int, int]) -> tuple[int, int]:
        ...     return (state[1], state[0] + state[1])
        >>>
        >>> fib_process = IcoProcess(fib_step, num_iterations=8)
        >>> result = fib_process((0, 1))
        >>> assert result == (21, 34)

        Iterate until convergence, with a safety limit:

        >>> def halve(x: float) -> float:
        ...     return x / 2
        >>> def small(x: float) -> bool:
        ...     return x < 1
        >>> shrink = IcoProcess(halve, until=small, max_iterations=100)
        >>> assert shrink(10.0) == 0.625

    Attributes:
        body: The wrapped operator that performs each iteration step.
        max_iterations: Exact iteration count in fixed mode, upper bound in
            conditional mode.
        until: Stop condition operator, or ``None`` in fixed mode.
        on_limit: Behaviour when the limit is reached in conditional mode.
    """

    __slots__ = ("body", "max_iterations", "until", "on_limit")

    body: IcoOperatorProtocol[C, C]
    max_iterations: int
    until: IcoOperatorProtocol[C, Any] | None
    on_limit: OnLimit

    def __init__(
        self,
        body: Callable[[C], C],
        *,
        num_iterations: int | None = None,
        until: Callable[[C], object] | None = None,
        max_iterations: int | None = None,
        on_limit: OnLimit = "return",
        name: str | None = None,
        strict: bool | None = None,
    ):
        """Initialize a process in fixed or conditional mode.

        Args:
            body: Callable that transforms C → C for each iteration step.
                  Will be automatically wrapped as an IcoOperator.
            num_iterations: Fixed mode: apply the body exactly this many
                times. Must be non-negative.
            until: Conditional mode: stop as soon as ``until(state)`` is
                truthy. Requires ``max_iterations``.
            max_iterations: Conditional mode: upper bound on iterations.
                Must be non-negative.
            on_limit: Conditional mode: ``"return"`` the last state or
                ``"raise"`` :class:`IcoProcessLimitError` when the limit is
                reached without meeting the condition.
            name: Optional name for this process (defaults to "process").
            strict: Check at runtime that the body output can be passed to
                ``until``. ``None`` follows the current
                ``strict_signatures()`` context (off by default).

        Raises:
            ValueError: On an invalid combination of arguments or a negative
                iteration count.
            IcoSignatureError: In strict mode, if ``until`` cannot accept the
                body output.
        """
        iterations = _validate_mode(num_iterations, until, max_iterations, on_limit)

        body_op = wrap_operator(body)
        until_op = wrap_operator(until) if until is not None else None
        process_name = name or "process"

        if until_op is not None:
            from ico.core.signature_check import (
                IcoSignatureError,
                check_link,
                is_strict,
            )

            if strict if strict is not None else is_strict():
                issue = check_link(body_op, until_op)
                if issue is not None and issue.is_error:
                    raise IcoSignatureError(issue)

        super().__init__(
            fn=self._process_fn,
            name=process_name,
            children=[body_op] if until_op is None else [body_op, until_op],
        )
        self.body = body_op
        self.max_iterations = iterations
        self.until = until_op
        self.on_limit = on_limit

    @property
    def num_iterations(self) -> int:
        """Iteration count in fixed mode; the upper bound when ``until`` is set.

        Kept for backward compatibility; equal to ``max_iterations``.
        """
        return self.max_iterations

    def _process_fn(self, context: C) -> C:
        """Execute the process by iteratively applying the body operator.

        Args:
            context: Initial context value of type C.

        Returns:
            The final context: after ``max_iterations`` steps in fixed mode,
            or once ``until`` holds (or the limit is reached) in
            conditional mode.

        Raises:
            IcoProcessLimitError: In conditional mode with
                ``on_limit="raise"`` when the limit is reached.

        Note:
            This is the internal implementation used by __call__.
            Each iteration passes the output of the previous as input
            to the next iteration.
        """
        until = self.until
        if until is None:
            for _ in range(self.max_iterations):
                check_cancelled()
                context = self.body(context)
            return context

        for _ in range(self.max_iterations):
            if until(context):
                return context
            check_cancelled()
            context = self.body(context)

        if until(context) or self.on_limit == "return":
            return context
        raise IcoProcessLimitError(self.name or "process", self.max_iterations, context)

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


def _validate_mode(
    num_iterations: int | None,
    until: object | None,
    max_iterations: int | None,
    on_limit: str,
) -> int:
    """Validate the process mode arguments and return the iteration bound."""
    if num_iterations is not None:
        if until is not None or max_iterations is not None:
            raise ValueError(
                "Use either num_iterations (fixed mode) or until with "
                "max_iterations (conditional mode), not both."
            )
        iterations = num_iterations
    elif until is not None:
        if max_iterations is None:
            raise ValueError(
                "until requires max_iterations as a safety limit; "
                "pass a large value if the loop is expected to run long."
            )
        iterations = max_iterations
    elif max_iterations is not None:
        raise ValueError(
            "max_iterations requires until; use num_iterations for a fixed count."
        )
    else:
        raise ValueError(
            "Either num_iterations or until with max_iterations is required."
        )

    if iterations < 0:
        raise ValueError(f"Iteration count must be non-negative, got {iterations}.")
    if on_limit not in ("return", "raise"):
        raise ValueError(f"on_limit must be 'return' or 'raise', got {on_limit!r}.")
    return iterations
