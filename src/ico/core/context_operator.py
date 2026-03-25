from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, Protocol, TypeVar, cast, overload, runtime_checkable

from ico.core.node import IcoNode, IcoNodeProtocol
from ico.core.operator import I, IContra, O
from ico.core.signature import IcoSignature

# ────────────────────────────────────────────────
# Generic type variables for ICO model
# ────────────────────────────────────────────────
C = TypeVar("C", contravariant=True)  # noqa: E741
CContra = TypeVar("CContra", contravariant=True)
OCovariant = TypeVar("OCovariant", covariant=True)  # noqa: E741

# ────────────────────────────────────────────────
# Operator Class
# ────────────────────────────────────────────────


@runtime_checkable
class IcoContextOperatorProtocol(
    IcoNodeProtocol, Protocol[IContra, CContra, OCovariant]
):
    @property
    def fn(self) -> Callable[[IContra, CContra], OCovariant]: ...

    def __call__(self, item: IContra, context: CContra) -> OCovariant: ...


class IcoContextOperator(Generic[I, C, O], IcoNode):
    """Context-aware operator that processes items while maintaining state.

    IcoContextOperator represents operations that need access to both an input item
    and a context object to produce output. This pattern enables stateful processing,
    accumulation, filtering based on context, and other context-dependent operations.
    Common use cases include training processes where context is a neural network model
    and input is a training batch, or aggregation operations with accumulated state.

    Generic Parameters:
        I: Input item type - the type of data items being processed.
        C: Context type - the type of state/context object used in processing.
        O: Output type - the type produced after processing item with context.

    ICO signature:
        (I, C) → O

    Example:
        >>> # Simple accumulation with context
        >>> def add_to_sum(num: int, total: int) -> int:
        ...     return total + num

        >>> accumulator = IcoContextOperator(add_to_sum, name="accumulator")
        >>> result = accumulator(5, 10)  # 5 + 10 = 15
        >>> assert result == 15

        >>> # Conditional processing based on context
        >>> def multiply_if_even(num: int, multiplier: int) -> int:
        ...     return num * multiplier if num % 2 == 0 else num

        >>> conditional = IcoContextOperator(multiply_if_even, name="conditional")
        >>> assert conditional(4, 3) == 12  # 4 is even: 4 * 3 = 12
        >>> assert conditional(5, 3) == 5   # 5 is odd: unchanged

        >>> # Context modification example
        >>> def update_counter(item: str, counter: dict[str, int]) -> dict[str, int]:
        ...     counter = counter.copy()  # Avoid mutation
        ...     counter[item] = counter.get(item, 0) + 1
        ...     return counter

        >>> counter_op = IcoContextOperator(update_counter, name="counter")
        >>> ctx = {'a': 1, 'b': 2}
        >>> new_ctx = counter_op('a', ctx)
        >>> assert new_ctx == {'a': 2, 'b': 2}

    Attributes:
        fn: The callable function that takes (I, C) and returns O.

    Note:
        Context operators are essential for stateful operations in ICO pipelines.
        They enable epochs, aggregations, and other patterns that need state.
    """

    fn: Callable[[I, C], O]

    def __init__(
        self,
        fn: Callable[[I, C], O],
        *,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNodeProtocol] | None = None,
    ):
        """Initialize a context operator with a callable function.

        Args:
            fn: Function that takes (item: I, context: C) and returns O.
               This is the core operation that will be performed.
            name: Optional name for this operator (useful for debugging/visualization).
            parent: Optional parent node in the computation tree.
            children: Optional child nodes in the computation tree.

        Note:
            The function signature determines the operator's type signature,
            which can be inferred automatically for type safety.
        """
        super().__init__(name=name, parent=parent, children=children)
        self.fn = fn

    def __call__(self, item: I, context: C) -> O:
        """Execute the context operator on given item and context.

        Args:
            item: Input item of type I to be processed.
            context: Context object of type C providing additional state/info.

        Returns:
            Result of type O after processing item with context.

        Note:
            This delegates to the wrapped function, enabling the operator
            to be used as a callable: operator(item, context).
        """
        return self.fn(item, context)

    @property
    def signature(self) -> IcoSignature:
        """Infer ICO signature of this operator."""
        from ico.core.signature_utils import (
            infer_from_callable,
            resolve_types_from_generic,
        )

        # 1. Infer from generic type parameters if available
        i_type, c_type, o_type = resolve_types_from_generic(
            self, IcoContextOperator, I, C, O
        )

        if i_type is not None and c_type is not None and o_type is not None:
            return IcoSignature(i=i_type, c=c_type, o=o_type, infered=True)

        # 2. Infer from callable signature
        signature = infer_from_callable(self.fn)
        if signature is not None:
            return signature

        # 3. Fallback to Any types
        return IcoSignature(i=type[Any], c=type[Any], o=type[Any], infered=False)


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


def context_operator() -> (
    Callable[[Callable[[I, C], C]], IcoContextOperatorProtocol[I, C, C]]
):
    """Decorator for creating context operators that update context.

    This decorator is specifically designed for functions that take an item and
    context, then return an updated context (I, C) → C pattern. This is the
    most common pattern for stateful processing in ICO pipelines.

    Returns:
        Decorator function that wraps callables into IcoContextOperator instances.

    Example:
        >>> @context_operator()
        ... def accumulate_sum(num: int, total: int) -> int:
        ...     return total + num

        >>> # accumulate_sum is now an IcoContextOperator[int, int, int]
        >>> result = accumulate_sum(5, 10)
        >>> assert result == 15

        >>> @context_operator()
        ... def track_max(num: int, current_max: int) -> int:
        ...     return max(num, current_max)

        >>> max_tracker = track_max
        >>> assert max_tracker(7, 5) == 7
        >>> assert max_tracker(3, 7) == 7

    Note:
        The decorated function maintains the same signature but becomes
        a proper ICO node that can participate in pipelines and epochs.
    """

    def decorator(fn: Callable[[I, C], C]) -> IcoContextOperatorProtocol[I, C, C]:
        return wrap_context_operator(fn)

    return decorator


@overload
def wrap_context_operator(
    fn: IcoContextOperatorProtocol[I, C, O],
) -> IcoContextOperatorProtocol[I, C, O]: ...


@overload
def wrap_context_operator(
    fn: Callable[[I, C], O],
) -> IcoContextOperatorProtocol[I, C, O]: ...


def wrap_context_operator(
    fn: Callable[[I, C], O] | IcoContextOperatorProtocol[I, C, O],
) -> IcoContextOperatorProtocol[I, C, O]:
    """Wrap callable into IcoContextOperator only when necessary.

    Utility function that ensures proper wrapping of callables into context
    operators while avoiding double-wrapping. This enables flexible APIs that
    accept both raw functions and pre-wrapped operators.

    Args:
        fn: Either a raw callable (I, C) → O or an existing IcoContextOperator.
           Raw callables will be wrapped, existing operators pass through.

    Returns:
        IcoContextOperator instance, either newly created or the input operator.

    Example:
        >>> def add_numbers(a: int, b: int) -> int:
        ...     return a + b

        >>> # Wrapping a raw function
        >>> op1 = wrap_context_operator(add_numbers)
        >>> assert isinstance(op1, IcoContextOperator)
        >>> assert op1(3, 7) == 10

        >>> # Wrapping an already-wrapped operator (no double-wrap)
        >>> op2 = wrap_context_operator(op1)
        >>> assert op2 is op1  # Same object, no double-wrapping

    Note:
        This function ensures type inference works correctly for both mypy and
        pyright while providing runtime safety against double-wrapping.
    """
    if isinstance(fn, IcoContextOperatorProtocol):
        # Suppress runtime type checker warning,
        # because we know the type is correct here from static analysis.
        return fn  # pyright: ignore[reportUnknownVariableType]
    return cast(IcoContextOperatorProtocol[I, C, O], IcoContextOperator[I, C, O](fn=fn))
