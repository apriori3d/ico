from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar, overload

from apriori.ico.core.node import IcoNode
from apriori.ico.core.signature import IcoSignature

# ────────────────────────────────────────────────
# Generic type variables for ICO model
# ────────────────────────────────────────────────

I = TypeVar("I")  # noqa: E741
O = TypeVar("O")  # noqa: E741


# variables for composition
I2 = TypeVar("I2")
O2 = TypeVar("O2")


# ────────────────────────────────────────────────
# Operator Class
# ────────────────────────────────────────────────


class IcoOperator(Generic[I, O], IcoNode):
    """
    A fundamental building block and base class for synchronous operations in the ICO framework.

    Generic Parameters:
        I: Input type - the type of data this operator accepts.
        O: Output type - the type of data this operator produces.

    ICO signature:
        I → O
        fn: I → O

    An `IcoOperator` wraps a callable and provides:
    • chainable transformations via `|` or `chain()`
    • wrapping into an `Iterable` with `.stream()`

    Example:
        >>> from apriori.ico import IcoOperator

        >>> to_float = IcoOperator(float)
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> to_str = IcoOperator(str)

        # Compose: I → O → O2 → O3 == I → O3
        >>> pipeline = to_float | scale | to_str
        >>> print(pipeline("21.0"))
        '42.0'

        # Lazy stream over iterable
        >>> stream = scale.stream()
        >>> for i in stream([1, 2, 3]):
                print(i)
        [2, 4, 6]

    """

    # Note:  __slots__ is not used here to allow dynamic inference of ICO-signature attributes

    fn: Callable[[I], O]

    def __init__(
        self,
        fn: Callable[[I], O],
        *,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        """Initialize an IcoOperator with a callable function.

        Args:
            fn: The callable function to wrap. Should accept type I and return type O.
            name: Optional human-readable identifier for this operator.
            parent: Optional parent node to establish hierarchy.
            children: Optional sequence of child nodes to connect.

        Note:
            The operator inherits tree structure functionality from IcoNode.
        """
        IcoNode.__init__(
            self,
            name=name,
            parent=parent,
            children=children,
        )
        self.fn = fn

    def __call__(self, item: I) -> O:
        """Execute the wrapped function with the given input.

        Args:
            item: The input value of type I.

        Returns:
            The output value of type O after applying the wrapped function.
        """
        return self.fn(item)

    # ────────────────────────────────────────────────
    # Compositions Protocols
    # ────────────────────────────────────────────────

    def chain(self, other: IcoOperator[O, O2]) -> IcoOperator[I, O2]:
        """Function chaining: (I → O, O → O2) == I → O2."""
        from apriori.ico.core.chain import chain

        return chain(self, other)

    def __or__(self, other: IcoOperator[O, O2]) -> IcoOperator[I, O2]:
        """Pipe composition operator: a | b == a.chain(b).

        Args:
            other: The operator to chain with this one.

        Returns:
            A new operator that applies this operator followed by the other.
        """
        return self.chain(other)

    def stream(self) -> IcoOperator[Iterator[I], Iterator[O]]:
        """Apply this operator element-wise over an iterable (lazy generator).

        Transforms: Iterable[I] → Iterable[O]

        Returns:
            An IcoStream that applies this operator to each element of an iterable.
        """
        from apriori.ico.core.stream import IcoStream

        return IcoStream(self)

    # ────────────────────────────────────────────────
    # Signature interface
    # ────────────────────────────────────────────────

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature of this operator.

        Attempts to determine input and output types through:
        1. Generic type parameters if available
        2. Callable signature inspection
        3. Fallback to Any types

        Returns:
            IcoSignature containing input, context, and output type information.
        """
        from apriori.ico.core.signature_utils import (
            get_generic_args,
            infer_from_callable,
        )

        # 1. Infer from generic type parameters if available
        args = get_generic_args(self)
        if args is not None:
            if len(args) == 1:
                return IcoSignature(i=args[0], c=None, o=args[0])
            if len(args) == 2:
                return IcoSignature(i=args[0], c=None, o=args[1])

        # 2. Infer from callable signature
        signature = infer_from_callable(self.fn)
        if signature is not None:
            return signature

        # 3. Fallback to Any types
        return IcoSignature(i=type(Any), c=None, o=type(Any), infered=False)


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


def operator() -> Callable[[Callable[[I], O]], IcoOperator[I, O]]:
    """Decorator to wrap a function as an IcoOperator.

    Returns:
        A decorator function that converts a callable into an IcoOperator.

    Example:
        >>> @operator()
        >>> def double(x: int) -> int:
        ...     return x * 2
    """

    def decorator(fn: Callable[[I], O]) -> IcoOperator[I, O]:
        return wrap_operator(fn)

    return decorator


@overload
def wrap_operator(fn: IcoOperator[I, O]) -> IcoOperator[I, O]: ...


@overload
def wrap_operator(fn: Callable[[I], O]) -> IcoOperator[I, O]: ...


def wrap_operator(fn: Callable[[I], O] | IcoOperator[I, O]) -> IcoOperator[I, O]:
    """
    Wrap a raw callable into an IcoOperator only when necessary.

    If the input is already an IcoOperator, returns it unchanged.
    Otherwise, wraps the callable in a new IcoOperator.

    Args:
        fn: A callable or existing IcoOperator to wrap.

    Returns:
        An IcoOperator wrapping the given callable.

    Note:
        Ensures proper type inference for both mypy and pyright.
    """
    if isinstance(fn, IcoOperator):
        # Suppress runtime type checker warning,
        # because we know the type is correct here from static analysis.
        return fn  # pyright: ignore[reportUnknownVariableType]
    return IcoOperator[I, O](fn=fn)
