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
    An atomic transformation unit following the ICO convention.

    ICO form:
        I → O
        fn: I → O

    An `IcoOperator` wraps a callable and provides:
    • chainable transformations via `|` or `chain()`
    • lazy mapping over iterables with `.map()`
    • structured graph representation for flow inspection

    Operators are the fundamental building blocks of ICO pipelines.
    They can represent stateless or stateful transformations,
    depending on the behavior of the wrapped callable.

    Example:
        >>> from apriori.ico import IcoOperator

        >>> to_float = IcoOperator(float)
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> to_str = IcoOperator(str)

        # Compose: I → O → O2 → O3 == I → O3
        >>> pipeline = to_float | scale | to_str
        >>> print(pipeline("21.0"))
        '42.0'

        # Lazy map over iterable
        >>> mapped = scale.map()
        >>> print(list(mapped([1, 2, 3])))
        [2, 4, 6]

        # Inspect flow
        >>> flow = pipeline.describe_flow()
        >>> print(flow.name)
        to_float | scale | to_string
    """

    # Note: do not use __slots__ here to allow dynamic inference of ICO-form attributes

    fn: Callable[[I], O]

    def __init__(
        self,
        fn: Callable[[I], O],
        *,
        name: str | None = None,
        parent: IcoNode | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        IcoNode.__init__(
            self,
            name=name,
            parent=parent,
            children=children,
        )
        self.fn = fn

    def __call__(self, item: I) -> O:
        return self.fn(item)

    # ────────────────────────────────────────────────
    # Compositions Protocols
    # ────────────────────────────────────────────────

    def chain(self, other: IcoOperator[O, O2]) -> IcoOperator[I, O2]:
        """Function chaining: (I → O, O → O2) == I → O2."""
        from apriori.ico.core.chain import chain

        return chain(self, other)

    def __or__(self, other: IcoOperator[O, O2]) -> IcoOperator[I, O2]:
        """Pipe composition: a | b == a.chain(b)."""
        return self.chain(other)

    def stream(self) -> IcoOperator[Iterator[I], Iterator[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """
        from apriori.ico.core.stream import IcoStream

        return IcoStream(self)

    # ────────────────────────────────────────────────
    # Signature interface
    # ────────────────────────────────────────────────

    @property
    def signature(self) -> IcoSignature:
        """Infer ICO signature of this operator."""
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
        return IcoSignature(i=Any, c=None, o=Any, infered=False)


# ─────────────────────────────────────────────
# Operator Wrapping Utility
# ─────────────────────────────────────────────


def operator() -> Callable[[Callable[[I], O]], IcoOperator[I, O]]:
    def decorator(fn: Callable[[I], O]) -> IcoOperator[I, O]:
        return wrap_operator(fn)

    return decorator


@overload
def wrap_operator(fn: IcoOperator[I, O]) -> IcoOperator[I, O]: ...


@overload
def wrap_operator(fn: Callable[[I], O]) -> IcoOperator[I, O]: ...


def wrap_operator(fn: Callable[[I], O] | IcoOperator[I, O]) -> IcoOperator[I, O]:
    """
    Wrap raw callable into IcoOperator only when necessary.
    Ensures type inference for both mypy and pyright.
    """
    if isinstance(fn, IcoOperator):
        # Suppress runtime type checker warning,
        # because we know the type is correct here from static analysis.
        return fn  # pyright: ignore[reportUnknownVariableType]
    return IcoOperator[I, O](fn=fn)
