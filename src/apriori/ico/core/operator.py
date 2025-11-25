from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar, overload

from apriori.ico.core.node import IcoNode

# ────────────────────────────────────────────────
# Generic type variables for ICO model
# ────────────────────────────────────────────────

I = TypeVar("I")  # noqa: E741
C = TypeVar("C")
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
        super().__init__(
            name=name,
            parent=parent,
            children=children,
        )
        self.fn = fn

        if name is None and hasattr(fn, "__name__"):
            self.name = fn.__name__

    def __str__(self) -> str:
        return self.name

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

    def iterate(self) -> IcoOperator[Iterator[I], Iterator[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """
        from apriori.ico.core.iterate import iterate

        return iterate(self)


# ────────────────────────────────────────────────
# Operator Wrapping Utility
# ────────────────────────────────────────────────


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
