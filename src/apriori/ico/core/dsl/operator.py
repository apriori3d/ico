from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Generic, overload

from apriori.ico.core.types import (
    O2,
    I,
    IcoNodeProtocol,
    IcoNodeType,
    IcoOperatorProtocol,
    O,
)

# ────────────────────────────────────────────────
# Operator Class
# ────────────────────────────────────────────────


class IcoOperator(Generic[I, O]):
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

    # IcoNodeProtocol attributes
    name: str
    node_type: IcoNodeType
    _parent: IcoNodeProtocol | None
    children: Sequence[IcoNodeProtocol]

    def __init__(
        self,
        fn: Callable[[I], O],
        *,
        name: str | None = None,
        node_type: IcoNodeType = IcoNodeType.operator,
        children: Sequence[IcoNodeProtocol] | None = None,
    ):
        super().__init__()
        self.fn = fn

        self.name = name or self.fn.__name__
        self.node_type = node_type

        self._parent = None
        self.children = children or []
        for child in self.children:
            child.parent = self

    def __str__(self) -> str:
        return self.name

    # ────────────────────────────────────────────────
    # IcoTree Protocol
    # ────────────────────────────────────────────────

    @property
    def parent(self) -> IcoNodeProtocol | None:
        return self._parent

    @parent.setter
    def parent(self, value: IcoNodeProtocol | None) -> None:
        self._parent = value

    # ────────────────────────────────────────────────
    # Computation Protocols
    # ────────────────────────────────────────────────

    def __call__(self, item: I) -> O:
        return self.fn(item)

    def chain(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperatorProtocol[I, O2]:
        """Function chaining: (I → O, O → O2) == I → O2."""

        def chained_fn(x: I) -> O2:
            output = self(x)
            return other(output)

        return IcoOperator[I, O2](
            fn=chained_fn,
            name="chain",
            node_type=IcoNodeType.chain,
            children=[self, other],
        )

    def __or__(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperatorProtocol[I, O2]:
        """Pipe composition: a | b == a.chain(b)."""
        return self.chain(other)

    def map(self) -> IcoOperatorProtocol[Iterator[I], Iterator[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """

        return IcoOperator[Iterator[I], Iterator[O]](
            fn=self._map_fn,
            name="map",
            node_type=IcoNodeType.map,
            children=[self],
        )

    def _map_fn(self, items: Iterator[I]) -> Iterator[O]:
        for item in items:
            yield self(item)


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
