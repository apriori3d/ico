from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

# ──── Generic type variables for composition ────

I2 = TypeVar("I2")
O2 = TypeVar("O2")


# ─── Operator Class ───
class IcoOperator(IcoOperatorProtocol[I, O], Generic[I, O]):
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

    __slots__ = ("fn", "name", "node_type", "children", "parent")

    fn: Callable[[I], O]
    name: str
    node_type: NodeType
    parent: IcoOperatorProtocol[Any, Any] | None
    children: list[IcoOperatorProtocol[Any, Any]]

    def __init__(
        self,
        fn: Callable[[I], O],
        *,
        name: str | None = None,
        node_type: NodeType = NodeType.operator,
        children: list[IcoOperatorProtocol[Any, Any]] | None = None,
    ):
        super().__init__()
        self.fn = fn
        self.name = name or self.__class__.__name__
        self.node_type = node_type
        self.parent = None
        self.children = children if children is not None else list()
        for child in self.children:
            child.parent = self

    # ─── Properties ───

    def __str__(self) -> str:
        return self.name

    # @overload
    # def __call__(self, item: I) -> O:
    #     # Method overload for standard call with input
    #     ...

    # @overload
    # def __call__(self) -> O:
    #     # Method overload for no-argument call in flow with IcoSource
    #     ...

    def __call__(self, item: I) -> O:
        return self.fn(item)

    # ─── Imperative async execution path ───

    async def run_async(self, item: I) -> O:
        """Asynchronous execution of the operator."""
        return self(item)

    # ─── Composition of operators ───

    # ─── Chaining ───

    def chain(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperator[I, O2]:
        """Function chaining: (I → O, O → O2) == I → O2."""

        def chained(x: I) -> O2:
            output = self(x)
            return other(output)

        return IcoOperator(
            fn=chained,
            name="chain",
            node_type=NodeType.chain,
            children=[self, other],
        )

    def __or__(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperator[I, O2]:
        """Pipe composition: a | b == a.chain(b)."""
        return self.chain(other)

    # ─── Map ───

    def map(self) -> IcoOperator[Iterator[I], Iterator[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """

        return IcoOperator(
            fn=self._map_fn,
            name="map",
            node_type=NodeType.map,
            children=[self],
        )

    def _map_fn(self, xs: Iterator[I]) -> Iterator[O]:
        for x in xs:
            yield self(x)


# ─── Operator Wrapping Utility ───

from typing import TypeGuard


def _is_operator(obj: object) -> TypeGuard[IcoOperator[I, O]]:
    return isinstance(obj, IcoOperator)


def wrap_operator(fn: Callable[[I], O] | IcoOperator[I, O]) -> IcoOperator[I, O]:
    """
    Wrap raw callable into IcoOperator only when necessary.
    Ensures type inference for both mypy and pyright.
    """
    if _is_operator(fn):
        return fn
    return IcoOperator(fn=fn)


# ─── Tree traversal Utilities ───


def iterate_nodes(
    node: IcoOperatorProtocol[Any, Any],
) -> Iterator[IcoOperatorProtocol[Any, Any]]:
    """Recursively yield all children operators in the flow tree."""
    yield node
    for c in node.children:
        yield from iterate_nodes(c)


def iterate_parents(
    node: IcoOperatorProtocol[Any, Any],
) -> Iterator[IcoOperatorProtocol[Any, Any]]:
    """Recursively yield all parent operators in the flow tree."""
    if node.parent is None:
        return

    yield node.parent
    yield from iterate_parents(node.parent)
