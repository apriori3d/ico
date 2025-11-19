from __future__ import annotations

from collections.abc import Coroutine, Iterator, Sequence
from enum import Enum, auto
from typing import Any, Protocol, TypeVar, runtime_checkable

# ────────────────────────────────────────────────
# Computational Protocols
# ────────────────────────────────────────────────


# Generic type variables for ICO model
I = TypeVar("I", contravariant=True)  # noqa: E741
C = TypeVar("C")
O = TypeVar("O", covariant=True)  # noqa: E741


# Generic type variables for composition
I2 = TypeVar("I2")
O2 = TypeVar("O2")


@runtime_checkable
class IcoCallableProtocol(Protocol[I, O]):
    """Protocol for ICO operators for execution and composition."""

    def __call__(self, item: I) -> O: ...


@runtime_checkable
class IcoAsyncCallableProtocol(Protocol[I, O]):
    """Protocol for ICO operators for execution and composition."""

    async def __call__(self, item: I) -> Coroutine[Any, Any, O]: ...

    # async def acall(self, item: I) -> O: ...


class IcoCompositionalProtocol(Protocol[I, O]):
    """Protocol for ICO operators for execution and composition."""

    def chain(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def map(self) -> IcoOperatorProtocol[Iterator[I], Iterator[O]]: ...


# ────────────────────────────────────────────────
# Operators Tree Protocols
# ────────────────────────────────────────────────


@runtime_checkable
class IcoNodeProtocol(Protocol):
    """Structural attributes for graph representation of ICO operators."""

    name: str
    node_type: IcoNodeType
    children: Sequence[IcoNodeProtocol]

    @property
    def parent(self) -> IcoNodeProtocol | None: ...

    @parent.setter
    def parent(self, value: IcoNodeProtocol | None) -> None: ...


class IcoNodeType(Enum):
    operator = auto()

    # Specialized nodes
    source = auto()
    sink = auto()
    runtime = auto()

    # Structural nodes
    pipeline = auto()
    process = auto()

    # Compositional nodes
    chain = auto()
    map = auto()
    stream = auto()

    unknown = auto()

    # agent = auto()


# ────────────────────────────────────────────────
# ICO Form Typed Node Protocol
# ────────────────────────────────────────────────


class IcoTypedNodeProtocol(Protocol):
    ico_input: type | None
    ico_context: type | None
    ico_output: type | None


# ────────────────────────────────────────────────
# Combined Protocols for Operator
# ────────────────────────────────────────────────


@runtime_checkable
class IcoOperatorProtocol(
    IcoNodeProtocol,
    IcoCallableProtocol[I, O],
    IcoCompositionalProtocol[I, O],
    Protocol[I, O],
):
    """Protocol for ICO operators combining computation and tree structure."""

    ...
