from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum, auto
from typing import Protocol, TypeVar, runtime_checkable

# ──── Generic type variables for ICO model ────

I = TypeVar("I", contravariant=True)  # noqa: E741
C = TypeVar("C")
O = TypeVar("O")  # noqa: E741

# ──── Generic type variables for composition ────

I2 = TypeVar("I2")
O2 = TypeVar("O2")


# ─── Node Types ───


class NodeType(Enum):
    operator = auto()
    chain = auto()
    map = auto()
    pipeline = auto()
    stream = auto()
    process = auto()
    source = auto()
    sink = auto()
    runtime = auto()
    agent_link = auto()
    agent = auto()


# ────────────────────────────────────────────────
# Operator Protocols
# ────────────────────────────────────────────────


# @runtime_checkable
class IcoOperatorProtocol(Protocol[I, O]):
    """Protocol for ICO operators for execution and composition."""

    def __call__(self, item: I) -> O: ...

    def chain(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def map(self) -> IcoOperatorProtocol[Iterator[I], Iterator[O]]: ...


@runtime_checkable
class IcoTreeProtocol(Protocol):
    """Structural attributes for graph representation of ICO operators."""

    name: str
    node_type: NodeType
    children: Sequence[IcoTreeProtocol]

    @property
    def parent(self) -> IcoTreeProtocol | None: ...

    @parent.setter
    def parent(self, value: IcoTreeProtocol | None) -> None: ...
