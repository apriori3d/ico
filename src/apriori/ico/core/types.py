from __future__ import annotations

from collections.abc import Callable, Iterator
from enum import Enum, auto
from typing import Any, Protocol, TypeVar, runtime_checkable

# ──── Generic type variables for ICO model ────

I = TypeVar("I")  # noqa: E741
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


# ─── Operator Protocol ───
@runtime_checkable
class IcoOperatorProtocol(Protocol[I, O]):
    """
    Protocol for ICO Operators, defining the expected interface.
    """

    fn: Callable[[I], O]

    # -── Structural attributes for graph representation ───

    name: str
    node_type: NodeType
    parent: IcoOperatorProtocol[Any, Any] | None
    children: list[IcoOperatorProtocol[Any, Any]]

    # ─── Declarative sync execution path ───

    def __call__(self, item: I) -> O: ...

    # ─── Operator composition ───

    def chain(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def __or__(
        self, other: IcoOperatorProtocol[O, O2]
    ) -> IcoOperatorProtocol[I, O2]: ...

    def map(self) -> IcoOperatorProtocol[Iterator[I], Iterator[O]]: ...
