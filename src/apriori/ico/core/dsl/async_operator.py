from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from typing import Any, Generic

from apriori.ico.core.types import (
    I,
    IcoNodeProtocol,
    IcoNodeType,
    O,
)

# ────────────────────────────────────────────────
# Asynchronous Operator Class
# ────────────────────────────────────────────────


class IcoAsyncOperator(Generic[I, O]):
    fn: Callable[[I], Coroutine[Any, Any, O]]

    # IcoNodeProtocol attributes
    name: str
    node_type: IcoNodeType
    _parent: IcoNodeProtocol | None
    children: Sequence[IcoNodeProtocol]

    def __init__(
        self,
        fn: Callable[[I], Coroutine[Any, Any, O]],
        *,
        name: str | None = None,
        node_type: IcoNodeType = IcoNodeType.operator,
        children: Sequence[IcoNodeProtocol] | None = None,
    ):
        super().__init__()
        self.fn = fn

        self.name = (
            name or self.fn.__name__
            if hasattr(self.fn, "__name__")
            else fn.__class__.__name__
        )
        self.node_type = node_type

        self._parent = None
        self.children = children or []
        for child in self.children:
            child.parent = self

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

    async def __call__(self, item: I) -> O:
        return await self.fn(item)
