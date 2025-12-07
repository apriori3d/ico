from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import ClassVar, Generic

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import (
    I,
    O,
)

# ────────────────────────────────────────────────
# Asynchronous Operator Class
# ────────────────────────────────────────────────


class IcoAsyncOperator(Generic[I, O], IcoNode):
    type_name: ClassVar[str] = "Async Operator"

    fn: Callable[[I], Awaitable[O]]

    def __init__(
        self,
        fn: Callable[[I], Awaitable[O]],
        *,
        name: str | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        super().__init__(
            name=name or fn.__name__ if hasattr(fn, "__name__") else None,
            children=children,
        )
        self.fn = fn

    async def __call__(self, item: I) -> O:
        return await self.fn(item)
