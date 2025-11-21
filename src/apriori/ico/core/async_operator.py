from __future__ import annotations

from collections.abc import Callable, Coroutine, Sequence
from typing import Any, Generic

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import (
    I,
    O,
)

# ────────────────────────────────────────────────
# Asynchronous Operator Class
# ────────────────────────────────────────────────


class IcoAsyncOperator(Generic[I, O], IcoNode):
    fn: Callable[[I], Coroutine[Any, Any, O]]

    def __init__(
        self,
        fn: Callable[[I], Coroutine[Any, Any, O]],
        *,
        name: str | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        super().__init__(
            name=name or self.fn.__name__ if hasattr(self.fn, "__name__") else None,
            children=children,
        )
        self.fn = fn

    async def __call__(self, item: I) -> O:
        return await self.fn(item)
