from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic

from apriori.ico.core.node import IcoNode
from apriori.ico.core.operator import (
    I,
    O,
)
from apriori.ico.core.signature import IcoSignature

# ────────────────────────────────────────────────
# Asynchronous Operator Class
# ────────────────────────────────────────────────


class IcoAsyncOperator(Generic[I, O], IcoNode):
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
