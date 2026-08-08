from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic

from ico.core.node import IcoNode
from ico.core.operator import (
    I,
    O,
)
from ico.core.signature import IcoSignature

# ────────────────────────────────────────────────
# Asynchronous Operator Class
# ────────────────────────────────────────────────


class IcoAsyncOperator(Generic[I, O], IcoNode):
    """Asynchronous ICO operator for non-blocking operations.

    IcoAsyncOperator represents async operations that return awaitable results.
    This enables non-blocking processing for I/O operations, network calls,
    database queries, and other async computations. These operators are primarily
    used within IcoAsyncStream for concurrent processing of stream items.

    Generic Parameters:
        I: Input type - the type of data this async operator accepts.
        O: Output type - the type of data this async operator produces.

    ICO signature:
        I → Awaitable[O]

    Example:
        >>> import asyncio

        >>> # Simple async computation
        >>> async def async_double(x: int) -> int:
        ...     await asyncio.sleep(0.1)  # Simulate async work
        ...     return x * 2

        >>> doubler = IcoAsyncOperator(async_double, name="async_doubler")
        >>> result = await doubler(5)  # Returns 10 after async delay
        >>> # assert result == 10

    Attributes:
        fn: The async callable function that takes I and returns Awaitable[O].

    Note:
        All operations must be awaited. These operators are typically used within
        IcoAsyncStream for concurrent processing rather than standalone execution.
    """

    fn: Callable[[I], Awaitable[O]]

    def __init__(
        self,
        fn: Callable[[I], Awaitable[O]],
        *,
        name: str | None = None,
        children: Sequence[IcoNode] | None = None,
    ):
        """Initialize an async operator with an async callable function.

        Args:
            fn: Async function that takes I and returns Awaitable[O].
               This is the core async operation that will be performed.
            name: Optional name for this operator. Auto-derived from function name
                 if not provided and function has __name__ attribute.
            children: Optional child nodes in the computation tree.

        Note:
            The function must be async (return Awaitable[O]). For sync functions,
            use regular IcoOperator instead. Name auto-derivation helps with debugging.
        """
        super().__init__(
            name=name or fn.__name__ if hasattr(fn, "__name__") else None,
            children=children,
        )
        self.fn = fn

    async def __call__(self, item: I) -> O:
        """Execute the async operator on given input.

        Args:
            item: Input item of type I to be processed asynchronously.

        Returns:
            Result of type O after awaiting the async operation.

        Note:
            This method must be awaited since it delegates to the async function.
            Used to make the operator callable: await operator(item).
        """
        return await self.fn(item)

    # ────────────────────────────────────────────────
    # Signature interface
    # ────────────────────────────────────────────────

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this async operator.

        Attempts to derive types from generic parameters or callable inspection.
        For async functions, the return type should be Awaitable[O], and this
        method extracts the inner type O for the signature.

        Returns:
            IcoSignature with input type I, no context (None), and output type O.
            Falls back to Any types if inference fails.

        Note:
            The signature represents I → O (not I → Awaitable[O]) since
            the awaiting is handled by the async call mechanism.
        """
        from ico.core.signature_utils import (
            infer_from_callable,
            resolve_types_from_generic,
        )

        # 1. Infer from generic type parameters if available
        i_type, o_type = resolve_types_from_generic(self, IcoAsyncOperator, I, O)

        if i_type is not None and o_type is not None:
            return IcoSignature(i=i_type, c=None, o=o_type, infered=True)

        # 2. Infer from callable signature
        signature = infer_from_callable(self.fn)
        if signature is not None:
            return signature

        # 3. Fallback to Any types
        return IcoSignature(i=type[Any], c=None, o=type[Any], infered=False)
