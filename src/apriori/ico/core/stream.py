from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, final

from apriori.ico.core.operator import (
    I,
    IcoOperator,
    O,
    wrap_operator,
)
from apriori.ico.core.signature import IcoSignature


@final
class IcoStream(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    """
    Applies a body operator element-wise to each item in a data stream.

    IcoStream enables lazy, element-by-element processing of iterators by
    wrapping a single-item operator and applying it to each element in sequence.
    This creates efficient streaming pipelines with minimal memory overhead.

    Generic Parameters:
        I: Input item type - the type of individual items in the input stream.
        O: Output item type - the type of individual items in the output stream.

    ICO signature:
        Iterator[I] → Iterator[O]

    Example:
        >>> # Typical usage through .stream() API
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> stream = scale.stream()  # Creates IcoStream automatically
        >>> result = list(stream(iter([1.0, 2.0, 3.0])))
        >>> assert result == [2.0, 4.0, 6.0]

        >>> # Chaining with sources and operators
        >>> source = IcoSource(lambda: [1.0, 2.0, 3.0])
        >>> transform = IcoOperator(lambda x: x * 2)
        >>> flow = source | transform.stream()
        >>> result = list(flow(None))
        >>> assert result == [2.0, 4.0, 6.0]

        >>> # Direct construction (less common)
        >>> stream = IcoStream(lambda x: str(x))
        >>> result = list(stream(iter([1, 2, 3])))
        >>> assert result == ['1', '2', '3']

    Attributes:
        body: The wrapped operator that processes individual items.

    Note:
        IcoStream provides lazy evaluation – items are processed on-demand
        as the output iterator is consumed, not when the stream is created.
    """

    __slots__ = ("body",)

    body: IcoOperator[I, O]

    def __init__(
        self,
        body: Callable[[I], O],
        *,
        name: str | None = None,
    ):
        """Initialize a stream with a body operator for element-wise processing.

        Args:
            body: Callable that transforms individual items I → O.
                  Will be automatically wrapped as an IcoOperator if needed.
            name: Optional name for this stream (useful for debugging/visualization).

        Note:
            The body operator is stored as a child node and will be wrapped
            in an IcoOperator if it isn't already one.
        """
        body_op = wrap_operator(body)

        super().__init__(
            fn=self._stream_fn,
            name=name,
            children=[body_op],
        )
        self.body = body_op

    def _stream_fn(self, items: Iterator[I]) -> Iterator[O]:
        """Internal implementation that applies the body operator to each item.

        Args:
            items: Iterator of input items to process.

        Yields:
            Transformed items after applying the body operator to each input.

        Note:
            This is the function used by __call__. It provides lazy evaluation
            by yielding items one at a time as they're requested.
        """
        for item in items:
            yield self.body(item)

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this stream.

        Derives the stream signature from the body operator's signature,
        wrapping the input/output types in Iterator containers.

        Returns:
            IcoSignature with Iterator[I] input and Iterator[O] output,
            where I and O are derived from the body operator's signature.

        Note:
            If the parent signature is not inferred, falls back to using
            the body operator's signature directly.
        """
        signature = super().signature

        # If signature is undefined, infer from body operator
        if not signature.infered:
            signature = self.body.signature

        return IcoSignature(
            i=Iterator[signature.i],
            c=None,
            o=Iterator[signature.o],
        )
