from __future__ import annotations

import operator
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Generic, cast, final

from ico.core.length import passing_length
from ico.core.operator import (
    I,
    IcoOperator,
    IcoOperatorProtocol,
    O,
    wrap_operator,
)
from ico.core.runtime.cancel import check_cancelled
from ico.core.signature import IcoSignature

if TYPE_CHECKING:
    from ico.core.runtime.progress import IcoProgress


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

    body: IcoOperatorProtocol[I, O]

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
        """Apply the body to each item, lazily. One output per input: the
        outputs keep the input's length hint, so progress totals survive a
        chain of streams."""
        return passing_length(items, self._iterate(items))

    def _iterate(self, items: Iterator[I]) -> Iterator[O]:
        counters = self._counters()
        if counters:
            length = operator.length_hint(items, -1)
            if length >= 0:
                for counter in counters:
                    counter.start(length)
        for item in items:
            check_cancelled()
            yield self.body(item)
        # The iteration is over: its progress is reported before the next.
        for progress in self._progress_steps():
            progress.flush()

    def _counters(self) -> list[IcoProgress[Any]]:
        """Progress nodes without a total among the body's own steps (not
        inside nested streams or forks): they count this stream's items."""
        return [p for p in self._progress_steps() if p.total is None]

    def _progress_steps(self) -> list[IcoProgress[Any]]:
        """Progress nodes among the body's own steps (not inside nested
        streams or forks)."""
        from ico.core.chain import IcoChain
        from ico.core.runtime.progress import IcoProgress
        from ico.registry.registry import get_meta

        found: list[IcoProgress[Any]] = []

        def visit(node: object) -> None:
            if get_meta(node) is None and isinstance(node, IcoChain):
                for child in cast(IcoChain[Any, Any, Any], node).children:
                    visit(child)
            elif isinstance(node, IcoProgress):
                found.append(cast(IcoProgress[Any], node))

        visit(self.body)
        return found

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
        if signature.infered:
            return signature

        # Infer from body operator's signature and wrap in Iterator
        signature = self.body.signature
        if signature.infered:
            return IcoSignature(
                i=Iterator[signature.i],  # type: ignore[name-defined]
                c=None,
                o=Iterator[signature.o],  # type: ignore[name-defined]
                infered=True,
            )

        return signature
