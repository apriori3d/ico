from collections.abc import Callable, Iterator
from typing import Any, Generic

from ico.core.operator import (
    I,
    IcoOperator,
)
from ico.core.signature import IcoSignature


class IcoSink(
    Generic[I],
    IcoOperator[Iterator[I], None],
):
    """A terminal operator that consumes data streams and ends the flow.

    IcoSink represents the final stage in an ICO data pipeline, consuming
    each item from an iterator without producing any output. This makes it
    the terminal point where data flows end and side effects occur.

    Generic Parameters:
        I: Input item type - the type of individual items being consumed.

    ICO signature:
        Iterator[I] → ()

    Example:
        >>> def print_item(x: int) -> None:
        ...     print(f"Processing: {x}")

        >>> sink = IcoSink(print_item)
        >>> sink(iter([1, 2, 3]))  # Prints each number

        >>> # Using as decorator
        >>> @sink(name="logger")
        >>> def log_item(x: str) -> None:
        ...     logger.info(x)

    Attributes:
        consumer: Function that processes each individual item from the stream.

    Note:
        Sinks are designed to handle side effects like logging, saving to files,
        or displaying results, without returning values to continue the pipeline.
    """

    consumer: Callable[[I], None]

    def __init__(
        self,
        consumer: Callable[[I], None],
        name: str | None = None,
    ) -> None:
        """Initialize a sink with a consumer function.

        Args:
            consumer: Function that processes individual items from the stream.
                     Should accept type I and return None (side effects only).
            name: Optional name for this sink (useful for debugging/logging).

        Note:
            The consumer function will be called once for each item in any
            iterator passed to this sink.
        """
        super().__init__(fn=self._sink_fn, name=name)
        self.consumer = consumer

    def _sink_fn(self, items: Iterator[I]) -> None:
        """Internal implementation that consumes all items from the iterator.

        Args:
            items: Iterator of items to consume.

        Note:
            This is the function used by __call__. It iterates through all
            items and applies the consumer function to each one.
        """
        for item in items:
            self.consumer(item)

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this sink.

        Attempts to determine the input type by analyzing the consumer function's
        signature, then wraps it in Iterator[T] → () format.

        Returns:
            IcoSignature with Iterator[I] input and None output.

        Note:
            Falls back to Iterator[Any] → () if type inference fails.
            The output is always None since sinks are terminal operations.
        """
        from ico.core.signature_utils import (
            infer_from_callable,
        )

        signature = super().signature

        if signature.infered:
            # Help mypy to understand this is a type, not just a variable
            i_type: Any = signature.i

            return IcoSignature(
                i=Iterator[i_type],
                c=None,
                o=type(None),
            )

        # Infer from consumer callable
        consumer_signature = infer_from_callable(self.consumer)

        if consumer_signature is not None:
            ci_type: Any = consumer_signature.i

            return IcoSignature(
                i=Iterator[ci_type],
                c=None,
                o=type(None),
            )

        # Fallback to Any types
        return IcoSignature(
            i=Iterator[Any],
            c=None,
            o=type(None),
            infered=False,
        )


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def sink(
    *,
    name: str | None = None,
) -> Callable[[Callable[[I], None]], IcoSink[I]]:
    """Decorator to create an IcoSink from a consumer function.

    Provides a convenient way to convert any function that processes individual
    items into a sink that can consume iterators in ICO pipelines.

    Args:
        name: Optional name for the created sink.

    Returns:
        A decorator function that converts consumer functions to IcoSink instances.

    Example:
        >>> @sink(name="printer")
        ... def print_numbers(x: int) -> None:
        ...     print(f"Number: {x}")

        >>> # print_numbers is now an IcoSink[int]
        >>> print_numbers(iter([1, 2, 3]))  # Prints: Number: 1, Number: 2, Number: 3

    Note:
        The decorated function should accept a single argument and return None.
        The resulting sink will handle iterator consumption automatically.
    """

    def decorator(fn: Callable[[I], None]) -> IcoSink[I]:
        return IcoSink(fn, name=name)

    return decorator
