from collections.abc import Callable, Iterable, Iterator
from typing import Any, Generic, final

from apriori.ico.core.operator import (
    IcoOperator,
    O,
)
from apriori.ico.core.signature import IcoSignature


@final
class IcoSource(
    Generic[O],
    IcoOperator[None, Iterator[O]],
):
    """
    A data source operator that produces data streams without requiring input.

    IcoSource represents the starting point of an ICO data pipeline, generating
    data from a provider function. It converts zero-input data providers into
    standardized iterator-producing operators.

    Generic Parameters:
        O: Output item type - the type of individual items produced by this source.

    ICO signature:
        () → Iterator[O]

    Example:
        >>> def get_numbers() -> list[float]:
        ...     return [1.0, 2.0, 3.0]

        >>> dataset = IcoSource(get_numbers, name="dataset")
        >>> scale = IcoOperator(lambda x: x * 2, name="scale")
        >>> sum_op = IcoOperator(sum, name="sum")

        >>> flow = dataset | scale.stream() | sum_op
        >>> result = flow(None)  # Takes None as input
        >>> assert result == 12.0

        >>> # Using as decorator
        >>> @source(name="test_data")
        >>> def load_test_data() -> range:
        ...     return range(5)

    Attributes:
        provider: Function that generates the source data when called.

    Note:
        The source always takes None as input and produces an Iterator[O].
        This allows sources to be the first operator in any ICO pipeline.
    """

    provider: Callable[[], Iterable[O]]

    def __init__(
        self,
        provider: Callable[[], Iterable[O]],
        *,
        name: str | None = None,
    ):
        """Initialize a source with a data provider function.

        Args:
            provider: Zero-argument function that returns an Iterable[O].
                     Called each time the source is executed to generate fresh data.
            name: Optional name for this source (useful for debugging/visualization).

        Note:
            The provider function should be side-effect free and return consistent
            data types. It will be called every time the source operator is executed.
        """
        super().__init__(fn=self._iterator_fn, name=name)
        self.provider = provider

    def _iterator_fn(self, _: None) -> Iterator[O]:
        """Internal implementation that converts provider output to iterator.

        Args:
            _: Ignored None input (sources don't use their input).

        Returns:
            Iterator yielding all items from the provider's output.

        Note:
            This is the function used by __call__. It calls the provider
            and yields each item, converting any Iterable to an Iterator.
        """
        yield from self.provider()

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this source.

        Attempts to determine the output type by analyzing the provider function's
        return type, then formats it as () → Iterator[O].

        Returns:
            IcoSignature with None input and Iterator[O] output.

        Note:
            Falls back to () → Iterator[Any] if type inference fails.
            The input is always None since sources generate data without input.
        """
        from types import GenericAlias
        from typing import get_args

        from apriori.ico.core.signature_utils import (
            infer_from_callable,
        )

        signature = super().signature
        if signature.infered:
            return IcoSignature(
                i=None,
                c=None,
                o=Iterator[signature.o],
            )

        # Infer from provider callable
        provider_signature = infer_from_callable(self.provider)

        # Provider returns an Iterable[T], we need to convert it to Iterator[T]
        # to match IcoSource signature
        if provider_signature is not None and isinstance(
            provider_signature.o, GenericAlias
        ):
            o_args = get_args(provider_signature.o)

            return IcoSignature(
                i=type(None),
                c=type(None),
                o=Iterator[o_args],
            )

        return IcoSignature(
            i=None,
            c=None,
            o=Iterator[Any],
            infered=False,
        )


# ─────────────────────────────────────────────
# Decorator
# ─────────────────────────────────────────────


def source(
    *,
    name: str | None = None,
) -> Callable[[Callable[[], Iterable[O]]], IcoSource[O]]:
    """Decorator to create an IcoSource from a provider function.

    Provides a convenient way to convert any zero-argument function that returns
    an iterable into a source operator for ICO pipelines.

    Args:
        name: Optional name for the created source.

    Returns:
        A decorator function that converts provider functions to IcoSource instances.

    Example:
        >>> @source(name="fibonacci")
        ... def fib_sequence() -> list[int]:
        ...     return [1, 1, 2, 3, 5, 8, 13]

        >>> # fib_sequence is now an IcoSource[int]
        >>> numbers = fib_sequence(None)  # Returns Iterator[int]
        >>> print(list(numbers))  # [1, 1, 2, 3, 5, 8, 13]

        >>> @source()
        ... def load_config() -> dict[str, Any]:
        ...     return {"key": "value"}

    Note:
        The decorated function should take no arguments and return an Iterable.
        The resulting source will handle None input automatically.
    """

    def decorator(fn: Callable[[], Iterable[O]]) -> IcoSource[O]:
        return IcoSource(fn, name=name)

    return decorator
