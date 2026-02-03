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
    A data source operator, produces a data generator without requiring any input.
    ICO Form:
        () → Iterator[O]

    Example:
        >>> dataset = IcoSource(lambda: (1.0, 2.0, 3.0), name="dataset")
        >>> scale = IcoOperator(lambda x: x * 2, name="scale")
        >>> to_sum = IcoOperator(sum, name="sum")

        >>> flow = dataset | IcoStream(body=scale) | to_sum
        >>> result = flow()
        >>> print(result)
        12.0
    """

    provider: Callable[[], Iterable[O]]

    def __init__(
        self,
        provider: Callable[[], Iterable[O]],
        *,
        name: str | None = None,
    ):
        super().__init__(fn=self._iterator_fn, name=name)
        self.provider = provider

    def _iterator_fn(self, _: None) -> Iterator[O]:
        yield from self.provider()

    @property
    def signature(self) -> IcoSignature:
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
    def decorator(fn: Callable[[], Iterable[O]]) -> IcoSource[O]:
        return IcoSource(fn, name=name)

    return decorator
