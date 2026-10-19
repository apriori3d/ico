from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic

from ico.core.length import materialized
from ico.core.operator import IcoOperator, IcoOperatorProtocol, IInv, wrap_operator
from ico.core.signature import IcoSignature


class IcoFilter(Generic[IInv], IcoOperator[Iterator[IInv], Iterator[IInv]]):
    """Lazily keep the stream items for which a predicate is truthy.

    Generic Parameters:
        I: Item type of the stream.

    ICO signature:
        Iterator[I] → Iterator[I]

    Example:
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> evens = keep(is_even)
        >>> assert list(evens(iter([1, 2, 3, 4]))) == [2, 4]

    Attributes:
        predicate: Operator deciding whether an item is kept.
    """

    __slots__ = ("predicate",)

    predicate: IcoOperatorProtocol[IInv, Any]

    def __init__(
        self,
        predicate: Callable[[IInv], object],
        *,
        name: str | None = None,
    ):
        """Initialize a filter.

        Args:
            predicate: Callable or operator; an item is kept when the result
                is truthy. Plain callables are wrapped as IcoOperators.
            name: Optional name for this filter (defaults to "keep").
        """
        predicate_op = wrap_operator(predicate)

        super().__init__(
            fn=self._filter_fn,
            name=name or "keep",
            children=[predicate_op],
        )
        self.predicate = predicate_op

    def _filter_fn(self, items: Iterator[IInv]) -> Iterator[IInv]:
        """Items already in memory are filtered at once, so the kept ones
        keep a length (a progress total downstream); a lazy input stays
        lazy (predicates are cheap, upstream work may not be)."""
        if materialized(items):
            return iter([item for item in items if self.predicate(item)])
        return self._lazy(items)

    def _lazy(self, items: Iterator[IInv]) -> Iterator[IInv]:
        for item in items:
            if self.predicate(item):
                yield item

    @property
    def signature(self) -> IcoSignature:
        """Infer the signature from the predicate input: Iterator[I] → Iterator[I]."""
        signature = super().signature
        if signature.infered:
            return signature

        predicate = self.predicate.signature
        if not predicate.infered:
            return predicate

        return IcoSignature(
            i=Iterator[predicate.i],  # type: ignore[name-defined]
            c=None,
            o=Iterator[predicate.i],  # type: ignore[name-defined]
        )


def keep(
    predicate: Callable[[IInv], object],
    *,
    name: str | None = None,
) -> IcoOperatorProtocol[Iterator[IInv], Iterator[IInv]]:
    """Create a lazy stream filter keeping items for which ``predicate`` is truthy.

    Named ``keep`` rather than ``filter`` so that ``from ico import *`` does
    not shadow the builtin.

    Example:
        >>> flow = IcoSource(numbers) | keep(is_even) | IcoSink(show)
    """
    return IcoFilter(predicate, name=name)
