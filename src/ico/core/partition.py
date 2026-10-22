from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic

from ico.core.operator import IcoOperator, IcoOperatorProtocol, IInv, wrap_operator
from ico.core.signature import IcoSignature


class IcoPartition(
    Generic[IInv],
    IcoOperator[Iterator[IInv], tuple[list[IInv], list[IInv]]],
):
    """Split a stream into items that match a predicate and the rest.

    Generic Parameters:
        I: Item type of the stream.

    ICO signature:
        Iterator[I] → tuple[list[I], list[I]]

    Example:
        >>> def is_valid(x: int) -> bool:
        ...     return x >= 0
        >>> valid, rejected = partition(is_valid)(iter([1, -2, 3]))
        >>> assert (valid, rejected) == ([1, 3], [-2])

    Note:
        The whole stream is consumed and materialized into two lists.
        A lazy split (e.g. via ``itertools.tee``) would silently buffer the
        lagging side anyway, so it is intentionally not offered.

    Attributes:
        predicate: Operator deciding which side an item goes to.
    """

    __slots__ = ("predicate",)

    predicate: IcoOperatorProtocol[IInv, Any]

    def __init__(
        self,
        predicate: Callable[[IInv], object],
        *,
        name: str | None = None,
    ):
        """Initialize a partition.

        Args:
            predicate: Callable or operator; items with a truthy result go to
                the first list, the others to the second.
            name: Optional name for this partition (defaults to "partition").
        """
        predicate_op = wrap_operator(predicate)

        super().__init__(
            fn=self._partition_fn,
            name=name or "partition",
            children=[predicate_op],
        )
        self.predicate = predicate_op

    def _partition_fn(self, items: Iterator[IInv]) -> tuple[list[IInv], list[IInv]]:
        matched: list[IInv] = []
        rest: list[IInv] = []
        for item in items:
            (matched if self.predicate(item) else rest).append(item)
        return matched, rest

    @property
    def signature(self) -> IcoSignature:
        """Infer from the predicate input: Iterator[I] → tuple[list[I], list[I]]."""
        signature = super().signature
        if signature.infered:
            return signature

        predicate = self.predicate.signature
        if not predicate.infered:
            return predicate

        item = predicate.i
        return IcoSignature(
            i=Iterator[item],  # type: ignore[valid-type]
            o=tuple[list[item], list[item]],  # type: ignore[valid-type]
        )


def partition(
    predicate: Callable[[IInv], object],
    *,
    name: str | None = None,
) -> IcoOperatorProtocol[Iterator[IInv], tuple[list[IInv], list[IInv]]]:
    """Create a partition: ``(items with truthy predicate, other items)``.

    Example:
        >>> valid, rejected = partition(is_valid)(iter(rows))
    """
    return IcoPartition(predicate, name=name)
