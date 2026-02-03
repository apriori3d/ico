from collections.abc import Iterator
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator
from apriori.ico.core.signature import IcoSignature
from apriori.ico.core.signature_utils import wrap_iterator_or_none


class IcoBatcher(
    Generic[I],
    IcoOperator[Iterator[I], Iterator[Iterator[I]]],
):
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        batch_size: int,
        drop_last: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._batch_fn,
            name=name or f"batcher({batch_size})",
        )
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _batch_fn(self, input: Iterator[I]) -> Iterator[Iterator[I]]:
        batch: list[I] = []

        for item in input:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield iter(batch)
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield iter(batch)

    @property
    def signature(self) -> IcoSignature:
        signature = super().signature

        return IcoSignature(
            i=wrap_iterator_or_none(signature.i),
            c=None,
            o=wrap_iterator_or_none(wrap_iterator_or_none(signature.i)),
        )
