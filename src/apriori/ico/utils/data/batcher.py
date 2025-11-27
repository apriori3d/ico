from collections.abc import Iterator
from typing import Generic

from apriori.ico.core.operator import I, IcoOperator


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
