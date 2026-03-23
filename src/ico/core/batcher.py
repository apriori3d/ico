from collections.abc import Iterator
from typing import Generic

from ico.core.operator import I, IcoOperator


class IcoBatcher(
    Generic[I],
    IcoOperator[Iterator[I], Iterator[Iterator[I]]],
):
    """Batches stream items into fixed-size chunks for batch processing.

    IcoBatcher transforms a continuous stream of items into batches (chunks) of
    a specified size. This is essential for efficient processing of large datasets,
    memory management, and interfacing with batch-oriented systems like ML models.

    Generic Parameters:
        I: Item type - the type of individual items being batched.

    ICO signature:
        Iterator[I] → Iterator[Iterator[I]]

    Example:
        >>> # Simple number batching
        >>> def numbers() -> Iterator[int]:
        ...     return iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        >>> # Create batches of size 3
        >>> batcher = IcoBatcher(batch_size=3, name="batch_3")
        >>> batches = list(batcher(numbers()))
        >>> batch_lists = [list(batch) for batch in batches]
        >>> assert batch_lists == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

        >>> # Drop incomplete last batch
        >>> strict_batcher = IcoBatcher(batch_size=3, drop_last=True)
        >>> strict_batches = list(strict_batcher(numbers()))
        >>> strict_lists = [list(batch) for batch in strict_batches]
        >>> assert strict_lists == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        >>> # Batch processing example
        >>> def sum_batch(batch: Iterator[int]) -> int:
        ...     return sum(batch)

        >>> # Process each batch to get batch sums
        >>> batch_sums = []
        >>> for batch in batcher(numbers()):
        ...     batch_sums.append(sum_batch(batch))
        >>> assert batch_sums == [6, 15, 24, 10]  # [1+2+3, 4+5+6, 7+8+9, 10]

    Attributes:
        batch_size: Number of items per batch.
        drop_last: Whether to drop the final batch if it's smaller than batch_size.

    Note:
        Each yielded batch is an Iterator[I], not a list. This preserves memory
        efficiency for large datasets by avoiding materialization until needed.
    """

    batch_size: int
    drop_last: bool

    def __init__(
        self,
        batch_size: int,
        drop_last: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize a batcher with specified batch size and behavior.

        Args:
            batch_size: Number of items to include in each batch. Must be positive.
            drop_last: If True, discard the final batch if it contains fewer than
                      batch_size items. If False (default), yield all items.
            name: Optional name for this batcher. Auto-generated if not provided.

        Note:
            The default name format is "batcher({batch_size})" for easy identification
            in debugging and visualization contexts.
        """
        super().__init__(
            fn=self._batch_fn,
            name=name or f"batcher({batch_size})",
        )
        self.batch_size = batch_size
        self.drop_last = drop_last

    def _batch_fn(self, input: Iterator[I]) -> Iterator[Iterator[I]]:
        """Internal implementation that performs the batching logic.

        Args:
            input: Source iterator to be batched.

        Yields:
            Iterator[I]: Each batch as an iterator over batch_size items
                        (or fewer for the last batch if drop_last=False).

        Note:
            This is the function used by __call__. It accumulates items in memory
            only up to batch_size, then yields and clears the batch for efficiency.
        """
        batch: list[I] = []

        for item in input:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield iter(batch)
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield iter(batch)
