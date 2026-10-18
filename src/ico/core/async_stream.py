from __future__ import annotations

import asyncio
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar, final, overload

from ico.core.async_operator import IcoAsyncOperator
from ico.core.operator import I, IcoOperator, O
from ico.core.runtime.cancel import check_cancelled
from ico.core.runtime.worker import IcoWorkerDied
from ico.core.signature import IcoSignature

T = TypeVar("T")

OnError: TypeAlias = Literal["raise", "skip", "collect"]


# ─── Error handling policies ───


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How a worker retries a failed item before ``on_error`` applies.

    The failed item is retried by the same worker, so ordering is preserved.
    The delay before retry ``k`` (1-based) is ``delay * backoff ** (k - 1)``.

    Attributes:
        max_retries: Extra attempts after the first one (0 disables retries).
        delay: Seconds to wait before the first retry.
        backoff: Multiplier applied to the delay after every retry.
        retry_on: Exception types that are retried; others fail immediately.

    Note:
        Retrying happens on the same worker. If a multiprocessing worker
        died, retries on it are likely to fail as well.
    """

    max_retries: int = 0
    delay: float = 0.0
    backoff: float = 1.0
    retry_on: tuple[type[Exception], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}."
            )
        if self.delay < 0:
            raise ValueError(f"delay must be non-negative, got {self.delay}.")
        if self.backoff <= 0:
            raise ValueError(f"backoff must be positive, got {self.backoff}.")

    def delay_before(self, retry: int) -> float:
        """Delay in seconds before the given retry (1-based)."""
        return self.delay * self.backoff ** (retry - 1)


@dataclass(frozen=True, slots=True)
class ItemError:
    """A stream item that failed after all attempts.

    Attributes:
        index: Position of the item in the input stream.
        item: The input item.
        error: The last exception raised for the item.
        attempts: Number of attempts made (1 + retries).
    """

    index: int
    item: Any
    error: Exception
    attempts: int

    def __str__(self) -> str:
        return f"[{self.index}] {type(self.error).__name__}: {self.error}"


class IcoStreamErrors(RuntimeError):
    """Raised at the end of a stream with ``on_error="collect"``.

    Raised after every successful result was emitted.

    Attributes:
        errors: Failed items, ordered by input index.
        stream: The stream that collected them, so a caller can locate it
            (e.g. ``ico.spec.run`` maps it to a spec path); ``None`` if unknown.
    """

    _SHOWN = 5

    def __init__(self, errors: list[ItemError], stream: object | None = None) -> None:
        shown = "; ".join(str(e) for e in errors[: self._SHOWN])
        more = len(errors) - self._SHOWN
        suffix = f"; and {more} more" if more > 0 else ""
        super().__init__(f"{len(errors)} stream item(s) failed: {shown}{suffix}")
        self.errors = errors
        self.stream = stream


@final
class IcoAsyncStream(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    """Asynchronous stream processor with concurrent worker pool.

    IcoAsyncStream enables concurrent processing of stream items using a pool of
    workers (operators). Items are distributed among workers for parallel execution,
    with results streamed back as soon as they're produced. Workers can be async
    operators for I/O-bound tasks or multiprocessing workers (MPWorker) for CPU-intensive
    computations. This provides significant performance improvements across different
    workload types.

    Generic Parameters:
        I: Input item type - the type of individual items in the input stream.
        O: Output item type - the type of individual items produced by workers.

    ICO signature:
        Iterator[I] → Iterator[O]

    Example:
        >>> import asyncio
        >>> from ico.core.async_operator import IcoAsyncOperator

        >>> # Create async workers
        >>> async def slow_process(x: int) -> int:
        ...     await asyncio.sleep(0.1)  # Simulate I/O work
        ...     return x * 2

        >>> # Pool of 3 workers for concurrent processing
        >>> workers = [IcoAsyncOperator(slow_process) for _ in range(3)]
        >>> stream_processor = IcoAsyncStream(workers, name="concurrent_doubler")

        >>> # Process numbers concurrently
        >>> numbers = iter([1, 2, 3, 4, 5, 6])
        >>> results = list(stream_processor(numbers))
        >>> # Results may be out of order: [4, 2, 6, 8, 10, 12]

        >>> # Ordered processing (preserves input order)
        >>> ordered_stream = IcoAsyncStream(workers, ordered=True)
        >>> ordered_results = list(ordered_stream(iter([1, 2, 3])))
        >>> # Results in order: [2, 4, 6]

        >>> # Factory-based worker creation with MPWorker for CPU-bound work
        >>> from ico.runtime.worker.mp.mp_worker import MPWorker
        >>> def make_worker() -> IcoAsyncOperator[int, int]:
        ...     return IcoAsyncOperator(slow_process)

        >>> def make_mp_worker() -> MPWorker[int, int]:
        ...     # MPWorker provides true multiprocessing for CPU-intensive operations
        ...     return MPWorker(lambda: IcoOperator(lambda x: x ** 2))

        >>> dynamic_stream = IcoAsyncStream(make_worker, pool_size=4)
        >>> # Creates 4 workers dynamically when needed

    Flow schema:
        input_stream(I) → producer → in_queue → workers → out_queue → yield(O)

    Key properties:
    • Fully asynchronous execution, converted back to sync iterator.
    • Configurable worker pool size for optimal concurrency.
    • Optional result ordering to preserve input sequence.
    • Graceful shutdown and fast-exit for empty input streams.
    • No external timeout or blocking assumptions.

    Attributes:
        pool: List of worker operators (sync, async, or multiprocessing workers).
        ordered: Whether to preserve input order in results.
        pool_size: Number of workers in the pool.

    Note:
        Best suited for I/O-bound tasks where concurrency provides significant
        performance gains. For CPU-bound tasks, consider using MPWorkers
        for true multiprocessing or other process-based parallelism approaches.
    """

    __slots__ = ("pool", "ordered", "on_error", "retry")

    pool: list[IcoOperator[I, O] | IcoAsyncOperator[I, O]]
    pool_from_factory: bool

    ordered: bool
    has_factory: bool
    pool_size: int
    on_error: OnError
    retry: RetryPolicy

    @overload
    def __init__(
        self,
        pool: Sequence[IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        ordered: bool = False,
        on_error: OnError = "raise",
        retry: int | RetryPolicy | None = None,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        pool: Callable[[], IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        pool_size: int | None = None,
        ordered: bool = False,
        on_error: OnError = "raise",
        retry: int | RetryPolicy | None = None,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        pool: Sequence[IcoOperator[I, O] | IcoAsyncOperator[I, O]]
        | Callable[[], IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        pool_size: int | None = None,
        ordered: bool = False,
        on_error: OnError = "raise",
        retry: int | RetryPolicy | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize async stream with worker pool or factory.

        Args:
            pool: Either a sequence of pre-created operators/workers or a factory function
                 that creates new workers. Supports IcoOperator, IcoAsyncOperator, and
                 MPWorker types. Factory enables dynamic worker creation.
            pool_size: Required when pool is a factory. Number of workers to create.
                      Ignored when pool is a sequence (uses sequence length).
            ordered: If True, results are yielded in the same order as input items.
                    If False, results are yielded as soon as workers complete.
            on_error: What to do with an item that failed after all retries:
                    ``"raise"`` (default) re-raises its error immediately (at the
                    item's position when ordered); ``"skip"`` drops it;
                    ``"collect"`` drops it and, after all results were emitted,
                    raises :class:`IcoStreamErrors` listing every failed item.
            retry: Retries per failed item: a number of retries or a
                    :class:`RetryPolicy`. Defaults to no retries.
            name: Optional name for this stream (useful for debugging/visualization).

        Raises:
            ValueError: If pool is a factory but pool_size is not specified,
                or if on_error / retry are invalid.

        Note:
            Factory-based pools allow for fresh worker instances, which can be
            useful for stateful operators, memory management, or process isolation.
        """
        if on_error not in ("raise", "skip", "collect"):
            raise ValueError(
                f"on_error must be 'raise', 'skip' or 'collect', got {on_error!r}."
            )
        retry_policy = (
            retry
            if isinstance(retry, RetryPolicy)
            else RetryPolicy(max_retries=retry or 0)
        )

        if callable(pool):
            if pool_size is None:
                raise ValueError(
                    "Pool_size must be specified when providing a flow factory."
                )
            worker_pool = [pool() for _ in range(pool_size)]
            self.pool_from_factory = True
        else:
            worker_pool = list(pool)
            pool_size = len(worker_pool)
            self.pool_from_factory = False

        super().__init__(fn=self._run_stream, name=name, children=worker_pool)
        self.pool = list(worker_pool)
        self.ordered = ordered
        self.pool_size = pool_size
        self.on_error = on_error
        self.retry = retry_policy

    # ─── Signature API ───

    @property
    def signature(self) -> IcoSignature:
        """Infer the ICO type signature for this async stream.

        Derives the stream signature from the first worker's signature, wrapping
        the input and output types in Iterator containers to reflect stream processing.

        Returns:
            IcoSignature with Iterator[I] input and Iterator[O] output types,
            derived from the worker operators' signatures.

        Note:
            All workers in the pool should have compatible signatures. The first
            worker's signature is used as the template for the entire stream.
        """
        # Explicit parametrization (IcoAsyncStream[I, O]) already resolves to
        # Iterator[I] → Iterator[O] through the IcoOperator base.
        signature = super().signature
        if signature.infered:
            return signature

        # Otherwise wrap the first worker's item signature.
        worker = self.pool[0].signature
        if not worker.infered:
            return worker

        i_type: Any = worker.i
        o_type: Any = worker.o
        return IcoSignature(
            i=Iterator[i_type],
            c=None,
            o=Iterator[o_type],
        )

    # ─── Synchronous entrypoint ───

    def _run_stream(self, input_stream: Iterator[I]) -> Iterator[O]:
        """Run the async stream inside a synchronous context.

        Args:
            input_stream: Iterator of input items to be processed concurrently.

        Yields:
            O: Processed results from the worker pool, either in order or as completed.

        Note:
            This is the synchronous entrypoint that bridges to the async implementation.
            It converts the async iterator to sync using async_to_sync_iter utility.
        """
        yield from async_to_sync_iter(self._run_stream_async(input_stream))

    # ─── Asynchronous execution ───

    async def _run_stream_async(self, input_stream: Iterator[I]) -> AsyncIterator[O]:
        """Internal async implementation driving producer, workers, and output emission.

        Flow:
            1. The scheduler feeds ``(index, item)`` pairs into the input queue
               and finally reports how many items it scheduled (``_End``).
            2. Workers turn every item into exactly one outcome on the results
               queue: a value or a ``_Failed`` wrapper.
            3. This coroutine emits outcomes (reordered if ``ordered``) and
               stops once it has received as many outcomes as items were
               scheduled.

        All per-run state is local, so a stream instance can be run several
        times, including concurrently.

        Args:
            input_stream: Iterator of input items to be processed.

        Yields:
            O: Processed results, either in input order (if ordered=True) or
               as soon as workers complete (if ordered=False).

        Raises:
            Exception: The first item error, or an error raised by the input
                iterator once all items scheduled before it were emitted.
        """
        # Max concurrency = number of operators
        input_queue = asyncio.Queue[tuple[int, I] | None](maxsize=self.pool_size)
        results_queue = asyncio.Queue[_Message[O]](maxsize=self.pool_size)

        # Backpressure for ordered mode: pause scheduling while the ordering
        # buffer is full.
        ordering_buffer_ready = asyncio.Event()
        ordering_buffer_ready.set()

        live = _Live(len(self.pool))
        executors = [
            asyncio.create_task(
                self._execute_operator(op, input_queue, results_queue, live)
            )
            for op in self.pool
        ]
        scheduler = asyncio.create_task(
            self._schedule_items(
                input_stream, input_queue, results_queue, ordering_buffer_ready
            )
        )

        completed = False
        try:
            received = 0
            total: int | None = None
            source_error: Exception | None = None
            next_index = 0
            ordering_buffer: dict[int, O | _Failed] = {}
            errors: list[ItemError] = []

            while total is None or received < total:
                message = await results_queue.get()
                results_queue.task_done()

                if isinstance(message, _End):
                    total, source_error = message.count, message.error
                    continue

                received += 1
                index, outcome = message

                if not self.ordered:
                    if not isinstance(outcome, _Failed):
                        yield outcome
                    else:
                        self._handle_failure(index, outcome, errors)
                    continue

                ordering_buffer[index] = outcome
                while next_index in ordering_buffer:
                    ready = ordering_buffer.pop(next_index)
                    if not isinstance(ready, _Failed):
                        yield ready
                    else:
                        self._handle_failure(next_index, ready, errors)
                    next_index += 1
                if len(ordering_buffer) < self.pool_size:
                    ordering_buffer_ready.set()
                else:
                    ordering_buffer_ready.clear()

            completed = True
            # An input iterator error takes precedence over collected item
            # errors: the stream itself is incomplete.
            if source_error is not None:
                raise source_error
            if errors:
                raise IcoStreamErrors(
                    sorted(errors, key=lambda e: e.index), stream=self
                )
        finally:
            if completed:
                # ─── Graceful shutdown ───
                # All items are done; workers exit on their termination signal.
                await asyncio.gather(scheduler, *executors, return_exceptions=True)
            else:
                # ─── Early exit: error or consumer stopped iterating ───
                scheduler.cancel()
                for w in executors:
                    w.cancel()
                await asyncio.gather(scheduler, *executors, return_exceptions=True)

    def _handle_failure(
        self, index: int, failure: _Failed, errors: list[ItemError]
    ) -> None:
        """Apply the on_error policy to a failed item (raise, skip or collect)."""
        if self.on_error == "raise":
            raise failure.error
        if self.on_error == "collect":
            errors.append(
                ItemError(index, failure.item, failure.error, failure.attempts)
            )

    # ─── Scheduler coroutine ───

    async def _schedule_items(
        self,
        input_stream: Iterator[I],
        in_queue: asyncio.Queue[tuple[int, I] | None],
        out_queue: asyncio.Queue[_Message[O]],
        ordering_buffer_ready: asyncio.Event,
    ) -> None:
        """Feed items to workers, then report the item count and stop workers.

        An exception raised by the input iterator is not an item error: it is
        reported with the count and re-raised by the consumer.
        """
        count = 0
        error: Exception | None = None
        try:
            for item in input_stream:
                # Cancelled: stop scheduling; items in progress finish.
                check_cancelled()
                await ordering_buffer_ready.wait()
                await in_queue.put((count, item))
                count += 1
        except Exception as e:
            error = e

        await out_queue.put(_End(count, error))

        # Termination signal for every executor
        for _ in range(self.pool_size):
            await in_queue.put(None)

    # ─── Executor coroutine ───

    async def _execute_operator(
        self,
        operator: IcoOperator[I, O] | IcoAsyncOperator[I, O],
        in_queue: asyncio.Queue[tuple[int, I] | None],
        out_queue: asyncio.Queue[_Message[O]],
        live: _Live,
    ) -> None:
        """Worker coroutine: process items until the termination signal.

        Every item produces exactly one outcome (after retries), and
        ``task_done()`` is called once per ``get()``, including the
        termination signal.

        A worker whose process died (``IcoWorkerDied``) stops taking items,
        so the others process the rest. If it was the last live worker, it
        keeps taking items and fails each one with the same error, so the
        stream ends instead of waiting for workers that are gone.
        """
        dead: IcoWorkerDied | None = None
        while True:
            task = await in_queue.get()
            try:
                if task is None:
                    return
                index, item = task
                if dead is not None:
                    outcome: O | _Failed = _Failed(dead, item, 0)
                else:
                    outcome = await self._call_with_retry(operator, item)
                await out_queue.put((index, outcome))
                if (
                    dead is None
                    and isinstance(outcome, _Failed)
                    and isinstance(outcome.error, IcoWorkerDied)
                ):
                    dead = outcome.error
                    live.count -= 1
                    if live.count > 0:
                        return  # leave the remaining items to live workers
            finally:
                in_queue.task_done()

    async def _call_with_retry(
        self,
        operator: IcoOperator[I, O] | IcoAsyncOperator[I, O],
        item: I,
    ) -> O | _Failed:
        """Call the operator, retrying failures according to the retry policy."""
        policy = self.retry
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._call_async(operator, item)
            except Exception as e:
                retry = attempt  # number of the retry that would follow
                if retry > policy.max_retries or not isinstance(e, policy.retry_on):
                    return _Failed(e, item, attempt)
                delay = policy.delay_before(retry)
                if delay > 0:
                    await asyncio.sleep(delay)

    # ─── Async operator execution ───

    async def _call_async(
        self,
        operator: IcoOperator[I, O] | IcoAsyncOperator[I, O],
        item: I,
    ) -> O:
        """Async wrapper around operator call (compatible with sync operators)."""
        if isinstance(operator, IcoAsyncOperator):
            result = await operator(item)
        else:
            result = await asyncio.to_thread(operator, item)
        return result


# ─── Internal messages ───


@dataclass(frozen=True, slots=True)
class _Failed:
    """A failed item outcome (kept distinct from results that are exceptions)."""

    error: Exception
    item: Any
    attempts: int


@dataclass(slots=True)
class _Live:
    """Number of workers of a run whose process is not known to be dead."""

    count: int


@dataclass(frozen=True, slots=True)
class _End:
    """Scheduler report: number of scheduled items and an input iterator error."""

    count: int
    error: Exception | None


_Message: TypeAlias = tuple[int, T | _Failed] | _End


# ─── Utility: bridge async → sync iterator ───


def async_to_sync_iter(async_iter: AsyncIterator[T]) -> Iterator[T]:
    """
    Convert an async iterator into a synchronous generator.

    This runs a dedicated event loop that pulls from the async iterator
    one item at a time. Used to integrate async operators inside
    synchronous ICO pipelines.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                yield loop.run_until_complete(async_iter.__anext__())
            except StopAsyncIteration:
                break
    finally:
        # Close the generator explicitly: it was first iterated outside the
        # running loop, so shutdown_asyncgens() does not track it. Without
        # this, an early-closed stream would run its cleanup (cancelling
        # workers) only at garbage collection, after the loop is closed.
        if isinstance(async_iter, AsyncGenerator):
            loop.run_until_complete(async_iter.aclose())
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)  # Explicitly detach the closed loop
        loop.close()
