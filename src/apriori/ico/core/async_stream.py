import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Generic, TypeVar, final, overload

from apriori.ico.core.async_operator import IcoAsyncOperator
from apriori.ico.core.operator import I, IcoOperator, O
from apriori.ico.core.signature import IcoSignature


@final
class IcoAsyncStream(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    """Asynchronous stream processor with concurrent worker pool.

    IcoAsyncStream enables concurrent processing of stream items using a pool of
    workers (operators). Items are distributed among workers for parallel execution,
    with results streamed back as soon as they're produced. Workers can be async
    operators for I/O-bound tasks or multiprocessing agents (MPAgent) for CPU-intensive
    computations. This provides significant performance improvements across different
    workload types.

    Generic Parameters:
        I: Input item type - the type of individual items in the input stream.
        O: Output item type - the type of individual items produced by workers.

    ICO signature:
        Iterator[I] → Iterator[O]

    Example:
        >>> import asyncio
        >>> from apriori.ico.core.async_operator import IcoAsyncOperator

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

        >>> # Factory-based worker creation with MPAgent for CPU-bound work
        >>> from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
        >>> def make_worker() -> IcoAsyncOperator[int, int]:
        ...     return IcoAsyncOperator(slow_process)

        >>> def make_mp_worker() -> MPAgent[int, int]:
        ...     # MPAgent provides true multiprocessing for CPU-intensive operations
        ...     return MPAgent(lambda: IcoOperator(lambda x: x ** 2))

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
        pool: List of worker operators (sync, async, or multiprocessing agents).
        ordered: Whether to preserve input order in results.
        pool_size: Number of workers in the pool.

    Note:
        Best suited for I/O-bound tasks where concurrency provides significant
        performance gains. For CPU-bound tasks, consider using MPAgent workers
        for true multiprocessing or other process-based parallelism approaches.
    """

    __slots__ = (
        "pool",
        "ordered",
        "_num_executors",
        "_has_job",
        "_next_index",
        "_ordering_buffer",
    )

    pool: list[IcoOperator[I, O] | IcoAsyncOperator[I, O]]
    pool_from_factory: bool

    ordered: bool
    has_factory: bool
    pool_size: int

    # ─── Async iterator state ───

    _has_job: bool  # An indicator whether input_stream produced any items
    _next_index: int  # Handle ordered result emission
    _ordering_buffer: dict[int, O | Exception]

    @overload
    def __init__(
        self,
        pool: Sequence[IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        ordered: bool = False,
        name: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        pool: Callable[[], IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        pool_size: int | None = None,
        ordered: bool = False,
        name: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        pool: Sequence[IcoOperator[I, O] | IcoAsyncOperator[I, O]]
        | Callable[[], IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        pool_size: int | None = None,
        ordered: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize async stream with worker pool or factory.

        Args:
            pool: Either a sequence of pre-created operators/agents or a factory function
                 that creates new workers. Supports IcoOperator, IcoAsyncOperator, and
                 MPAgent types. Factory enables dynamic worker creation.
            pool_size: Required when pool is a factory. Number of workers to create.
                      Ignored when pool is a sequence (uses sequence length).
            ordered: If True, results are yielded in the same order as input items.
                    If False, results are yielded as soon as workers complete.
            name: Optional name for this stream (useful for debugging/visualization).

        Raises:
            ValueError: If pool is a factory but pool_size is not specified.

        Note:
            Factory-based pools allow for fresh worker instances, which can be
            useful for stateful operators, memory management, or process isolation.
        """
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

        # Async iterator state
        self._has_job = False
        self._next_index = 0
        self._ordering_buffer: dict[int, O | Exception] = {}

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
        signature = super().signature

        # If signature is undefined, infer from body operator
        if not signature.infered:
            signature = self.pool[0].signature

        i_type: Any = signature.i
        o_type: Any = signature.o
        return IcoSignature(
            i=Iterator[i_type],
            c=None,
            o=Iterator[o_type],
        )

    # ─── Async iterator state management ───

    def _reset_iterator_state(self) -> None:
        """Reset per-iteration state for stream reuse.

        Clears internal state variables that track iteration progress and ordering.
        This is important when the same stream instance is used multiple times.

        Note:
            Called automatically before each stream execution to ensure clean state.
        """
        self._has_job = False
        self._next_index = 0
        self._ordering_buffer.clear()

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

        Coordinates the complete async processing lifecycle:
        1. Producer fills input queue with items from input_stream
        2. Worker pool consumes from input queue and processes items concurrently
        3. Results are collected in output queue and yielded as they become available
        4. Graceful shutdown ensures all workers complete and resources are cleaned up

        Args:
            input_stream: Iterator of input items to be processed.

        Yields:
            O: Processed results, either in input order (if ordered=True) or
               as soon as workers complete (if ordered=False).

        Note:
            Handles empty input streams with fast-exit optimization and ensures
            proper resource cleanup even if processing is interrupted.
        """
        self._reset_iterator_state()

        # Max concurrency = number of operators
        input_queue = asyncio.Queue[tuple[int, I] | None](maxsize=self.pool_size)
        results_queue = asyncio.Queue[tuple[int, O | Exception]](maxsize=self.pool_size)

        # Create one worker task per operator
        executors = [
            asyncio.create_task(self._execute_operator(op, input_queue, results_queue))
            for op in self.pool
        ]

        # Set indicator results buffer size is below max_concurrency
        ordering_buffer_ready = asyncio.Event()
        ordering_buffer_ready.set()  # initially allow producing

        scheduler_active = asyncio.Event()
        scheduler_active.clear()

        producer_task = asyncio.create_task(
            self._schedule_items(
                input_stream, input_queue, scheduler_active, ordering_buffer_ready
            )
        )

        try:
            # Wait for producer to finish pushing items and termination signals.
            # Corner case: if the input stream is empty, out_queue will stay empty.
            await scheduler_active.wait()

            # ─── Fast-exit ───
            # If no jobs were produced, exit immediately (no worker output expected).
            if not self._has_job:
                return

            # ─── Main result loop ───
            async for output_item in self._produce_output_async(
                results_queue, ordering_buffer_ready
            ):
                yield output_item
                # Update flag: continue while at least one worker is still active
                self._has_job = not all(w.done() for w in executors)

        finally:
            # ─── Graceful shutdown ───
            producer_task.cancel()
            for w in executors:
                w.cancel()
            await asyncio.gather(*executors, return_exceptions=True)

    # ─── Producer coroutine ───

    async def _produce_output_async(
        self,
        results_queue: asyncio.Queue[tuple[int, O | Exception]],
        ordering_buffer_ready: asyncio.Event,
    ) -> AsyncIterator[O]:
        """
        Pulls results from results_queue and yields them downstream.
        Maintains result ordering if required.
        Backpressure (results_buffer_ready) is toggled based on buffer size.
        """
        while not results_queue.empty() or self._has_job:
            # Wait for next available result
            i, result = await results_queue.get()
            results_queue.task_done()

            if self.ordered:
                # Store result by index
                self._ordering_buffer[i] = result

                # Yield all ready results in order
                while self._next_index in self._ordering_buffer:
                    result = self._ordering_buffer.pop(self._next_index)

                    # update results buffer flag
                    if len(self._ordering_buffer) < self.pool_size:
                        ordering_buffer_ready.set()
                    else:
                        ordering_buffer_ready.clear()

                    # Yield output item or raise exception
                    if isinstance(result, Exception):
                        raise result
                    yield result
                    self._next_index += 1
            else:
                # Yield result immediately
                if isinstance(result, Exception):
                    raise result
                yield result

    # ─── Scheduler coroutine ───

    async def _schedule_items(
        self,
        input_stream: Iterator[I],
        in_queue: asyncio.Queue[tuple[int, I] | None],
        scheduler_active: asyncio.Event,
        results_buffer_ready: asyncio.Event,
    ) -> None:
        scheduler_active.set()

        for i, item in enumerate(input_stream):
            await results_buffer_ready.wait()
            self._has_job = True
            # Put item with its index for ordered processing
            await in_queue.put((i, item))

        # Send termination signals for all executors
        for _ in range(self.pool_size):
            await in_queue.put(None)

    # ─── Executor coroutine ───

    async def _execute_operator(
        self,
        operator: IcoOperator[I, O] | IcoAsyncOperator[I, O],
        in_queue: asyncio.Queue[tuple[int, I] | None],
        out_queue: asyncio.Queue[tuple[int, O | Exception]],
    ) -> None:
        """
        Worker coroutine that continuously pulls items from in_queue,
        executes its operator, and pushes results to out_queue.
        """
        try:
            while True:
                task = await in_queue.get()
                if task is None:
                    break
                i, item = task
                try:
                    result = await self._call_async(operator, item)
                    await out_queue.put((i, result))
                except Exception as e:
                    await out_queue.put((i, e))

        except asyncio.CancelledError:
            # Normal cancellation during shutdown
            pass
        finally:
            in_queue.task_done()

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


# ─── Utility: bridge async → sync iterator ───

T = TypeVar("T")


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
        loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)  # Explicitly detach the closed loop
        loop.close()
