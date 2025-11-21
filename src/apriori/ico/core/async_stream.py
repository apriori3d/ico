import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Generic, TypeVar, final

from apriori.ico.core.async_operator import IcoAsyncOperator
from apriori.ico.core.operator import I, IcoOperator, O


@final
class IcoAsyncStream(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
):
    """
    Asynchronous stream operator.

    ICO form:
        Iterator[I] → Iterator[O]

    Executes multiple operators concurrently on incoming items.
    Each operator acts as an independent worker consuming items
    from a shared async queue. Results are streamed back as soon
    as they are produced.

    Flow schema:
        input_stream(I) → producer → in_queue → workers → out_queue → yield(O)

    Key properties:
    • Fully asynchronous execution, converted back to sync iterator.
    • No external timeout or blocking assumptions.
    • Graceful shutdown and fast-exit for empty input streams.
    """

    __slots__ = (
        "operators",
        "ordered",
        "_num_executors",
        "_has_job",
        "_next_index",
        "_ordering_buffer",
    )
    operators: list[IcoOperator[I, O] | IcoAsyncOperator[I, O]]
    ordered: bool
    _num_executors: int

    # ─── Async iterator state ───

    _has_job: bool  # An indicator whether input_stream produced any items
    _next_index: int  # Handle ordered result emission
    _ordering_buffer: dict[int, O | Exception]

    def __init__(
        self,
        operators: Sequence[IcoOperator[I, O] | IcoAsyncOperator[I, O]],
        *,
        ordered: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._run_stream,
            name=name or "async_stream",
            children=[op for op in operators],
        )
        self.operators = list(operators)
        self.ordered = ordered
        self._num_executors = len(self.operators)

        # Async iterator state
        self._has_job = False
        self._next_index = 0
        self._ordering_buffer: dict[int, O | Exception] = {}

    def _reset_iterator_state(self) -> None:
        """Reset per-iteration state (important when stream reused across runs)."""
        self._has_job = False
        self._next_index = 0
        self._ordering_buffer.clear()

    # ─── Synchronous entrypoint ───

    def _run_stream(self, input_stream: Iterator[I]) -> Iterator[O]:
        """Run the async stream inside a synchronous context."""
        yield from async_to_sync_iter(self._run_stream_async(input_stream))

    # ─── Asynchronous execution ───

    async def _run_stream_async(self, input_stream: Iterator[I]) -> AsyncIterator[O]:
        """
        Internal async loop driving producer, workers, and output emission.

        The lifecycle:
            1. Producer fills in_queue with items.
            2. Workers consume in_queue, push results to out_queue.
            3. Main loop yields items from out_queue as they appear.
            4. All workers receive termination signals (None).
        """
        self._reset_iterator_state()

        # Max concurrency = number of operators
        input_queue = asyncio.Queue[tuple[int, I] | None](maxsize=self._num_executors)
        results_queue = asyncio.Queue[tuple[int, O | Exception]](
            maxsize=self._num_executors
        )

        # Create one worker task per operator
        executors = [
            asyncio.create_task(self._execute_operator(op, input_queue, results_queue))
            for op in self.operators
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
                    if len(self._ordering_buffer) < self._num_executors:
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
        for _ in range(self._num_executors):
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
