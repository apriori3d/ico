import multiprocessing
import time
from collections.abc import Iterator

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.source import IcoSource
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent
from apriori.ico.utils.data.batcher import IcoBatcher
from examples.visual.cifar.data_flow import (
    CifarBatch,
    CifarItemsResolver,
    IndexedAugmentationFlowFactory,
    create_augmentation_flow,
    create_data_input_flow,
)
from examples.visual.cifar.dataset import CifarInMemoryDataset


def create_indexed_augmentation_flow(
    dataset: CifarInMemoryDataset,
) -> IcoOperator[Iterator[int], CifarBatch]:
    resolver = CifarItemsResolver(dataset)
    resolver_op = IcoOperator(resolver, name="CIFAR10 Items Resolver")

    augment_flow = create_augmentation_flow()
    resolved_flow = resolver_op | augment_flow
    return resolved_flow


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    batch_size = 128
    num_workers = 4
    num_epochs = 1

    data_input_flow = create_data_input_flow(batch_size=batch_size)

    # 1. Run augmentation in main process
    augment_flow = create_augmentation_flow()
    single_data_flow = data_input_flow | augment_flow.stream()
    single_data_flow.name = "Single Process Data Flow with Augmentation"
    single_data_flow.describe()

    start = time.perf_counter()

    for _ in range(num_epochs):
        for _ in single_data_flow(None):
            pass
    end = time.perf_counter()

    single_time = end - start
    print(f"Single process data flow took {single_time:.2f} seconds.")

    # 2. Run augmentation in multiple MP processes
    workers_pool = [MPAgent(create_augmentation_flow) for _ in range(num_workers)]

    async_stream = IcoAsyncStream(
        workers_pool, ordered=False, name="Async Augmentation Stream"
    )

    parallel_data_flow = data_input_flow | async_stream
    parallel_data_flow.name = "Parallel Data Flow with Augmentation"

    runtime = parallel_data_flow.runtime()
    runtime.activate()
    parallel_data_flow.describe()

    start = time.perf_counter()
    for _ in range(num_epochs):
        for _ in parallel_data_flow(None):
            pass
    end = time.perf_counter()

    parallel_time = end - start
    print(f"Parallel data flow took {parallel_time:.2f} seconds.")

    print(f"Speedup: {single_time / parallel_time:.2f}x using {num_workers} workers.")

    runtime.deactivate()

    # 3. Run augmentation in multiple MP processes with shared dataset and references
    # to avoid dataset copy overhead.

    dataset = CifarInMemoryDataset()
    indices = list(range(len(dataset)))
    source = IcoSource[int](lambda: iter(indices), name="CIFAR10 indices source")
    batcher = IcoBatcher[int](batch_size=batch_size)

    input_flow_indices = source | batcher
    input_flow_indices.name = "CIFAR10 Indices Input Flow"

    augmentation_flow_factory = IndexedAugmentationFlowFactory(dataset)
    workers_pool = [MPAgent(augmentation_flow_factory) for _ in range(num_workers)]

    async_stream = IcoAsyncStream(
        workers_pool, ordered=False, name="Async Augmentation Stream"
    )

    parallel_data_flow_indices = input_flow_indices | async_stream
    parallel_data_flow_indices.name = "Parallel Data Flow with Augmentation by Indices"

    runtime = parallel_data_flow_indices.runtime()
    runtime.activate()
    parallel_data_flow_indices.describe()

    start = time.perf_counter()
    for _ in range(num_epochs):
        for _ in parallel_data_flow_indices(None):
            pass
    end = time.perf_counter()

    parallel_time_indices = end - start
    print(f"Parallel data flow took {parallel_time_indices:.2f} seconds.")

    print(
        f"Speedup: {single_time / parallel_time_indices:.2f}x using {num_workers} workers."
    )
    runtime.deactivate()
