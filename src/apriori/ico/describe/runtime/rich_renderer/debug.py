from collections.abc import Iterable, Iterator

from apriori.ico.core.async_stream import IcoAsyncStream
from apriori.ico.core.batcher import IcoBatcher
from apriori.ico.core.operator import IcoOperator, operator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.runtime.progress import IcoProgress
from apriori.ico.core.runtime.runtime import IcoRuntime
from apriori.ico.core.sink import sink
from apriori.ico.core.source import source
from apriori.ico.describe.runtime.rich_renderer.tree_renderer import RuntimeTreeRenderer
from apriori.ico.runtime.agent.mp.mp_agent import MPAgent

# ──── 1. Define a batched data source ────
data = [0.5, 1.0, 1.5, 2.0, 0.8, 0.2, 1.0, 1.2, 0.9]


@source()
def indices() -> Iterable[int]:
    return list(range(len(data)))


@operator()
def fetch_data_item(index: int) -> float:
    return data[index]


# ──── 2. Define augmentation & collation pipelines per item ────


@operator()
def scale(x: float) -> float:
    return x * 1.1


@operator()
def shift(x: float) -> float:
    return x + 0.1


@operator()
def collate_max(batch: Iterator[float]) -> str:
    return f"max={max(batch)}"


def create_augment_flow() -> IcoOperator[Iterator[int], str]:
    augment = IcoPipeline(
        scale,
        shift,
        IcoProgress[float](name="Augment Progress", total=9),
        name="Item Augment Pipeline",
    )

    # Augment each item in the batch, then collate to single value (mimic data loader collate)
    augment_stream = (fetch_data_item | augment).stream() | collate_max
    augment_stream.name = "Train Batch producer"

    return augment_stream


if __name__ == "__main__":
    """
    ──── 3. Compose into a full data stream ────
    • dataset: () → Iterable[Iterable[float]]
    • stream: Iterable[Iterable[float]] → Iterable[float]
    • augment.map(): Iterable[float] → Iterable[float]
    • augment: float → float,
    • collate: Iterable[float] → float
    • dataflow: () → Iterable[float]
    """

    batcher = IcoBatcher[int](batch_size=3)
    agents = [MPAgent(create_augment_flow) for _ in range(2)]
    workers_pool = IcoAsyncStream(
        agents,
        # pool_size=2,
        name="Workers pool",
    )

    data_stream = indices | batcher | workers_pool
    data_stream.name = "Data stream"

    # ──── 4. Define training pipeline ────
    # We collate batch into single float via max, so input into training stream is Iterable[float],
    # without batch dimension. In real scenarios, item could be tensors with batch dim.
    # Each batch item passes through a process that transforms floats ≤ 1 via pow(2)
    # train_step: float -> float, ICO Form: I → O
    # train_process: float -> float, ICO Form: C → C
    # I → C → O = Iterable[float] → Iterable[float] → Iterable[float]

    def optimize(values: str) -> str:
        return f"{values} x2"

    def log_metrics(values: str) -> str:
        return values

    train_iter = IcoPipeline(optimize, log_metrics)
    train_stream = train_iter.stream()
    train_stream.name = "Train stream"

    @sink()
    def save_result(item: str) -> None:
        print("Saved result:", item)

    # ──── 5. Combine all into the full flow ────
    full_flow = data_stream | train_stream | save_result
    full_flow.name = "Example flow"

    renderer = RuntimeTreeRenderer()

    runtime = IcoRuntime(full_flow, name="full_flow_runtime")
    runtime.activate()
    renderer.render(runtime)

    runtime.run()

    agents[1].deactivate()
    renderer.render(runtime)

    # ──── 6. Visualize ────

    runtime.deactivate()
    renderer.render(runtime)

    print("Done")
