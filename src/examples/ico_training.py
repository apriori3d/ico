"""The first example of the README: a training process in miniature."""

from collections.abc import Iterator
from dataclasses import dataclass

from ico import (
    IcoAsyncStream,
    IcoFold,
    IcoOperator,
    IcoProcess,
    IcoProgress,
    IcoRuntime,
    IcoSource,
    MPWorker,
    RichProgressTool,
    operator,
)


@dataclass(frozen=True)
class Model:
    weight: float = 0.0
    steps: int = 0


def indices() -> Iterator[int]:
    return iter(range(1000))


def load(index: int) -> float:  # heavy work: runs in worker processes
    return index / 1000


def loader() -> IcoOperator[int, float]:
    return IcoOperator(load)


def train_step(pair: tuple[float, Model]) -> Model:
    x, model = pair
    return Model(model.weight + 0.01 * (x - model.weight), model.steps + 1)


data = (
    IcoSource(indices)
    | IcoProgress[int](total=1000, name="items").stream()
    | IcoAsyncStream(lambda: MPWorker(loader), pool_size=4)
)
epoch = IcoFold(data, train_step)  # Model → Model


@operator()
def init(_: None) -> Model:
    return Model()


@operator()
def report(model: Model) -> None:
    print(model)


flow = init | IcoProcess(epoch, num_iterations=3) | report
flow.name = "training"

if __name__ == "__main__":
    flow.describe()
    IcoRuntime(flow, tools=[RichProgressTool()]).activate().run().deactivate()
