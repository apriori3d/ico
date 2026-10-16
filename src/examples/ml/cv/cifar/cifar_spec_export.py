"""Export the CIFAR-10 multiprocessing data loader as a JSON spec.

The flow mirrors the train data flow of `ico_cifar_complete_flow_mp.py`:

    indices | shuffle | progress.stream() | batcher
        | IcoAsyncStream(pool of MPWorkers)

where every MPWorker runs

    (progress | fetch | augmentations).stream() | collate

but is built from the registered operators of `cifar_ops`, so it can be
exported in strict mode and rebuilt elsewhere with `cifar_spec_run.py`.

Usage:
    poetry run python src/examples/ml/cv/cifar/cifar_spec_export.py [out.spec.json]
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

from examples.ml.cv.cifar.cifar_ops import (
    CifarIndices,
    FetchCifarItem,
    RandomBrightness,
    RandomContrast,
    RandomHorizontalFlip,
)
from examples.ml.cv.cifar.ico_cifar_complete_flow_mp import (
    CifarBatch,
    collate,
    shuffle_indices,
)
from ico import IcoAsyncStream, IcoBatcher, IcoPipeline, IcoProgress, MPWorker
from ico.core.operator import IcoOperator, IcoOperatorProtocol
from ico.spec import to_json, to_spec

DEFAULT_OUT = Path(__file__).parent / "cifar_data_flow.spec.json"


def make_worker_flow(batch_size: int) -> IcoOperatorProtocol[Iterator[int], CifarBatch]:
    """Flow run by every MPWorker runner process: indices batch → collated CifarBatch."""
    # Progress nodes are not parametrized: a spec carries no Python types, so
    # build() recreates them unparametrized and the round trip stays exact.
    worker_progress: IcoProgress[int] = IcoProgress(total=batch_size, name="Worker")
    augment = IcoPipeline(
        RandomHorizontalFlip(p=0.5),
        RandomBrightness(p=0.2, factor_max_delta=0.2),
        RandomContrast(p=0.2, factor_max_delta=0.2),
    )
    worker_flow = (
        worker_progress | FetchCifarItem(root="data") | augment
    ).stream() | collate
    worker_flow.name = "Worker flow"
    return worker_flow


def make_data_flow(
    batch_size: int = 8,
    num_workers: int = 4,
    limit: int | None = 2_000,
) -> IcoOperatorProtocol[None, Iterator[CifarBatch]]:
    """The train data loader with a pool of MPWorker runner processes."""
    indices = CifarIndices(split="train", train_ratio=0.8, seed=0, limit=limit)
    total = limit if limit is not None else 40_000
    track_progress: IcoProgress[int] = IcoProgress(
        total=total, name="Train Epoch Progress"
    )
    batcher: IcoBatcher[int] = IcoBatcher(batch_size=batch_size)

    def create_mp_worker() -> IcoOperator[Iterator[int], CifarBatch]:
        # Only used to describe the worker: build() recreates workers and
        # workers from the spec with a SpecFactory.
        return MPWorker(lambda: make_worker_flow(batch_size))

    workers_pool = IcoAsyncStream(create_mp_worker, pool_size=num_workers)

    data_flow = (
        indices | shuffle_indices | track_progress.stream() | batcher | workers_pool
    )
    data_flow.name = "Train data flow"
    return data_flow


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    data_flow = make_data_flow()
    data_flow.describe()

    # strict=True: fails on any anonymous callable, i.e. anything that could
    # not be rebuilt from the spec.
    spec = to_spec(data_flow, strict=True)
    out.write_text(to_json(spec) + "\n")
    print(f"Spec written to {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
