"""Load the CIFAR-10 data loader from a JSON spec and run it.

Nothing here knows how the flow was built: the structure comes from the
spec written by `cifar_spec_export.py`, and the operators come from the
registry populated by importing `cifar_ops`. The MPWorkers are
recreated from the spec in spawned processes.

The run checks that every selected sample arrived exactly once, by
comparing label counts with the labels of the indices from the spec's own
index source.

Usage:
    poetry run python src/examples/ml/cv/cifar/cifar_spec_run.py [flow.spec.json]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import cast

# Registers the cifar.* operators that the spec refers to.
import examples.ml.cv.cifar.cifar_ops as cifar_ops
from examples.ml.cv.cifar.ico_cifar_complete_flow_mp import CifarBatch
from ico import IcoRuntime, IcoSink, RichProgressTool
from ico.core.operator import IcoOperatorProtocol
from ico.spec import build, from_json

DEFAULT_SPEC = Path(__file__).parent / "cifar_data_flow.spec.json"


def expected_label_counts(
    source: IcoOperatorProtocol[None, Iterator[int]],
) -> Counter[int]:
    """Labels of the samples the index source selects."""
    dataset = cifar_ops.open_cifar("data")
    return Counter(int(dataset.labels[i]) for i in source(None))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SPEC
    spec = from_json(path.read_text())

    data_flow = cast(IcoOperatorProtocol[None, Iterator[CifarBatch]], build(spec))
    print(f"Flow rebuilt from {path}:")
    data_flow.describe()

    batches = 0
    shapes: set[tuple[int, ...]] = set()
    labels: Counter[int] = Counter()

    def consume(batch: CifarBatch) -> None:
        nonlocal batches
        batches += 1
        shapes.add(tuple(batch.images.shape))
        labels.update(int(label) for label in batch.labels)

    # The runtime activates the MPWorkers and shows their progress.
    closed_flow = data_flow | IcoSink(consume)
    runtime = IcoRuntime(closed_flow, tools=[RichProgressTool()])
    t0 = time.perf_counter()
    runtime.activate()  # spawns the MPWorker runner processes
    t1 = time.perf_counter()
    runtime.run()
    t2 = time.perf_counter()
    runtime.deactivate()  # stops the workers
    t3 = time.perf_counter()

    # The index source alone, rebuilt from the same spec.
    source = cast(
        IcoOperatorProtocol[None, Iterator[int]], build(spec.root.children[0])
    )
    expected = expected_label_counts(source)

    print(f"Batches: {batches}, image batch shapes: {sorted(shapes)}")
    print(
        f"Samples: {sum(labels.values())} | start workers {t1 - t0:.1f}s, "
        f"run {t2 - t1:.1f}s, stop workers {t3 - t2:.1f}s"
    )
    print(f"Every selected sample arrived exactly once: {labels == expected}")
    if labels != expected:
        sys.exit(1)


if __name__ == "__main__":
    main()
