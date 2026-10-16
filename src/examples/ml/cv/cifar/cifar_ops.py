"""Registered, parametric CIFAR-10 operators: the "crystallized" data loader.

`ico_cifar_complete_flow_mp.py` builds its data loader from closures and
local functions (the dataset captured by `WorkerFlowFactory`, the indices
captured by `get_train_indices`, `fetch_item` defined inside `__call__`).
That works in-process but cannot be described by a spec. This module
rewrites the same steps as registered operators whose configuration is
plain JSON constructor kwargs:

* `cifar.indices`   - `CifarIndices`: seeded train/validation index source;
* `cifar.fetch`     - `FetchCifarItem`: index → item, dataset opened lazily;
* `cifar.hflip`, `cifar.brightness`, `cifar.contrast` - augmentations;
* `cifar.shuffle`, `cifar.collate` - the original functions, registered.

Importing this module registers them in `ico.registry.default_registry`.
A process that builds a flow from a spec (or a spawned MPWorker runner process that
unpickles a SpecFactory) imports it to resolve the references.

Note:
    Each process opens the dataset itself (resources cannot travel in a
    spec), so the shared-memory dataset of the original example is lost.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor
from torchvision.datasets import CIFAR10  # type: ignore

from examples.ml.cv.cifar.ico_cifar_complete_flow_mp import (
    AdjustBrightness,
    AdjustContrast,
    CifarItem,
    CifarTransform,
    HorizontalFlip,
    collate,
    shuffle_indices,
)
from ico import IcoOperator, register

# Relative roots are resolved against this directory, so specs stay portable.
_HERE = Path(__file__).parent
CIFAR_TRAIN_SIZE = 50_000


# ─────────────────────────────────────────────────────────────────────────────
# Lazily opened dataset (one per process and root)
# ─────────────────────────────────────────────────────────────────────────────


class CifarTensors:
    """CIFAR-10 train split as tensors, same layout as `ToTensor()` output."""

    images: Tensor  # (N, 3, 32, 32), float in [0, 1]
    labels: Tensor  # (N,)

    def __init__(self, root: str) -> None:
        path = Path(root) if Path(root).is_absolute() else _HERE / root
        dataset = CIFAR10(root=str(path), train=True, download=True)
        # Raw uint8 arrays: much faster than converting 50k PIL images.
        data = torch.from_numpy(dataset.data)  # type: ignore[arg-type]  # (N, 32, 32, 3)
        self.images = data.permute(0, 3, 1, 2).float().div(255.0)
        self.labels = torch.tensor(dataset.targets)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> CifarItem:
        return CifarItem(image=self.images[index], label=self.labels[index])


_DATASETS: dict[str, CifarTensors] = {}


def open_cifar(root: str) -> CifarTensors:
    """The dataset for `root`, loaded on first use in this process."""
    if root not in _DATASETS:
        _DATASETS[root] = CifarTensors(root)
    return _DATASETS[root]


# ─────────────────────────────────────────────────────────────────────────────
# Source and item operators
# ─────────────────────────────────────────────────────────────────────────────


@register("cifar.indices", 1, tags={"cifar", "source"})
class CifarIndices(IcoOperator[None, Iterator[int]]):
    """Seeded train or validation indices of the CIFAR-10 train split.

    Replaces the `get_train_indices` / `get_val_indices` closures over a
    randomly shuffled index list. A plain operator from `None` rather than
    an `IcoSource` subclass, since `IcoSource` is final.
    """

    def __init__(
        self,
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 0,
        limit: int | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.limit = limit
        super().__init__(fn=self._indices, name=f"CIFAR10 {split} indices")

    def _indices(self, _: None) -> Iterator[int]:
        indices = list(range(CIFAR_TRAIN_SIZE))
        random.Random(self.seed).shuffle(indices)
        cut = int(CIFAR_TRAIN_SIZE * self.train_ratio)
        selected = indices[:cut] if self.split == "train" else indices[cut:]
        return iter(selected[: self.limit] if self.limit is not None else selected)


@register("cifar.fetch", 1, tags={"cifar"}, caps={"disk"})
class FetchCifarItem(IcoOperator[int, CifarItem]):
    """Fetch a CIFAR-10 item by index; the dataset is opened lazily.

    Replaces `fetch_item`, a local function closing over a preloaded
    dataset. `delay` simulates slow I/O, as in the original example.
    """

    def __init__(self, root: str = "data", delay: float | None = None) -> None:
        self.root = root
        self.delay = delay
        super().__init__(fn=self._fetch)

    def _fetch(self, index: int) -> CifarItem:
        if self.delay:
            time.sleep(self.delay)
        return open_cifar(self.root)[index]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations: the original transforms as parametric operators
# ─────────────────────────────────────────────────────────────────────────────


class _Augmentation(IcoOperator[CifarItem, CifarItem]):
    """Wraps an original `CifarTransform`; its params become spec params."""

    transform_type: type[CifarTransform]

    def __init__(self, p: float = 0.5, factor_max_delta: float = 1.0) -> None:
        self.p = p
        self.factor_max_delta = factor_max_delta
        super().__init__(fn=self.transform_type(p=p, factor_max_delta=factor_max_delta))


@register("cifar.hflip", 1, tags={"cifar", "augmentation"})
class RandomHorizontalFlip(_Augmentation):
    """Randomly flip the image horizontally with probability `p`."""

    transform_type = HorizontalFlip


@register("cifar.brightness", 1, tags={"cifar", "augmentation"})
class RandomBrightness(_Augmentation):
    """Randomly scale brightness by up to `factor_max_delta` with probability `p`."""

    transform_type = AdjustBrightness


@register("cifar.contrast", 1, tags={"cifar", "augmentation"})
class RandomContrast(_Augmentation):
    """Randomly scale contrast by up to `factor_max_delta` with probability `p`."""

    transform_type = AdjustContrast


# ─────────────────────────────────────────────────────────────────────────────
# Original module-level operators, registered as they are
# ─────────────────────────────────────────────────────────────────────────────

register("cifar.shuffle", 1, tags={"cifar"})(shuffle_indices)
register("cifar.collate", 1, tags={"cifar"})(collate)
