from collections.abc import Iterator
from typing import TypeAlias, final

import torch
from torch import Tensor

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.source import IcoSource
from apriori.ico.utils.data.batcher import IcoBatcher
from examples.visual.cifar.augment import (
    AdjustBrightness,
    AdjustContrast,
    HorizontalFlip,
    VerticalFlip,
)
from examples.visual.cifar.dataset import CifarInMemoryDataset, CifarItem

CifarItemBatch: TypeAlias = Iterator[CifarItem]


@final
class CifarBatch:
    __slots__ = ("images", "labels")

    images: Tensor  # (B, 3, 32, 32)
    labels: Tensor  # (B,)

    def __init__(self, items: Iterator[CifarItem]):
        items_list = list(items)
        self.images = torch.stack([item.image for item in items_list])  # (B, 3, 32, 32)
        self.labels = torch.tensor([item.label for item in items_list])  # (B,)


def collate(items: CifarItemBatch) -> CifarBatch:
    return CifarBatch(items)


def to_shared_memory(batch: CifarBatch) -> CifarBatch:
    batch.images.share_memory_()
    batch.labels.share_memory_()
    return batch


class CifarItemsResolver:
    dataset: CifarInMemoryDataset

    def __init__(self, dataset: CifarInMemoryDataset):
        self.dataset = dataset

    def __call__(self, batch_ref: Iterator[int]) -> Iterator[CifarItem]:
        for index in batch_ref:
            yield self.dataset[index]


class IndexedAugmentationFlowFactory:
    def __init__(self, dataset: CifarInMemoryDataset):
        self.dataset = dataset

    def __call__(self) -> IcoOperator[Iterator[int], CifarBatch]:
        resolver = CifarItemsResolver(self.dataset)
        resolver_op = IcoOperator(resolver, name="CIFAR10 Items Resolver")

        augment_flow = create_augmentation_flow()
        resolved_flow = resolver_op | augment_flow
        return resolved_flow


def create_augmentation_flow(
    share_tensors: bool = True,
) -> IcoOperator[CifarItemBatch, CifarBatch]:
    item_aug_flow = IcoPipeline(
        HorizontalFlip(),
        VerticalFlip(),
        AdjustBrightness(),
        AdjustContrast(),
    )
    item_aug_flow.name = "Item augmentation flow"
    flow = item_aug_flow.stream() | IcoOperator(collate)

    if share_tensors:
        flow = flow | IcoOperator(to_shared_memory)

    flow.name = "Full Augmentation flow"
    return flow


def create_data_input_flow(
    batch_size: int,
    drop_last: bool = False,
) -> IcoOperator[None, Iterator[CifarItemBatch]]:
    dataset = CifarInMemoryDataset()
    cifar_source = IcoSource[CifarItem](lambda: iter(dataset), name="CIFAR10 dataset")
    batcher = IcoBatcher[CifarItem](batch_size=batch_size, drop_last=drop_last)

    data_input_flow = cifar_source | batcher
    data_input_flow.name = "CIFAR10 In-Memory Data Input Flow"

    return data_input_flow
