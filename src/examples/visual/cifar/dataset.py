from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from torchvision.datasets import CIFAR10  # type: ignore
from torchvision.transforms import ToTensor  # type: ignore


@dataclass(slots=True, frozen=True)
class CifarItem:
    image: Tensor  # (3, 32, 32)
    label: Tensor  # ()


class CifarInMemoryDataset:
    images: Tensor  # (N, 3, 32, 32)
    labels: Tensor  # (N,)

    def __init__(self, root: str = "data/"):
        dataset = CIFAR10(root=root, download=True, transform=ToTensor())

        # Preload all data into memory as tensors
        self.images = torch.stack(
            [dataset[i][0] for i in range(len(dataset))]
        )  # (N, 3, 32, 32)
        self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])  # (N,)

        # Share memory for multi-process data loading
        self.images.share_memory_()
        self.labels.share_memory_()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> CifarItem:
        return CifarItem(image=self.images[index], label=self.labels[index])

    def __iter__(self) -> Iterator[CifarItem]:
        for i in range(len(self)):
            yield self[i]
