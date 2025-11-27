from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torchvision.transforms import (  # type: ignore
    functional as F,
)

from examples.visual.cifar.dataset import CifarItem


class CifarTransform(ABC):
    p: float

    def __init__(self, p: float = 1.0):
        self.p = p

    @abstractmethod
    def _image_transform(self, image: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, item: CifarItem) -> CifarItem:
        if torch.rand(1) >= self.p:
            return item

        return CifarItem(image=self._image_transform(item.image), label=item.label)


class HorizontalFlip(CifarTransform):
    def _image_transform(self, image: Tensor) -> Tensor:
        return F.hflip(image)


class VerticalFlip(CifarTransform):
    def _image_transform(self, image: Tensor) -> Tensor:
        return F.vflip(image)


class AdjustBrightness(CifarTransform):
    def __init__(self, p: float = 0.5, factor: float = 0.2):
        super().__init__(p)
        self.factor = factor

    def _image_transform(self, image: Tensor) -> Tensor:
        return F.adjust_brightness(image, self.factor)


class AdjustContrast(CifarTransform):
    def __init__(self, p: float = 0.5, factor: float = 0.2):
        super().__init__(p)
        self.factor = factor

    def _image_transform(self, image: Tensor) -> Tensor:
        return F.adjust_contrast(image, self.factor)
