"""What the pool knows of a GPU: its free memory (all processes on it
count), emptying the allocator's cache, and out-of-memory errors.

Through torch, and only when the process already uses CUDA: the core does
not import torch, and a probe must not create a CUDA context (it takes
hundreds of megabytes of the GPU). Without one, :meth:`Device.memory` is
``None`` and the pool falls back to moving idle objects to RAM.
"""

from __future__ import annotations

import gc
import sys
from typing import Any, Protocol, cast


class Device(Protocol):
    """The GPU side of a pool (tests give a fake one)."""

    def memory(self, device: str) -> tuple[int, int] | None:
        """``(free, total)`` bytes of ``device``, or ``None`` if unknown."""
        ...

    def empty_cache(self, device: str) -> None:
        """Return memory freed by dropped objects to the device (collecting
        garbage first)."""
        ...


class TorchDevice:
    """:class:`Device` of CUDA devices, through torch when the process has
    initialized CUDA."""

    def memory(self, device: str) -> tuple[int, int] | None:
        torch = _torch(device)
        if torch is None:
            return None
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        free, total = cast(tuple[int, int], torch.cuda.mem_get_info(index))
        return free, total

    def empty_cache(self, device: str) -> None:
        torch = _torch(device)
        if torch is not None:
            # A dropped object in a reference cycle (a TensorRT program in an
            # fx GraphModule) holds its device memory until collected: 2.5 of
            # the 4.8 GB of an engine.
            gc.collect()
            torch.cuda.empty_cache()


def _torch(device: str) -> Any:
    """``torch`` if ``device`` is a CUDA device and this process has
    initialized CUDA, else ``None``."""
    if not device.startswith("cuda"):
        return None
    torch = cast(Any, sys.modules.get("torch"))
    if torch is None or not torch.cuda.is_initialized():
        return None
    return torch


def is_out_of_memory(error: BaseException) -> bool:
    """Whether ``error`` is a GPU out-of-memory error (torch's
    ``OutOfMemoryError``, or a runtime error that says so)."""
    if type(error).__name__ == "OutOfMemoryError":
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
