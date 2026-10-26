"""What the pool knows of a GPU and of the other processes on it: its free
memory (all processes count), the room other processes ask for, emptying
the allocator's cache, and out-of-memory errors.

Free memory comes through torch, and only when the process already uses
CUDA: the core does not import torch, and a probe must not create a CUDA
context (it takes hundreds of megabytes of the GPU). Without one,
:meth:`Device.memory` is ``None`` and the pool falls back to moving idle
objects to RAM.

A process that cannot make room for a load asks the others: a file
``<tmp>/ico-gpu-asks/<device>.<pid>`` with the bytes it needs and until
when. Pools of other processes on the machine (its worker processes, other
runs) read the asks in their headroom checks and free that much more.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, cast

ASKS = Path(tempfile.gettempdir()) / "ico-gpu-asks"


class Device(Protocol):
    """The GPU side of a pool (tests give a fake one)."""

    def memory(self, device: str) -> tuple[int, int] | None:
        """``(free, total)`` bytes of ``device``, or ``None`` if unknown."""
        ...

    def ask(self, device: str, need: int, seconds: float) -> None:
        """Ask other processes for ``need`` bytes of ``device`` above their
        headroom for ``seconds``; ``need`` 0 withdraws the ask."""
        ...

    def asked(self, device: str) -> int:
        """Bytes of ``device`` that other processes ask for now."""
        ...

    def empty_cache(self, device: str) -> None:
        """Return memory freed by dropped objects to the device."""
        ...


class TorchDevice:
    """:class:`Device` of CUDA devices, through torch when the process has
    initialized CUDA; asks are files under ``asks``."""

    def __init__(self, asks: Path = ASKS) -> None:
        self.asks = asks

    def memory(self, device: str) -> tuple[int, int] | None:
        torch = _torch(device)
        if torch is None:
            return None
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        free, total = cast(tuple[int, int], torch.cuda.mem_get_info(index))
        return free, total

    def ask(self, device: str, need: int, seconds: float) -> None:
        path = self.asks / f"{device}.{os.getpid()}"
        with contextlib.suppress(OSError):
            if need <= 0:
                path.unlink(missing_ok=True)
                return
            self.asks.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(f"{need} {time.time() + seconds}")
            tmp.replace(path)

    def asked(self, device: str) -> int:
        total = 0
        now = time.time()
        own = f"{device}.{os.getpid()}"
        for path in self.asks.glob(f"{device}.*"):
            if path.name == own or path.suffix == ".tmp":
                continue
            try:
                need, until = path.read_text().split()
                if float(until) >= now:
                    total += int(need)
                else:
                    path.unlink(missing_ok=True)  # left by a process that died
            except (OSError, ValueError):
                continue
        return total

    def empty_cache(self, device: str) -> None:
        torch = _torch(device)
        if torch is not None:
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
