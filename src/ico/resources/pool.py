"""The pool of a process: where loaded resources stay between uses.

The operator decides the order (``with res.use("cuda") as model: ...``);
the pool decides where a released resource stays:

- ``use(res, device)`` takes the object from the pool, or loads it (into
  RAM, ``"cpu"``), and moves it to ``device``; the same key is one object
  for every operator of the process;
- leaving the block releases it: an idle object goes back to RAM unless
  the device budget keeps it there (``gpu``, 0 by default: the GPU holds
  only what is in use), and idle objects in RAM beyond the ``ram`` budget
  are dropped, least recently used first;
- an object in use is never moved or dropped; two users at once must ask
  for the same device.

Budgets: ``ICO_RESOURCES_RAM`` and ``ICO_RESOURCES_GPU`` (bytes, or
``"48GB"``, ``"512MB"``). RAM may also be a share of the memory the process
may use (``"80%"``: the physical memory, or the cgroup limit of a container
when lower) and defaults to ``"50%"``. Sizes come from
:meth:`Resource.size` (unknown counts as 0).

Every load, move, hit, release (with the time in use) and drop is an
event for listeners (the profiler).
"""

from __future__ import annotations

import contextlib
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

from ico.resources.base import Resource

T = TypeVar("T")

RAM_ENV = "ICO_RESOURCES_RAM"
GPU_ENV = "ICO_RESOURCES_GPU"
CGROUP = Path("/sys/fs/cgroup")
CPU = "cpu"

Action = Literal["load", "move", "hit", "release", "drop"]


class PoolEvent(TypedDict):
    ref: str
    label: str
    """The reference and the fields that tell the resource apart."""
    key: str
    action: Action
    device: str
    """Where the object is after the action (``"cpu"`` after a load)."""
    seconds: float
    """Of the load, move or drop; for ``release``, how long the block
    held the resource."""
    bytes: int | None


class Held(TypedDict):
    ref: str
    key: str
    device: str
    bytes: int | None
    users: int


@dataclass
class _Entry:
    resource: Resource[Any]
    obj: Any = None
    device: str | None = None
    """``None``: not loaded."""
    size: int | None = None
    users: int = 0
    use_device: str | None = None
    last_used: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class ResourcePool:
    """Loaded resources of a process, by key (see the module docs)."""

    def __init__(
        self,
        *,
        ram: int | None = None,
        gpu: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ram = ram if ram is not None else ram_budget(os.environ.get(RAM_ENV))
        self.gpu = gpu if gpu is not None else _budget(GPU_ENV, 0)
        self.clock = clock
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._listeners: list[Callable[[PoolEvent], None]] = []

    # ─── Listeners ───

    def add_listener(self, listener: Callable[[PoolEvent], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[PoolEvent], None]) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    # ─── Use ───

    @contextlib.contextmanager
    def use(self, res: Resource[T], device: str = CPU) -> Iterator[T]:
        """The object of ``res`` on ``device`` while the block runs.

        Raises:
            RuntimeError: The resource is in use on another device.
            Exception: Whatever ``load`` or ``to`` raised (nothing stays
                in the pool for a failed load).
        """
        key = res.key
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = self._entries[key] = _Entry(res)
            if entry.users and entry.use_device != device:
                raise RuntimeError(
                    f"resource {res.ref} is in use on {entry.use_device}, asked for {device}"
                )
            entry.users += 1
            entry.use_device = device
        try:
            with entry.lock:
                self._ready(entry, device)
                obj = entry.obj
        except BaseException:
            self._release(entry, failed=entry.obj is None)
            raise
        started = self.clock()
        try:
            yield obj
        finally:
            self._emit(entry, "release", self.clock() - started)
            self._release(entry)

    def preload(self, res: Resource[Any]) -> None:
        """Load ``res`` into RAM, if it is not loaded."""
        with self.use(res, CPU):
            pass

    def held(self) -> list[Held]:
        """What the pool holds, and where."""
        with self._lock:
            return [
                {
                    "ref": e.resource.ref,
                    "key": key,
                    "device": e.device or CPU,
                    "bytes": e.size,
                    "users": e.users,
                }
                for key, e in sorted(self._entries.items())
                if e.device is not None
            ]

    def clear(self) -> None:
        """Drop every idle resource."""
        with self._lock:
            idle = [e for e in self._entries.values() if not e.users]
        for entry in idle:
            self._drop(entry)

    # ─── Internals ───

    def _ready(self, entry: _Entry, device: str) -> None:
        res = entry.resource
        acted = False
        if entry.obj is None:
            start = self.clock()
            entry.obj = res.load()
            entry.device = CPU
            entry.size = res.size(entry.obj)
            self._emit(entry, "load", self.clock() - start)
            acted = True
        if entry.device != device:
            self._move(entry, device)
            acted = True
        if not acted:
            self._emit(entry, "hit", 0.0)

    def _move(self, entry: _Entry, device: str) -> None:
        start = self.clock()
        entry.obj = entry.resource.to(entry.obj, device)
        entry.device = device
        self._emit(entry, "move", self.clock() - start)

    def _release(self, entry: _Entry, *, failed: bool = False) -> None:
        with self._lock:
            entry.users -= 1
            entry.last_used = self.clock()
            if not entry.users:
                entry.use_device = None
            if failed and not entry.users:
                self._entries.pop(entry.resource.key, None)
                return
            back, drop = self._plan()
        for e in back:
            with e.lock:
                if not e.users and e.device not in (None, CPU):
                    self._move(e, CPU)
        for e in drop:
            self._drop(e)

    def _plan(self) -> tuple[list[_Entry], list[_Entry]]:
        """Idle entries to move back to RAM and to drop (under the lock).
        Budgets count what is in use too; only idle entries move or go,
        the least recently used first."""
        loaded = [e for e in self._entries.values() if e.device is not None]
        idle = sorted((e for e in loaded if not e.users), key=lambda e: e.last_used)
        back: list[_Entry] = []
        on_device = sum(e.size or 0 for e in loaded if e.device != CPU)
        for e in idle:
            if e.device != CPU and on_device > self.gpu:
                back.append(e)
                on_device -= e.size or 0
        in_ram = sum(e.size or 0 for e in loaded if e.device == CPU or e in back)
        drop: list[_Entry] = []
        for e in idle:
            if in_ram <= self.ram:
                break
            if e.device == CPU or e in back:
                drop.append(e)
                in_ram -= e.size or 0
        return [e for e in back if e not in drop], drop

    def _drop(self, entry: _Entry) -> None:
        with entry.lock:
            if entry.users or entry.obj is None:
                return
            start = self.clock()
            entry.resource.close(entry.obj)
            entry.obj = None
            entry.device = None
            self._emit(entry, "drop", self.clock() - start, device=CPU)
        with self._lock:
            if not entry.users and entry.obj is None:
                self._entries.pop(entry.resource.key, None)

    def _emit(
        self,
        entry: _Entry,
        action: Action,
        seconds: float,
        *,
        device: str | None = None,
    ) -> None:
        event: PoolEvent = {
            "ref": entry.resource.ref,
            "label": entry.resource.label,
            "key": entry.resource.key,
            "action": action,
            "device": device or entry.device or CPU,
            "seconds": seconds,
            "bytes": entry.size,
        }
        with self._lock:  # never called with the lock held
            listeners = list(self._listeners)
        for listener in listeners:
            listener(event)


_pool: ResourcePool | None = None
_pool_lock = threading.Lock()


def pool() -> ResourcePool:
    """The pool of this process (created on first use)."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ResourcePool()
        return _pool


def set_pool(new: ResourcePool | None) -> ResourcePool | None:
    """Replace the pool of this process (tests); returns the old one."""
    global _pool
    with _pool_lock:
        old, _pool = _pool, new
        return old


_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?B?)\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "B": 1,
    "K": 1e3,
    "KB": 1e3,
    "M": 1e6,
    "MB": 1e6,
    "G": 1e9,
    "GB": 1e9,
    "T": 1e12,
    "TB": 1e12,
}


def parse_size(text: str) -> int:
    """Bytes of ``"48GB"``, ``"512MB"``, ``"1000"``.

    Raises:
        ValueError: Not a size.
    """
    match = _SIZE.match(text)
    if match is None:
        raise ValueError(f"not a size: {text!r} (e.g. 48GB)")
    return int(float(match.group(1)) * _UNITS[match.group(2).upper()])


def _budget(env: str, default: int) -> int:
    value = os.environ.get(env)
    return parse_size(value) if value else default


def ram_budget(text: str | None, cgroup: Path = CGROUP) -> int:
    """Bytes of a RAM budget: a size, or a share of :func:`memory`
    (``"80%"``); ``None`` or empty is ``"50%"``.

    Raises:
        ValueError: Neither a size nor a share in (0, 100].
    """
    text = (text or "50%").strip()
    if not text.endswith("%"):
        return parse_size(text)
    try:
        share = float(text[:-1])
    except ValueError:
        raise ValueError(f"not a share: {text!r} (e.g. 80%)") from None
    if not 0 < share <= 100:
        raise ValueError(f"not a share: {text!r} (e.g. 80%)")
    return int(memory(cgroup) * share / 100)


def memory(cgroup: Path = CGROUP) -> int:
    """Bytes this process may use: the physical memory, or the memory limit
    of its cgroup (v2 ``memory.max``, v1 ``memory.limit_in_bytes``) when
    lower. A container sees the memory of its host, not its limit."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        total = 16 * 10**9
    for path in (cgroup / "memory.max", cgroup / "memory" / "memory.limit_in_bytes"):
        try:
            limit = path.read_text().strip()
        except OSError:
            continue
        if limit.isdigit():  # v2 "max": no limit
            total = min(total, int(limit))
    return total
