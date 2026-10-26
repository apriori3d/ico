"""The pool of a process: where loaded resources stay between uses.

The operator decides the order (``with res.use("cuda") as model: ...``);
the pool decides where a released resource stays:

- ``use(res, device)`` takes the object from the pool, or loads it (into
  RAM, ``"cpu"``), and moves it to ``device``; the same key is one object
  for every operator of the process. A resident resource
  (:attr:`Resource.resident`, e.g. a TensorRT engine) loads on its device
  and is used only there;
- leaving the block releases it: an idle object stays on the GPU while the
  GPU keeps a free **headroom** (``ICO_RESOURCES_GPU_FREE``, ``"20%"`` of
  its memory by default); below it, idle objects of this process leave
  the GPU, least recently used first: back to RAM, or dropped if resident.
  The free memory is the device's, so every process on a GPU (the main
  one and its worker processes) keeps the same headroom;
- before an object loads on or moves to a GPU, the pool makes room for
  it the same way (its size is known once it has loaded); if its own idle
  objects are not enough, it asks the other processes on the GPU for the
  rest and waits up to ``ROOM_WAIT`` seconds (see
  :mod:`ico.resources.device`), then goes on;
- while the pool holds objects on a GPU, a thread checks every ``WATCH``
  seconds that the GPU has its headroom plus what other processes ask
  for, so an idle process (a worker waiting for its next item) gives
  memory back when another one needs it;
- after a GPU out-of-memory error in a block, every idle object of this
  process leaves that GPU (a retry of the item then finds room);
- idle objects in RAM beyond the ``ram`` budget (``ICO_RESOURCES_RAM``:
  bytes, ``"48GB"``, or a share of the memory the process may use,
  ``"80%"``, ``"50%"`` by default) are dropped, least recently used first;
- an object in use is never moved or dropped; two users at once must ask
  for the same device.

The GPU's free memory comes from torch when the process has initialized
CUDA (:mod:`ico.resources.device`); without it (a process without torch,
before its first CUDA use) idle objects go back to RAM at once and
resident ones stay. Sizes come from :meth:`Resource.size` (unknown counts
as 0).

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
from ico.resources.device import Device, TorchDevice, is_out_of_memory

T = TypeVar("T")

RAM_ENV = "ICO_RESOURCES_RAM"
HEADROOM_ENV = "ICO_RESOURCES_GPU_FREE"
DEFAULT_HEADROOM = "20%"
ROOM_WAIT = 3.0
"""Seconds a load or move waits for other processes to free GPU memory
(they check every ``WATCH`` seconds)."""
ROOM_POLL = 0.1
WATCH = 1.0
"""Seconds between checks of the headroom while the pool holds objects on
a GPU."""
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
    """Where the object is after the action (``"cpu"`` after a load, the
    device of a resident resource)."""
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

    @property
    def resident(self) -> str | None:
        return type(self.resource).resident


class ResourcePool:
    """Loaded resources of a process, by key (see the module docs).

    Args:
        ram: The RAM budget in bytes; ``ICO_RESOURCES_RAM`` by default.
        headroom: The share of a GPU's memory to keep free;
            ``ICO_RESOURCES_GPU_FREE`` by default.
        device: Free memory of GPUs; torch's by default.
        clock: Time for LRU order and waits (tests).
        sleep: Waits for room on a GPU (tests).
        watch: Seconds between headroom checks; ``None``: no thread.
    """

    def __init__(
        self,
        *,
        ram: int | None = None,
        headroom: float | None = None,
        device: Device | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        watch: float | None = WATCH,
    ) -> None:
        self.ram = ram if ram is not None else ram_budget(os.environ.get(RAM_ENV))
        self.headroom = (
            headroom
            if headroom is not None
            else gpu_headroom(os.environ.get(HEADROOM_ENV))
        )
        self.device: Device = device if device is not None else TorchDevice()
        self.clock = clock
        self.sleep = sleep
        self.watch = watch
        self._entries: dict[str, _Entry] = {}
        self._sizes: dict[str, int] = {}
        """Sizes of keys loaded before (the room a load needs)."""
        self._watching = False
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
            RuntimeError: The resource is in use on another device, or it
                is resident on another device.
            Exception: Whatever ``load`` or ``to`` raised (nothing stays
                in the pool for a failed load).
        """
        key = res.key
        resident = type(res).resident
        if resident is not None and device != resident:
            raise RuntimeError(
                f"resource {res.ref} lives on {resident}, asked for {device}"
            )
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
        except BaseException as e:
            self._release(entry, failed=entry.obj is None)
            self._after_error(e, device)
            raise
        started = self.clock()
        error: BaseException | None = None
        try:
            yield obj
        except BaseException as e:
            error = e
            raise
        finally:
            self._emit(entry, "release", self.clock() - started)
            self._release(entry)
            if error is not None:
                self._after_error(error, device)

    def preload(self, res: Resource[Any]) -> None:
        """Load ``res`` (into RAM, or on its device if it is resident), if
        it is not loaded."""
        with self.use(res, type(res).resident or CPU):
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
            if entry.resident is not None:
                self._make_room(entry.resident, self._sizes.get(res.key, 0))
            start = self.clock()
            entry.obj = res.load()
            entry.device = entry.resident or CPU
            entry.size = res.size(entry.obj)
            if entry.size is not None:
                self._sizes[res.key] = entry.size
            self._emit(entry, "load", self.clock() - start)
            acted = True
        if entry.device != device:
            self._make_room(device, entry.size or 0)
            self._move(entry, device)
            acted = True
        if entry.device != CPU:
            self._watch()
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
            devices = self._gpus(idle=True)
        for device in devices:
            self._keep_headroom(device)
        with self._lock:
            drop = self._over_ram()
        for e in drop:
            self._drop(e)

    def _after_error(self, error: BaseException, device: str) -> None:
        """After a GPU out-of-memory error, every idle object of this
        process leaves that GPU."""
        if device != CPU and is_out_of_memory(error):
            self._keep_headroom(device, everything=True)

    # ─── The GPU ───

    def _gpus(self, *, idle: bool = False) -> set[str]:
        """Devices other than RAM that hold objects (idle ones only, with
        ``idle``); under the lock."""
        return {
            e.device
            for e in self._entries.values()
            if e.device is not None and e.device != CPU and not (idle and e.users)
        }

    def _keep_headroom(self, device: str, *, everything: bool = False) -> None:
        """Idle objects leave ``device`` until it has its headroom free,
        plus what other processes ask for; without its free memory, idle
        movable ones go back to RAM (resident ones stay); ``everything``:
        every idle one leaves."""
        while True:
            memory = None if everything else self.device.memory(device)
            if memory is not None:
                free, total = memory
                if free >= self.headroom * total + self.device.asked(device):
                    return
            victim = self._idle_on(device, movable=memory is None and not everything)
            if victim is None:
                return
            self._evict(victim, device)

    def _make_room(self, device: str, need: int) -> None:
        """Free ``need`` bytes on ``device`` above its headroom: idle
        objects of this process leave first, then other processes are
        asked and have ``ROOM_WAIT`` seconds to free theirs; the caller
        then goes on."""
        if device == CPU:
            return
        deadline = self.clock() + ROOM_WAIT
        asking = False
        try:
            while True:
                memory = self.device.memory(device)
                if memory is None:
                    return
                free, total = memory
                if free - need >= self.headroom * total:
                    return
                victim = self._idle_on(device, movable=False)
                if victim is not None:
                    self._evict(victim, device)
                    continue
                if self.clock() >= deadline:
                    return
                if not asking:
                    asking = True
                    self.device.ask(device, need, ROOM_WAIT)
                self.sleep(ROOM_POLL)
        finally:
            if asking:
                self.device.ask(device, 0, 0)

    def _idle_on(self, device: str, *, movable: bool) -> _Entry | None:
        """The least recently used idle object on ``device`` (a movable
        one only, with ``movable``)."""
        with self._lock:
            idle = [
                e
                for e in self._entries.values()
                if e.device == device
                and not e.users
                and not (movable and e.resident is not None)
            ]
        return min(idle, key=lambda e: e.last_used, default=None)

    def _evict(self, entry: _Entry, device: str) -> None:
        """An idle object leaves ``device``: back to RAM, or dropped if it
        is resident."""
        with entry.lock:
            if entry.users or entry.device != device:
                return
            if entry.resident is None:
                self._move(entry, CPU)
            else:
                self._close(entry)
        self._forget(entry)
        self.device.empty_cache(device)

    def _watch(self) -> None:
        """Start the headroom checks, if they do not run."""
        if self.watch is None:
            return
        with self._lock:
            if self._watching:
                return
            self._watching = True
        thread = threading.Thread(
            target=self._watch_loop, name="ico-resource-pool", daemon=True
        )
        thread.start()

    def _watch_loop(self) -> None:
        assert self.watch is not None
        while True:
            time.sleep(self.watch)
            with self._lock:
                devices = self._gpus()
                if not devices:
                    self._watching = False
                    return
            for device in devices:
                self._keep_headroom(device)

    # ─── RAM ───

    def _over_ram(self) -> list[_Entry]:
        """Idle objects in RAM to drop, the least recently used first, so
        that RAM holds at most the budget (what is in use counts too);
        under the lock."""
        in_ram = [e for e in self._entries.values() if e.device == CPU]
        idle = sorted((e for e in in_ram if not e.users), key=lambda e: e.last_used)
        total = sum(e.size or 0 for e in in_ram)
        drop: list[_Entry] = []
        for e in idle:
            if total <= self.ram:
                break
            drop.append(e)
            total -= e.size or 0
        return drop

    def _drop(self, entry: _Entry) -> None:
        with entry.lock:
            device = entry.device
            self._close(entry)
        self._forget(entry)
        if device is not None and device != CPU:
            self.device.empty_cache(device)

    def _close(self, entry: _Entry) -> None:
        """Drop the object (the entry's lock is held)."""
        if entry.users or entry.obj is None:
            return
        start = self.clock()
        entry.resource.close(entry.obj)
        entry.obj = None
        entry.device = None
        self._emit(entry, "drop", self.clock() - start, device=CPU)

    def _forget(self, entry: _Entry) -> None:
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


def gpu_headroom(text: str | None) -> float:
    """The share of a GPU's memory to keep free (``"20%"``); ``None`` or
    empty is ``"20%"``.

    Raises:
        ValueError: Not a share in [0, 100).
    """
    text = (text or DEFAULT_HEADROOM).strip()
    try:
        share = float(text.removesuffix("%")) if text.endswith("%") else -1.0
    except ValueError:
        share = -1.0
    if not 0 <= share < 100:
        raise ValueError(f"not a share of the GPU: {text!r} (e.g. 20%)")
    return share / 100


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
