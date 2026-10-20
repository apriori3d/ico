"""Resource rows of a profile: what each resource cost and how it was used.

Collected from the events of the process's resource pool while the
profiler is active. A row answers three questions:

- **Is it needed?** ``uses`` and ``in_use`` (seconds inside ``use()``).
- **How much waiting?** ``loads`` / ``load_seconds`` and ``moves`` /
  ``move_seconds`` (RAM ↔ device). When they come near ``in_use``, the
  model is moved or loaded more than it works.
- **Is the budget too small?** ``reloads``: loads after it was dropped in
  the same run; ``drops`` counts the drops.

``bytes`` is the size on its device; the snapshot's ``resource_peaks``
gives the most the pool held at once per device during the run.
Resources of worker processes are not included yet (their pools are their
own).
"""

from __future__ import annotations

import threading
from typing import TypedDict

from ico.resources.pool import PoolEvent


class ResourceProfile(TypedDict):
    ref: str
    key: str
    uses: int
    in_use: float
    loads: int
    load_seconds: float
    reloads: int
    moves: int
    move_seconds: float
    drops: int
    bytes: int | None
    devices: list[str]
    """Devices it was used on."""


class ResourceStats:
    """A pool listener that aggregates :class:`ResourceProfile` rows."""

    def __init__(self) -> None:
        self._rows: dict[str, ResourceProfile] = {}
        self._where: dict[str, tuple[str, int]] = {}
        """Key → (device, bytes) of what the pool holds now."""
        self._peaks: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self, event: PoolEvent) -> None:
        key, action = event["key"], event["action"]
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                row = self._rows[key] = {
                    "ref": event["ref"], "key": key, "uses": 0, "in_use": 0.0,
                    "loads": 0, "load_seconds": 0.0, "reloads": 0, "moves": 0,
                    "move_seconds": 0.0, "drops": 0, "bytes": None, "devices": [],
                }  # fmt: skip
            if event["bytes"] is not None:
                row["bytes"] = event["bytes"]
            if action == "load":
                row["reloads"] += 1 if row["loads"] else 0
                row["loads"] += 1
                row["load_seconds"] += event["seconds"]
            elif action == "move":
                row["moves"] += 1
                row["move_seconds"] += event["seconds"]
            elif action == "release":
                row["uses"] += 1
                row["in_use"] += event["seconds"]
                if event["device"] not in row["devices"]:
                    row["devices"].append(event["device"])
            elif action == "drop":
                row["drops"] += 1
            if action == "drop":
                self._where.pop(key, None)
            else:
                self._where[key] = (event["device"], event["bytes"] or 0)
            held: dict[str, int] = {}
            for device, size in self._where.values():
                held[device] = held.get(device, 0) + size
            for device, size in held.items():
                self._peaks[device] = max(self._peaks.get(device, 0), size)

    def rows(self) -> list[ResourceProfile]:
        with self._lock:
            return [
                {**row, "devices": list(row["devices"])}
                for _, row in sorted(self._rows.items())
            ]

    def peaks(self) -> dict[str, int]:
        with self._lock:
            return dict(self._peaks)
