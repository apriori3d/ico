"""Where assets are on this machine, and the cheap check of :func:`asset`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ico.assets.manifest import AssetLock, AssetSpec, Manifest, load_manifest

MANIFEST_ENV = "ICO_ASSETS_MANIFEST"
ROOT_ENV = "ICO_ASSETS_ROOT"
DEFAULT_ROOT = "~/.cache/ico/assets"

Reason = Literal["no_manifest", "unknown", "unlocked", "missing", "size"]


class AssetMissing(Exception):
    """An asset is not usable here; the message says how to get it.

    Attributes:
        id: The asset id.
        reason: ``no_manifest``, ``unknown`` (not in ``assets.toml``),
            ``unlocked`` (not in ``assets.lock``), ``missing`` (files not
            on disk) or ``size`` (a file of the wrong size: partial or
            replaced).
        size: Bytes of the asset per the lock, when known.
        files: The missing or wrong files.
    """

    def __init__(
        self,
        id: str,
        reason: Reason,
        message: str,
        *,
        size: int | None = None,
        files: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.id = id
        self.reason: Reason = reason
        self.size = size
        self.files = files


@dataclass(frozen=True)
class Assets:
    """The assets of a manifest under a root directory."""

    manifest: Manifest
    root: Path

    @classmethod
    def from_env(cls) -> Assets:
        """The manifest of ``ICO_ASSETS_MANIFEST`` (else ``assets.toml`` in
        the working directory or a parent) under ``ICO_ASSETS_ROOT`` (else
        ``~/.cache/ico/assets``).

        Raises:
            FileNotFoundError: No manifest.
        """
        configured = os.environ.get(MANIFEST_ENV)
        path = Path(configured).expanduser() if configured else _find(Path.cwd())
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"no assets.toml (set {MANIFEST_ENV}, or work in the domain repository)"
            )
        root = Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT).expanduser()
        return _cached(path.resolve(), root)

    def location(self, id: str) -> Path:
        """Where the asset lives (its file, or its tree's directory),
        present or not."""
        return self.root / self._spec(id).path

    def path(self, id: str) -> Path:
        """The path of a present asset (sizes checked, no hashing).

        Raises:
            AssetMissing: Unknown, not locked, missing or of a wrong size.
        """
        spec = self._spec(id)
        lock = self.manifest.locks.get(id)
        if lock is None:
            raise AssetMissing(
                id,
                "unlocked",
                f"asset {id!r} is not locked: run `ico assets lock {id}`",
            )
        location = self.root / spec.path
        missing, wrong = _check(location, lock)
        if missing or wrong:
            gb = lock.size / 1e9
            what = "is missing" if not wrong else "has files of a wrong size"
            raise AssetMissing(
                id,
                "size" if wrong else "missing",
                f"asset {id!r} ({gb:.2f} GB) {what} under {self.root}: "
                f"run `ico assets fetch {id}`",
                size=lock.size,
                files=tuple(missing + wrong),
            )
        return location

    def _spec(self, id: str) -> AssetSpec:
        spec = self.manifest.assets.get(id)
        if spec is None:
            known = ", ".join(sorted(self.manifest.assets)) or "none"
            raise AssetMissing(
                id,
                "unknown",
                f"unknown asset {id!r} in {self.manifest.path}; known: {known}",
            )
        return spec


def asset(id: str) -> Path:
    """The local path of an asset: its file, or its tree's directory.

    Reads the manifest and root from the environment (see
    :meth:`Assets.from_env`); never downloads.

    Raises:
        AssetMissing: The asset is not usable here; the message says why
            and gives the fetch command.
    """
    try:
        assets = Assets.from_env()
    except FileNotFoundError as e:
        raise AssetMissing(id, "no_manifest", str(e)) from None
    return assets.path(id)


def _check(location: Path, lock: AssetLock) -> tuple[list[str], list[str]]:
    base = location.parent if lock.kind == "file" else location
    missing: list[str] = []
    wrong: list[str] = []
    for file in lock.files:
        path = location if lock.kind == "file" else base / file.name
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            missing.append(file.name)
            continue
        if size != file.size:
            wrong.append(file.name)
    return missing, wrong


def _find(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / "assets.toml"
        if candidate.is_file():
            return candidate
    return None


Stamp = tuple[int, int, int, int]
_cache: dict[tuple[Path, Path], tuple[Stamp, Assets]] = {}


def _cached(path: Path, root: Path) -> Assets:
    """Parse a manifest once per change of it or its lock."""
    lock = path.with_name("assets.lock")
    m = path.stat()
    k = lock.stat() if lock.is_file() else None
    stamp = (
        m.st_mtime_ns,
        m.st_size,
        k.st_mtime_ns if k else 0,
        k.st_size if k else -1,
    )
    hit = _cache.get((path, root))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    assets = Assets(load_manifest(path), root)
    _cache[(path, root)] = (stamp, assets)
    return assets
