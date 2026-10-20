"""``ico assets status / verify / fetch`` under an asset root.

The root keeps its bookkeeping in ``<root>/.ico``: ``partial/<sha256>``
(downloads in progress), ``locks/<id>.lock`` (one fetcher per asset,
``flock``: other processes and pods on a shared volume wait), and
``verified/<id>.json`` (size, mtime and sha256 of each file at its last
full check).

A fetch downloads only what is missing or of a wrong size, into a
partial file hashed while it streams; a partial file is continued with
a range request after hashing what is there. Only a file of the right
size and sha256 is ``fsync``-ed and renamed into place, so a crash never
leaves a wrong file at the final path. A run never fetches.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ico.assets.hub import HubError, check_online, hf_url, open_url
from ico.assets.manifest import AssetFile, AssetLock, AssetSpec
from ico.assets.store import Assets

CHUNK = 1 << 20

State = Literal["present", "missing", "partial", "wrong_size", "unlocked"]
Progress = Callable[[str, int, int], None]
"""``(asset id, bytes done, bytes total)`` while a fetch streams."""


class FetchError(RuntimeError):
    """A download failed or did not match the lock."""


@dataclass(frozen=True)
class Status:
    """The state of an asset under a root.

    ``present``: every file has its size (a full hash is :func:`verify`);
    ``partial``: something is missing and part of it is downloading or
    was interrupted; ``missing``: nothing of the missing files on disk;
    ``wrong_size``: a file at its final path has another size;
    ``unlocked``: not in ``assets.lock``.
    """

    id: str
    state: State
    size: int
    """Bytes of the asset (0 when unlocked)."""
    to_fetch: int
    """Bytes still to download (partial parts counted as done)."""
    verified: bool
    """Fully hashed and unchanged since."""


@dataclass(frozen=True)
class _File:
    spec: AssetSpec
    entry: AssetFile
    path: Path
    url: str


def status(assets: Assets, ids: list[str] | None = None) -> list[Status]:
    return [_status(assets, id) for id in _ids(assets, ids)]


def _status(assets: Assets, id: str) -> Status:
    lock = assets.manifest.locks.get(id)
    if lock is None:
        assets.location(id)  # unknown ids raise
        return Status(id, "unlocked", 0, 0, False)
    to_fetch, partial, wrong = 0, False, False
    for f in _files(assets, id, lock):
        size = _size(f.path)
        if size == f.entry.size:
            continue
        wrong = wrong or size is not None
        done = _size(_partial(assets.root, f.entry)) or 0
        partial = partial or done > 0
        to_fetch += f.entry.size - min(done, f.entry.size)
    state: State = (
        "present"
        if not to_fetch and not wrong
        else "wrong_size"
        if wrong
        else "partial"
        if partial
        else "missing"
    )
    verified = state == "present" and _verified(assets, id, lock)
    return Status(id, state, lock.size, to_fetch, verified)


def verify(
    assets: Assets, ids: list[str] | None = None, *, full: bool = False
) -> dict[str, list[str]]:
    """Check sizes (and the sha256 of every file with ``full``); returns
    the bad files per asset (absent: all good). A full check that passes
    is recorded in ``.ico/verified``."""
    bad: dict[str, list[str]] = {}
    for id in _ids(assets, ids):
        lock = assets.manifest.locks.get(id)
        if lock is None:
            bad[id] = ["(not locked)"]
            continue
        wrong: list[str] = []
        for f in _files(assets, id, lock):
            if _size(f.path) != f.entry.size or (
                full and _sha256(f.path) != f.entry.sha256
            ):
                wrong.append(f.entry.name)
        if wrong:
            bad[id] = wrong
        elif full:
            _record(assets, id, lock)
    return bad


def fetch(
    assets: Assets,
    ids: list[str] | None = None,
    *,
    dry_run: bool = False,
    progress: Progress | None = None,
) -> list[Status]:
    """Download what is missing; returns the status before (with
    ``dry_run``) or after.

    Raises:
        HubError: Offline, or a source failed.
        FetchError: Not enough free space, an unlocked asset, or a
            download that does not match the lock (its partial file is
            removed).
    """
    chosen = _ids(assets, ids)
    before = status(assets, chosen)
    unlocked = [s.id for s in before if s.state == "unlocked"]
    if unlocked:
        raise FetchError(
            f"not locked: {', '.join(unlocked)}; run `ico assets lock` first"
        )
    needed = sum(s.to_fetch for s in before)
    if dry_run or not needed:
        return before
    check_online("ico assets fetch")
    assets.root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(assets.root).free
    if needed > free:
        raise FetchError(
            f"{needed / 1e9:.2f} GB to fetch, {free / 1e9:.2f} GB free under {assets.root}"
        )
    for s in before:
        if s.to_fetch or s.state == "wrong_size":
            _fetch_one(assets, s.id, progress)
    return status(assets, chosen)


def _fetch_one(assets: Assets, id: str, progress: Progress | None) -> None:
    lock = assets.manifest.locks[id]
    with _locked(assets.root / ".ico" / "locks" / f"{id}.lock"):
        files = [f for f in _files(assets, id, lock) if _size(f.path) != f.entry.size]
        total = sum(f.entry.size for f in files)
        done = 0
        for f in files:
            token = None
            hf = f.spec.hf
            if hf is not None and hf.token_env:
                token = os.environ.get(hf.token_env)

            def step(n: int, base: int = done) -> None:
                if progress is not None:
                    progress(id, base + n, total)

            _download(assets.root, f, token, step)
            done += f.entry.size
        _record(assets, id, lock)


def _download(
    root: Path, f: _File, token: str | None, step: Callable[[int], None]
) -> None:
    part = _partial(root, f.entry)
    part.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    have = 0
    if part.exists():
        have = part.stat().st_size
        if have > f.entry.size:
            part.unlink()
            have = 0
        else:
            with part.open("rb") as existing:
                while chunk := existing.read(CHUNK):
                    digest.update(chunk)
    if have < f.entry.size:
        with open_url(f.url, token, start=have) as response:
            if have and response.status != 206:
                have, digest = 0, hashlib.sha256()  # no range support: start over
            with part.open("r+b" if have else "wb") as out:
                out.seek(have)
                out.truncate()
                while chunk := response.read(CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
                    have += len(chunk)
                    step(have)
                    if have > f.entry.size:
                        break
                out.flush()
                os.fsync(out.fileno())
    sha256 = digest.hexdigest()
    if have != f.entry.size or sha256 != f.entry.sha256:
        part.unlink(missing_ok=True)
        raise FetchError(
            f"{f.spec.id}: {f.entry.name} does not match the lock "
            f"({have} bytes, sha256 {sha256[:12]}…; expected {f.entry.size}, "
            f"{f.entry.sha256[:12]}…): the source changed or the download broke"
        )
    f.path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, f.path)


# ─── Helpers ───


def _ids(assets: Assets, ids: list[str] | None) -> list[str]:
    chosen = ids if ids else sorted(assets.manifest.assets)
    for id in chosen:
        assets.location(id)  # unknown ids raise AssetMissing
    return chosen


def _files(assets: Assets, id: str, lock: AssetLock) -> list[_File]:
    spec = assets.manifest.assets[id]
    location = assets.location(id)
    files: list[_File] = []
    for entry in lock.files:
        if spec.url is not None:
            url = spec.url
        else:
            assert spec.hf is not None
            name = spec.hf.file if spec.kind == "file" and spec.hf.file else entry.name
            url = hf_url(spec.hf.repo, spec.hf.revision, name)
        path = location if spec.kind == "file" else location / entry.name
        files.append(_File(spec, entry, path, url))
    return files


def _partial(root: Path, entry: AssetFile) -> Path:
    return root / ".ico" / "partial" / entry.sha256


def _size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _record(assets: Assets, id: str, lock: AssetLock) -> None:
    record = {
        f.entry.name: {
            "size": f.entry.size,
            "mtime_ns": f.path.stat().st_mtime_ns,
            "sha256": f.entry.sha256,
        }
        for f in _files(assets, id, lock)
    }
    path = assets.root / ".ico" / "verified" / f"{id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))


def _verified(assets: Assets, id: str, lock: AssetLock) -> bool:
    path = assets.root / ".ico" / "verified" / f"{id}.json"
    try:
        record = json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return False
    for f in _files(assets, id, lock):
        seen = record.get(f.entry.name)
        if not isinstance(seen, dict) or seen != {
            "size": f.entry.size,
            "mtime_ns": f.path.stat().st_mtime_ns,
            "sha256": f.entry.sha256,
        }:
            return False
    return True


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One fetcher per asset; others wait (``flock``: local disks, NFSv4,
    CephFS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


__all__ = ["FetchError", "HubError", "Status", "fetch", "status", "verify"]
