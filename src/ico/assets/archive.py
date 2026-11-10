"""Archives of ``url`` assets with ``unpack``: their files, hashed, and
the unpacking of locked files (a fetch)."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from ico.assets.manifest import AssetFile

CHUNK = 1 << 20


class ArchiveError(ValueError):
    """An archive that cannot be an asset: an unsafe member path, or files
    that do not match the lock."""


def members(archive: Path) -> tuple[AssetFile, ...]:
    """The files of a zip archive (directories left out), by name, with
    their sizes and sha256.

    Raises:
        ArchiveError: A member path is absolute or leaves the archive
            (``..``), or two members have the same name.
    """
    files: list[AssetFile] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            _safe(info.filename)
            digest = hashlib.sha256()
            with zf.open(info) as member:
                while chunk := member.read(CHUNK):
                    digest.update(chunk)
            files.append(AssetFile(info.filename, info.file_size, digest.hexdigest()))
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        raise ArchiveError(f"{archive.name}: members with the same name")
    return tuple(sorted(files, key=lambda f: f.name))


def unpack(archive: Path, dest: Path, files: tuple[AssetFile, ...]) -> None:
    """Unpack the given files of a zip archive under ``dest``, each checked
    against its size and sha256 while it is written.

    Raises:
        ArchiveError: A file is not in the archive or does not match.
    """
    with zipfile.ZipFile(archive) as zf:
        for entry in files:
            _safe(entry.name)
            try:
                info = zf.getinfo(entry.name)
            except KeyError:
                raise ArchiveError(f"{archive.name}: no {entry.name!r}") from None
            target = dest / entry.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with zf.open(info) as member, target.open("wb") as out:
                while chunk := member.read(CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != entry.size or digest.hexdigest() != entry.sha256:
                raise ArchiveError(
                    f"{archive.name}: {entry.name!r} does not match the lock "
                    f"({size} bytes, sha256 {digest.hexdigest()[:12]}…; "
                    f"expected {entry.size}, {entry.sha256[:12]}…)"
                )


def place(unpacked: Path, location: Path, files: tuple[AssetFile, ...]) -> None:
    """Move unpacked files to the asset's directory (each by ``os.replace``)
    and remove what is left of ``unpacked``."""
    for entry in files:
        target = location / entry.name
        target.parent.mkdir(parents=True, exist_ok=True)
        (unpacked / entry.name).replace(target)
    shutil.rmtree(unpacked, ignore_errors=True)


def _safe(name: str) -> None:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ArchiveError(f"unsafe path in the archive: {name!r}")


__all__ = ["ArchiveError", "members", "place", "unpack"]
