"""Storage for the data of jobs: files go by reference (URIs), never
through the job's JSON.

The runner downloads the storage URIs of a job's input and params into the
job's work directory (stage in) and uploads its outputs (stage out). Only
``file://`` is built in (tests, a shared disk); ``s3://`` comes with the
``ico[remote]`` extra.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlsplit


class Storage(Protocol):
    """Files and directories behind URIs of one scheme."""

    def get(self, uri: str, path: Path) -> None:
        """Download ``uri`` (a file, or a directory when it ends with ``/``
        or is one) to ``path``."""
        ...

    def put(self, path: Path, uri: str) -> None:
        """Upload the file or directory ``path`` to ``uri``."""
        ...


class FileStorage:
    """``file://`` URIs: a local or shared directory."""

    def get(self, uri: str, path: Path) -> None:
        source = _local(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, path, dirs_exist_ok=True)
        else:
            shutil.copy2(source, path)

    def put(self, path: Path, uri: str) -> None:
        target = _local(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)


_STORAGES: dict[str, Storage] = {"file": FileStorage()}


def register_storage(scheme: str, storage: Storage) -> None:
    """Make URIs of ``scheme`` (``"s3"``) known to the runner."""
    _STORAGES[scheme] = storage


def storage_for(uri: str) -> Storage | None:
    """The storage of a URI, or ``None`` if it is not a storage URI."""
    scheme = urlsplit(uri).scheme
    return _STORAGES.get(scheme) if scheme and "://" in uri else None


def join(prefix: str, relative: str) -> str:
    """``prefix`` (a URI) and a relative path."""
    return prefix.rstrip("/") + "/" + relative.lstrip("/")


def basename(uri: str) -> str:
    """The last part of a URI's path (a directory's name for ``.../d/``)."""
    return Path(unquote(urlsplit(uri).path).rstrip("/")).name or "data"


def _local(uri: str) -> Path:
    parts = urlsplit(uri)
    return Path(unquote(parts.netloc + parts.path))
