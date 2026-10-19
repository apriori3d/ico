"""``s3://`` storage on any S3-compatible service (AWS S3, Cloudflare R2,
Backblaze B2, MinIO), through ``boto3`` (the ``ico[remote]`` extra).

The client is configured by the environment, as boto3 does it: keys in
``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``, the service in
``AWS_ENDPOINT_URL_S3`` (e.g. ``https://<account>.r2.cloudflarestorage.com``
for R2), never in a job. ``s3://bucket/key`` is a file; a key that ends
with ``/`` or under which objects lie is a directory.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class S3Storage:
    """``s3://bucket/key`` URIs.

    Args:
        client: A boto3 S3 client; one from the environment by default
            (created on first use).
    """

    def __init__(self, client: Any = None) -> None:
        self._client: Any = client

    @property
    def client(self) -> Any:
        if self._client is None:
            boto3: Any = importlib.import_module("boto3")  # untyped
            self._client = boto3.client("s3")
        client: Any = self._client
        return client

    def get(self, uri: str, path: Path) -> None:
        bucket, key = _split(uri)
        keys = (
            []
            if key and not key.endswith("/") and self._is_file(bucket, key)
            else self._keys(bucket, key)
        )
        if not keys:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket, key, str(path))
            return
        prefix = key if key.endswith("/") or not key else key + "/"
        for found in keys:
            target = path / found[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(bucket, found, str(target))

    def put(self, path: Path, uri: str) -> None:
        bucket, key = _split(uri)
        if path.is_file():
            self.client.upload_file(str(path), bucket, key)
            return
        prefix = key.rstrip("/")
        for root, _, files in os.walk(path):
            for name in sorted(files):
                local = Path(root) / name
                relative = local.relative_to(path).as_posix()
                target = f"{prefix}/{relative}" if prefix else relative
                self.client.upload_file(str(local), bucket, target)

    def _is_file(self, bucket: str, key: str) -> bool:
        try:
            self.client.head_object(Bucket=bucket, Key=key)
        except Exception:
            return False
        return True

    def _keys(self, bucket: str, key: str) -> list[str]:
        """The keys under ``key`` taken as a directory."""
        prefix = key if key.endswith("/") or not key else key + "/"
        found: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            found.extend(str(item["Key"]) for item in page.get("Contents", []))
        return found


def _split(uri: str) -> tuple[str, str]:
    parts = urlsplit(uri)
    return parts.netloc, parts.path.lstrip("/")
