"""Remote sources of assets over plain HTTPS (standard library only).

Hugging Face: file lists with sizes and LFS sha256 from
``<endpoint>/api/models/<repo>/revision/<revision>?blobs=true``, files from
``<endpoint>/<repo>/resolve/<revision>/<file>``; the endpoint is
``HF_ENDPOINT`` (as for ``huggingface_hub``), else ``https://huggingface.co``.
A token goes only to the endpoint, never to the CDN it redirects to.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import cast

OFFLINE_ENV = "ICO_ASSETS_OFFLINE"
HF_ENDPOINT = "https://huggingface.co"
TIMEOUT = 60


class HubError(RuntimeError):
    """A remote source failed or is not reachable."""


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int
    sha256: str | None
    """``None`` for a small file stored in git (not LFS): hash it."""


def offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "") not in ("", "0")


def check_online(what: str) -> None:
    if offline():
        raise HubError(f"{what} needs the network, but {OFFLINE_ENV} is set")


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", HF_ENDPOINT).rstrip("/")


def hf_files(repo: str, revision: str, token: str | None) -> list[RemoteFile]:
    """The files of a repository at a commit."""
    url = f"{hf_endpoint()}/api/models/{repo}/revision/{revision}?blobs=true"
    with open_url(url, token) as response:
        data = cast(dict[str, object], json.loads(response.read()))
    files: list[RemoteFile] = []
    for raw in cast(list[dict[str, object]], data.get("siblings") or []):
        lfs = cast(dict[str, object] | None, raw.get("lfs"))
        files.append(
            RemoteFile(
                name=str(raw["rfilename"]),
                size=int(cast(int, raw.get("size", 0))),
                sha256=str(lfs["sha256"]) if lfs else None,
            )
        )
    return files


def hf_url(repo: str, revision: str, name: str) -> str:
    return f"{hf_endpoint()}/{repo}/resolve/{revision}/{urllib.parse.quote(name)}"


def remote_size(url: str, token: str | None = None) -> int:
    """The size of a file by ``HEAD`` (redirects followed)."""
    with open_url(url, token, method="HEAD") as response:
        length = response.headers.get("content-length")
    if length is None:
        raise HubError(f"{url}: no Content-Length")
    return int(length)


def open_url(
    url: str, token: str | None = None, *, start: int = 0, method: str = "GET"
) -> HTTPResponse:
    """An open response; ``start`` asks for a range (check ``status``:
    206 means the range was honoured)."""
    request = urllib.request.Request(url, method=method)
    request.add_header("user-agent", "ico-assets")
    if token:
        # Unredirected: the token never follows a redirect to a CDN.
        request.add_unredirected_header("authorization", f"Bearer {token}")
    if start:
        request.add_header("range", f"bytes={start}-")
    try:
        return cast(HTTPResponse, urllib.request.urlopen(request, timeout=TIMEOUT))
    except urllib.error.HTTPError as e:
        hint = " (gated: set the token)" if e.code in (401, 403) else ""
        raise HubError(f"{method} {url}: HTTP {e.code}{hint}") from None
    except urllib.error.URLError as e:
        raise HubError(f"{method} {url}: {e.reason}") from None
