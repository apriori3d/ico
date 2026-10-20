"""lock / status / verify / fetch against a fake Hugging Face hub on
127.0.0.1 (metadata API, resolve with ranges and a CDN redirect)."""

import hashlib
import json
import textwrap
import threading
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ico.assets import Assets, asset, load_manifest
from ico.assets.fetch import FetchError, fetch, status, verify
from ico.assets.hub import HubError
from ico.assets.lockfile import lock
from ico.cli import main

REV = "ee6f4a40737a995bf5818954cfce6d59443b0f04"
TOKEN = "hf-test-token"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHub:
    """Repositories ``org/wan`` (LFS weights, a git config, a tree) and
    ``org/gated`` (needs the token); plain files under ``/files``."""

    def __init__(self) -> None:
        self.repos: dict[str, dict[str, bytes]] = {
            "org/wan": {
                "split/high.safetensors": b"H" * 3000,
                "text_encoder/model.safetensors": b"T" * 2000,
                "text_encoder/config.json": b'{"d": 1}',
                "tokenizer/spiece.model": b"S" * 500,
                "README.md": b"readme",
            },
            "org/gated": {"weights.bin": b"G" * 700},
        }
        self.files = {"ProPainter.pth": b"P" * 1200}
        self.lfs = {"split/high.safetensors", "text_encoder/model.safetensors",
                    "tokenizer/spiece.model", "weights.bin"}  # fmt: skip
        self.ranges = True
        self.gets: Counter[str] = Counter()
        self.ranged: list[str] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_HEAD(self) -> None:
                fake.serve(self, head=True)

            def do_GET(self) -> None:
                fake.serve(self, head=False)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def serve(self, request: BaseHTTPRequestHandler, *, head: bool) -> None:
        path = request.path.split("?")[0]
        if path.startswith("/api/models/"):
            repo = path.removeprefix("/api/models/").split("/revision/")[0]
            siblings = [
                {"rfilename": name, "size": len(data),
                 **({"lfs": {"sha256": sha(data), "size": len(data)}} if name in self.lfs else {})}
                for name, data in self.repos[repo].items()
            ]  # fmt: skip
            return self.send(request, json.dumps({"siblings": siblings}).encode(), head)
        if path.startswith("/cdn/"):
            assert (
                "authorization" not in request.headers
            ), "the token followed a redirect"
            repo, name = path.removeprefix("/cdn/").split("::")
            return self.content(request, self.repos[repo][name], head, key=name)
        if path.startswith("/files/"):
            name = path.removeprefix("/files/")
            return self.content(request, self.files[name], head, key=name)
        repo, rest = path[1:].split("/resolve/")
        name = rest.split("/", 1)[1]
        if (
            repo == "org/gated"
            and request.headers.get("authorization") != f"Bearer {TOKEN}"
        ):
            return self.send(request, b"gated", head, code=401)
        request.send_response(302)
        request.send_header("location", f"/cdn/{repo}::{name}")
        request.send_header("content-length", "0")
        request.end_headers()

    def content(
        self, request: BaseHTTPRequestHandler, data: bytes, head: bool, key: str
    ) -> None:
        if not head:
            self.gets[key] += 1
        wanted = request.headers.get("range")
        if wanted and self.ranges and not head:
            start = int(wanted.removeprefix("bytes=").rstrip("-"))
            self.ranged.append(key)
            return self.send(request, data[start:], head, code=206)
        self.send(request, data, head)

    def send(
        self, request: BaseHTTPRequestHandler, data: bytes, head: bool, code: int = 200
    ) -> None:
        request.send_response(code)
        request.send_header("content-length", str(len(data)))
        request.end_headers()
        if not head:
            request.wfile.write(data)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeHub]:
    fake = FakeHub()
    monkeypatch.setenv("HF_ENDPOINT", fake.url)
    monkeypatch.delenv("ICO_ASSETS_OFFLINE", raising=False)
    yield fake
    fake.server.shutdown()


@pytest.fixture
def repo(tmp_path: Path, hub: FakeHub, monkeypatch: pytest.MonkeyPatch) -> Path:
    manifest = tmp_path / "assets.toml"
    manifest.write_text(textwrap.dedent(f'''
        [assets.wan-high]
        hf = {{ repo = "org/wan", revision = "{REV}", file = "split/high.safetensors" }}
        path = "wan22/high.safetensors"

        [assets.umt5]
        hf = {{ repo = "org/wan", revision = "{REV}", include = ["text_encoder/*", "tokenizer/*"] }}

        [assets.propainter]
        url = "{hub.url}/files/ProPainter.pth"
        sha256 = "{sha(hub.files["ProPainter.pth"])}"

        [assets.gated]
        hf = {{ repo = "org/gated", revision = "{REV}", file = "weights.bin", token_env = "HF_TOKEN" }}
    '''))  # fmt: skip
    monkeypatch.setenv("ICO_ASSETS_MANIFEST", str(manifest))
    monkeypatch.setenv("ICO_ASSETS_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    return tmp_path


def store() -> Assets:
    return Assets.from_env()


def test_lock_takes_lfs_hashes_from_metadata(repo: Path, hub: FakeHub) -> None:
    locks = lock(store().manifest)

    high, umt5 = locks["wan-high"], locks["umt5"]
    assert [(f.name, f.size) for f in high.files] == [("high.safetensors", 3000)]
    assert high.files[0].sha256 == sha(hub.repos["org/wan"]["split/high.safetensors"])
    assert [f.name for f in umt5.files] == [
        "text_encoder/config.json", "text_encoder/model.safetensors", "tokenizer/spiece.model",
    ]  # fmt: skip
    assert umt5.files[0].sha256 == sha(b'{"d": 1}')
    assert locks["propainter"].files[0].size == 1200
    # only the small git file was downloaded to be hashed
    assert hub.gets == Counter({"text_encoder/config.json": 1})
    assert load_manifest(repo / "assets.toml").locks == locks


def test_lock_names_what_the_repository_lacks(repo: Path) -> None:
    text = (repo / "assets.toml").read_text().replace("tokenizer/*", "vae/*")
    (repo / "assets.toml").write_text(text)
    with pytest.raises(Exception, match="has nothing for vae/"):
        lock(store().manifest, ["umt5"])


def test_fetch_downloads_verifies_and_stops_when_present(
    repo: Path, hub: FakeHub
) -> None:
    lock(store().manifest)
    before = {s.id: s for s in status(store())}
    assert before["umt5"].state == "missing" and before["umt5"].to_fetch == 2508
    seen: list[tuple[str, int, int]] = []

    after = fetch(
        store(), progress=lambda id, done, total: seen.append((id, done, total))
    )

    assert all(s.state == "present" and s.verified for s in after)
    assert asset("umt5") / "tokenizer" / "spiece.model"
    assert (asset("wan-high")).read_bytes() == hub.repos["org/wan"][
        "split/high.safetensors"
    ]
    assert ("gated", 700, 700) in seen
    gets = sum(hub.gets.values())
    fetch(store())
    assert sum(hub.gets.values()) == gets
    assert not any((repo / "root" / ".ico" / "partial").iterdir())


def test_an_interrupted_download_continues(repo: Path, hub: FakeHub) -> None:
    lock(store().manifest, ["wan-high"])
    data = hub.repos["org/wan"]["split/high.safetensors"]
    part = repo / "root" / ".ico" / "partial" / sha(data)
    part.parent.mkdir(parents=True)
    part.write_bytes(data[:1000])
    assert status(store(), ["wan-high"])[0].state == "partial"

    fetch(store(), ["wan-high"])
    assert hub.ranged == ["split/high.safetensors"]
    assert asset("wan-high").read_bytes() == data

    hub.ranges = False  # a server without ranges: start over
    asset("wan-high").unlink()
    part.write_bytes(data[:1000])
    fetch(store(), ["wan-high"])
    assert asset("wan-high").read_bytes() == data


def test_a_changed_source_never_reaches_the_final_path(
    repo: Path, hub: FakeHub
) -> None:
    lock(store().manifest, ["wan-high"])
    hub.repos["org/wan"]["split/high.safetensors"] = b"X" * 3000
    with pytest.raises(FetchError, match="does not match the lock"):
        fetch(store(), ["wan-high"])
    assert not (repo / "root" / "wan22" / "high.safetensors").exists()
    assert not any((repo / "root" / ".ico" / "partial").iterdir())


def test_no_space_no_fetch(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock(store().manifest, ["wan-high"])

    class Usage:
        free = 10

    def usage(path: object) -> Usage:
        return Usage()

    monkeypatch.setattr("ico.assets.fetch.shutil.disk_usage", usage)
    with pytest.raises(FetchError, match="GB free"):
        fetch(store(), ["wan-high"])


def test_offline_refuses_the_network(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ICO_ASSETS_OFFLINE", "1")
    with pytest.raises(HubError, match="ICO_ASSETS_OFFLINE"):
        lock(store().manifest)
    monkeypatch.delenv("ICO_ASSETS_OFFLINE")
    lock(store().manifest, ["wan-high"])
    monkeypatch.setenv("ICO_ASSETS_OFFLINE", "1")
    assert fetch(store(), ["wan-high"], dry_run=True)[0].to_fetch == 3000
    with pytest.raises(HubError):
        fetch(store(), ["wan-high"])


def test_a_gated_file_needs_its_token(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock(store().manifest, ["gated"])  # metadata is public
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(HubError, match="HTTP 401"):
        fetch(store(), ["gated"])
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    assert fetch(store(), ["gated"])[0].state == "present"


def test_verify_full_finds_a_corrupt_file_of_the_right_size(repo: Path) -> None:
    lock(store().manifest, ["wan-high", "umt5"])
    fetch(store(), ["wan-high", "umt5"])
    assert verify(store(), full=True) == {
        "propainter": ["(not locked)"],
        "gated": ["(not locked)"],
    }

    asset("wan-high").write_bytes(b"Z" * 3000)
    assert verify(store(), ["wan-high"]) == {}
    assert verify(store(), ["wan-high"], full=True) == {
        "wan-high": ["high.safetensors"]
    }
    assert not status(store(), ["wan-high"])[0].verified


def test_concurrent_fetches_download_once(repo: Path, hub: FakeHub) -> None:
    lock(store().manifest, ["umt5"])
    threads = [
        threading.Thread(target=fetch, args=(store(), ["umt5"])) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert hub.gets["text_encoder/model.safetensors"] == 1
    assert status(store(), ["umt5"])[0].state == "present"


def test_the_command_line(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["assets", "lock", "wan-high"]) == 0
    assert main(["assets", "status", "wan-high"]) == 2
    assert "wan-high: missing (0.00 GB, 0.00 GB to fetch)" in capsys.readouterr().out
    assert main(["assets", "fetch", "wan-high"]) == 0
    assert "fetched" in capsys.readouterr().out
    assert main(["assets", "verify", "--full", "wan-high"]) == 0
    assert main(["assets", "status", "nope"]) == 1
    assert "unknown asset 'nope'" in capsys.readouterr().err
