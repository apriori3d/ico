"""Prefetch jobs: assets downloaded into the worker's asset root, events
and results; runs that carry the manifest resolve asset(id)."""

import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

import ico.remote.runpod_handler as runpod_handler
from ico.core.runtime.cancel import IcoCancelToken
from ico.remote import (
    FetchJob,
    Job,
    LocalExecutor,
    Package,
    fetch_job,
    load_job,
    result,
)
from ico.remote.job import JsonObject
from tests.apriori.ico.assets.test_fetch import REV, FakeHub, sha

ASSET_OPS = textwrap.dedent(
    '''
    from ico.assets import AssetId, asset
    from ico.registry import register
    from ico import IcoOperator


    @register("jobs.asset_text", 1)
    class AssetText(IcoOperator[str, str]):
        """The text of a file asset."""

        def __init__(self, notes: AssetId = "notes") -> None:
            self.notes = notes
            super().__init__(fn=self._apply)

        def _apply(self, prefix: str) -> str:
            return prefix + asset(self.notes).read_text()
    '''
)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeHub]:
    fake = FakeHub()
    fake.repos["org/wan"]["notes.txt"] = b"hello"
    monkeypatch.setenv("HF_ENDPOINT", fake.url)
    monkeypatch.setenv("ICO_ASSETS_ROOT", str(tmp_path / "volume" / "assets"))
    monkeypatch.delenv("ICO_ASSETS_OFFLINE", raising=False)
    monkeypatch.delenv("ICO_ASSETS_MANIFEST", raising=False)
    yield fake
    fake.server.shutdown()


def manifest(hub: FakeHub, tmp_path: Path) -> Path:
    data = hub.repos["org/wan"]
    path = tmp_path / "domain" / "assets.toml"
    path.parent.mkdir()
    path.write_text(textwrap.dedent(f'''
        [assets.wan-high]
        hf = {{ repo = "org/wan", revision = "{REV}", file = "split/high.safetensors" }}

        [assets.notes]
        hf = {{ repo = "org/wan", revision = "{REV}", file = "notes.txt" }}
    '''))  # fmt: skip
    high, notes = data["split/high.safetensors"], data["notes.txt"]
    path.with_name("assets.lock").write_text(textwrap.dedent(f'''
        version = 1

        [[asset]]
        id = "notes"
        kind = "file"
        files = [ {{ name = "notes.txt", size = {len(notes)}, sha256 = "{sha(notes)}" }} ]

        [[asset]]
        id = "wan-high"
        kind = "file"
        files = [ {{ name = "high.safetensors", size = {len(high)}, sha256 = "{sha(high)}" }} ]
    '''))  # fmt: skip
    return path


def events(handle: Any) -> list[dict[str, Any]]:
    return [cast(dict[str, Any], e) for e in handle.events()]


def test_a_prefetch_fills_the_asset_root(hub: FakeHub, tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path / "jobs")
    job = FetchJob.from_files(manifest(hub, tmp_path))
    first = events(executor.submit(job))
    gets = sum(hub.gets.values())
    second = events(executor.submit(FetchJob.from_files(manifest_path(tmp_path))))

    fetched = [e for e in first if e["event"] == "fetch"]
    assert {e["asset"] for e in fetched} == {"notes", "wan-high"}
    assert fetched[-1]["done"] == fetched[-1]["total"]
    states = {a["id"]: a["state"] for a in first[-1]["result"]["assets"]}
    assert states == {"notes": "present", "wan-high": "present"}
    root = tmp_path / "volume" / "assets"
    assert (root / "notes" / "notes.txt").read_bytes() == b"hello"
    assert first[-1]["result"]["root"] == str(root)
    assert second[-1]["event"] == "result" and sum(hub.gets.values()) == gets


def manifest_path(tmp_path: Path) -> Path:
    return tmp_path / "domain" / "assets.toml"


def test_a_run_resolves_the_assets_it_carries(hub: FakeHub, tmp_path: Path) -> None:
    path = manifest(hub, tmp_path)
    executor = LocalExecutor(tmp_path / "jobs")
    result(executor.submit(FetchJob.from_files(path, ["notes"])))
    run = Job(
        spec={"kind": "op", "op": "jobs.asset_text@1"},
        input="> ",
        packages=(Package("asset_ops", "draft"),),
        sources={
            "asset_ops/__init__.py": "",
            "asset_ops/ops.py": ASSET_OPS,
            "assets.toml": path.read_text(),
            "assets.lock": path.with_name("assets.lock").read_text(),
        },
    )
    assert result(executor.submit(run))["outputs"] == ["> hello"]


def test_a_run_without_its_assets_ends_at_once(hub: FakeHub, tmp_path: Path) -> None:
    path = manifest(hub, tmp_path)
    run = Job(
        spec={"kind": "op", "op": "jobs.asset_text@1"},
        input="> ",
        packages=(Package("asset_ops", "draft"),),
        sources={
            "asset_ops/__init__.py": "",
            "asset_ops/ops.py": ASSET_OPS,
            "assets.toml": path.read_text(),
            "assets.lock": path.with_name("assets.lock").read_text(),
        },
    )
    [error] = events(LocalExecutor(tmp_path / "jobs").submit(run))

    assert error["code"] == "missing_assets"
    assert error["assets"] == [{"id": "notes", "reason": "missing", "size": 5}]
    assert "prefetch them first" in error["message"]


def test_prefetch_errors_and_cancel(hub: FakeHub, tmp_path: Path) -> None:
    path = manifest(hub, tmp_path)
    seen: list[JsonObject] = []
    fetch_job(FetchJob.from_files(path, ["nope"]), seen.append, tmp_path / "w1")
    token = IcoCancelToken()
    token.cancel("cancelled")
    fetch_job(
        FetchJob.from_files(path, ["wan-high"]), seen.append, tmp_path / "w2", token
    )

    assert (seen[0]["code"], "unknown asset 'nope'" in str(seen[0]["message"])) == (
        "fetch_error",
        True,
    )
    assert seen[-1]["code"] == "cancelled"
    assert not (tmp_path / "volume" / "assets" / "wan-high").exists()


def test_jobs_round_trip_by_kind(tmp_path: Path) -> None:
    fetch = FetchJob(manifest="m", lock="l", ids=("a",))
    run = Job(spec={"kind": "op", "op": "x@1"})
    assert load_job(fetch.to_dict()) == fetch
    assert load_job(run.to_dict()) == run


def test_the_handler_puts_assets_on_the_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICO_ASSETS_ROOT", raising=False)
    monkeypatch.setattr(runpod_handler, "VOLUME", tmp_path)
    assert runpod_handler._asset_env() == {"ICO_ASSETS_ROOT": str(tmp_path / "assets")}  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setenv("ICO_ASSETS_ROOT", "/data")
    assert runpod_handler._asset_env() == {}  # pyright: ignore[reportPrivateUsage]
