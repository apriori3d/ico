"""assets.toml / assets.lock and the cheap check of asset()."""

import textwrap
from pathlib import Path

import pytest

from ico.assets import AssetMissing, Assets, ManifestError, asset, load_manifest

REV = "ee6f4a40737a995bf5818954cfce6d59443b0f04"
SHA = "c" * 64
MANIFEST = f"""
version = 1

[assets.wan-high]
description = "Wan2.2 high-noise expert"
hf = {{ repo = "Comfy-Org/Wan", revision = "{REV}", file = "split/high.safetensors" }}
path = "wan22/high.safetensors"

[assets.umt5]
hf = {{ repo = "Wan-AI/VACE", revision = "{REV}", include = ["text_encoder/*", "tokenizer/*"] }}

[assets.propainter]
url = "https://example.com/v0.1.0/ProPainter.pth"
sha256 = "{SHA}"
"""
LOCK = f"""
version = 1

[[asset]]
id = "wan-high"
kind = "file"
files = [ {{ name = "high.safetensors", size = 5, sha256 = "{SHA}" }} ]

[[asset]]
id = "umt5"
kind = "tree"
files = [
  {{ name = "text_encoder/model.safetensors", size = 3, sha256 = "{SHA}" }},
  {{ name = "tokenizer/spiece.model", size = 2, sha256 = "{SHA}" }},
]
"""


def write(directory: Path, manifest: str = MANIFEST, lock: str | None = LOCK) -> Path:
    path = directory / "assets.toml"
    path.write_text(textwrap.dedent(manifest))
    if lock is not None:
        (directory / "assets.lock").write_text(textwrap.dedent(lock))
    return path


def put(root: Path, relative: str, size: int) -> None:
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_bytes(b"x" * size)


def test_the_manifest_and_its_lock(tmp_path: Path) -> None:
    manifest = load_manifest(write(tmp_path))

    high, umt5, pp = (manifest.assets[i] for i in ("wan-high", "umt5", "propainter"))
    assert (high.kind, high.path) == ("file", "wan22/high.safetensors")
    assert (umt5.kind, umt5.path) == ("tree", "umt5")
    assert umt5.hf is not None and umt5.hf.include == ("text_encoder/*", "tokenizer/*")
    assert (pp.kind, pp.path, pp.sha256) == ("file", "propainter/ProPainter.pth", SHA)
    assert manifest.locks["umt5"].size == 5
    assert "propainter" not in manifest.locks


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ('hf = { repo = "r", revision = "main", file = "f" }', "commit sha"),
        ('url = "https://x/y"', "needs its 'sha256'"),
        (
            f'url = "https://x/y"\nsha256 = "{SHA}"\nhf = {{ repo = "r", revision = "{REV}", file = "f" }}',
            "one source",
        ),
        (
            f'hf = {{ repo = "r", revision = "{REV}" }}',
            "'file' (one file) or 'include'",
        ),
        (
            f'hf = {{ repo = "r", revision = "{REV}", file = "f" }}\nsize = 3',
            "unknown key(s) size",
        ),
        (
            f'hf = {{ repo = "r", revision = "{REV}", file = "f" }}\npath = "../x"',
            "inside the root",
        ),
    ],
)
def test_bad_entries_are_named(tmp_path: Path, entry: str, message: str) -> None:
    write(tmp_path, f"[assets.bad]\n{entry}\n", lock=None)
    with pytest.raises(ManifestError, match="bad") as raised:
        load_manifest(tmp_path / "assets.toml")
    assert message in str(raised.value)


def test_a_lock_must_match_the_manifest(tmp_path: Path) -> None:
    write(
        tmp_path,
        lock=LOCK.replace('id = "umt5"\nkind = "tree"', 'id = "umt5"\nkind = "file"'),
    )
    with pytest.raises(ManifestError, match="a file asset has exactly one file"):
        load_manifest(tmp_path / "assets.toml")
    write(tmp_path, lock=LOCK.replace('"umt5"', '"gone"'))
    with pytest.raises(ManifestError, match="'gone' is not in assets.toml"):
        load_manifest(tmp_path / "assets.toml")


def test_present_assets_resolve_to_their_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    put(root, "wan22/high.safetensors", 5)
    put(root, "umt5/text_encoder/model.safetensors", 3)
    put(root, "umt5/tokenizer/spiece.model", 2)
    assets = Assets(load_manifest(write(tmp_path)), root)

    assert assets.path("wan-high") == root / "wan22/high.safetensors"
    assert assets.path("umt5") == root / "umt5"


def test_missing_assets_say_why_and_how_to_fetch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    put(root, "umt5/text_encoder/model.safetensors", 1)  # partial
    assets = Assets(load_manifest(write(tmp_path)), root)

    def missing(id: str) -> AssetMissing:
        with pytest.raises(AssetMissing) as raised:
            assets.path(id)
        return raised.value

    gone, partial = missing("wan-high"), missing("umt5")
    assert gone.reason == "missing" and gone.size == 5
    assert "ico assets fetch wan-high" in str(gone)
    assert partial.reason == "size"
    assert set(partial.files) == {
        "text_encoder/model.safetensors",
        "tokenizer/spiece.model",
    }
    assert missing("propainter").reason == "unlocked"
    unknown = missing("wan-hgih")
    assert unknown.reason == "unknown" and "known: propainter, umt5, wan-high" in str(
        unknown
    )


def test_asset_reads_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    put(root, "wan22/high.safetensors", 5)
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    write(repo)
    monkeypatch.setenv("ICO_ASSETS_ROOT", str(root))
    monkeypatch.delenv("ICO_ASSETS_MANIFEST", raising=False)
    monkeypatch.chdir(repo / "sub")

    assert asset("wan-high") == root / "wan22/high.safetensors"
    (root / "wan22/high.safetensors").write_bytes(b"xx")
    with pytest.raises(AssetMissing, match="wrong size"):
        asset("wan-high")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(AssetMissing) as raised:
        asset("wan-high")
    assert raised.value.reason == "no_manifest"
    monkeypatch.setenv("ICO_ASSETS_MANIFEST", str(repo / "assets.toml"))
    (root / "wan22/high.safetensors").write_bytes(b"xxxxx")
    assert asset("wan-high").is_file()


def test_a_changed_lock_is_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    put(root, "wan22/high.safetensors", 7)
    write(tmp_path)
    monkeypatch.setenv("ICO_ASSETS_ROOT", str(root))
    monkeypatch.setenv("ICO_ASSETS_MANIFEST", str(tmp_path / "assets.toml"))
    with pytest.raises(AssetMissing):
        asset("wan-high")
    write(tmp_path, lock=LOCK.replace("size = 5", "size = 7"))
    assert asset("wan-high").is_file()
