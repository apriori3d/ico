"""Assets: pinned, verifiable files that operators need (model weights,
LoRAs, tokenizers, tables).

A domain lists its assets in ``assets.toml`` (where each one comes from)
and a generated ``assets.lock`` (the size and sha256 of every file).
:func:`asset` returns the local path of an asset after a cheap check
(presence and sizes); it never downloads. See
``docs/proposals/2026-10-20-assets-and-resources.md``.
"""

from ico.assets.manifest import (
    AssetFile,
    AssetLock,
    AssetSpec,
    HfSource,
    Manifest,
    ManifestError,
    load_lock,
    load_manifest,
)
from ico.assets.store import AssetMissing, Assets, asset

__all__ = [
    "AssetFile",
    "AssetLock",
    "AssetMissing",
    "AssetSpec",
    "Assets",
    "HfSource",
    "Manifest",
    "ManifestError",
    "asset",
    "load_lock",
    "load_manifest",
]
