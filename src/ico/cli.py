"""``ico``: the command line of ICO (``ico assets ...``)."""

from __future__ import annotations

import argparse
import sys
import time

from ico.assets import AssetMissing, Assets, ManifestError
from ico.assets.fetch import FetchError, Status, fetch, status, verify
from ico.assets.hub import HubError
from ico.assets.lockfile import lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ico")
    groups = parser.add_subparsers(dest="group", required=True)
    assets = groups.add_parser("assets", help="pinned files of a domain (assets.toml)")
    commands = assets.add_subparsers(dest="command", required=True)
    for name, help in [
        ("lock", "write sizes and sha256 to assets.lock (no large downloads)"),
        ("status", "what is present, missing or partial, and how much to fetch"),
        ("verify", "check sizes, or every sha256 with --full"),
        ("fetch", "download what is missing, verified"),
    ]:
        command = commands.add_parser(name, help=help)
        command.add_argument("ids", nargs="*", help="asset ids (all by default)")
        if name == "verify":
            command.add_argument("--full", action="store_true", help="hash every file")
        if name == "fetch":
            command.add_argument(
                "--dry-run", action="store_true", help="only show the plan"
            )
    args = parser.parse_args(argv)
    try:
        return _assets(args)
    except (AssetMissing, FetchError, FileNotFoundError, HubError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _assets(args: argparse.Namespace) -> int:
    store = Assets.from_env()
    ids: list[str] = args.ids
    if args.command == "lock":
        for entry in lock(store.manifest, ids).values():
            print(f"{entry.id}: {len(entry.files)} file(s), {_gb(entry.size)}")
        return 0
    if args.command == "status":
        report = status(store, ids)
        _print(report)
        return 0 if all(s.state == "present" for s in report) else 2
    if args.command == "verify":
        bad = verify(store, ids, full=args.full)
        for id, files in bad.items():
            print(f"{id}: bad {', '.join(files)}")
        print("ok" if not bad else f"{len(bad)} asset(s) failed")
        return 0 if not bad else 2
    started = time.monotonic()
    shown: dict[str, int] = {}

    def progress(id: str, done: int, total: int) -> None:
        percent = done * 100 // max(total, 1)
        if percent >= shown.get(id, -10) + 10:
            shown[id] = percent
            print(f"{id}: {_gb(done)} / {_gb(total)}", flush=True)

    before = status(store, ids)
    needed = sum(s.to_fetch for s in before)
    if args.dry_run or not needed:
        _print(before)
        return 0
    after = fetch(store, ids, progress=progress)
    seconds = max(time.monotonic() - started, 1e-3)
    _print(after)
    print(
        f"fetched {_gb(needed)} in {seconds:.0f} s ({needed / seconds / 1e6:.1f} MB/s)"
    )
    return 0


def _print(report: list[Status]) -> None:
    for s in report:
        extra = f", {_gb(s.to_fetch)} to fetch" if s.to_fetch else ""
        checked = ", verified" if s.verified else ""
        print(f"{s.id}: {s.state} ({_gb(s.size)}{extra}{checked})")
    total = sum(s.to_fetch for s in report)
    if total:
        print(f"to fetch: {_gb(total)}")


def _gb(size: int) -> str:
    return f"{size / 1e9:.2f} GB"


if __name__ == "__main__":
    sys.exit(main())
