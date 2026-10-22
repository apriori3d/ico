"""``ico``: the command line of ICO (``ico assets ...``)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import cast

from ico.assets import AssetMissing, Assets, ManifestError
from ico.assets.fetch import FetchError, Status, fetch, status, verify
from ico.assets.hub import HubError
from ico.assets.lockfile import lock
from ico.remote.job import FetchJob, RemoteError
from ico.remote.runpod import API, RunPod
from ico.remote.runpod_api import RunPodApi, RunPodApiError


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
    runpod = groups.add_parser("runpod", help="RunPod: network volumes, prefetch")
    actions = runpod.add_subparsers(dest="command", required=True)
    volume = actions.add_parser("volume", help="network volumes of an order")
    volume_actions = volume.add_subparsers(dest="action", required=True)
    up = volume_actions.add_parser(
        "up", help="create a volume and attach it to the endpoint"
    )
    up.add_argument("--endpoint", required=True)
    up.add_argument("--size", type=int, required=True, help="GB")
    up.add_argument("--datacenter", required=True, help="e.g. EU-RO-1")
    up.add_argument("--name", default="ico-assets")
    down = volume_actions.add_parser("down", help="detach the volume and delete it")
    down.add_argument("--endpoint", required=True)
    down.add_argument("--volume", required=True)
    volume_actions.add_parser("list", help="the volumes of the account")
    prefetch = actions.add_parser(
        "prefetch", help="fetch assets onto the endpoint's volume"
    )
    prefetch.add_argument("--endpoint", required=True)
    prefetch.add_argument("--manifest", type=Path, default=Path("assets.toml"))
    prefetch.add_argument("ids", nargs="*", help="asset ids (all by default)")
    args = parser.parse_args(argv)
    try:
        return _assets(args) if args.group == "assets" else _runpod(args)
    except (AssetMissing, FetchError, FileNotFoundError, HubError, ManifestError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (RunPodApiError, RemoteError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _runpod(args: argparse.Namespace) -> int:
    api = RunPodApi.from_env()
    if args.command == "volume":
        if args.action == "list":
            for v in api.volumes():
                print(f"{v.id}  {v.name}  {v.size} GB  {v.datacenter}")
            return 0
        if args.action == "up":
            v = api.volume_up(args.endpoint, args.size, args.datacenter, args.name)
            print(
                f"volume {v.id} ({v.size} GB, {v.datacenter}) attached to {args.endpoint}"
            )
            print(f"storage: {v.uri()}  S3 API: {v.s3_endpoint}")
            return 0
        api.volume_down(args.endpoint, args.volume)
        print(f"volume {args.volume} detached from {args.endpoint} and deleted")
        return 0
    executor = RunPod(args.endpoint, base_url=os.environ.get("ICO_RUNPOD_API") or API)
    handle = executor.submit(FetchJob.from_files(args.manifest, args.ids))
    shown: dict[str, int] = {}
    for event in handle.events():
        kind = event.get("event")
        if kind == "fetch":
            id, done, total = (
                str(event["asset"]),
                int(cast(int, event["done"])),
                int(cast(int, event["total"])),
            )
            percent = done * 100 // max(total, 1)
            if percent >= shown.get(id, -10) + 10:
                shown[id] = percent
                print(f"{id}: {_gb(done)} / {_gb(total)}", flush=True)
        elif kind == "queue":
            print(f"runpod: {event.get('status')}", flush=True)  # e.g. IN_QUEUE
        elif kind == "result":
            result = cast(dict[str, object], event["result"])
            for item in cast(list[dict[str, object]], result["assets"]):
                print(f"{item['id']}: {item['state']}")
            return 0
        elif kind == "error":
            print(
                f"error: {event.get('code')}: {event.get('message')}", file=sys.stderr
            )
            return 1
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
