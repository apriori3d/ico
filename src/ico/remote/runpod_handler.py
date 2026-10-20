"""The RunPod worker: ``python -m ico.remote.runpod_handler`` in the image.

Each RunPod job carries an ICO job (``{"input": {"job": ...}}``); the
handler runs it in a runner subprocess (``python -m ico.remote.runner``)
and yields its events, which ``/stream`` delivers to the executor. When
RunPod cancels the job, the runner gets SIGTERM (it stops between items)
and is killed after ``CANCEL_GRACE`` seconds.

Environment: ``ICO_WORKDIRS`` (where work directories go; by default on
the endpoint's network volume when one is mounted, which makes jobs
resumable: cancel one to stop billing, send it again with the same id to
continue), ``ICO_CODE`` (the runner's working directory: the domain's
packages in the image).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from ico.remote.job import JsonObject

CANCEL_GRACE = 10.0
VOLUME = Path("/runpod-volume")
"""Where RunPod mounts an endpoint's network volume."""
LINE_LIMIT = 64 * 1024 * 1024


async def handler(request: JsonObject) -> AsyncIterator[JsonObject]:
    """Run the ICO job of a RunPod job; yield its events."""
    job = cast(JsonObject, cast(JsonObject, request["input"])["job"])
    root = workdirs()
    workdir = root / str(job.get("id") or request.get("id"))
    workdir.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "ico.remote.runner",
        "--workdir",
        str(workdir),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        cwd=os.environ.get("ICO_CODE"),
        limit=LINE_LIMIT,
    )
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(job).encode())
    await process.stdin.drain()
    process.stdin.close()
    try:
        async for line in process.stdout:
            if line.strip():
                yield cast(JsonObject, json.loads(line))
        await process.wait()
    except asyncio.CancelledError:
        await _stop(process)
        raise


def workdirs() -> Path:
    """``ICO_WORKDIRS``, else the network volume, else ``/tmp``."""
    configured = os.environ.get("ICO_WORKDIRS")
    if configured:
        return Path(configured)
    return VOLUME / "ico-jobs" if VOLUME.is_dir() else Path("/tmp/ico-jobs")


async def _stop(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        await asyncio.wait_for(process.wait(), CANCEL_GRACE)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()


def main() -> None:
    runpod: Any = importlib.import_module("runpod")  # installed in the image
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
