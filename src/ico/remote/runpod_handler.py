"""The RunPod worker: ``python -m ico.remote.runpod_handler`` in the image.

Each RunPod job carries an ICO job (``{"input": {"job": ...}}``); the
handler runs it in a job server (``python -m ico.remote --serve``, see
:mod:`ico.remote.server`) and yields its events, which ``/stream``
delivers to the executor. The server lives between jobs, so a warm worker
keeps the models it loaded; the handler starts a new one for a job with
other code (:meth:`~ico.remote.job.Job.code_fingerprint`), after a job
that ended with ``run_error`` or ``load_error``, after
``ICO_SERVE_MAX_JOBS`` jobs (unlimited by default), and after a server
that crashed or was killed. Jobs run one at a time.

When RunPod cancels a job, the server gets SIGTERM: the job stops between
items, and the server waits for the next one. A job that does not stop
within ``CANCEL_GRACE`` seconds is killed with its server. The runner
removes the work directory of a job with ``cleanup``; the handler does it
for a job it killed.

Environment: ``ICO_WORKDIRS`` (where work directories go; by default on
the endpoint's network volume when one is mounted, which makes jobs
resumable: cancel one to stop billing, send it again with the same id to
continue), ``ICO_CODE`` (the server's working directory: the domain's
packages in the image), ``ICO_ASSETS_ROOT`` (assets; by default
``assets`` on the network volume, filled by prefetch jobs).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO, Any, cast

from ico.remote.job import FINAL, Job, JsonObject
from ico.remote.server import RESTART

CANCEL_GRACE = 10.0
SERVER_EXIT = 30.0
"""Seconds a server that gets no more jobs has to exit before it is
killed."""
MAX_JOBS_ENV = "ICO_SERVE_MAX_JOBS"
VOLUME = Path("/runpod-volume")
"""Where RunPod mounts an endpoint's network volume."""
POLL = 0.02


async def handler(request: JsonObject) -> AsyncIterator[JsonObject]:
    """Run the ICO job of a RunPod job; yield its events."""
    job = cast(JsonObject, cast(JsonObject, request["input"])["job"])
    root = workdirs()
    workdir = root / str(job.get("id") or request.get("id"))
    workdir.mkdir(parents=True, exist_ok=True)
    async with _one_at_a_time():
        server = await _server_for(_fingerprint(job))
        server.send({"job": job, "workdir": str(workdir)})
        final: JsonObject | None = None
        try:
            while (event := await server.event()) is not None:
                if event.get("event") in FINAL:
                    final = event
                yield event
                if final is not None:
                    break
            if final is None:
                yield {
                    "event": "error",
                    "code": "crashed",
                    "message": f"the job server exited (code "
                    f"{server.process.wait()}) without a result",
                }
        except asyncio.CancelledError:
            final = await _cancel(server)
            if final is None and cast(JsonObject, job.get("options") or {}).get(
                "cleanup"
            ):
                shutil.rmtree(workdir, ignore_errors=True)
            raise
        finally:
            _after_job(server, final)


def workdirs() -> Path:
    """``ICO_WORKDIRS``, else the network volume, else ``/tmp``."""
    configured = os.environ.get("ICO_WORKDIRS")
    if configured:
        return Path(configured)
    return VOLUME / "ico-jobs" if VOLUME.is_dir() else Path("/tmp/ico-jobs")


def stop_server() -> None:
    """Stop the job server, if one runs (tests; the worker's exit closes
    its stdin, which stops it too)."""
    global _server
    server, _server = _server, None
    if server is not None:
        server.retire()
        server.wait()


def _asset_env() -> dict[str, str]:
    """``ICO_ASSETS_ROOT`` on the network volume, unless it is set."""
    if os.environ.get("ICO_ASSETS_ROOT") or not VOLUME.is_dir():
        return {}
    return {"ICO_ASSETS_ROOT": str(VOLUME / "assets")}


def _fingerprint(job: JsonObject) -> str | None:
    """The code fingerprint of a run job; ``None`` for a prefetch, which
    any server runs."""
    if job.get("kind") == "fetch":
        return None
    return Job.from_dict(job).code_fingerprint()


def _max_jobs() -> int | None:
    value = os.environ.get(MAX_JOBS_ENV)
    return int(value) if value else None


# ─── The server ───


class _Server:
    """A job server process; its output lines are read on a thread, so any
    event loop can wait for them."""

    def __init__(self, fingerprint: str | None) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", "ico.remote", "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=os.environ.get("ICO_CODE"),
            env={**os.environ, **_asset_env()},
        )
        self.fingerprint = fingerprint
        self.jobs = 0
        self.retired = False
        self._lines: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        stdout = cast(IO[bytes], self.process.stdout)
        for line in iter(stdout.readline, b""):
            if line.strip():
                self._lines.put(line)
        self._lines.put(None)

    def takes(self, fingerprint: str | None) -> bool:
        """Whether the next job, of ``fingerprint``, may run here."""
        if self.retired or self.process.poll() is not None:
            return False
        return fingerprint is None or self.fingerprint in (None, fingerprint)

    def send(self, request: JsonObject) -> None:
        stdin = cast(IO[bytes], self.process.stdin)
        stdin.write((json.dumps(request) + "\n").encode())
        stdin.flush()
        self.jobs += 1

    async def event(self, timeout: float | None = None) -> JsonObject | None:
        """The next event; ``None`` when the server's output ended.

        Raises:
            TimeoutError: No event within ``timeout`` seconds.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError from None
                await asyncio.sleep(POLL)
                continue
            return None if line is None else cast(JsonObject, json.loads(line))

    def retire(self) -> None:
        """No more jobs: close its stdin, and an idle server exits."""
        self.retired = True
        with contextlib.suppress(OSError):
            cast(IO[bytes], self.process.stdin).close()

    def kill(self) -> None:
        self.retired = True
        with contextlib.suppress(ProcessLookupError):
            self.process.kill()

    def wait(self) -> None:
        """Wait for the process to exit; kill it after ``SERVER_EXIT``."""
        try:
            self.process.wait(SERVER_EXIT)
        except subprocess.TimeoutExpired:
            self.kill()
            self.process.wait()


_server: _Server | None = None
_busy = threading.Lock()


@contextlib.asynccontextmanager
async def _one_at_a_time() -> AsyncIterator[None]:
    """Jobs run one at a time, whatever RunPod's concurrency (each job may
    wait on an event loop of its own)."""
    while not _busy.acquire(blocking=False):
        await asyncio.sleep(POLL)
    try:
        yield
    finally:
        _busy.release()


async def _server_for(fingerprint: str | None) -> _Server:
    """The running server if it takes the job, else a new one; an old
    server exits first, so that its models leave the GPU before the new
    one loads its own."""
    global _server
    server = _server
    if server is not None and server.takes(fingerprint):
        if server.fingerprint is None:
            server.fingerprint = fingerprint
        return server
    if server is not None:
        server.retire()
        await asyncio.to_thread(server.wait)
    _server = _Server(fingerprint)
    return _server


async def _cancel(server: _Server) -> JsonObject | None:
    """Cancel the job in progress; its final event, or ``None`` if the
    server had to be killed (or exited)."""
    with contextlib.suppress(ProcessLookupError):
        server.process.terminate()  # the job stops between items
    deadline = time.monotonic() + CANCEL_GRACE
    try:
        while (
            event := await server.event(max(0.0, deadline - time.monotonic()))
        ) is not None:
            if event.get("event") in FINAL:
                return event
    except TimeoutError:
        server.kill()
        await asyncio.to_thread(server.process.wait)
    return None


def _after_job(server: _Server, final: JsonObject | None) -> None:
    """Keep the server for the next job, or retire it."""
    if final is None:
        server.kill()  # the job did not end: the server is not idle
        return
    max_jobs = _max_jobs()
    if (final.get("event") == "error" and final.get("code") in RESTART) or (
        max_jobs is not None and server.jobs >= max_jobs
    ):
        server.retire()


def main() -> None:
    runpod: Any = importlib.import_module("runpod")  # installed in the image
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
