"""A job server: one process that runs jobs one after another, so the
resource pool, the imported operators and the CUDA context of the process
stay between them (``docs/proposals/2026-10-25-job-server.md``).

``python -m ico.remote --serve`` reads requests from stdin, one JSON line
each (``{"job": <job>, "workdir": "<path>"}``), and writes the events of
every job to stdout as JSON lines, ending with its ``result`` or
``error``, as a runner of one job does. It exits when stdin closes.

- The code is loaded once: the first run job writes its sources into the
  server's own code directory (not the job's work directory, which a
  ``cleanup`` job removes while the next jobs still import from it) and
  loads its packages. A later job with the same
  :meth:`~ico.remote.job.Job.code_fingerprint` reuses them; one with
  another fingerprint ends with the error ``code_changed`` (the caller
  starts a new server for it). Prefetch jobs carry no code.
- The environment is restored after every job (``ICO_RUN_ID``,
  ``ICO_ASSETS_MANIFEST``, and what operators set).
- SIGTERM cancels the job in progress (it stops between items and its
  result says so); between jobs it does nothing.

Whether to keep a server after a job is the caller's decision
(:mod:`ico.remote.runpod_handler`): a server whose job ended with
``run_error`` or ``load_error`` may hold a broken state (CUDA, half
imported modules) and should not get the next job.
"""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import IO, cast

from ico.core.runtime.cancel import IcoCancelToken
from ico.remote.job import AnyJob, FetchJob, JsonObject, load_job
from ico.remote.runner import Emit, fetch_job, run_job

RESTART = ("run_error", "load_error")
"""Error codes after which a server should not run another job."""


class JobServer:
    """Runs jobs in this process; loads their code once (see the module
    docs).

    Args:
        code: The server's code directory.
    """

    def __init__(self, code: Path) -> None:
        self.code = code
        self.fingerprint: str | None = None
        """Of the code this process loaded; ``None`` before the first run
        job."""

    def run(
        self, job: AnyJob, workdir: Path, emit: Emit, cancel: IcoCancelToken
    ) -> None:
        """Run one job; every outcome is an event."""
        if isinstance(job, FetchJob):
            fetch_job(job, emit, workdir, cancel)
            return
        fingerprint = job.code_fingerprint()
        if self.fingerprint is not None and fingerprint != self.fingerprint:
            emit(
                {
                    "event": "error",
                    "code": "code_changed",
                    "message": "this job server runs other code; "
                    "send the job to a new one",
                }
            )
            return
        load = self.fingerprint is None
        self.fingerprint = fingerprint
        environ = dict(os.environ)
        try:
            run_job(job, emit, workdir, cancel, code=self.code, load=load)
        finally:
            os.environ.clear()
            os.environ.update(environ)


def serve_jobs(requests: IO[str], out: IO[str], code: Path) -> None:
    """Run the jobs of ``requests`` until it closes, writing their events
    to ``out`` as JSON lines."""
    lock = threading.Lock()

    def emit(event: JsonObject) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with lock:
            out.write(line + "\n")
            out.flush()

    current: list[IcoCancelToken] = []

    def cancel(signum: int, frame: object) -> None:
        for token in current:
            token.cancel("cancelled")

    signal.signal(signal.SIGTERM, cancel)
    server = JobServer(code)
    while line := requests.readline():
        if not line.strip():
            continue
        try:
            request = cast(JsonObject, json.loads(line))
            job = load_job(cast(JsonObject, request["job"]))
            workdir = Path(str(request["workdir"]))
        except Exception as e:
            emit(
                {
                    "event": "error",
                    "code": "internal",
                    "message": f"not a job request: {type(e).__name__}: {e}",
                }
            )
            continue
        token = IcoCancelToken()
        current.append(token)
        try:
            server.run(job, workdir, emit, token)
        finally:
            current.remove(token)
