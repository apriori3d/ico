"""Executors send jobs somewhere and follow their events.

``LocalExecutor`` runs each job in a subprocess of this machine (``python
-m ico.remote.runner``): for tests, and for running on a GPU machine at
hand. Cloud executors (RunPod) follow the same protocol.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import IO, Protocol, cast

from ico.remote.job import (
    FINAL,
    AnyJob,
    Job,
    JsonObject,
    Package,
    RemoteError,
    spec_json,
)
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec


class JobHandle(Protocol):
    """A job sent to an executor."""

    @property
    def id(self) -> str: ...

    def events(self) -> Iterator[JsonObject]:
        """The job's events, until ``result`` or ``error`` (included)."""
        ...

    def cancel(self) -> None:
        """Ask the job to stop between items; its result comes as usual."""
        ...


class Executor(Protocol):
    """Where jobs run."""

    def submit(self, job: AnyJob) -> JobHandle: ...


def result(handle: JobHandle) -> JsonObject:
    """Follow a job to its end and return its result.

    Raises:
        RemoteError: The job's ``error`` event.
    """
    for event in handle.events():
        if event.get("event") == "result":
            return cast(JsonObject, event["result"])
        if event.get("event") == "error":
            raise RemoteError(str(event.get("code")), str(event.get("message")), event)
    raise RemoteError("internal", "the job ended without a result")


def submit(
    spec: PipelineSpec | NodeSpec | JsonObject,
    input: JsonValue = None,
    *,
    executor: Executor,
    params: JsonObject | None = None,
    packages: Sequence[Package] = (),
    sources: dict[str, str] | None = None,
    out: str | None = None,
    profile: bool = False,
    recent: bool = False,
    max_items: int | None = None,
    id: str | None = None,
) -> JobHandle:
    """Send one run of ``spec`` to ``executor``; see :class:`Job`. The same
    ``id`` again resumes the job from its work directory."""
    job = Job(
        spec=spec_json(spec),
        input=input,
        params=params,
        packages=tuple(packages),
        sources=dict(sources or {}),
        out=out,
        profile=profile,
        recent=recent,
        max_items=max_items,
    )
    if id is not None:
        job = replace(job, id=id)
    return executor.submit(job)


class LocalExecutor:
    """Runs jobs in subprocesses of this machine.

    Args:
        workdirs: Where the work directories (``<workdirs>/<job id>``) go;
            a temporary directory by default. A job sent again with the same
            id finds its files there.
        python: The interpreter (its environment has the operators'
            dependencies).
        cwd: The runner's working directory: where packages that the job
            does not carry are found.
    """

    def __init__(
        self,
        workdirs: Path | None = None,
        *,
        python: str = sys.executable,
        cwd: Path | None = None,
    ) -> None:
        self.workdirs = workdirs or Path(tempfile.mkdtemp(prefix="ico-jobs-"))
        self.python = python
        self.cwd = cwd

    def submit(self, job: AnyJob) -> LocalJob:
        workdir = self.workdirs / job.id
        workdir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [self.python, "-m", "ico.remote.runner", "--workdir", str(workdir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=self.cwd,
        )
        stdin = cast(IO[str], process.stdin)
        stdin.write(json.dumps(job.to_dict()))
        stdin.close()
        return LocalJob(job.id, process, workdir)


class LocalJob:
    """A job running in a subprocess."""

    def __init__(self, id: str, process: subprocess.Popen[str], workdir: Path) -> None:
        self._id = id
        self.process = process
        self.workdir = workdir

    @property
    def id(self) -> str:
        return self._id

    def events(self) -> Iterator[JsonObject]:
        stdout = cast(IO[str], self.process.stdout)
        ended = False
        for line in stdout:
            if not line.strip():
                continue
            event = cast(JsonObject, json.loads(line))
            yield event
            if event.get("event") in FINAL:
                ended = True
                break
        code = self.process.wait()
        if not ended:
            yield {
                "event": "error",
                "code": "crashed",
                "message": f"the runner exited (code {code}) without a result",
            }

    def cancel(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.process.pid, signal.SIGTERM)
