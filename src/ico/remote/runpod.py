"""Jobs on RunPod Serverless: the executor (any machine) and the worker's
handler (the image).

The executor speaks the queue API with the standard library: ``POST
/run`` with ``{"input": {"job": <job>}, "policy": {...}}``, the job's
events from ``GET /stream/<id>`` (each chunk's ``output`` is one event),
``GET /status/<id>``, ``POST /cancel/<id>``. The key comes from
``RUNPOD_API_KEY``. A dropped connection or a server error on a read is
retried for ``retry`` seconds (``ICO_REMOTE_RETRY``, 10 minutes by
default), so a short outage on the client does not end a long job; a job
whose client went away is found again by its RunPod id
(:meth:`RunPod.job`).

The handler (``python -m ico.remote.runpod_handler`` in the image, with
the ``runpod`` package installed there) runs the job in a runner
subprocess and yields its events as the stream.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

from ico.remote.job import FINAL, AnyJob, JsonObject
from ico.remote.runpod_api import USER_AGENT
from ico.spec.model import JsonValue

API = "https://api.runpod.ai/v2"
DONE = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")
"""Final statuses of a RunPod job."""
RETRIES = 5
"""Attempts of a POST that RunPod throttled (429)."""
RETRY_ENV = "ICO_REMOTE_RETRY"
RETRY = 600.0
"""Seconds a read keeps being retried, unless ``ICO_REMOTE_RETRY`` says."""


def _retry_window() -> float:
    return float(os.environ.get(RETRY_ENV) or RETRY)


@dataclass(frozen=True)
class RunPod:
    """An executor for one RunPod Serverless endpoint.

    Args:
        endpoint: The endpoint id.
        api_key: The API key; ``RUNPOD_API_KEY`` by default.
        execution_timeout: Seconds a job may run on a worker (RunPod's
            default of 10 minutes is too short for videos).
        ttl: Seconds a job lives in total, queue included; a day more than
            ``execution_timeout`` by default.
        poll: Seconds between stream reads while no event comes.
        base_url: The API root (a fake one in tests).
        retry: Seconds a read (``GET``) or a cancel keeps being retried
            on a dropped connection, a DNS failure or an HTTP 429/5xx.
    """

    endpoint: str
    api_key: str | None = None
    execution_timeout: float = 3 * 3600
    ttl: float | None = None
    poll: float = 1.0
    base_url: str = API
    retry: float = field(default_factory=_retry_window)

    def submit(self, job: AnyJob) -> RunPodJob:
        ttl = self.ttl if self.ttl is not None else self.execution_timeout + 86400
        body: JsonObject = {
            "input": {"job": job.to_dict()},
            "policy": {
                "executionTimeout": int(self.execution_timeout * 1000),
                "ttl": int(ttl * 1000),
            },
        }
        answer = self.call("POST", "/run", body)
        return RunPodJob(self, str(answer["id"]), job.id)

    def job(self, id: str, job_id: str = "") -> RunPodJob:
        """The handle of a job submitted earlier, by RunPod's ``id``: its
        events go on from where RunPod's stream is (``job_id`` is the ICO
        job's id, when known)."""
        return RunPodJob(self, id, job_id)

    def call(
        self, method: str, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        key = self.api_key or os.environ.get("RUNPOD_API_KEY", "")
        url = f"{self.base_url.rstrip('/')}/{self.endpoint}{path}"
        data = json.dumps(body).encode() if body is not None else None
        # A read or a cancel changes nothing on RunPod, so it is retried
        # for the whole window; a POST /run is retried only when RunPod
        # throttled it (it did not take the job).
        idempotent = method == "GET" or path.startswith("/cancel/")
        deadline = time.monotonic() + self.retry
        attempt = 0
        while True:
            request = urllib.request.Request(url, data=data, method=method)
            request.add_header("authorization", f"Bearer {key}")
            request.add_header("content-type", "application/json")
            request.add_header("user-agent", USER_AGENT)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return cast(JsonObject, json.loads(response.read() or b"{}"))
            except urllib.error.HTTPError as e:
                transient = e.code == 429 or (idempotent and e.code >= 500)
                if not transient or not self._again(idempotent, attempt, deadline):
                    raise RuntimeError(
                        f"RunPod {method} {path}: HTTP {e.code}"
                    ) from None
            except (urllib.error.URLError, OSError) as e:
                # A dropped connection or a DNS failure (a flaky network)
                # must not end a long job; a POST /run is not retried, it
                # may have reached RunPod.
                if not idempotent or not self._again(True, attempt, deadline):
                    raise RuntimeError(
                        f"RunPod {method} {path}: {type(e).__name__}: {e}"
                    ) from e
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
            attempt += 1

    @staticmethod
    def _again(idempotent: bool, attempt: int, deadline: float) -> bool:
        """Whether a failed call is tried once more."""
        if idempotent:
            return time.monotonic() < deadline
        return attempt < RETRIES - 1


class RunPodJob:
    """A job on a RunPod endpoint (``id`` is RunPod's)."""

    def __init__(self, executor: RunPod, id: str, job_id: str) -> None:
        self.executor = executor
        self._id = id
        self.job_id = job_id

    @property
    def id(self) -> str:
        return self._id

    def events(self) -> Iterator[JsonObject]:
        """The job's events; a ``queue`` event when RunPod's status of the
        job changes (``IN_QUEUE``: waiting for a worker)."""
        last = None
        while True:
            answer = self.executor.call("GET", f"/stream/{self._id}")
            status = str(answer.get("status", ""))
            if status and status != last and status not in DONE:
                last = status
                yield {"event": "queue", "status": status}
            chunks = _chunks(answer)
            for chunk in chunks:
                event = cast(JsonObject, chunk)
                yield event
                if event.get("event") in FINAL:
                    return
            if status in DONE and not chunks:
                # Ended without our final event: failed, cancelled, timed out.
                details = self.executor.call("GET", f"/status/{self._id}")
                yield {
                    "event": "error",
                    "code": status.lower(),
                    "message": f"the RunPod job ended {status}",
                    "status": cast(JsonValue, details),
                }
                return
            if not chunks:
                time.sleep(self.executor.poll)

    def status(self) -> JsonObject:
        """RunPod's view of the job: ``status`` (``IN_QUEUE``,
        ``IN_PROGRESS``, ``COMPLETED``, ...) and its details. RunPod keeps
        a finished job's status for a limited time only."""
        return self.executor.call("GET", f"/status/{self._id}")

    def cancel(self) -> None:
        """Cancel the RunPod job. Unlike a local job, RunPod stops the
        handler, so the events end with an ``error`` "cancelled" and the
        outputs so far are lost (resumable jobs are the next step)."""
        self.executor.call("POST", f"/cancel/{self._id}")


def _chunks(answer: JsonObject | list[JsonValue]) -> list[JsonValue]:
    """The events of a ``/stream`` answer (``{"stream": [{"output": e}]}``,
    or a bare list of chunks)."""
    raw = answer if isinstance(answer, list) else answer.get("stream") or []
    events: list[JsonValue] = []
    for chunk in cast(list[JsonValue], raw):
        if isinstance(chunk, dict) and "output" in chunk:
            events.append(chunk["output"])
    return events
