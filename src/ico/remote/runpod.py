"""Jobs on RunPod Serverless: the executor (any machine) and the worker's
handler (the image).

The executor speaks the queue API with the standard library: ``POST
/run`` with ``{"input": {"job": <job>}, "policy": {...}}``, the job's
events from ``GET /stream/<id>`` (each chunk's ``output`` is one event),
``POST /cancel/<id>``. The key comes from ``RUNPOD_API_KEY``.

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
from dataclasses import dataclass
from typing import cast

from ico.remote.job import FINAL, AnyJob, JsonObject
from ico.spec.model import JsonValue

API = "https://api.runpod.ai/v2"
DONE = ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT")
"""Final statuses of a RunPod job."""
RETRIES = 5


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
    """

    endpoint: str
    api_key: str | None = None
    execution_timeout: float = 3 * 3600
    ttl: float | None = None
    poll: float = 1.0
    base_url: str = API

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

    def call(
        self, method: str, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        key = self.api_key or os.environ.get("RUNPOD_API_KEY", "")
        url = f"{self.base_url.rstrip('/')}/{self.endpoint}{path}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(RETRIES):
            request = urllib.request.Request(url, data=data, method=method)
            request.add_header("authorization", f"Bearer {key}")
            request.add_header("content-type", "application/json")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return cast(JsonObject, json.loads(response.read() or b"{}"))
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == RETRIES - 1:
                    raise RuntimeError(
                        f"RunPod {method} {path}: HTTP {e.code}"
                    ) from None
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        raise AssertionError("unreachable")


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
        while True:
            answer = self.executor.call("GET", f"/stream/{self._id}")
            chunks = _chunks(answer)
            for chunk in chunks:
                event = cast(JsonObject, chunk)
                yield event
                if event.get("event") in FINAL:
                    return
            status = str(answer.get("status", ""))
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
