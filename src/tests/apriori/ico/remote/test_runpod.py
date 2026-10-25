"""RunPod jobs against a fake RunPod API on 127.0.0.1 that runs the real
worker handler in process (the ``runpod`` package itself is not needed)."""

import asyncio
import itertools
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

import ico.remote.runpod as runpod_module
import ico.remote.runpod_handler as runpod_handler
from ico.remote import Job, RemoteError, RunPod, result, submit
from ico.remote.job import JsonObject
from ico.remote.runpod_handler import handler
from tests.apriori.ico.remote.test_jobs import (
    PACKAGE,
    SOURCES,
    op,
    wait_until_gone,
    words_then,
)

KEY = "test-key"


@dataclass
class FakeJob:
    body: dict[str, Any]
    status: str = "IN_QUEUE"
    chunks: list[JsonObject] = field(default_factory=lambda: [])
    read: int = 0
    loop: asyncio.AbstractEventLoop | None = None
    task: "asyncio.Task[None] | None" = None


class FakeRunPod:
    """``/run``, ``/stream``, ``/status`` and ``/cancel`` of one endpoint;
    each job runs ``handler`` on a thread of its own."""

    def __init__(self) -> None:
        self.jobs: dict[str, FakeJob] = {}
        self.gpu = threading.Event()
        """Cleared: jobs wait in the queue (no worker free)."""
        self.gpu.set()
        self.throttle = 0
        self.drop = 0
        """Stream reads to answer by closing the connection."""
        self.ids = itertools.count(1)
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                fake.serve(self)

            def do_POST(self) -> None:
                fake.serve(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def serve(self, request: BaseHTTPRequestHandler) -> None:
        if request.headers["authorization"] != f"Bearer {KEY}":
            return self.reply(request, 401, {"error": "unauthorized"})
        _, endpoint, action, *rest = request.path.split("/")
        assert endpoint == "ep"
        if action == "run":
            if self.throttle:
                self.throttle -= 1
                return self.reply(request, 429, {"error": "throttled"})
            length = int(request.headers["content-length"])
            body = json.loads(request.rfile.read(length))
            id = f"rp-{next(self.ids)}"
            job = self.jobs[id] = FakeJob(body)
            threading.Thread(target=self.work, args=(id, job), daemon=True).start()
            return self.reply(request, 200, {"id": id, "status": job.status})
        job = self.jobs[rest[0]]
        if action == "stream" and self.drop:
            self.drop -= 1
            request.close_connection = True
            return
        if action == "stream":
            new = job.chunks[job.read :]
            job.read += len(new)
            stream = [{"output": chunk} for chunk in new]
            return self.reply(request, 200, {"status": job.status, "stream": stream})
        if action == "status":
            return self.reply(request, 200, {"id": rest[0], "status": job.status})
        assert action == "cancel"
        job.status = "CANCELLED"
        if job.loop is not None and job.task is not None:
            job.loop.call_soon_threadsafe(job.task.cancel)
        self.reply(request, 200, {"id": rest[0], "status": job.status})

    def reply(self, request: BaseHTTPRequestHandler, code: int, body: object) -> None:
        data = json.dumps(body).encode()
        request.send_response(code)
        request.send_header("content-type", "application/json")
        request.send_header("content-length", str(len(data)))
        request.end_headers()
        request.wfile.write(data)

    def work(self, id: str, job: FakeJob) -> None:
        self.gpu.wait()

        async def go() -> None:
            async for chunk in handler({"id": id, "input": job.body["input"]}):
                job.chunks.append(chunk)

        async def main() -> None:
            job.loop = asyncio.get_running_loop()
            job.task = asyncio.current_task()
            job.status = "IN_PROGRESS"
            try:
                await go()
                job.status = "COMPLETED"
            except asyncio.CancelledError:
                pass
            except Exception:
                job.status = "FAILED"

        asyncio.run(main())


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeRunPod]:
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("ICO_WORKDIRS", str(tmp_path / "workdirs"))
    server = FakeRunPod()
    yield server
    server.gpu.set()
    server.server.shutdown()


def executor(fake: FakeRunPod, **kwargs: Any) -> RunPod:
    kwargs.setdefault("poll", 0.05)
    return RunPod("ep", base_url=fake.url, **kwargs)


def test_a_job_runs_on_the_endpoint(fake: FakeRunPod, tmp_path: Path) -> None:
    handle = submit(
        words_then(op("jobs.shout@1")), "a b c", executor=executor(fake),
        packages=[PACKAGE], sources=SOURCES,
    )  # fmt: skip
    events = [cast(dict[str, Any], e) for e in handle.events()]
    events = [e for e in events if e["event"] != "queue"]  # RunPod's status

    assert events[0]["event"] == "plan" and events[-1]["event"] == "result"
    assert events[-1]["result"]["outputs"] == ["A", "B", "C"]
    body = fake.jobs[handle.id].body
    assert body["input"]["job"]["spec"] == words_then(op("jobs.shout@1"))
    assert (tmp_path / "workdirs" / body["input"]["job"]["id"] / "code").is_dir()


def test_the_policy_gives_long_jobs_time(fake: FakeRunPod) -> None:
    handle = executor(fake, execution_timeout=7200).submit(
        _job(op("jobs.shout@1"), "x")
    )
    result(handle)

    assert fake.jobs[handle.id].body["policy"] == {
        "executionTimeout": 7200 * 1000,
        "ttl": (7200 + 86400) * 1000,
    }


def test_job_errors_come_as_events(fake: FakeRunPod) -> None:
    handle = submit(
        op("jobs.fail@1"), "x", executor=executor(fake),
        packages=[PACKAGE], sources=SOURCES,
    )  # fmt: skip
    with pytest.raises(RemoteError) as raised:
        result(handle)
    assert raised.value.code == "run_error"


def test_a_cancelled_job_ends_with_an_error(fake: FakeRunPod) -> None:
    words = " ".join(f"w{i}" for i in range(200))
    handle = submit(
        words_then(op("jobs.slow@1")), words, executor=executor(fake),
        packages=[PACKAGE], sources=SOURCES,
    )  # fmt: skip
    final: dict[str, Any] = {}
    for raw in handle.events():
        event = cast(dict[str, Any], raw)
        if event["event"] == "progress" and event["snapshot"]["done"] >= 3:
            handle.cancel()
        final = event

    assert final["event"] == "error" and final["code"] == "cancelled"
    assert final["status"]["status"] == "CANCELLED"


def test_throttled_submits_back_off(
    fake: FakeRunPod, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        real_sleep(0.01)

    monkeypatch.setattr(runpod_module, "time", _Clock(sleep))
    fake.throttle = 2
    handle = executor(fake, poll=0.2).submit(_job(op("jobs.shout@1"), "x"))

    assert result(handle)["outputs"] == ["X"]
    assert 1 <= sleeps[0] < 2 and 2 <= sleeps[1] < 3


def test_dropped_stream_reads_are_retried(
    fake: FakeRunPod, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sleep(seconds: float) -> None:
        time.sleep(0.01)

    monkeypatch.setattr(runpod_module, "time", _Clock(sleep))
    fake.drop = 2
    handle = executor(fake).submit(_job(op("jobs.shout@1"), "x"))

    assert result(handle)["outputs"] == ["X"] and fake.drop == 0


def test_queue_events_tell_a_job_waits_for_a_worker(fake: FakeRunPod) -> None:
    fake.gpu.clear()
    threading.Timer(0.5, fake.gpu.set).start()  # a worker frees up
    events = executor(fake).submit(_job(op("jobs.shout@1"), "x")).events()
    first = next(events)
    rest = list(events)

    assert first == {"event": "queue", "status": "IN_QUEUE"}
    queue = [e["status"] for e in rest if e["event"] == "queue"]
    assert queue == ["IN_PROGRESS"]  # once per change
    assert rest[-1]["event"] == "result"


def test_a_wrong_key_is_an_http_error(fake: FakeRunPod) -> None:
    with pytest.raises(RuntimeError, match="HTTP 401"):
        RunPod("ep", api_key="wrong", base_url=fake.url).submit(
            _job(op("jobs.shout@1"), "x")
        )


def _job(spec: dict[str, Any], input: str) -> Job:
    return Job(spec=spec, input=input, packages=(PACKAGE,), sources=dict(SOURCES))


@dataclass
class _Clock:
    sleep: Any


def test_work_directories_go_to_the_network_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ICO_WORKDIRS", raising=False)
    monkeypatch.setattr(runpod_handler, "VOLUME", tmp_path / "missing")
    assert runpod_handler.workdirs() == Path("/tmp/ico-jobs")
    monkeypatch.setattr(runpod_handler, "VOLUME", tmp_path)
    assert runpod_handler.workdirs() == tmp_path / "ico-jobs"
    monkeypatch.setenv("ICO_WORKDIRS", "/data/jobs")
    assert runpod_handler.workdirs() == Path("/data/jobs")


def test_the_handler_cleans_up_after_a_runner_it_killed(
    fake: FakeRunPod, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runpod_handler, "CANCEL_GRACE", 0.0)  # kill at once
    words = " ".join(f"w{i}" for i in range(200))
    handle = submit(
        words_then(op("jobs.slow@1")), words, executor=executor(fake),
        packages=[PACKAGE], sources=SOURCES, id="killed", cleanup=True,
    )  # fmt: skip
    for raw in handle.events():
        event = cast(dict[str, Any], raw)
        if event["event"] == "progress" and event["snapshot"]["done"] >= 3:
            handle.cancel()

    assert wait_until_gone(tmp_path / "workdirs" / "killed")
