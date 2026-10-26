"""The job server: jobs one after another in one process, which keeps its
resource pool; through the RunPod handler (without ``runpod``), and the
server's own protocol (``python -m ico.remote --serve``)."""

import asyncio
import contextlib
import json
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

import ico.remote.runpod_handler as runpod_handler
from ico.remote import Job, Package
from ico.remote.runpod_handler import handler

OPS = textwrap.dedent(
    '''
    import os
    import time
    from collections.abc import Iterator
    from dataclasses import dataclass

    from ico.registry import register
    from ico.resources import Resource, resource

    LOADS = [0]


    @resource("serve.model", 1)
    @dataclass(frozen=True)
    class Model(Resource[str]):
        """A model that counts its loads in this process."""

        def load(self) -> str:
            LOADS[0] += 1
            return "model"


    MODEL = Model()


    @register("serve.pid", 1)
    def pid(text: str) -> str:
        """The process id and the model's loads in it."""
        with MODEL.use():
            return f"{os.getpid()}:{LOADS[0]}"


    @register("serve.words", 1)
    def words(text: str) -> Iterator[str]:
        """Split a text into words."""
        return iter(text.split())


    @register("serve.slow", 1)
    def slow(word: str) -> str:
        """Pass a word on after a pause."""
        time.sleep(0.05)
        return word


    @register("serve.stuck", 1)
    def stuck(word: str) -> str:
        """Hang in one item."""
        time.sleep(60)
        return word


    @register("serve.fail", 1)
    def fail(text: str) -> str:
        """Fail."""
        raise ValueError("boom")


    @register("serve.crash", 1)
    def crash(text: str) -> str:
        """Kill the process."""
        os._exit(3)


    @register("serve.setenv", 1)
    def setenv(name: str) -> str:
        """Set an environment variable; its value before."""
        before = os.environ.get(name, "")
        os.environ[name] = "set"
        return before
    '''
)
SOURCES = {"serve_ops/__init__.py": "", "serve_ops/ops.py": OPS}
PACKAGE = Package("serve_ops", "draft")


def job(spec: dict[str, Any], input: str = "x", **kwargs: Any) -> Job:
    kwargs.setdefault("sources", SOURCES)
    return Job(spec=spec, input=input, packages=(PACKAGE,), **kwargs)


def op(ref: str) -> dict[str, Any]:
    return {"kind": "op", "op": ref}


def words_then(ref: str) -> dict[str, Any]:
    progress = {"kind": "progress", "name": "words"}
    inner = {"kind": "seq", "children": [progress, op(ref)]}
    return {
        "kind": "seq",
        "children": [op("serve.words@1"), {"kind": "stream", "children": [inner]}],
    }


def request(j: Job) -> dict[str, Any]:
    return {"id": j.id, "input": {"job": j.to_dict()}}


def run(j: Job) -> list[dict[str, Any]]:
    async def go() -> list[dict[str, Any]]:
        return [cast(dict[str, Any], e) async for e in handler(request(j))]

    return asyncio.run(go())


def pid_and_loads(events: list[dict[str, Any]]) -> tuple[int, int]:
    assert events[-1]["event"] == "result", events[-1]
    pid, loads = str(events[-1]["result"]["outputs"][0]).split(":")
    return int(pid), int(loads)


def run_cancelled(j: Job, after: int) -> list[dict[str, Any]]:
    """Run ``j`` and cancel it (as RunPod does) after ``after`` items."""
    events: list[dict[str, Any]] = []

    async def consume() -> None:
        async for event in handler(request(j)):
            events.append(cast(dict[str, Any], event))

    async def go() -> None:
        task = asyncio.create_task(consume())
        while not any(
            e["event"] == "progress" and e["snapshot"]["done"] >= after for e in events
        ):
            await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    return events


@pytest.fixture(autouse=True)
def workdirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("ICO_WORKDIRS", str(tmp_path / "workdirs"))
    yield tmp_path / "workdirs"
    runpod_handler.stop_server()


def test_jobs_with_the_same_code_share_the_process_and_its_pool() -> None:
    first = pid_and_loads(run(job(op("serve.pid@1"))))
    second = pid_and_loads(run(job(op("serve.pid@1"))))

    assert first[1] == 1
    assert second == first  # the same process, the model loaded once


def test_a_job_with_other_code_gets_a_new_server() -> None:
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    changed = {**SOURCES, "serve_ops/ops.py": OPS + "\n# edited\n"}
    second, loads = pid_and_loads(run(job(op("serve.pid@1"), sources=changed)))

    assert second != first and loads == 1


def test_a_run_error_retires_the_server() -> None:
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    failed = run(job(op("serve.fail@1")))
    second, _ = pid_and_loads(run(job(op("serve.pid@1"))))

    assert failed[-1]["code"] == "run_error"
    assert second != first


def test_other_errors_keep_the_server() -> None:
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    bad = run(job(op("serve.missing@1")))
    second = pid_and_loads(run(job(op("serve.pid@1"))))

    assert bad[-1]["code"] == "spec_error"
    assert second == (first, 1)


def test_a_cancelled_job_keeps_the_server() -> None:
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    words = " ".join(f"w{i}" for i in range(200))
    run_cancelled(job(words_then("serve.slow@1"), words), after=3)
    second = pid_and_loads(run(job(op("serve.pid@1"))))

    assert second == (first, 1)


def test_a_job_that_does_not_stop_is_killed_with_its_server(
    monkeypatch: pytest.MonkeyPatch, workdirs: Path
) -> None:
    monkeypatch.setattr(runpod_handler, "CANCEL_GRACE", 0.5)
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    stuck = job(words_then("serve.stuck@1"), "a b", cleanup=True)
    run_cancelled(stuck, after=0)
    second, loads = pid_and_loads(run(job(op("serve.pid@1"))))

    assert second != first and loads == 1
    assert not (workdirs / stuck.id).exists()


def test_a_crashed_server_is_an_error_and_the_next_job_runs() -> None:
    first, _ = pid_and_loads(run(job(op("serve.pid@1"))))
    crashed = run(job(op("serve.crash@1")))
    second, _ = pid_and_loads(run(job(op("serve.pid@1"))))

    assert crashed[-1]["event"] == "error" and crashed[-1]["code"] == "crashed"
    assert second != first


def test_max_jobs_retires_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runpod_handler.MAX_JOBS_ENV, "2")
    pids = [pid_and_loads(run(job(op("serve.pid@1"))))[0] for _ in range(3)]

    assert pids[0] == pids[1] != pids[2]


def test_jobs_keep_their_sources_in_their_work_directories(workdirs: Path) -> None:
    jobs = [job(op("serve.pid@1")), job(op("serve.pid@1"), cleanup=True)]
    for j in jobs:
        run(j)

    assert (workdirs / jobs[0].id / "code" / "serve_ops" / "ops.py").is_file()
    assert not (workdirs / jobs[1].id).exists()
    third = pid_and_loads(run(job(op("serve.pid@1"))))  # imports still work
    assert third[1] == 1


def test_worker_processes_import_the_code_after_a_cleanup() -> None:
    server, _ = pid_and_loads(run(job(op("serve.pid@1"), cleanup=True)))
    worker = {
        "kind": "worker",
        "params": {"type": "ico.runtime.worker.mp.mp_worker:MPWorker"},
        "children": [op("serve.pid@1")],
    }
    pid, loads = pid_and_loads(run(job(worker)))

    assert pid != server and loads == 1  # spawned, imports by name


# ─── The server's protocol ───


def serve(*jobs: Job, workdir: Path) -> list[list[dict[str, Any]]]:
    """Events of ``jobs`` sent to one ``python -m ico.remote --serve``."""
    lines = "".join(
        json.dumps({"job": j.to_dict(), "workdir": str(workdir / j.id)}) + "\n"
        for j in jobs
    )
    done = subprocess.run(
        [sys.executable, "-m", "ico.remote", "--serve"],
        input=lines,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    runs: list[list[dict[str, Any]]] = [[]]
    for line in done.stdout.splitlines():
        event = cast(dict[str, Any], json.loads(line))
        runs[-1].append(event)
        if event["event"] in ("result", "error"):
            runs.append([])
    return runs[:-1]


def test_the_server_refuses_other_code(tmp_path: Path) -> None:
    changed = {**SOURCES, "serve_ops/ops.py": OPS + "\n# edited\n"}
    first, second = serve(
        job(op("serve.pid@1")), job(op("serve.pid@1"), sources=changed),
        workdir=tmp_path,
    )  # fmt: skip

    assert first[-1]["event"] == "result"
    assert second[-1]["code"] == "code_changed"


def test_the_server_restores_the_environment(tmp_path: Path) -> None:
    runs = serve(
        job(op("serve.setenv@1"), "ICO_TEST_VAR"),
        job(op("serve.setenv@1"), "ICO_TEST_VAR"),
        workdir=tmp_path,
    )

    outputs = [r[-1]["result"]["outputs"] for r in runs]
    assert outputs == [[""], [""]]  # the first job's value is gone
