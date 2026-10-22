"""Jobs end to end: a runner subprocess, operators by value or on disk,
file:// staging, errors, cancellation, recent samples and profiles."""

import fcntl
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

from ico.remote import Job, LocalExecutor, Package, RemoteError, result, submit

OPS = textwrap.dedent(
    '''
    import time
    from collections.abc import Iterator
    from pathlib import Path

    from ico import IcoOperator
    from ico.registry import register


    @register("jobs.words", 1)
    def words(text: str) -> Iterator[str]:
        """Split a text into words."""
        return iter(text.split())


    @register("jobs.shout", 1)
    def shout(word: str) -> str:
        """Upper-case a word (and print, which must not break the events)."""
        print("noise on stdout")
        return word.upper()


    @register("jobs.slow", 1)
    def slow(word: str) -> str:
        """Pass a word on after a pause."""
        time.sleep(0.05)
        return word


    @register("jobs.fail", 1)
    def fail(word: str) -> str:
        """Fail."""
        raise ValueError("boom")


    @register("jobs.once", 1)
    class Once(IcoOperator[str, str]):
        """Process a word once per work directory (a mark file per word)."""

        def __init__(self, work_dir: str) -> None:
            self.work_dir = work_dir
            super().__init__(fn=self._apply)

        def _apply(self, word: str) -> str:
            mark = Path(self.work_dir) / word
            if mark.exists():
                return "cached:" + word
            time.sleep(0.05)
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.touch()
            return word


    @register("jobs.upper_file", 1)
    class UpperFile(IcoOperator[str, str]):
        """Write an upper-cased copy of a text file into out_dir."""

        def __init__(self, out_dir: str) -> None:
            self.out_dir = out_dir
            super().__init__(fn=self._apply)

        def _apply(self, path: str) -> str:
            target = Path(self.out_dir) / Path(path).name
            target.write_text(Path(path).read_text().upper())
            return str(target)
    '''
)
SOURCES = {"jobs_ops/__init__.py": "", "jobs_ops/ops.py": OPS}
PACKAGE = Package("jobs_ops", "draft")


def op(ref: str, **params: Any) -> dict[str, Any]:
    node: dict[str, Any] = {"kind": "op", "op": ref}
    if params:
        node["params"] = params
    return node


def words_then(*body: dict[str, Any]) -> dict[str, Any]:
    progress = {"kind": "progress", "name": "words"}
    inner = {"kind": "seq", "children": [progress, *body]}
    return {
        "kind": "seq",
        "children": [op("jobs.words@1"), {"kind": "stream", "children": [inner]}],
    }


def run_events(executor: LocalExecutor, **kwargs: Any) -> list[dict[str, Any]]:
    handle = submit(executor=executor, packages=[PACKAGE], sources=SOURCES, **kwargs)
    return [cast(dict[str, Any], event) for event in handle.events()]


def test_a_job_runs_with_its_events(tmp_path: Path) -> None:
    events = run_events(
        LocalExecutor(tmp_path), spec=words_then(op("jobs.shout@1")), input="a b c"
    )

    kinds = [e["event"] for e in events]
    assert kinds[0] == "plan" and kinds[-1] == "result"
    assert "progress" in kinds
    assert events[-1]["result"]["outputs"] == ["A", "B", "C"]
    last = [e["snapshot"] for e in events if e["event"] == "progress"][-1]
    assert (last["done"], last["total"]) == (3, 3)


def test_files_go_by_reference(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("hello")
    spec = op("jobs.upper_file@1", out_dir={"$param": "out_dir"})
    handle = submit(
        spec,
        f"file://{tmp_path}/data/a.txt",
        executor=LocalExecutor(tmp_path / "jobs"),
        params={"out_dir": "$JOB/out"},
        packages=[PACKAGE],
        sources=SOURCES,
        out=f"file://{tmp_path}/results",
    )
    outputs = result(handle)["outputs"]

    assert outputs == [f"file://{tmp_path}/results/a.txt"]
    assert (tmp_path / "results" / "a.txt").read_text() == "HELLO"
    assert (tmp_path / "jobs" / handle.id / "in" / "a.txt").read_text() == "hello"


def test_packages_from_the_working_directory(tmp_path: Path) -> None:
    for relative, text in SOURCES.items():
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(text)
    handle = submit(
        op("jobs.shout@1"),
        "hi",
        executor=LocalExecutor(tmp_path / "jobs", cwd=tmp_path),
        packages=[PACKAGE],
    )
    assert result(handle)["outputs"] == ["HI"]


def test_errors_are_events(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    spec_error = run_events(executor, spec=op("jobs.nope@1"), input="x")[-1]
    run_error = run_events(executor, spec=op("jobs.fail@1"), input="x")[-1]
    broken = submit(
        op("jobs.shout@1"), "x", executor=executor, packages=[PACKAGE],
        sources={**SOURCES, "jobs_ops/bad.py": "this is not python"},
    )  # fmt: skip

    assert spec_error["code"] == "spec_error"
    assert (
        run_error["code"] == "run_error"
        and "ValueError: boom" in run_error["traceback"]
    )
    try:
        result(broken)
    except RemoteError as e:
        assert e.code == "load_error" and "bad.py" in str(e.data["errors"])
    else:
        raise AssertionError("expected a load error")


def test_a_cancelled_job_returns_what_it_produced(tmp_path: Path) -> None:
    words = " ".join(f"w{i}" for i in range(200))
    handle = submit(
        words_then(op("jobs.slow@1")),
        words,
        executor=LocalExecutor(tmp_path),
        packages=[PACKAGE],
        sources=SOURCES,
    )
    final: dict[str, Any] = {}
    for raw in handle.events():
        event = cast(dict[str, Any], raw)
        if event["event"] == "progress" and event["snapshot"]["done"] >= 3:
            handle.cancel()
        final = event

    assert final["event"] == "result"
    assert final["result"]["cancelled"] == "cancelled"
    assert 3 <= len(final["result"]["outputs"]) < 200


def test_a_job_sent_again_resumes(tmp_path: Path) -> None:
    executor = LocalExecutor(tmp_path)
    spec = words_then(op("jobs.once@1", work_dir={"$param": "marks"}))
    words = " ".join(f"w{i}" for i in range(40))
    first = submit(
        spec, words, executor=executor, packages=[PACKAGE], sources=SOURCES,
        params={"marks": "$JOB/marks"}, id="video-1",
    )  # fmt: skip
    for raw in first.events():
        event = cast(dict[str, Any], raw)
        if event["event"] == "progress" and event["snapshot"]["done"] >= 3:
            first.cancel()
    again = submit(
        spec, words, executor=executor, packages=[PACKAGE], sources=SOURCES,
        params={"marks": "$JOB/marks"}, id="video-1",
    )  # fmt: skip
    outputs = cast(list[str], result(again)["outputs"])

    cached = [o for o in outputs if o.startswith("cached:")]
    assert 3 <= len(cached) < 40 and len(outputs) == 40
    assert outputs[: len(cached)] == [f"cached:w{i}" for i in range(len(cached))]


def test_a_resumed_job_keeps_its_inputs(tmp_path: Path) -> None:
    source = tmp_path / "a.txt"
    source.write_text("hello")
    executor = LocalExecutor(tmp_path / "jobs")

    def send() -> list[Any]:
        handle = submit(
            op("jobs.upper_file@1", out_dir={"$param": "out_dir"}),
            f"file://{source}", params={"out_dir": "$JOB/out"},
            executor=executor, packages=[PACKAGE], sources=SOURCES, id="j",
        )  # fmt: skip
        return cast(list[Any], result(handle)["outputs"])

    first = send()
    source.write_text("changed")
    assert send() == first
    staged = tmp_path / "jobs" / "j" / "in"
    assert (staged / "a.txt").read_text() == "hello"
    assert not (staged / "a-1.txt").exists()


def test_one_runner_per_work_directory(tmp_path: Path) -> None:
    (tmp_path / "j").mkdir()
    with (tmp_path / "j" / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = run_events(
            LocalExecutor(tmp_path), spec=op("jobs.shout@1"), input="x", id="j"
        )
    assert events[-1]["code"] == "busy"


def test_recent_samples_and_profile(tmp_path: Path) -> None:
    events = run_events(
        LocalExecutor(tmp_path), spec=words_then(op("jobs.slow@1")), input="a b c d",
        recent=True, profile=True,
    )  # fmt: skip

    samples = [e["snapshot"] for e in events if e["event"] == "sample"]
    assert samples and all(s["recent"] for s in samples)
    assert any(e["event"] == "profile" for e in events)
    assert "nodes" in events[-1]["result"]["profile"]


def test_a_job_round_trips_through_json() -> None:
    job = Job(
        spec=op("jobs.shout@1"), input="x", params={"p": 1}, packages=(PACKAGE,),
        sources={"a/b.py": ""}, out="s3://b/o", profile=True, recent=True, max_items=5,
    )  # fmt: skip
    assert Job.from_dict(job.to_dict()) == job


def test_the_runner_starts_without_warnings(tmp_path: Path) -> None:
    job = Job(spec=op("jobs.shout@1"), input="x", packages=(PACKAGE,), sources=SOURCES)
    done = subprocess.run(
        [sys.executable, "-W", "error", "-m", "ico.remote", "--workdir", str(tmp_path)],
        input=json.dumps(job.to_dict()),
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Warning" not in done.stderr
    assert json.loads(done.stdout.splitlines()[-1])["result"]["outputs"] == ["X"]
