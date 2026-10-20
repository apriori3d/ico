"""The runner of a job: anywhere, without knowing who sent it.

``run_job(job, emit, workdir)`` runs one job in a work directory and passes
its events to ``emit``:

1. writes the job's ``sources`` under ``<workdir>/code`` and loads its
   operator packages (from there, else from the working directory);
2. downloads the storage URIs of the input and params into
   ``<workdir>/in`` and puts the local paths in their place; ``$JOB`` in a
   string is the work directory;
3. sends the plan, runs the spec (progress, recent samples, profile and
   cancellation as asked), and sends the result;
4. uploads ``<workdir>/out`` to the job's ``out`` and puts URIs in place
   of the output paths under it.

``python -m ico.remote.runner --workdir DIR`` reads a job (JSON) from stdin
and writes its events as JSON lines to stdout (operators' prints go to
stderr); SIGTERM cancels the run between items.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import IO, cast

from ico.core.runtime.cancel import IcoCancelToken
from ico.describe.plan.json_renderer import plan_view
from ico.registry.loader import load_operators
from ico.registry.registry import Status
from ico.remote.job import Job, JsonObject
from ico.remote.storage import basename, join, storage_for
from ico.spec.builders import build
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError
from ico.spec.run import DEFAULT_MAX_ITEMS, run
from ico.tools.profiler import ProfilerTool, ProfileSnapshot
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot
from ico.tools.sample import JsonSampleTool, SampleSnapshot

Emit = Callable[[JsonObject], None]
JOB_VAR = "$JOB"
TRACEBACK_LIMIT = 8000


def run_job(
    job: Job, emit: Emit, workdir: Path, cancel: IcoCancelToken | None = None
) -> None:
    """Run ``job`` in ``workdir``; every outcome is an event (never raises
    for the job's own errors)."""
    try:
        _run(job, emit, workdir, cancel)
    except SpecError as e:
        emit(
            {
                "event": "error",
                "code": "spec_error",
                "message": str(e),
                "reason": e.reason,
                "path": str(e.path) if e.path is not None else None,
            }
        )
    except _LoadError as e:
        emit(
            {
                "event": "error",
                "code": "load_error",
                "message": str(e),
                "errors": e.errors,
            }
        )
    except Exception as e:
        emit(
            {
                "event": "error",
                "code": "run_error",
                "message": f"{type(e).__name__}: {e}",
                "traceback": "".join(traceback.format_exception(e))[-TRACEBACK_LIMIT:],
            }
        )


class _LoadError(Exception):
    def __init__(self, message: str, errors: list[JsonValue]) -> None:
        super().__init__(message)
        self.errors = errors


def _run(job: Job, emit: Emit, workdir: Path, cancel: IcoCancelToken | None) -> None:
    workdir = workdir.resolve()
    (workdir / "out").mkdir(parents=True, exist_ok=True)
    _load(job, workdir)
    staged: dict[str, str] = {}
    input = _stage_in(job.input, workdir, staged)
    params = cast(JsonObject | None, _stage_in(job.params, workdir, staged))
    spec = _parse(job.spec)

    with contextlib.suppress(Exception):  # a spec that does not build fails below
        emit(
            {
                "event": "plan",
                "plan": cast(JsonValue, plan_view(build(spec, params=params))),
            }
        )

    def progress(snapshot: ProgressSnapshot) -> None:
        emit({"event": "progress", "snapshot": cast(JsonValue, dict(snapshot))})

    def sample(snapshot: SampleSnapshot) -> None:
        emit({"event": "sample", "snapshot": cast(JsonValue, dict(snapshot))})

    def profile(snapshot: ProfileSnapshot) -> None:
        emit({"event": "profile", "snapshot": cast(JsonValue, dict(snapshot))})

    profiler = ProfilerTool(profile) if job.profile else None
    result = run(
        spec,
        input,
        max_items=job.max_items or DEFAULT_MAX_ITEMS,
        tools=[JsonProgressTool(progress), JsonSampleTool(sample)],
        params=params,
        profiler=profiler,
        cancel=cancel,
        recent=job.recent,
    )
    out = result.to_dict()
    if profiler is not None:
        out["profile"] = cast(JsonValue, dict(profiler.snapshot()))
    if job.out is not None:
        out["outputs"] = _stage_out(out["outputs"], workdir / "out", job.out)
    emit({"event": "result", "result": out})


def _parse(spec: JsonObject) -> PipelineSpec | NodeSpec:
    if "root" in spec:
        return PipelineSpec.from_dict(spec)
    return NodeSpec.from_dict(spec)


def _load(job: Job, workdir: Path) -> None:
    """Write the sources and load the operator packages."""
    code = workdir / "code"
    for relative, text in job.sources.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise _LoadError(f"invalid source path {relative!r}", [relative])
        target = code.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    errors: list[JsonValue] = []
    for package in job.packages:
        carried = code / package.path
        directory = carried if carried.is_dir() else Path(package.path)
        if carried.is_dir():
            (carried / "__init__.py").touch()
        report = load_operators(
            directory.resolve(),
            status=cast(Status | None, package.status),
            modules=package.modules,
        )
        errors.extend({"file": f, "error": message} for f, message in report.errors)
    if errors:
        raise _LoadError("operator modules failed to import", errors)


def _stage_in(value: JsonValue, workdir: Path, staged: dict[str, str]) -> JsonValue:
    """``value`` with ``$JOB`` expanded and storage URIs downloaded."""
    if isinstance(value, str):
        text = value.replace(JOB_VAR, str(workdir))
        storage = storage_for(text)
        if storage is None:
            return text
        if text not in staged:
            local = _free(workdir / "in" / basename(text))
            storage.get(text, local)
            staged[text] = str(local)
        return staged[text]
    if isinstance(value, list):
        return [_stage_in(v, workdir, staged) for v in value]
    if isinstance(value, dict):
        return {k: _stage_in(v, workdir, staged) for k, v in value.items()}
    return value


def _free(path: Path) -> Path:
    """``path``, or a numbered sibling if it is taken."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _stage_out(value: JsonValue, out_dir: Path, uri: str) -> JsonValue:
    """Upload ``out_dir`` to ``uri``; output paths under it become URIs."""
    storage = storage_for(uri)
    if storage is None:
        raise ValueError(f"not a storage URI: {uri!r}")
    for child in sorted(out_dir.iterdir()):
        storage.put(child, join(uri, child.name))
    prefix = str(out_dir)

    def replace(item: JsonValue) -> JsonValue:
        if isinstance(item, str) and (
            item == prefix or item.startswith(prefix + os.sep)
        ):
            relative = os.path.relpath(item, prefix).replace(os.sep, "/")
            return join(uri, relative) if relative != "." else uri
        if isinstance(item, list):
            return [replace(v) for v in item]
        if isinstance(item, dict):
            return {k: replace(v) for k, v in item.items()}
        return item

    return replace(value)


# ─── Process ───


def serve(job: Job, workdir: Path, out: IO[str]) -> None:
    """Run ``job``, writing its events to ``out`` as JSON lines."""
    lock = threading.Lock()

    def emit(event: JsonObject) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with lock:
            out.write(line + "\n")
            out.flush()

    token = IcoCancelToken()
    signal.signal(signal.SIGTERM, lambda signum, frame: token.cancel("cancelled"))
    run_job(job, emit, workdir, token)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m ico.remote.runner")
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args(argv)
    # The events own the original stdout; operators' output goes to stderr.
    events = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    job = Job.from_dict(cast(JsonObject, json.loads(sys.stdin.read())))
    serve(job, cast(Path, args.workdir), events)


if __name__ == "__main__":
    main()
