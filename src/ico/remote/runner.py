"""The runner of a job: anywhere, without knowing who sent it.

``run_job(job, emit, workdir)`` runs one job in a work directory and passes
its events to ``emit``:

1. writes the job's ``sources`` under ``<workdir>/code`` and loads its
   operator packages (from there, else from the working directory);
2. downloads the storage URIs of the input and params into
   ``<workdir>/in`` and puts the local paths in their place; ``$JOB`` in a
   string is the work directory and ``$RUN`` the job's id (also in
   ``ICO_RUN_ID``, which the run's worker processes inherit);
3. sends the plan, runs the spec (progress, recent samples, profile and
   cancellation as asked), and sends the result;
4. uploads ``<workdir>/out`` to the job's ``out`` and puts URIs in place
   of the output paths under it;
5. writes the final event (``result`` or ``error``) to the job's
   ``status`` URI as well, when it has one, so the outcome is in the
   storage even if nobody read the stream.

A job sent again with the same id into the same work directory resumes:
its operators find the files they left, and every URI comes back to the
local path it was downloaded to (``in/.staged.json``), without downloading
it again. One runner at a time holds a work directory (``.lock``); another
one ends with the error ``busy``.

A prefetch (:class:`~ico.remote.job.FetchJob`, :func:`fetch_job`) writes
its manifest and lock into ``<workdir>/assets`` and downloads the assets
into ``ICO_ASSETS_ROOT`` (the worker's volume), with ``fetch`` events per
percent of each asset and a result with the state of every asset. A run
job that carries ``assets.toml`` in its sources resolves ``asset(id)``
against it.

``python -m ico.remote --workdir DIR`` (:mod:`ico.remote.__main__`) reads a job (JSON) from stdin
and writes its events as JSON lines to stdout (operators' prints go to
stderr); SIGTERM cancels the run between items.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import IO, cast

from ico.core.runtime.cancel import IcoCancelToken
from ico.describe.plan.json_renderer import plan_view
from ico.registry.loader import load_operators
from ico.registry.registry import Status
from ico.remote.job import FINAL, AnyJob, FetchJob, Job, JsonObject
from ico.remote.storage import basename, join, mount_bucket, storage_for
from ico.spec.builders import build
from ico.spec.model import JsonValue, NodeSpec, PipelineSpec, SpecError
from ico.spec.run import DEFAULT_MAX_ITEMS, run
from ico.tools.profiler import ProfilerTool, ProfileSnapshot
from ico.tools.progress.json_progress_tool import JsonProgressTool, ProgressSnapshot
from ico.tools.sample import JsonSampleTool, SampleSnapshot

Emit = Callable[[JsonObject], None]
JOB_VAR = "$JOB"
RUN_VAR = "$RUN"
RUN_ENV = "ICO_RUN_ID"
"""The job's id, for operators and the worker processes they start."""
VOLUME_ROOT = "/runpod-volume"
VOLUME_ENV = "ICO_VOLUME_ROOT"
"""Where the worker's network volume is mounted (tests set another)."""
TRACEBACK_LIMIT = 8000


def run_job(
    job: Job,
    emit: Emit,
    workdir: Path,
    cancel: IcoCancelToken | None = None,
    *,
    code: Path | None = None,
    load: bool = True,
) -> None:
    """Run ``job`` in ``workdir``; every outcome is an event (never raises
    for the job's own errors).

    Args:
        code: Where the job's sources are written and loaded from;
            ``<workdir>/code`` by default. A job server keeps one for the
            jobs it runs (:mod:`ico.remote.server`).
        load: Write the sources and load the packages; ``False`` when this
            process already loaded the same code.
    """
    if job.status is not None:
        emit = _recording(emit, job.status)
    try:
        _run(job, emit, workdir, cancel, code, load)
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
    except _BusyError as e:
        emit({"event": "error", "code": "busy", "message": str(e)})
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


def _recording(emit: Emit, uri: str) -> Emit:
    """``emit`` that also writes the final event to ``uri``. A ``busy``
    error is not written: another runner of the same job owns the outcome.
    A failed write is reported on stderr and does not fail the job (the
    stream still carries the event)."""

    def recorded(event: JsonObject) -> None:
        if event.get("event") in FINAL and event.get("code") != "busy":
            try:
                write_status(event, uri)
            except Exception as e:  # botocore raises its own types
                print(f"ico: writing {uri}: {type(e).__name__}: {e}", file=sys.stderr)
        emit(event)

    return recorded


def write_status(event: JsonObject, uri: str) -> None:
    """Write a job's final event to the storage URI ``uri`` (JSON)."""
    storage = storage_for(uri)
    if storage is None:
        raise ValueError(f"not a storage URI: {uri!r}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "status.json"
        path.write_text(json.dumps(event, ensure_ascii=False, default=str))
        storage.put(path, uri)


class _BusyError(Exception):
    pass


class _LoadError(Exception):
    def __init__(self, message: str, errors: list[JsonValue]) -> None:
        super().__init__(message)
        self.errors = errors


def _run(
    job: Job,
    emit: Emit,
    workdir: Path,
    cancel: IcoCancelToken | None,
    code: Path | None,
    load: bool,
) -> None:
    workdir = workdir.resolve()
    (workdir / "out").mkdir(parents=True, exist_ok=True)
    with _locked(workdir):
        try:
            _run_locked(job, emit, workdir, cancel, code or workdir / "code", load)
        finally:
            if job.cleanup:
                remove_workdir(workdir)


def remove_workdir(workdir: Path) -> None:
    """Remove a job's work directory (what is left of it, too)."""
    shutil.rmtree(workdir, ignore_errors=True)


@contextlib.contextmanager
def _locked(workdir: Path) -> Iterator[None]:
    """Hold the work directory; released when the process dies, too."""
    with (workdir / ".lock").open("w") as lock:
        if sys.platform != "win32":
            import fcntl

            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise _BusyError(
                    f"another runner works in {workdir} (the same job id)"
                ) from None
        yield


def _run_locked(
    job: Job,
    emit: Emit,
    workdir: Path,
    cancel: IcoCancelToken | None,
    code: Path,
    load: bool,
) -> None:
    os.environ[RUN_ENV] = job.id
    if code != workdir / "code":
        _write_sources(job, workdir / "code")  # what the job ran, for people
    if load:
        _load(job, code)
    volume_root = Path(os.environ.get(VOLUME_ENV) or VOLUME_ROOT)
    if job.volume is not None and volume_root.is_dir():
        mount_bucket(job.volume, volume_root)
    manifest = code / "assets.toml"
    if manifest.is_file():
        os.environ["ICO_ASSETS_MANIFEST"] = str(manifest)
        missing = _missing_assets(_parse(job.spec), job.params)
        if missing:
            ids = ", ".join(str(cast(JsonObject, m)["id"]) for m in missing)
            size = sum(
                int(cast(int, cast(JsonObject, m).get("size") or 0)) for m in missing
            )
            emit(
                {
                    "event": "error",
                    "code": "missing_assets",
                    "message": f"assets not here: {ids} ({size / 1e9:.2f} GB); "
                    "prefetch them first",
                    "assets": missing,
                }
            )
            return
    staged = _Staged(workdir / "in" / ".staged.json")
    input = _stage_in(job.input, workdir, job.id, staged)
    params = cast(JsonObject | None, _stage_in(job.params, workdir, job.id, staged))
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


def _missing_assets(
    spec: PipelineSpec | NodeSpec, params: JsonObject | None
) -> list[JsonValue]:
    """The assets the spec needs that are not usable under the asset root
    (``{"id", "reason", "size"}``), checked before anything is staged or
    run; a spec whose assets cannot be told fails later, in the run."""
    from ico.assets import AssetMissing, Assets
    from ico.spec.assets import spec_assets

    try:
        needed = spec_assets(spec, params)
    except SpecError:
        return []
    if not needed:
        return []
    store = Assets.from_env()
    missing: list[JsonValue] = []
    for id in needed:
        try:
            store.path(id)
        except AssetMissing as e:
            missing.append({"id": id, "reason": e.reason, "size": e.size})
    return missing


def _parse(spec: JsonObject) -> PipelineSpec | NodeSpec:
    if "root" in spec:
        return PipelineSpec.from_dict(spec)
    return NodeSpec.from_dict(spec)


def _load(job: Job, code: Path) -> None:
    """Write the sources under ``code`` and load the operator packages."""
    _write_sources(job, code)
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


def _write_sources(job: Job, code: Path) -> None:
    for relative, text in job.sources.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise _LoadError(f"invalid source path {relative!r}", [relative])
        target = code.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


class _Staged:
    """The local paths of downloaded URIs, kept in the work directory so
    that a resumed job finds them."""

    def __init__(self, manifest: Path) -> None:
        self.manifest = manifest
        self.paths: dict[str, str] = {}
        if manifest.is_file():
            self.paths = cast(dict[str, str], json.loads(manifest.read_text()))

    def get(self, uri: str) -> str | None:
        path = self.paths.get(uri)
        return path if path is not None and Path(path).exists() else None

    def add(self, uri: str, path: str) -> None:
        self.paths[uri] = path
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_text(json.dumps(self.paths, indent=1))


def _stage_in(
    value: JsonValue, workdir: Path, run_id: str, staged: _Staged
) -> JsonValue:
    """``value`` with ``$JOB`` and ``$RUN`` expanded and storage URIs
    downloaded."""
    if isinstance(value, str):
        text = value.replace(JOB_VAR, str(workdir)).replace(RUN_VAR, run_id)
        storage = storage_for(text)
        if storage is None:
            return text
        local = staged.get(text)
        if local is None:
            target = _free(workdir / "in" / basename(text))
            storage.get(text, target)
            local = str(target)
            staged.add(text, local)
        return local
    if isinstance(value, list):
        return [_stage_in(v, workdir, run_id, staged) for v in value]
    if isinstance(value, dict):
        return {k: _stage_in(v, workdir, run_id, staged) for k, v in value.items()}
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


# ─── Prefetch ───


class _FetchCancelled(Exception):
    pass


def fetch_job(
    job: FetchJob, emit: Emit, workdir: Path, cancel: IcoCancelToken | None = None
) -> None:
    """Download the assets of a prefetch; every outcome is an event."""
    from ico.assets import AssetMissing, Assets, ManifestError, load_manifest
    from ico.assets.fetch import FetchError, fetch
    from ico.assets.hub import HubError
    from ico.assets.store import DEFAULT_ROOT, ROOT_ENV

    try:
        directory = workdir.resolve() / "assets"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "assets.toml").write_text(job.manifest, encoding="utf-8")
        (directory / "assets.lock").write_text(job.lock, encoding="utf-8")
        root = Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT).expanduser()
        store = Assets(load_manifest(directory / "assets.toml"), root)
        ids = list(job.ids) or None
        shown: dict[str, int] = {}

        def progress(id: str, done: int, total: int) -> None:
            if cancel is not None and cancel.cancelled:
                raise _FetchCancelled
            percent = done * 100 // max(total, 1)
            if percent > shown.get(id, -1):
                shown[id] = percent
                emit({"event": "fetch", "asset": id, "done": done, "total": total})

        with _locked(workdir.resolve()):
            after = fetch(store, ids, progress=progress)
        assets: list[JsonValue] = [
            {"id": s.id, "state": s.state, "size": s.size, "verified": s.verified}
            for s in after
        ]
        emit({"event": "result", "result": {"root": str(root), "assets": assets}})
    except _FetchCancelled:
        emit({"event": "error", "code": "cancelled", "message": "the fetch was cancelled; "
              "a later fetch continues the files"})  # fmt: skip
    except _BusyError as e:
        emit({"event": "error", "code": "busy", "message": str(e)})
    except (AssetMissing, FetchError, HubError, ManifestError) as e:
        emit({"event": "error", "code": "fetch_error", "message": str(e)})
    except Exception as e:
        emit(
            {
                "event": "error",
                "code": "run_error",
                "message": f"{type(e).__name__}: {e}",
                "traceback": "".join(traceback.format_exception(e))[-TRACEBACK_LIMIT:],
            }
        )


# ─── Process ───


def serve(job: AnyJob, workdir: Path, out: IO[str]) -> None:
    """Run ``job``, writing its events to ``out`` as JSON lines."""
    lock = threading.Lock()

    def emit(event: JsonObject) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with lock:
            out.write(line + "\n")
            out.flush()

    token = IcoCancelToken()
    signal.signal(signal.SIGTERM, lambda signum, frame: token.cancel("cancelled"))
    if isinstance(job, FetchJob):
        fetch_job(job, emit, workdir, token)
    else:
        run_job(job, emit, workdir, token)
