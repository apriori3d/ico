"""A job: one run of a spec, described as JSON, that a runner executes
anywhere (a subprocess, a cloud GPU).

A job carries what the run needs and nothing of the machine that sent it:
the spec (without ``pipeline_ref`` nodes: inline them first), the input
and the parameter values, the operator packages to load and, optionally,
their sources (code by value), the run options and where to put the
outputs. See ``docs/design/remote.md``.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

from ico.spec.model import JsonValue, NodeSpec, PipelineSpec

JsonObject = dict[str, JsonValue]

EventKind = Literal["plan", "progress", "profile", "sample", "result", "error"]
"""Events of a job, in order: ``plan`` first, then any number of
``progress`` / ``profile`` / ``sample``, then one ``result`` or ``error``."""

FINAL: tuple[EventKind, ...] = ("result", "error")


@dataclass(frozen=True, slots=True)
class Package:
    """An operator package to load: a directory with ``__init__.py``.

    Attributes:
        path: The package directory, relative to the job's code directory
            when the job carries its sources, else to the runner's working
            directory (the package is in the image).
        status: The registry status of its entries (``stable``, ``draft``),
            or ``None`` for the modules' own.
        modules: The modules to import (``None``: all of them).
    """

    path: str
    status: str | None = None
    modules: tuple[str, ...] | None = None

    def to_dict(self) -> JsonObject:
        out: JsonObject = {"path": self.path}
        if self.status is not None:
            out["status"] = self.status
        if self.modules is not None:
            out["modules"] = list[JsonValue](self.modules)
        return out

    @classmethod
    def from_dict(cls, data: JsonObject) -> Package:
        modules = data.get("modules")
        status = data.get("status")
        return cls(
            path=str(data["path"]),
            status=status if isinstance(status, str) else None,
            modules=tuple(str(m) for m in cast(list[JsonValue], modules))
            if isinstance(modules, list)
            else None,
        )


@dataclass(frozen=True, slots=True)
class Job:
    """One run of a spec.

    Attributes:
        spec: A spec document or node, as JSON, without ``pipeline_ref``.
        input: The JSON input. Strings that are storage URIs (``file://``,
            ``s3://``) are downloaded by the runner and replaced by local
            paths; ``$JOB`` in a string is the job's work directory.
        params: Values of the pipeline parameters (same rules as ``input``).
        packages: The operator packages to load.
        sources: Files of operator packages by their relative path
            (``"video_draft/scan.py": "..."``): code by value.
        out: Where outputs go (a storage URI prefix): files under
            ``$JOB/out`` are uploaded there and output paths become URIs.
            ``None`` leaves them on the runner's machine.
        profile: Profile the run (``profile`` events, a profile in the
            result).
        recent: Preview recent results of every leaf operator (``sample``
            events).
        max_items: At most this many output items in the result.
        id: The job id; a job sent again with the same id and a persistent
            work directory resumes from the files it left.
    """

    spec: JsonObject
    input: JsonValue = None
    params: JsonObject | None = None
    packages: tuple[Package, ...] = ()
    sources: dict[str, str] = field(default_factory=dict[str, str])
    out: str | None = None
    profile: bool = False
    recent: bool = False
    max_items: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> JsonObject:
        out: JsonObject = {
            "id": self.id,
            "spec": self.spec,
            "input": self.input,
            "packages": [p.to_dict() for p in self.packages],
        }
        if self.params is not None:
            out["params"] = self.params
        if self.sources:
            out["sources"] = dict[str, JsonValue](self.sources)
        if self.out is not None:
            out["out"] = self.out
        options: JsonObject = {"profile": self.profile, "recent": self.recent}
        if self.max_items is not None:
            options["max_items"] = self.max_items
        out["options"] = options
        return out

    @classmethod
    def from_dict(cls, data: JsonObject) -> Job:
        options = cast(JsonObject, data.get("options") or {})
        params = data.get("params")
        sources = cast(dict[str, JsonValue], data.get("sources") or {})
        max_items = options.get("max_items")
        out = data.get("out")
        return cls(
            spec=cast(JsonObject, data["spec"]),
            input=data.get("input"),
            params=cast(JsonObject, params) if isinstance(params, dict) else None,
            packages=tuple(
                Package.from_dict(cast(JsonObject, p))
                for p in cast(list[JsonValue], data.get("packages") or [])
            ),
            sources={str(k): str(v) for k, v in sources.items()},
            out=out if isinstance(out, str) else None,
            profile=bool(options.get("profile", False)),
            recent=bool(options.get("recent", False)),
            max_items=max_items if isinstance(max_items, int) else None,
            id=str(data.get("id") or uuid.uuid4().hex),
        )


@dataclass(frozen=True)
class FetchJob:
    """A prefetch: download assets into the worker's asset root (a network
    volume), as ``ico assets fetch`` does on any machine. The manifest and
    its lock travel in the job; runs never fetch.

    Attributes:
        manifest: The text of ``assets.toml``.
        lock: The text of ``assets.lock``.
        ids: The assets to fetch; all of the manifest when empty.
        id: The job id (its work directory).
    """

    manifest: str
    lock: str
    ids: tuple[str, ...] = ()
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_files(cls, manifest: Path, ids: Sequence[str] = ()) -> FetchJob:
        """A prefetch of ``manifest`` (``assets.toml``) and the
        ``assets.lock`` beside it."""
        lock = manifest.with_name("assets.lock")
        return cls(
            manifest=manifest.read_text(encoding="utf-8"),
            lock=lock.read_text(encoding="utf-8"),
            ids=tuple(ids),
        )

    def to_dict(self) -> JsonObject:
        ids: list[JsonValue] = []
        ids.extend(self.ids)
        return {
            "kind": "fetch",
            "id": self.id,
            "manifest": self.manifest,
            "lock": self.lock,
            "ids": ids,
        }

    @classmethod
    def from_dict(cls, data: JsonObject) -> FetchJob:
        return cls(
            manifest=str(data["manifest"]),
            lock=str(data["lock"]),
            ids=tuple(str(i) for i in cast(list[JsonValue], data.get("ids") or [])),
            id=str(data.get("id") or uuid.uuid4().hex),
        )


AnyJob = Job | FetchJob
"""What executors send: a run of a spec, or a prefetch of assets."""


def load_job(data: JsonObject) -> AnyJob:
    """A job from its JSON (``kind`` ``fetch``, else a run)."""
    return (
        FetchJob.from_dict(data) if data.get("kind") == "fetch" else Job.from_dict(data)
    )


def spec_json(spec: PipelineSpec | NodeSpec | JsonObject) -> JsonObject:
    """A spec as the JSON a job carries."""
    if isinstance(spec, PipelineSpec | NodeSpec):
        return spec.to_dict()
    return spec


class RemoteError(Exception):
    """A job ended with an ``error`` event.

    Attributes:
        code: ``spec_error``, ``run_error``, ``internal``, or a transport
            error of an executor.
        data: The error event.
    """

    def __init__(self, code: str, message: str, data: JsonObject | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data: JsonObject = data or {}
