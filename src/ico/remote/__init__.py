"""Remote runs: a spec as a job, run by a runner anywhere (a subprocess, a
cloud GPU). See ``docs/design/remote.md``.

Example:
    >>> job = submit(spec, "file:///data/a.mp4", executor=LocalExecutor(),
    ...              packages=[Package("video", "stable")])
    >>> for event in job.events(): ...
"""

from ico.remote.executor import (
    Executor,
    JobHandle,
    LocalExecutor,
    LocalJob,
    result,
    submit,
)
from ico.remote.job import Job, Package, RemoteError
from ico.remote.runner import run_job
from ico.remote.storage import FileStorage, Storage, register_storage

__all__ = [
    "Executor",
    "FileStorage",
    "Job",
    "JobHandle",
    "LocalExecutor",
    "LocalJob",
    "Package",
    "RemoteError",
    "Storage",
    "register_storage",
    "result",
    "run_job",
    "submit",
]
