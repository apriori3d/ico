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
from ico.remote.job import AnyJob, FetchJob, Job, Package, RemoteError, load_job
from ico.remote.runner import fetch_job, run_job
from ico.remote.runpod import RunPod, RunPodJob
from ico.remote.runpod_api import RunPodApi, RunPodApiError, Volume
from ico.remote.storage import FileStorage, Storage, mount_bucket, register_storage

__all__ = [
    "AnyJob",
    "Executor",
    "FetchJob",
    "FileStorage",
    "Job",
    "JobHandle",
    "LocalExecutor",
    "LocalJob",
    "Package",
    "RemoteError",
    "RunPod",
    "RunPodApi",
    "RunPodApiError",
    "RunPodJob",
    "Volume",
    "Storage",
    "register_storage",
    "fetch_job",
    "load_job",
    "mount_bucket",
    "result",
    "run_job",
    "submit",
]
