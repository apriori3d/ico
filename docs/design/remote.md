# Remote runs: a spec on a GPU in the cloud

Date: 2026-10-20. Status: design for review (ico roadmap items 22–25,
ico-mcp item 32).

## Why

A run of the video domain (one 30 s video) takes about 15 minutes on one
GPU: OCR 4–6 min on the CPU, DiffuEraser up to 15 min, the rest seconds.
Videos are independent, so the throughput grows with the number of GPUs.
A serverless GPU queue (RunPod Serverless) gives 0..N GPUs, billed per
second while a worker runs, nothing while it is scaled down: about
$0.2–0.3 per video on a 24 GB GPU (3090/A5000/4090), less with OCR on the
GPU. Problem scenes can go to 48–80 GB GPUs (better models, bf16, more
steps) for about $1 a scene.

Two ways to start a run must both work:

- **from ICO** (a script, later the Telegram bot): `submit(spec, input,
  params, target)`; the operators are in the image;
- **from ico-mcp** (an agent): `run(..., target="cloud")`; the operators
  are drafts on the agent's machine, often seconds old.

## The unit: a job

A job is one run of `ico.spec.run`, the same thing the ico-mcp kernel does
today, moved into ICO so that both sides share it.

**Request** (JSON):

```jsonc
{"spec": {...},                   // inlined: no pipeline_ref
 "input": "s3://bucket/in/01.mp4",
 "params": {"work_dir": "$JOB/work", "out_dir": "$JOB/out"},
 "packages": [{"path": "video", "status": "stable"},
              {"path": "video_draft", "status": "draft",
               "modules": ["text_scan", "text_clean"]}],
 "sources": {"video_draft/text_scan.py": "...", ...},   // optional
 "options": {"profile": false, "recent": true, "max_items": 20},
 "storage": {"out": "s3://bucket/jobs/<id>/"}}
```

**Events** (JSON, in order): `plan` (the plan view, first), then any of
`progress`, `profile`, `sample` (recent previews) as snapshots, then one
`result` (`RunResult.to_dict`, profile, draft fingerprints) or `error`
(`spec_error`, `run_error` with a traceback, `cancelled` with the result
so far). These are the snapshots the ico-mcp kernel sends today, so the
Live Board and the MCP tools show a remote run as they show a local one.

## The runner (`ico.remote.runner`)

Runs one job in a work directory (`$JOB`):

1. **Code.** `sources` are written into `$JOB/code/<package>/...`
   (code by value: the drafts of ico-mcp); packages without sources are
   looked up in the image. `load_operators(path, status=, modules=)` loads
   what the spec uses.
2. **Stage in.** Every string of `input` and `params` that is a storage
   URI (`s3://...`) is downloaded into `$JOB/in/` and replaced by the local
   path (directories by prefix). `$JOB` in a string is the work directory.
3. **Run.** `ico.spec.run` with `JsonProgressTool`, `JsonSampleTool`
   (recent), the profiler if asked, and a cancel token; events go to an
   `emit` callback.
4. **Stage out.** Files the outputs point to and everything under
   `$JOB/out` are uploaded to `storage.out`; paths in the result are
   replaced by their URIs.
5. **Resume.** The work directory may be a persistent volume
   (`storage.work`): a job sent again with the same id continues from the
   files its operators left (VACE windows, cleaned scenes) — this is how a
   run is "paused": cancel it (billing stops), send it again later.

The runner knows nothing of a provider: it reads a request and calls
`emit`.

## Executors (`ico.remote.executor`)

```python
class Executor(Protocol):
    def submit(self, job: Job) -> JobHandle: ...

class JobHandle(Protocol):
    def events(self) -> Iterator[JsonObject]: ...   # until result/error
    def cancel(self) -> None: ...
```

- `LocalExecutor`: the runner in a subprocess (`python -m
  ico.remote.runner`, request on stdin, events as JSON lines on stdout).
  For tests and for running the same job on cube; it replaces the run part
  of the ico-mcp kernel later.
- `RunPodExecutor`: `POST /run` with the request, events from `/stream`
  (the handler is a generator that yields events), `POST /cancel`;
  standard library HTTP, no SDK. The worker side is
  `ico.remote.runpod_handler`, the image's entry point.
- Modal or a plain SSH pod later, by the same protocol.

## Storage (`ico.remote.storage`)

`get(uri, path)` / `put(path, uri)` for `s3://` URIs on an S3-compatible
service (Cloudflare R2: no egress fees), and `file://` for tests. `boto3`
is an optional extra (`ico[remote]`); credentials come from the
environment, never from a request.

## Targets

A target names where jobs go: provider, endpoint, storage, timeouts.

```toml
[targets.default]   # 24 GB: 4090 → 3090/A5000
provider = "runpod"
endpoint = "abc123"
storage = "s3://ico-video"
timeout = 10800      # execution timeout, s
[targets.hard]      # problem scenes: A100/H100
provider = "runpod"
endpoint = "def456"
storage = "s3://ico-video"
```

ICO takes a target as an object; ico-mcp reads them from `ico.toml`.
Secrets (RunPod key, S3 keys) are environment variables.

## Scenario 1: from ICO

```python
from ico.remote import RunPod, submit
job = submit(spec, "s3://ico-video/in/01.mp4", params=..., target=RunPod(...),
             packages=[Package("video", "stable")])
for event in job.events(): ...
result = job.result()          # outputs as s3:// URIs
```

The operators are in the image (built from the domain repository). A
pipeline saved by ico-mcp is a file: the script inlines it
(`ico.spec.inline` with a library over `pipelines/`).

## Scenario 2: from ico-mcp

`run(pipeline=..., input=..., target="default")`:

- the server inlines the spec, asks the main kernel for the modules it
  uses (item 20) and puts their **sources** into the job (with the
  closure of their relative imports): a draft written a minute ago runs
  in the cloud without rebuilding the image; the image is the environment
  only and is rebuilt when dependencies change;
- local paths in `input` / `params` that exist on the server's machine
  are uploaded under `<storage>/runs/<run_id>/in/` (size limit in
  `ico.toml`); the outputs are downloaded into `runs/<run_id>/out/`;
- `run_status`, `wait_run`, `cancel_run`, the dashboard and traces work as
  for a local run; `sample_run` gives `recent` only (a job has no input
  channel).

## The image (domain repository)

`ico-mcp init` adds a `Dockerfile`: the domain's environment (`uv sync`),
`ffmpeg`, CUDA base; entry point `python -m ico.remote.runpod_handler`.
Model weights come from RunPod's model cache (Hugging Face) or a network
volume, not from the image. OCR runs on the GPU there (`paddlepaddle-gpu`
in its own worker process).

## Costs (RunPod, 2026-10)

| GPU | memory | $/h |
|---|---|---|
| L4 / A5000 / 3090 | 24 GB | 0.68 |
| 4090 PRO | 24 GB | 1.12 |
| A6000 / A40 | 48 GB | 1.22 |
| A100 | 80 GB | 2.74 |
| H100 PRO | 80 GB | 4.18 |

Billed from a worker's start (cold start: loading models) to its stop,
including the idle timeout (default 5 s); a scaled-down worker is free.
Defaults to change: execution timeout 10 min (to hours), max workers 3
(5 without a deposit). Job results are kept 30 min, so outputs always go
to storage.

## Order

1. ico 22: job, events, runner, `LocalExecutor`, `file://` storage — no
   network, tested end to end on the fixture registry.
2. ico 23: S3 storage (`ico[remote]`), stage in / out of directories.
3. ico 24: `RunPodExecutor` and `runpod_handler`, tested against a fake
   RunPod HTTP server.
4. ico 25: resume and cancel semantics (work on a volume, job ids).
5. ico-mcp 32: targets in `ico.toml`, `run(target=)`, code by value,
   upload / download of local files.
6. Domain: Dockerfile, endpoint, one real video; measure.

## Not now

- A remote worker inside a pipeline (scenes of one video on several GPUs
  at once): only if one video must be faster. Per-scene jobs to the
  `hard` target cover problem scenes.
- `sample` on request in the cloud (needs an input channel).
