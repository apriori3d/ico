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
 "options": {"profile": false, "recent": true, "max_items": 20,
             "cleanup": true},
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
   path (directories by prefix). `$JOB` in a string is the work directory
   and `$RUN` the job's id, which is also `ICO_RUN_ID` in the runner's
   environment (the worker processes of the run inherit it): a per-run
   directory outside the job, e.g. `/tmp/ico-work/$RUN` on a warm worker.
3. **Run.** `ico.spec.run` with `JsonProgressTool`, `JsonSampleTool`
   (recent), the profiler if asked, and a cancel token; events go to an
   `emit` callback.
4. **Stage out.** Files the outputs point to and everything under
   `$JOB/out` are uploaded to `storage.out`; paths in the result are
   replaced by their URIs.
5. **Resume.** The work directory may be a persistent volume
   (`storage.work`): a job sent again with the same id continues from the
   files its operators left (VACE windows, cleaned scenes) — this is how a
   run is "paused": cancel it (billing stops), send it again later
   (`submit(..., id=...)`). URIs come back to the local paths they were
   downloaded to (`$JOB/in/.staged.json`), without downloading them again,
   so operators that key their files by input path find them. One runner
   at a time holds a work directory (`flock` on `$JOB/.lock`, released when
   the process dies); a second one ends with the error `busy`. On RunPod
   the work directories go to the endpoint's network volume
   (`/runpod-volume/ico-jobs`) when one is mounted. A RunPod cancel loses
   the outputs so far (RunPod stops the handler), not the files.
6. **Cleanup.** A job with `options.cleanup` removes its work directory
   when it ends, after the stage out, whether it succeeded, failed or was
   cancelled (it cannot be resumed then); on RunPod the handler removes it
   for a runner it had to kill. Without it the directory stays: a
   network volume fills up with them.
7. **Status.** A job with `status` (a storage URI; ico-mcp gives
   `runs/<run_id>/result.json` beside `in/` and `out/`) also gets its
   final event written there, the same JSON as on the stream, after the
   stage out: a client that lost the stream learns the outcome and the
   outputs from the storage alone
   (`docs/proposals/2026-10-26-remote-results.md`). A `busy` error is not
   written, and a failed write does not fail the job.

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
  ico.remote`, request on stdin, events as JSON lines on stdout).
  For tests and for running the same job on cube; it replaces the run part
  of the ico-mcp kernel later.
- `RunPodExecutor`: `POST /run` with the request, events from `/stream`
  (the handler is a generator that yields events) and a `queue` event
  whenever RunPod's status of the job changes (`IN_QUEUE` while it waits
  for a worker, `IN_PROGRESS`), `GET /status`, `POST /cancel`;
  standard library HTTP, no SDK. Reads and cancels are retried for
  `ICO_REMOTE_RETRY` seconds (600 by default) on dropped connections, DNS
  failures and HTTP 429/5xx, then raise `RuntimeError`; `RunPod.job(id)`
  is the handle of a job submitted earlier, by RunPod's id (a client that
  restarted reads its events on). The worker side is
  `ico.remote.runpod_handler`, the image's entry point. It runs jobs in a
  **job server** (`python -m ico.remote --serve`, `ico.remote.server`):
  one process that lives between jobs, so a warm worker keeps its
  resource pool, imported operators and CUDA context. The server loads a
  job's code once, into a directory of its own, and runs only jobs with
  the same code fingerprint (packages and sources); the handler starts a
  new server for other code, after `run_error` or `load_error`, after
  `ICO_SERVE_MAX_JOBS` jobs, and after a crash or a kill (a cancel that
  did not stop within the grace period). A cancelled job that stops
  keeps the server. See `docs/proposals/2026-10-25-job-server.md`.
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
Model weights are not in the image (next section). OCR runs on the GPU
there (`paddlepaddle-gpu` in its own worker process).

## Assets on the remote side

Weights are assets (`docs/proposals/2026-10-20-assets-and-resources.md`):
pinned to a source revision and a sha256 in the domain's `assets.toml` /
`assets.lock`. So they are fetched again from their source into the
cloud, never uploaded from a machine that has them: cube uploads at a
home connection's speed (117 GB: hours), a datacenter reads Hugging Face
far faster, and the lock proves both copies equal.

- **Where.** A RunPod network volume, mounted at `/runpod-volume` on
  serverless workers; assets under `/runpod-volume/assets` (the worker's
  `ICO_ASSETS_ROOT`), work directories beside them. Standard tier $0.07
  per GB and month (117 GB ≈ $8/month); reads 200–400 MB/s.
- **Datacenter.** A volume binds the endpoint to its datacenter. More
  datacenters: one volume each (RunPod allows one per datacenter and does
  not sync them), filled by its own prefetch from the same source.
- **Prefetch job.** A job of its own kind on the same endpoint: the
  manifest and lock travel in the job, the worker runs `ico.assets.fetch`
  on the volume and streams `fetch` events (bytes per asset) and a result
  with the state of each asset; resumable and verified as on any machine.
  One prefetch per volume at a time: RunPod warns against concurrent
  writes from several workers, and `flock` on a network volume is not
  documented. Cost: the worker's time while downloading, about $0.1–0.3
  for the full set, once per volume.
- **A volume per order.** The volume exists only while there is work:
  `RunPodApi.volume_up(endpoint, size, datacenter)` creates it and
  attaches it to the endpoint (`PATCH /endpoints/<id>`: `networkVolumeId`
  and `dataCenterIds`), then a prefetch fills it; `volume_down` detaches
  and deletes it with everything on it (the outputs were brought back by
  then). `ico runpod volume up|down|list` and `ico runpod prefetch` do the
  same from a shell. Keeping 150 GB costs about $10.5 a month; filling it
  again about $0.2–0.3 and 15–20 minutes: per order unless orders come
  every few days.
- **The volume is also the storage** of runs: its S3 API
  (`https://s3api-<datacenter>.runpod.io`, bucket = the volume id, RunPod
  S3 keys in `AWS_*`) takes the inputs from outside and gives the outputs
  back; a job names its volume (`Job.volume`) and the worker reads and
  writes `s3://<volume>/...` at its mount `/runpod-volume` directly,
  without the network or keys. No second storage provider.
- **Runs never fetch.** A run job carries the manifest in its `sources`,
  so `asset(id)` resolves on the volume; a missing asset fails at the
  operator that loads it. ico-mcp checks `spec_assets` against the
  target before a run and proposes a prefetch with its size (the human's
  word, as `fetch_assets`).
- **Not RunPod's cached models.** They hold one Hugging Face model per
  endpoint, a whole repository (all its files), without a documented
  revision; our sets are subsets of several repositories at fixed
  commits.
- **Files without a public source** (own fine-tuned weights) are uploaded
  once to S3 and get an `s3://` source in the manifest — when the first
  such file appears.
- **Cold starts.** Reading the Wan2.2 experts (~60 GB) from the volume
  takes minutes. The job server keeps the resource pool of a job's main
  process between jobs, so a warm worker loads those models once.
  Models in a worker process (`MPWorker`, e.g. a `gpu` worker) still load
  once per run: the worker's runner process lives one run (item 31,
  phase 2).

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
5. ico: prefetch jobs (assets on a volume).
6. ico-mcp 32: targets in `ico.toml`, `run(target=)`, code by value,
   upload / download of local files, assets checked on the target.
7. Domain: Dockerfile, endpoint with a network volume, prefetch, one real
   video; measure (cold start included).

## Not now

- A remote worker inside a pipeline (scenes of one video on several GPUs
  at once): only if one video must be faster. Per-scene jobs to the
  `hard` target cover problem scenes.
- `sample` on request in the cloud (needs an input channel).
