# Proposal: results of remote runs without a live stream

Date: 2026-10-26. Status: proposal, nothing implemented. It touches ico
(`ico.remote`: the runner, the RunPod executor) and ico-mcp (`remote.py`,
`jobs.py`, `traces.py`, a new tool). Each item of section 6 is a separate
PR.

## 1. Summary

The client learns that a remote run ended, and what it produced, from a
single place: the final `result` or `error` event read over RunPod's
`GET /stream/<id>`. If the client loses that stream for about 30 s, the
run fails locally. The job keeps running on the worker, its outputs land
in the storage, and its work directory is removed. But nothing on the
client records where to look. The RunPod job id is not saved anywhere,
and the result (outputs, summary, profile) exists only in the lost event.

The proposal:

1. The runner writes the final event to the storage too:
   `runs/<run_id>/result.json`, next to `in/` and `out/`.
2. The client saves the RunPod job id and the storage prefix as soon as
   the job is submitted.
3. A dropped stream is retried for minutes, not seconds.
4. ico-mcp gets a `fetch_run(run_id)` tool that finishes such a run from
   `result.json`, or reattaches to the stream while the job still runs.

## 2. Case

Run `fb99e719fe57` (`clean_video`, video 01, target `runpod`, ico-mcp
0.1.2, 2026-10-25):

- At 30 of 31 scenes, cube's DNS (through a VPN) failed for
  `api.runpod.ai`. The run failed on cube with `URLError: Temporary
  failure in name resolution`, raised in `RunPodJob.events()` →
  `RunPod.call("GET", "/stream/…")`.
- The worker finished the job, uploaded `$JOB/out`, and removed
  `ico-jobs/fb99e719fe57` (checked on the volume).
- The client has no RunPod job id for the run. Its trace line in
  `runs/2026-10-25.jsonl` has no run id and no target, so the run cannot
  even be found by its id in `runs/`. ico-mcp has no tool to fetch
  anything afterwards.

## 3. How it works today

Ids:

| id | made by | where it lives |
|---|---|---|
| ico-mcp `run_id` (12 hex) | `jobs.py` (`Job.id`) | the in-memory job table, the board, `runs/<run_id>/out/` (outputs only); **not** in the trace line |
| ico `Job.id` | the same value (`remote.py:88`, `id=run_id`) | the job JSON; the worker's `ico-jobs/<id>`; `ICO_RUN_ID` |
| RunPod job id | `POST /run` (`runpod.py:67`) | `RunPodJob._id`, in memory only |

Submit (ico-mcp `remote.run_remote`, `src/ico_mcp/remote.py:35`):

- Local inputs are uploaded to `<storage>/runs/<run_id>/in/` (`:62`).
- The outputs go to `out_uri = <storage>/runs/<run_id>/out` (`:72`).
  This key is deterministic, but it is not documented and not persisted.
- `RunPod.submit` posts the job (`ico/remote/runpod.py:58`) and returns a
  `RunPodJob` with RunPod's id. Nothing writes that id down.

Worker (`ico/remote/runpod_handler.py`, `runner.py`):

- The handler starts `python -m ico.remote --workdir ico-jobs/<id>` and
  yields every JSON line of its stdout as a stream chunk
  (`runpod_handler.py:63`). It is a generator handler without
  `return_aggregate_stream`, so `/status` carries no output, only the
  status.
- The runner (`runner.py:154 _run_locked`) sends `plan`, `progress`,
  `profile` and `sample` events. At the end it uploads `<workdir>/out` to
  `job.out` (`_stage_out`, `:332`), rewrites output paths to URIs, and
  emits one `result` event (outputs, totals, errors, `profile`, `:219`).
  Errors become one `error` event (`run_job`, `:73`).
- `cleanup` removes the work directory in a `finally` after that
  (`:128`). The storage objects are never removed by anyone.

Client (`RunPodJob.events`, `runpod.py:111`; `run_remote`, `remote.py:100`):

- It polls `GET /stream/<id>` and yields chunks until a `result` or
  `error`. If RunPod says `COMPLETED`/`FAILED`/… without our final event,
  it asks `/status` and makes an `error`.
- `RunPod.call` retries a GET 5 times on `URLError`/`OSError`, with
  backoff of 1, 2, 4 and 8 s plus jitter. So it gives up after about
  15–20 s of a DNS failure, which fails at once, or after about 4–5 min
  of timeouts. Then the `URLError` propagates. `run_remote` catches only
  `RuntimeError`, so the kernel reports `internal`. That is what
  `fb99e719fe57` shows.
- On `result`, `_download` copies URIs under `out_uri` into
  `runs/<run_id>/out/` and replaces them by local paths (`:117`).
- `jobs._run` then writes the trace line (`jobs.py:383`,
  `traces.write`): spec, input, params, result summary, error, progress
  and profile, without `run_id`, `target` or the RunPod id. The
  in-memory job is evicted later; after a server restart only the trace
  remains.

What survives a lost stream:

- the outputs, under `<storage>/runs/<run_id>/out/` (found by listing);
- the RunPod job, for its retention (`/status` keeps a finished async
  job about 30 min; whether `/stream` still returns chunks the client
  did not read is not documented and needs a test);
- nothing else: the result summary, totals, item errors and the final
  profile are gone, and so is the fact that the run succeeded.

## 4. Design

### 4.1 The runner writes the final event to the storage (ico)

- New optional `Job.status` field: a storage URI. ico-mcp sets it to
  `<storage>/runs/<run_id>/result.json`.
- In `run_job`/`fetch_job`, `emit` is wrapped so that the final event
  (`result` or `error`, the same JSON as the stream's) is also written
  there. For a `result`, this happens after `_stage_out`, so the outputs
  are in place before the marker exists.
- For a job killed by the handler (cancel grace, OOM), the handler
  writes an `error` (`cancelled` or `crashed`) if the runner did not.
- If writing the marker fails, the job does not fail: the stream still
  carries the event. The failure is logged to stderr.
- A `started` marker is optional: `runs/<run_id>/started.json` with
  RunPod's job id and the worker id, written by the handler. It shows
  that the job reached a worker.

Compatibility with the job server (`2026-10-25-job-server.md`): the
change sits in `run_job`, which the server calls unchanged. The handler
part, writing `crashed` for a killed process, goes into the handler of
that design. It does not depend on it, and it is the last PR of section 6
so that it lands after the server.

### 4.2 The client persists what it needs to come back (ico + ico-mcp)

- `RunPod` gains `job(id) -> RunPodJob`: a handle for an existing RunPod
  job, for reattaching. `RunPodJob` gains `status() -> JsonObject`
  (`/status`).
- ico-mcp writes `runs/<run_id>/remote.json` right after `submit`:
  target, endpoint, RunPod job id, `storage`, `out_uri`, `status_uri`,
  submit time. It also adds `run_id` and `target` to the trace line, so
  traces can be joined with runs.

### 4.3 Retries that outlast a short outage (ico)

- A GET on a transient error (`URLError`, `OSError`, HTTP 429/5xx) is
  retried by time, not by count: `ICO_REMOTE_RETRY` seconds (default
  600), backoff up to 30 s. A POST stays unretried, except `/cancel`,
  which is idempotent.
- RunPod's `/stream` hands out chunks once. Whether chunks produced
  during the outage come back on the next read is to be tested first
  (§7). If they do not, the run loses some progress and profile events
  but never its outcome, because the final event is also in
  `result.json` (4.1).
- `run_remote` catches `URLError`/`OSError` too, and reports
  `remote_error` with `{"run_id", "runpod_id", "status_uri"}`, so the
  agent knows the run can be fetched.

### 4.4 `fetch_run(run_id)` in ico-mcp

- It reads `runs/<run_id>/remote.json`. If `result.json` exists in the
  storage, it downloads the outputs as `_download` does and records the
  run as finished: board, trace, `run_status`.
- If there is no `result.json` and RunPod says the job is still queued
  or running, it reattaches (`RunPod.job(id).events()`) and goes on as a
  normal run (`wait_run`, dashboard).
- If RunPod says the job ended and there is no `result.json` (killed
  before 4.1 existed, or a crash), it reports `/status` and what is
  under `out/`.
- It also accepts a run with no `remote.json` (from before this change)
  when the target is given: it looks under
  `<storage>/runs/<run_id>/` only. That covers `fb99e719fe57`'s outputs,
  though without a summary.
- Optionally the server tries `fetch_run` by itself for runs that failed
  with `remote_error` (a background poll of `result.json` for up to
  `ICO_REMOTE_RETRY`). That is a fork, see §5.

## 5. Forks for the user to decide

1. **Source of truth for the outcome.**
   - *A (recommended):* `result.json` in the storage (4.1). It does not
     depend on the provider, outlives RunPod's retention, and works the
     same for an order volume or Timeweb.
   - *B:* `return_aggregate_stream=True` in the handler, so `/status`
     carries every event. No storage write, but it lasts only RunPod's
     retention (about 30 min), and a long run's aggregate holds every
     progress event, which can be large.
2. **Recovery: automatic or by a tool.**
   - *A (recommended):* longer retries (4.3) plus an explicit
     `fetch_run`.
   - *B:* also an automatic background fetch after a `remote_error`.
     That is more magic, and a run "failed" in the trace may later turn
     into a success.
3. **Where the client keeps what it needs:** `runs/<run_id>/remote.json`
   (recommended, one file per run) or a column in the daily trace,
   written at submit and again at the end.
4. **Storage retention.** Today nobody deletes `runs/<run_id>/` in S3,
   and with `result.json` there is a reason to keep it. Choose between a
   bucket lifecycle rule (e.g. 14 days, outside ico), a `cleanup` of the
   storage prefix after a successful download, or keeping everything.
5. **Default retry window** (`ICO_REMOTE_RETRY`): 600 s is proposed. A
   longer window keeps a run "running" through a long outage.

## 6. Plan of work (one PR each)

1. ico: `RunPod.job(id)`, `RunPodJob.status()`, time-based GET retries,
   `/cancel` retried; tests with the fake API (dropped connections,
   DNS errors).
2. ico: `Job.status` and the final event written there by the runner;
   tests with `file://` storage.
3. ico-mcp: `remote.json` at submit; `run_id` and `target` in traces;
   `remote_error` for transport errors, carrying the ids.
4. ico-mcp: the `fetch_run` tool and guide text (§5 forks as decided).
5. ico: the handler writes `crashed`/`cancelled` for a killed runner;
   after the job server lands.
6. Releases, then ico-mcp-video locks them; a test on RunPod with a
   network cut on the client in the middle of a run.

## 7. To check before building

- Does `GET /stream/<id>` return chunks produced while nobody read, or
  only new ones? Does it answer at all after the job ends, and for how
  long?
- RunPod's retention of `/status` for an async job on this endpoint.
- Whether Timeweb S3 allows a lifecycle rule on a prefix, if fork 4
  picks that.
